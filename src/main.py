# ---------------------------------------------------------------------------
# Silence pytorch's "triton not found; flop counting will not work for triton
# kernels" log line.
# This MUST stay before `import torch`.
import logging as _logging
_logging.getLogger('torch.utils.flop_counter').setLevel(_logging.ERROR)


from pytorch_lightning import seed_everything
import xarray as xr
import torch
from einops import rearrange
from torch.nn.functional import mse_loss as mse
from pytorch_lightning.callbacks import ModelCheckpoint

# config
from config.constants import COORD, TARGET
from config.opts import get_opts

# datasets
from datasets.spherical_reg import SphericalDataset
from base_coord_system import BaseCoordSystem, run_main
from metrics.psnr import psnr
from metrics.breakdowns import compute_psnr_breakdowns


class ImgRegCoordSystem(BaseCoordSystem):
    def setup(self, stage=None):
        ce_kwargs = dict(getattr(self.hparams, 'encoding_kwargs', {}) or {})
        held_out_path = getattr(self.hparams, 'held_out_path', '') or None
        self.dataset = SphericalDataset(
            self.hparams.data_path,
            coordinate_encoding=self.hparams.ce,
            encoding_kwargs=ce_kwargs,
            held_out_file_path=held_out_path)

    def training_step(self, batch, batch_idx):
        pred = self(batch[COORD])['model_out']
        
        loss = mse(pred, batch[TARGET])
        psnr_ = psnr(pred, batch[TARGET]) 

        self.log('lr', self.opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch[COORD])['model_out']
        
        loss = mse(pred, batch[TARGET], reduction='none')

        log = {'val_loss': loss,
               'gt': batch[TARGET],
               'pred': pred
            }
        
        self.validation_step_outputs.append(log)

    def on_validation_epoch_end(self):
        mean_loss = torch.cat([x['val_loss'] for x in self.validation_step_outputs]).mean()
        mean_psnr = -10 * torch.log10(mean_loss)
        # Reshape flat (N, C) → (C, H, W) for TensorBoard image logging.
        # The H, W come from the dataset itself.
        H, W = self.dataset.height, self.dataset.width
        gt = torch.cat([x['gt'] for x in self.validation_step_outputs])
        gt = rearrange(gt, '(h w) c -> c h w', h=H, w=W)
        pred = torch.cat([x['pred'] for x in self.validation_step_outputs])
        pred = rearrange(pred, '(h w) c -> c h w', h=H, w=W)

        self.logger.experiment.add_images('val/gt_pred',
                                          torch.stack([gt, pred]),
                                          self.global_step)

        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/psnr', mean_psnr, prog_bar=True)

        if self.hparams.save_vis:
            import os
            from torchvision.utils import save_image
            pred_path = os.path.join(self.logger.log_dir, "pred")
            os.makedirs(pred_path, exist_ok=True)
            
            metrics_txt = []
            metrics_txt.append(f"epoch: {self.current_epoch}\n")
            metrics_txt.append(f"val/psnr:  {mean_psnr}\n\n")

            with open(os.path.join(self.logger.log_dir, "metrics.txt"), "a") as file:
                file.writelines(metrics_txt)

            save_image(pred, os.path.join(pred_path, f"pred_{self.current_epoch}.png"))
        
        self.validation_step_outputs.clear()  # free memory
        
    
    # ----- End-of-training evaluation -----------------------
    
    def _eval_in_batches(self, coords: torch.Tensor) -> torch.Tensor:
        """Run the model on flat (N, D) coords in batches; return (N, C) on CPU."""
        self.eval()
        batch_size = int(self.hparams.batch_size)
        N = coords.shape[0]
        preds = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                x = coords[i:i + batch_size].to(self.device, non_blocking=True)
                out = self(x)['model_out'].detach()
                preds.append(out.cpu())
        return torch.cat(preds, dim=0)
    
    
    def on_train_end(self):
        """End-of-training evaluation against the best checkpoint.

        Computes two PSNR metrics, each with overall / polar / equatorial /
        per-channel breakdowns:

          * reconstruction_psnr: over the training pixels (the full (H,W) grid).
            INR-Bench-comparable — same set the loss was computed on.
          * held_out_psnr: over the ((H-1)×(W-1)) half-pixel-offset grid, with
            bilinear-interp ground truth (preregistration §3.4). Pure
            generalization metric — the model never saw these positions.

        The best checkpoint is selected by `val/loss` (training-pixel loss).
        Held-out PSNR is NEVER used to drive checkpoint or early-stop
        decisions — that would constitute test-set leakage.

        All scalar metrics are stashed on `self` so the CSV-logging callback
        can pick them up without re-running inference.
        """
        # ---- 1. Load best checkpoint ----
        best_path = None
        for cb in self.trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                best_path = getattr(cb, 'best_model_path', None) or None
                break
        if best_path:
            print(f"[on_train_end] Loading best checkpoint: {best_path}")
            state = torch.load(best_path, map_location=self.device,
                               weights_only=False)
            self.load_state_dict(state['state_dict'])
        else:
            print("[on_train_end] No best checkpoint; using final model state.")

        # ---- 2. Reconstruction (training-pixel) evaluation ----
        # Re-run inference over training pixels with the BEST weights, so the
        # reconstruction metric corresponds to the checkpointed model — not
        # the post-final-step model, which may differ slightly.
        train_preds = self._eval_in_batches(self.dataset.coords)
        recon = compute_psnr_breakdowns(
            train_preds,
            self.dataset.targets,
            self.dataset.lats_deg,
            self.dataset.width,
        )
        self.reconstruction_psnr = recon['overall']
        self.reconstruction_psnr_polar = recon['polar']
        self.reconstruction_psnr_equatorial = recon['equatorial']
        if self.dataset.num_channels == 3:
            self.reconstruction_psnr_r = recon['channel_r']
            self.reconstruction_psnr_g = recon['channel_g']
            self.reconstruction_psnr_b = recon['channel_b']

        # ---- 3. Held-out (offset-grid) evaluation ----
        coords_held, targets_held = self.dataset.make_held_out_eval()
        held_preds = self._eval_in_batches(coords_held)
        held = compute_psnr_breakdowns(
            held_preds,
            targets_held,
            self.dataset._held_out_lats_deg,
            self.dataset.held_out_width,
        )
        self.held_out_psnr = held['overall']
        self.held_out_psnr_polar = held['polar']
        self.held_out_psnr_equatorial = held['equatorial']
        if self.dataset.num_channels == 3:
            self.held_out_psnr_r = held['channel_r']
            self.held_out_psnr_g = held['channel_g']
            self.held_out_psnr_b = held['channel_b']

        # ---- 4. Log to TensorBoard ----
        if self.logger is not None:
            for k, v in recon.items():
                self.logger.experiment.add_scalar(
                    f'reconstruction/psnr_{k}', v, self.global_step,
                )
            for k, v in held.items():
                self.logger.experiment.add_scalar(
                    f'held_out/psnr_{k}', v, self.global_step,
                )

        # ---- 5. Console summary ----
        print(f"\n[end-of-training metrics]")
        print(f"  reconstruction  overall={recon['overall']:7.3f} dB | "
              f"polar={recon['polar']:7.3f} | "
              f"equatorial={recon['equatorial']:7.3f}")
        if self.dataset.num_channels == 3:
            print(f"                  R={recon['channel_r']:7.3f}  "
                  f"G={recon['channel_g']:7.3f}  B={recon['channel_b']:7.3f}")
        print(f"  held_out        overall={held['overall']:7.3f} dB | "
              f"polar={held['polar']:7.3f} | "
              f"equatorial={held['equatorial']:7.3f}")
        if self.dataset.num_channels == 3:
            print(f"                  R={held['channel_r']:7.3f}  "
                  f"G={held['channel_g']:7.3f}  B={held['channel_b']:7.3f}")
        print(f"  held-out grid: ({self.dataset.held_out_height}×"
              f"{self.dataset.held_out_width}) = {coords_held.shape[0]} samples")


def main() -> None:
    """Training entrypoint."""
    hparams = get_opts()
    seed_everything(int(getattr(hparams, 'seed', 42)), workers=True)
    system = ImgRegCoordSystem(hparams)
    run_main(system, hparams)

if __name__ == '__main__':
   main()