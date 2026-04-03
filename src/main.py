import xarray as xr
import torch
from einops import rearrange
from torch.nn.functional import mse_loss as mse

# datasets
from datasets.spherical_reg import SphericalDataset, COORD, TARGET
from metrics import psnr
from base_coord_system import BaseCoordSystem, run_main
from config.opts import get_opts





class ImgRegCoordSystem(BaseCoordSystem):
    def setup(self, stage=None):
        self.dataset = SphericalDataset(hparams.data_path)

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
        mean_psnr = -10*torch.log10(mean_loss)
        gt = torch.cat([x['gt'] for x in self.validation_step_outputs])
        gt = rearrange(gt, '(h w) c -> c h w',
                           h=hparams.img_wh[1],
                           w=hparams.img_wh[0])
        pred = torch.cat([x['pred'] for x in self.validation_step_outputs])
        pred = rearrange(pred, '(h w) c -> c h w',
                             h=hparams.img_wh[1],
                             w=hparams.img_wh[0])

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


if __name__ == '__main__':
    
    #dataset = SphericalDataset(ELEVATION_DATA_PATH)
    #print("Dataset length: ", len(dataset))
    #sample = dataset[0]
    #print("Sample keys: ", sample.keys())
    #print("Coordinate shape: ", sample[COORD].shape)
    #print("Target shape: ", sample[TARGET].shape)
    
    hparams = get_opts()
    system = ImgRegCoordSystem(hparams)
    run_main(system, hparams)