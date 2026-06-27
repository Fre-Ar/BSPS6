"""PyTorch Lightning module + training-run helper for every benchmark cell.

The Lightning module is abstract (training/validation hooks are filled in by
`src/main.py:ImgRegCoordSystem`); `run_main` builds the Trainer with the
device-aware accelerator, the INR-Bench-compatible callback stack, and the
per-run CSV logger.
"""

import os
import abc
from argparse import Namespace
import torch
from torch.utils.data import DataLoader
from argparse import Namespace

# pytorch lightning
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# callbacks
from callbacks.runs_csv import RunsCSVLogger

# models
from models.INR import INR

from utils.device import (
    detect_device, lightning_accelerator, device_display_name, is_cuda,
)


def _dataloader_num_workers() -> int:
    """Per-platform default for DataLoader workers.

    CUDA on Linux/Windows benefits from a few workers feeding the GPU; MPS on
    macOS is single-process and tends to be slower with workers due to fork
    overhead; CPU runs anywhere want 0 (workers add memory pressure with no
    parallelism win since the model itself is on CPU).
    """
    d = detect_device()
    if d == 'cuda':
        return 4
    return 0

class BaseCoordSystem(LightningModule, abc.ABC):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        self.dataset = None

        self.model = INR(hparams)

        print("Model: ", self.model)

    def forward(self, x):
        return self.model(x)

    @abc.abstractmethod
    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=True,
            num_workers=_dataloader_num_workers(),
            batch_size=self.hparams.batch_size,
            pin_memory=is_cuda(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            shuffle=False,
            num_workers=_dataloader_num_workers(),
            batch_size=self.hparams.batch_size,
            pin_memory=is_cuda(),
        )
    
    def configure_optimizers(self):
        from torch.optim import Adam
        from torch.optim.lr_scheduler import CosineAnnealingLR
        # eta_min = lr/100, matching INR-Bench (preregistration §3.4).
        self.opt = Adam(self.model.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(
            self.opt, self.hparams.num_epochs, self.hparams.lr / 1e2,
        )
        return [self.opt], [scheduler]
    
    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abc.abstractmethod
    def on_validation_epoch_end(self):
        pass


def run_main(system: BaseCoordSystem, hparams: Namespace):
    """Top-level training entry point. Builds the Trainer and runs `.fit`."""
    
    seed = int(getattr(hparams, 'seed', 42))
    seed_everything(seed, workers=True)
    
    # CUDA-only matmul precision hint; harmless no-op elsewhere.
    if is_cuda():
        torch.set_float32_matmul_precision('high')
        
    print(f"[device] running on {device_display_name()}")

    logger = TensorBoardLogger(
        save_dir=hparams.save_dir,
        name=hparams.act,
        default_hp_metric=False,
    )
    
    pbar = TQDMProgressBar(refresh_rate=1)
    early_stopping_callback = EarlyStopping(
        monitor='val/loss',
        patience=5,
        verbose=True,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath=os.path.join(logger.log_dir, 'ckpt'),
        filename='best_model_{epoch:02d}',
        save_top_k=1,
        mode='min',
        verbose=False
    )

    callbacks = [pbar, early_stopping_callback, checkpoint_callback]
    
    # Append a per-run row to runs.csv. 
    # Disabled by setting --runs_csv to an empty string.
    runs_csv_path = getattr(hparams, 'runs_csv', '') or ''
    if runs_csv_path:
        callbacks.append(RunsCSVLogger(runs_csv_path))
    
    # Deterministic single-seed numbers. `deterministic='warn'`
    # downgrades non-deterministic-kernel errors to warnings (some interpolation
    # kernels we may hit during held-out eval lack a deterministic CUDA path).
    # We diverge from INR-Bench's `benchmark=True` on purpose.
    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=True,
        accelerator=lightning_accelerator(),
        devices=1,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        benchmark=False,
        deterministic='warn',
    )
    
    trainer.fit(system)