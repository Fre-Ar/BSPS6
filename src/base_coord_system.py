import os
import abc
import torch
from torch.utils.data import DataLoader
from argparse import Namespace

# models
from models.INR import INR

# pytorch lightning
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

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
        return DataLoader(self.dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)
    

    def configure_optimizers(self):
        if self.hparams.opt == "adam":
            from torch.optim import Adam
            from torch.optim.lr_scheduler import CosineAnnealingLR

            self.opt = Adam(self.model.parameters(), lr=self.hparams.lr)
            scheduler = CosineAnnealingLR(self.opt, self.hparams.num_epochs, self.hparams.lr/1e2)

            return [self.opt] , [scheduler]
    
    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abc.abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abc.abstractmethod
    def on_validation_epoch_end(self):
        pass

    def compute_ntk(self, X):
        """Compute the Neural Tangent Kernel (NTK).
        x: [B, N]
        """
        X.requires_grad = True
        jacobian = torch.autograd.functional.jacobian(lambda x: self.model(x), X) # [B, out_dim, in_dim]
        ntk = jacobian.permute(0, 2, 1) @ jacobian # [B, out_dim, out_dim]
        
        return ntk.mean(dim=0) # [out_dim, out_dim]
    
    def compute_ntk_eigenvalues(self, ntk):
        if ntk.is_cuda:
            eigenvalues, _ = torch.linalg.eigvalsh(ntk)
        else:
            from scipy.linalg import eigh
            eigenvalues = eigh(ntk.cpu().numpy())[0]
        
        return eigenvalues

def run_main(system: BaseCoordSystem, hparams: Namespace):
    seed_everything(42, workers=True)

    torch.set_float32_matmul_precision('high')

    logger = TensorBoardLogger(save_dir=hparams.save_dir,
                               name=f"{hparams.kan_act}_{hparams.mlp_act}" if hparams.arch == "kamp" else hparams.act,
                               default_hp_metric=False)

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

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=True,
                      accelerator='auto',
                      devices=1,
                      num_sanity_val_steps=0,
                      log_every_n_steps=1,
                      check_val_every_n_epoch=hparams.check_val_every_n_epoch,
                      benchmark=True)

    trainer.fit(system)