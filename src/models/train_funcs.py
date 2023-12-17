import os
from argparse import Namespace
from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

#os.environ["WANDB_DISABLE_SERVICE"]="True"

def train_model(model: pl.LightningModule, hparams: Namespace) -> Tuple[pl.LightningModule, pl.Trainer]:

    wandb_logger = WandbLogger(project="Nigeria-final") if hparams.wandb else True
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=hparams.patience,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        default_root_dir=hparams.data_folder,
        max_epochs=hparams.max_epochs,
        early_stop_callback=early_stop_callback,
        gpus=hparams.gpus,
        logger=wandb_logger,
    )
    trainer.fit(model)

    return trainer
