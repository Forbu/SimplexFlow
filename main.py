from simplexflow import datasets


import torch
import torch.nn as nn

import lightning.pytorch as pl


from simplexflow.trainer import MnistTrainer

# import wandb key
import wandb

# wandb logger
from lightning.pytorch.loggers import WandbLogger

if __name__ == "__main__":

    pl_model_train = MnistTrainer(hidden_dim=128, num_bins=3, nb_time_steps=100)

    # initialize wandb
    wandb.login(key="662770e6ed60fbc120d72a32362283aa1fc4349b")
    dataset = datasets.DiscretizeMnist(num_bins=3)

    logger = WandbLogger(project="simplexflow")

    # train the model
    trainer = pl.Trainer(max_epochs=100, accelerator="auto", logger=logger)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=4
    )

    trainer.fit(pl_model_train, train_dataloaders=loader)
