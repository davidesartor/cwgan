import warnings
from lightning import LightningModule, Trainer
from lightning.pytorch import loggers, callbacks
from matplotlib import pyplot as plt
import numpy as np
import plotly.figure_factory
import torch
from torch import nn
import seaborn as sns
import pandas as pd
import plotly

from vision.datamodules import MNIST
from networks.utils import MLP


def fairness_penalty(x: torch.Tensor, k=1024):
    # x: (..., batch_size, dim) -> latent representation for one group
    dim = x.shape[-1]

    # sample the evaluation points
    weighting_kernel = torch.distributions.Normal(0, 1)
    cov_root = 1 * torch.eye(dim).to(x.device)  # we can optimize this as a parameter
    t = cov_root @ weighting_kernel.sample((dim, k)).to(x.device)

    # evaluate characteristic functions at t
    phi_empirical = torch.exp(x @ t * 1j).mean(-2)  # (..., eval_points)
    phi_target = torch.exp(-0.5 * ((cov_root @ t) ** 2).sum(dim=0))  # (eval_points,)

    # compute distance between characteristic functions (integral norm of difference)
    loss = ((phi_empirical - phi_target).abs() ** 2).mean()
    return loss


class FairMLP(LightningModule):
    def __init__(self, theta_dim=2):
        super().__init__()
        self.target_classes = 5
        self.protected_classes = 2

        self.encoder = MLP(28 * 28, theta_dim, num_layers=2, preflatten=True)
        self.head = MLP(theta_dim, self.target_classes)

    def target(self, label):
        return label % self.target_classes

    def protected_attribute(self, label):
        return label % self.protected_classes

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def prediction_loss(self, theta, target):
        logits = self.head(theta)
        loss = nn.functional.cross_entropy(logits, target)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        self.log("task_accuracy", accuracy, prog_bar=True)
        self.log("task_loss", loss)
        return loss

    def fairness_loss(self, theta, protected):
        loss = sum(
            fairness_penalty(theta[protected == i])
            for i in range(self.protected_classes)
        )
        self.log("fairness_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        theta = self.encoder(x)
        loss = self.prediction_loss(theta, self.target(y))
        loss += self.fairness_loss(theta, self.protected_attribute(y))
        self.log("loss", loss)
        return loss


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer = Trainer(
            max_epochs=1000,
            devices="2,",
            callbacks=[
                callbacks.RichProgressBar(),
                callbacks.RichModelSummary(-1),
                callbacks.ModelCheckpoint(
                    "checkpoints/", monitor="fairness_loss", save_last=True
                ),
            ],
            logger=loggers.WandbLogger(project="fairmlp", log_model=True),
        )
        model = FairMLP()
        dm = MNIST(batch_size=2048, num_workers=32, pin_memory=True)
        trainer.fit(model, dm)
