import numpy as np
import torch
from lisa import LisaDataModule, LisaCWGAN
from lightning.pytorch import callbacks
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class WandBCustomLogs(callbacks.Callback):
    def on_fit_start(self, trainer, pl_module):
        trainer.logger.watch(pl_module)  # type: ignore

    def on_fit_end(self, trainer, pl_module):
        trainer.logger.experiment.unwatch(pl_module)  # type: ignore

    def plot_contour(self, critic, x_fake, x_true, conditioning):
        x, y = x_fake[:, 0], x_fake[:, 1]
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        xmargin, ymargin = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
        xrange = torch.linspace(xmin - xmargin, xmax + xmargin, 32, device=x.device)
        yrange = torch.linspace(ymin - ymargin, ymax + ymargin, 32, device=x.device)
        xx, yy = torch.meshgrid(xrange, yrange)
        xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        z = critic(conditioning.expand(xy.shape[0], -1, -1), xy).reshape(xx.shape)

        x, y = x_fake[:, 0].cpu(), x_fake[:, 1].cpu()
        scatter = go.Scatter(x=x, y=y, mode="markers", marker=dict(color="black"))
        reference = go.Scatter(
            x=x_true[:, 0].cpu(), y=x_true[:, 1].cpu(), mode="markers"
        )
        contour = go.Contour(
            z=z.cpu(),
            x=xrange.cpu(),
            y=yrange.cpu(),
            colorscale="rdbu",
            colorbar=dict(title="Score"),
        )
        fig = go.Figure(
            data=[scatter, reference, contour], layout=dict(width=600, height=600)
        )
        return fig

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for prefix, data in {
            "Delta": outputs["Generated"] - outputs["Real"],
            "Real": outputs["Real"],
            "Generated": outputs["Generated"],
        }.items():
            log_dict = {
                f"{prefix}/{param}": d
                for param, d in zip(LisaDataModule.parameter_names, data.T)
            }
            pl_module.logger.experiment.log(log_dict)

        pl_module.logger.experiment.log(
            {k: v for k, v in outputs.items() if k.startswith("Scores/")}
        )
        self.on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        x_real, y = batch
        x_real, y = x_real[:1], y[:1]

        x_fake = pl_module.generator(y.expand(1024, -1, -1))
        fig = self.plot_contour(pl_module.critic, x_fake, x_real, y)
        pl_module.logger.experiment.log({"plot": fig})
