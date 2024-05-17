import math
import torch
import torch.nn as nn
from lightning.pytorch import loggers
from networks.utils import MLP
from cwgan import CWGAN


class Generator(nn.Module):
    def __init__(
        self, signal_shape, params_size, encoded_size=128, latent_size=32, **kwargs
    ):
        super().__init__()
        self.latent_size = latent_size
        self.mlp = MLP(
            input_dim=math.prod(signal_shape),
            output_dim=encoded_size,
            num_layers=2,
            spectral_norm=False,
        )
        self.out_proj = MLP(
            input_dim=latent_size + encoded_size,
            output_dim=params_size,
            num_layers=2,
            spectral_norm=False,
        )

    def forward(self, signal):
        z = torch.randn(signal.size(0), self.latent_size, device=signal.device)
        x = self.mlp(signal)
        x = torch.cat([x, z], dim=-1)
        x = self.out_proj(x)
        return x


class Critic(nn.Module):
    def __init__(self, signal_shape, params_size, encoded_size=128, **kwargs):
        super().__init__()
        self.mlp = MLP(
            input_dim=math.prod(signal_shape),
            output_dim=encoded_size,
            num_layers=2,
            spectral_norm=True,
        )
        self.out_proj = MLP(
            input_dim=params_size + encoded_size,
            output_dim=1,
            num_layers=2,
            spectral_norm=True,
        )

    def forward(self, signal, params):
        x = self.mlp(signal)
        x = torch.cat([x, params], dim=-1)
        x = self.out_proj(x)
        return x


class SinCWGAN(CWGAN):
    def __init__(self, signal_shape, parameter_names, **kwargs):
        generator = Generator(signal_shape, len(parameter_names), **kwargs)
        critic = Critic(signal_shape, len(parameter_names), **kwargs)
        super().__init__(generator, critic, **kwargs)
        self.parameter_names = parameter_names
        self.save_hyperparameters()

    def validation_step(self, batch, batch_idx):
        params_real, signal = batch
        params_generated = self.generator(signal)

        if isinstance(self.logger, loggers.WandbLogger):
            log_dict = {
                f"{prefix}/{name}": d
                for prefix, data in (
                    ("Real", params_real),
                    ("Generated", params_generated),
                    ("Delta", params_generated - params_real),
                )
                for name, d in zip(self.parameter_names, data.T)
            }
            self.logger.experiment.log(log_dict)
