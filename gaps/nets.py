import torch
import torch.nn as nn
from lightning.pytorch import loggers
from networks.timeseries import MambaSignalVectorEncoder
from cwgan import CWGAN


class Generator(MambaSignalVectorEncoder):
    def __init__(self, signal_shape, params_size, z_dim=32, **kwargs):
        self.z_dim = z_dim
        super().__init__(
            signal_shape=signal_shape,
            vector_size=z_dim,
            encoded_size=params_size,
        )

    def sample_z(self, signal):
        z = torch.randn(signal.size(0), self.z_dim, device=signal.device)
        return z

    def forward(self, signal, z=None):
        if z is None:
            z = self.sample_z(signal)
        return super().forward(signal, z)


class Critic(MambaSignalVectorEncoder):
    def __init__(self, signal_shape, params_size, **kwargs):
        super().__init__(
            signal_shape=signal_shape,
            vector_size=params_size,
            encoded_size=1,
        )


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

        scores_real = self.critic(signal, params_real)
        scores_generated = self.critic(signal, params_generated)

        if isinstance(self.logger, loggers.WandbLogger):
            params_log_dict = {
                f"{prefix}/{name}": d
                for prefix, data in (
                    ("Real", params_real),
                    ("Generated", params_generated),
                    ("Delta", params_generated - params_real),
                )
                for name, d in zip(self.parameter_names, data.T)
            }

            scores_log_dict = {
                "Score/Real": scores_real,
                "Score/Generated": scores_generated,
            }

            self.logger.experiment.log({**params_log_dict, **scores_log_dict})

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
