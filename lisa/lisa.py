import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule

from lisa.nets import GRUEncoder, NormMLP
from cwgan import CWGAN


class LisaDataset(Dataset):
    def __init__(self, batch_size, n_batches=1):
        self.times = np.linspace(0, 1, 1024)
        self.examples = batch_size * n_batches

    def __len__(self):
        return self.examples

    def __getitem__(self, idx):
        A = 10 ** np.random.uniform(-1, 1)
        omega = 10 ** np.random.uniform(0, 1) * 2 * np.pi
        phi = 0.0  # np.random.uniform(0, np.pi)
        params = np.array([A, omega])

        h1 = A * np.sin(omega * self.times + phi)
        clean_signal = np.stack([h1], axis=-1)
        noise = np.random.normal(0, 1, clean_signal.shape)

        signal = torch.as_tensor(clean_signal + noise, dtype=torch.float32)
        params = torch.as_tensor(params, dtype=torch.float32)
        return params, signal


class LisaDataModule(LightningDataModule):
    parameter_names = ["A", "ω"]  # "φ"]
    channel_names = ["h+"]

    def __init__(self, batch_size=1024, num_workers=8, pin_memory=True, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        return DataLoader(LisaDataset(self.hparams["batch_size"], 128), **self.hparams)

    def val_dataloader(self):
        return DataLoader(LisaDataset(self.hparams["batch_size"]), **self.hparams)

    def test_dataloader(self):
        return DataLoader(LisaDataset(self.hparams["batch_size"]), **self.hparams)


class Generator(torch.nn.Module):
    def __init__(self, z_dim=1, encoded_dim=32, mlp_hidden=512, mlp_layers=2, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        # self.encoder = GRUEncoder(len(LisaDataModule.channel_names), encoded_dim)
        self.encoder = NormMLP(
            input_dim=1024 * len(LisaDataModule.channel_names),
            output_dim=encoded_dim,
            hidden_dim=mlp_hidden,
            num_layers=mlp_layers,
            norm=False,
        )
        self.mlp = NormMLP(
            input_dim=z_dim + encoded_dim,
            output_dim=len(LisaDataModule.parameter_names),
            hidden_dim=mlp_hidden,
            num_layers=mlp_layers,
            norm=False,
        )

    def sample_z(self, signal):
        factory_params = dict(device=signal.device, dtype=signal.dtype)
        return torch.randn(signal.size(0), self.z_dim, **factory_params)

    def forward(self, signal, z=None):
        signal = signal.reshape(signal.size(0), -1)
        x = self.encoder(signal)
        z = z or self.sample_z(signal)
        x = torch.cat([x, z], dim=-1)
        x = self.mlp(x)
        return x


class Critic(torch.nn.Module):
    def __init__(self, encoded_dim=32, mlp_hidden=512, mlp_layers=2, **kwargs):
        super().__init__()
        # self.encoder = GRUEncoder(len(LisaDataModule.channel_names), encoded_dim)
        self.encoder = NormMLP(
            input_dim=1024 * len(LisaDataModule.channel_names),
            output_dim=encoded_dim,
            hidden_dim=mlp_hidden,
            num_layers=mlp_layers,
            norm=False,
        )
        self.mlp = NormMLP(
            input_dim=len(LisaDataModule.parameter_names) + encoded_dim,
            output_dim=1,
            hidden_dim=mlp_hidden,
            num_layers=mlp_layers,
            norm=True,
        )

    def forward(self, signal, params):
        signal = signal.reshape(signal.size(0), -1)
        x = self.encoder(signal)
        x = torch.cat([x, params], dim=-1)
        x = self.mlp(x)
        return x


class LisaCWGAN(CWGAN):
    def __init__(self, **kwargs):
        super().__init__(
            generator=Generator(**kwargs), critic=Critic(**kwargs), **kwargs
        )
