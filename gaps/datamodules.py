import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule


class SinusoidsDataset(Dataset):
    parameter_names = ["A", "ω", "φ"]
    channel_names = ["h+", "hx"]
    signal_lenght = 1024

    def __init__(self, batch_size, n_batches=1):
        self.dataset_size = batch_size * n_batches
        self.times = torch.linspace(0, 1, self.signal_lenght)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        params = self.sample_params()
        clean_signal = self.clean_signal(params, self.times)
        noisy_signal = self.add_noise(clean_signal)
        return params, noisy_signal

    @staticmethod
    def sample_params():
        # A = 10 ** np.random.normal(0.5, 0.25)
        # omega = 2 * np.pi * 10 ** np.random.normal(0.5, 0.25)
        # phi = np.random.normal(0, 0.5 * np.pi)
        A = 10 ** np.random.uniform(0, 1)
        omega = 2 * np.pi * 10 ** np.random.uniform(0, 1)
        phi = np.random.uniform(0, 2 * np.pi)
        params = np.array([A, omega, phi])
        return torch.as_tensor(params, dtype=torch.float32)

    @staticmethod
    def clean_signal(params: torch.Tensor, times: torch.Tensor):
        A, omega, phi = params.tensor_split(3, dim=-1)
        h1 = A * torch.sin(omega * times + phi)
        h2 = A * torch.cos(omega * times + phi)
        return torch.stack([h1, h2], dim=-1)

    @staticmethod
    def add_noise(clean_signal: torch.Tensor):
        noise = torch.randn_like(clean_signal)
        return clean_signal + noise


class SinusoidsDatamodule(LightningDataModule):
    def __init__(self, batch_size=256, num_workers=8, pin_memory=True, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = SinusoidsDataset(self.hparams["batch_size"], 1024)
        self.test_dataset = SinusoidsDataset(self.hparams["batch_size"])

    def update_batch_size(self, batch_size):
        self.hparams.update(batch_size=batch_size)
        self.__init__(**self.hparams)

    @property
    def channel_names(self):
        return self.train_dataset.channel_names

    @property
    def parameter_names(self):
        return self.train_dataset.parameter_names

    @property
    def signal_shape(self):
        return (self.train_dataset.signal_lenght, len(self.channel_names))

    @property
    def params_size(self):
        return len(self.parameter_names)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.hparams)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, **self.hparams)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.hparams)
