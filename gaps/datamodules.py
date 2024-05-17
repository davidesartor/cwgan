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
        self.times = np.linspace(0, 1, self.signal_lenght)

    def __len__(self):
        return self.dataset_size

    def sample_params(self):
        A = 10 ** np.random.uniform(-1, 1)
        omega = 2 * np.pi * 10 ** np.random.uniform(0, 0.5)
        phi = np.random.uniform(0, np.pi)
        return A, omega, phi

    def simulate_signal(self, A, omega, phi):
        h1 = A * np.sin(omega * self.times + phi)
        h2 = A * np.cos(omega * self.times + phi)
        clean_signal = np.stack([h1, h2], axis=-1)
        noise = np.random.normal(0, 1, clean_signal.shape)
        return clean_signal + noise

    def __getitem__(self, idx):
        params = self.sample_params()
        signal = self.simulate_signal(*params)
        params = torch.as_tensor(params, dtype=torch.float32)
        signal = torch.as_tensor(signal, dtype=torch.float32)
        return params, signal


class SinusoidsDatamodule(LightningDataModule):
    def __init__(self, batch_size=1024, num_workers=8, pin_memory=True, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = SinusoidsDataset(self.hparams["batch_size"], 128)
        self.test_dataset = SinusoidsDataset(self.hparams["batch_size"])

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
