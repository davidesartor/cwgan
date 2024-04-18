import torch
from torch import nn


class NormMLP(nn.Sequential):
    """
    MLP with spectral normalization for critics.
    'SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS'
    https://arxiv.org/pdf/1802.05957.pdf
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=512,
        num_layers=1,
        activation=nn.SiLU(),
        spectral_norm=True,
    ):
        layers = []
        layer_dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        for in_dim, out_dim in zip(layer_dims, layer_dims[1:]):
            layer = nn.Linear(in_dim, out_dim)
            layers.append(nn.utils.spectral_norm(layer) if spectral_norm else layer)
            layers.append(activation)
        super().__init__(*layers[:-1])


class DownPool(nn.Module):
    def __init__(self, channels, pool=4, expand=2, spectral_norm=False):
        super().__init__()
        self.pool = pool
        self.expand = expand
        self.proj = nn.Linear(pool * channels, expand * channels)
        if spectral_norm:
            self.proj = nn.utils.spectral_norm(self.proj)

    def __call__(self, x):
        x = x.reshape(*x.shape[:-2], -1, x.shape[-1] // self.pool)
        return self.proj(x.transpose(-2, -1)).transpose(-2, -1)


class UpPool(nn.Module):
    def __init__(self, channels, pool=4, expand=2, spectral_norm=False):
        super().__init__()
        self.pool = pool
        self.expand = expand
        self.proj = nn.Linear(channels, pool // expand * channels)
        if spectral_norm:
            self.proj = nn.utils.spectral_norm(self.proj)

    def __call__(self, x):
        x = self.proj(x.transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(*x.shape[:-2], -1, x.shape[-1] * self.pool)
        return x


class GRUEncoder(nn.Module):
    def __init__(
        self,
        channels,
        output_dim,
        hidden_dim=32,
        num_layers=1,
        bidirectional=True,
        **kwargs,
    ):
        super().__init__()
        kwargs.update(dict(num_layers=num_layers, bidirectional=bidirectional))
        rnn_out_dim = hidden_dim * num_layers * (2 if bidirectional else 1)
        self.in_proj = nn.Linear(channels, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True, **kwargs)
        self.out_proj = nn.Linear(rnn_out_dim, output_dim)

    def forward(self, sequence):
        x = self.in_proj(sequence)
        x = self.norm(x)
        _, x = self.rnn(x)
        x = x.transpose(0, 1).reshape(x.shape[1], -1)
        x = self.out_proj(x)
        return x
