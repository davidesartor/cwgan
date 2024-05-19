from torch import nn
from mamba_ssm.models.mixer_seq_simple import MixerModel
import torch

from networks.utils import MLP


class Mamba(MixerModel):
    def __init__(self, input_dim, num_layers, state_dim, hidden_dim=None, **kwargs):
        hidden_dim = hidden_dim or input_dim
        super().__init__(
            d_model=hidden_dim,
            n_layer=num_layers,
            ssm_cfg=dict(d_state=state_dim),
            rms_norm=True,
            fused_add_norm=True,
            residual_in_fp32=True,
            vocab_size=32,  # dummy we replace the embedding layer
        )
        if input_dim != hidden_dim:
            self.embedding = nn.Linear(input_dim, hidden_dim, bias=False)
        else:
            self.embedding = nn.Identity()


class DownSample(nn.Module):
    def __init__(self, channels, pool=4, expand=2):
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(channels * pool, channels * expand)

    def forward(self, x):
        x = x.reshape(*x.shape[:-2], x.shape[-2] // self.pool, x.shape[-1] * self.pool)
        x = self.proj(x)
        return x


class MambaSignalVectorEncoder(nn.Sequential):
    def __init__(
        self,
        signal_shape,
        vector_size,
        encoded_size,
        hidden_dim=16,
        state_dim=16,
    ):
        super().__init__()
        time, channels = signal_shape

        self.projection = nn.Linear(channels, hidden_dim, bias=False)
        channels = hidden_dim

        layers = []
        while time > 8:
            layers.append(Mamba(channels, num_layers=1, state_dim=state_dim))
            layers.append(DownSample(channels, pool=8, expand=2))
            time, channels = time // 8, 2 * channels
        layers.append(Mamba(channels, num_layers=1, state_dim=state_dim))
        self.encoder = nn.Sequential(*layers)

        self.mpl = MLP(
            input_dim=time * channels + vector_size,
            output_dim=encoded_size,
            num_layers=2,
        )

    def forward(self, signal, vector):
        x = self.encoder(self.projection(signal))
        x = torch.cat([x.flatten(1), vector], dim=-1)
        x = self.mpl(x)
        return x
