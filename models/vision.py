import math
import torch
from torch import nn
from .utils import MLP, CIN


class ResidualConvBlock(nn.Sequential):
    def __init__(self, channels, activation=nn.SiLU(), spectral_norm=False):
        super().__init__()
        conv1 = nn.Conv2d(channels, channels, 3, padding="same")
        norm = nn.LayerNorm(channels)
        conv2 = nn.Conv2d(channels, channels, 3, padding="same")
        if spectral_norm:
            conv1 = nn.utils.spectral_norm(conv1)
            conv2 = nn.utils.spectral_norm(conv2)
        super().__init__(activation, conv1, norm, activation, conv2)

    def forward(self, x):
        return x + super().forward(x)


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU()):
        conv_t = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        conv = nn.Conv2d(out_channels, out_channels, 5, padding="same")
        super().__init__(conv_t, activation, conv)


class Generator(nn.Module):
    def __init__(
        self,
        shape,
        classes,
        noise_dim=1,
        hidden=64,
        activation=nn.SiLU(),
        **kwargs,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.classes = classes
        self.activation = activation
        out_channels, *out_shape = shape
        stages = math.ceil(math.log2(max(out_shape)))
        out_kernel_size = tuple(1 + 2**stages - s for s in out_shape)

        self.upsample_modules = nn.ModuleList(
            UpsampleBlock(1 if i == 0 else hidden, hidden, activation)
            for i in range(stages)
        )
        self.inception_modules = nn.ModuleList(
            CIN(hidden, condition_dim=noise_dim + classes) for _ in range(stages)
        )
        self.residual_blocks = nn.ModuleList(
            ResidualConvBlock(hidden, activation) for _ in range(stages)
        )
        self.head = nn.Conv2d(hidden, out_channels, out_kernel_size)

    def forward(self, y):
        z = torch.randn(y.size(0), self.noise_dim, device=y.device)
        y = nn.functional.one_hot(y, self.classes).float()
        cond = torch.cat((z, y), dim=-1)

        x = torch.randn(y.size(0), 1, 1, 1, device=y.device)
        for upsample, inception, block in zip(
            self.upsample_modules, self.inception_modules, self.residual_blocks
        ):
            x = upsample(x)
            x = self.activation(x)
            x = inception(x, cond)
            x = block(x)
            x = self.activation(x)
        x = self.head(x)
        return x


class Critic(nn.Module):
    def __init__(
        self,
        shape,
        classes,
        hidden=64,
        activation=nn.SiLU(),
        **kwargs,
    ):
        super().__init__()
        self.classes = classes
        self.activation = activation
        in_channels, *in_shape = shape
        stages = math.ceil(math.log2(max(in_shape)))
        out_kernel_size = tuple(1 + 2**stages - s for s in in_shape)

        self.expand = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_channels, hidden, out_kernel_size)
        )
        self.residual_blocks = nn.ModuleList(
            ResidualConvBlock(hidden, activation, spectral_norm=True)
            for i in range(stages)
        )
        self.inception_modules = nn.ModuleList(
            CIN(hidden, condition_dim=classes, spectral_norm=True)
            for _ in range(stages)
        )
        self.downsample_modules = nn.ModuleList(
            nn.utils.spectral_norm(
                nn.Conv2d(hidden, hidden, kernel_size=4, stride=2, padding=1)
            )
            for _ in range(stages)
        )
        self.head = MLP(hidden, 1, hidden_dim=hidden * 4, spectral_norm=True)

    def forward(self, y, x):
        cond = nn.functional.one_hot(y, self.classes).float()
        x = self.expand(x)
        for downsample, inception, block in zip(
            self.downsample_modules, self.inception_modules, self.residual_blocks
        ):
            x = inception(x, cond)
            x = block(x)
            x = self.activation(x)
            x = downsample(x)
            x = self.activation(x)
        x = self.head(x.view(x.size(0), -1))
        return x
