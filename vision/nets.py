import math
import torch
from torch import nn
from networks.utils import MLP, CIN, ConditionalSequential
from networks.conv2d import ResidualBlock, UpSampleBlock, DownSampleBlock
from cwgan import CWGAN


class Block(nn.Module):
    def __init__(
        self,
        channels,
        condition_dim,
        kernel_size=5,
        activation=nn.SiLU(),
        spectral_norm=False,
        upsample=False,
        downsample=False,
    ):
        super().__init__()
        self.upsample = (
            UpSampleBlock(channels, kernel_size, activation, spectral_norm)
            if upsample
            else None
        )
        self.downsample = (
            DownSampleBlock(channels, kernel_size, activation, spectral_norm)
            if downsample
            else None
        )
        self.cin = CIN(channels, condition_dim, spectral_norm)
        self.convblock = ResidualBlock(channels, kernel_size, activation, spectral_norm)
        self.activation = activation

    def forward(self, x, cond):
        if self.upsample is not None:
            x = self.upsample(x)
        x = self.cin(x, cond)
        x = self.convblock(x)
        x = self.activation(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        shape,
        classes,
        latent_dim=1,
        channels=128,
        kernel_size=5,
        activation=nn.SiLU(),
        **kwargs,
    ):
        super().__init__()
        out_channels, *out_shape = shape
        n_stages = math.ceil(math.log2(max(out_shape)))

        self.latent_dim = latent_dim
        self.classes = classes
        self.channels = channels
        self.stages = ConditionalSequential(
            Block(
                channels,
                condition_dim=classes + latent_dim,
                kernel_size=kernel_size,
                activation=activation,
                upsample=True,
            )
            for _ in range(n_stages)
        )
        out_kernel_size = tuple(1 + 2**n_stages - s for s in out_shape)
        self.final_conv = nn.Conv2d(channels, out_channels, out_kernel_size)

    def forward(self, y):
        z = torch.randn(y.size(0), self.latent_dim, device=y.device)
        y = nn.functional.one_hot(y, self.classes).float()
        cond = torch.cat((z, y), dim=-1)

        x = torch.ones(y.size(0), self.channels, 1, 1, device=y.device)
        x = self.stages(x, cond)
        x = self.final_conv(x)
        return x


class Critic(nn.Module):
    def __init__(
        self,
        shape,
        classes,
        channels=64,
        kernel_size=5,
        activation=nn.SiLU(),
        **kwargs,
    ):
        super().__init__()
        in_channels, *in_shape = shape
        n_stages = math.ceil(math.log2(max(in_shape)))

        self.classes = classes
        self.channels = channels

        expand_kernel_size = tuple(1 + 2**n_stages - s for s in in_shape)
        self.initial_conv = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_channels, channels, expand_kernel_size)
        )
        self.stages = ConditionalSequential(
            Block(
                channels,
                condition_dim=classes,
                kernel_size=kernel_size,
                activation=activation,
                downsample=True,
            )
            for _ in range(n_stages)
        )
        self.head = MLP(
            input_dim=channels,
            output_dim=1,
            hidden_dim=channels // 2,
            num_layers=2,
            spectral_norm=True,
        )

    def forward(self, y, x):
        x = self.initial_conv(x)
        cond = nn.functional.one_hot(y, self.classes).float()
        x = self.stages(x, cond)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class VisionCWGAN(CWGAN):
    def __init__(self, shape, classes, latent_dim=32, **kwargs):
        generator = Generator(shape, classes, latent_dim, **kwargs)
        critic = Critic(shape, classes, **kwargs)
        super().__init__(generator, critic, **kwargs)
        self.save_hyperparameters()
