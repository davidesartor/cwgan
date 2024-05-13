import math
import torch
from torch import nn
from .utils import MLP, CIN


class ResidualConvBlock(nn.Sequential):
    def __init__(self, channels, activation=nn.SiLU(), spectral_norm=False):
        super().__init__()
        conv1 = nn.Conv2d(channels, channels, 5, padding="same")
        norm = nn.InstanceNorm2d(channels)
        conv2 = nn.Conv2d(channels, channels, 5, padding="same")
        if spectral_norm:
            conv1 = nn.utils.spectral_norm(conv1)
            conv2 = nn.utils.spectral_norm(conv2)
        super().__init__(activation, conv1, norm, activation, conv2)

    def forward(self, x):
        return x + super().forward(x)


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation=nn.SiLU()):
        conv_t = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        conv = nn.Conv2d(in_channels, out_channels, 5, padding="same")
        super().__init__(conv_t, activation, conv)


class Generator(nn.Module):
    def __init__(
        self,
        shape,
        classes,
        noise_dim=1,
        latent_channels=1024,
        activation=nn.SiLU(),
        **kwargs,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.classes = classes
        self.activation = activation
        self.latent_channels = latent_channels

        self.upsample_modules = nn.ModuleList()
        self.inception_modules = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        img_dim = 1
        out_channels, *out_shape = shape
        while img_dim < max(out_shape):
            self.upsample_modules.append(
                UpsampleBlock(latent_channels, latent_channels // 2, activation)
            )
            latent_channels //= 2
            img_dim *= 2
            self.inception_modules.append(
                CIN(latent_channels, condition_dim=noise_dim + classes)
            )
            self.residual_blocks.append(ResidualConvBlock(latent_channels, activation))

        out_kernel_size = tuple(1 + img_dim - s for s in out_shape)
        self.head = nn.Conv2d(latent_channels, out_channels, out_kernel_size)

    def forward(self, y):
        z = torch.randn(y.size(0), self.noise_dim, device=y.device)
        y = nn.functional.one_hot(y, self.classes).float()
        cond = torch.cat((z, y), dim=-1)

        x = torch.randn(y.size(0), self.latent_channels, 1, 1, device=y.device)
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
        latent_channels=1024,
        activation=nn.SiLU(),
        **kwargs,
    ):
        super().__init__()
        self.classes = classes
        self.activation = activation

        self.downsample_modules = nn.ModuleList()
        self.inception_modules = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()

        in_channels, *in_shape = shape
        stages = math.ceil(math.log2(max(in_shape)))
        hidden = latent_channels // 2**stages
        out_kernel_size = tuple(1 + 2**stages - s for s in in_shape)

        self.expand = nn.utils.spectral_norm(
            nn.ConvTranspose2d(in_channels, hidden, out_kernel_size)
        )
        img_dim = 2**stages
        while img_dim > 1:
            self.inception_modules.append(
                CIN(hidden, condition_dim=classes, spectral_norm=True)
            )
            self.residual_blocks.append(
                ResidualConvBlock(hidden, activation, spectral_norm=True)
            )
            self.downsample_modules.append(
                nn.utils.spectral_norm(
                    nn.Conv2d(hidden, 2 * hidden, kernel_size=4, stride=2, padding=1)
                )
            )
            hidden *= 2
            img_dim //= 2
        self.head = MLP(hidden, 1, hidden_dim=hidden // 2, spectral_norm=True)

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
