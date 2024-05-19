import torch
from torch import nn
from networks.utils import CIN


class ResidualBlock(nn.Sequential):
    def __init__(
        self, channels, kernel_size=5, activation=nn.SiLU(), spectral_norm=False
    ):
        conv1 = nn.Conv2d(channels, channels, kernel_size, padding="same")
        norm = nn.InstanceNorm2d(channels)
        conv2 = nn.Conv2d(channels, channels, kernel_size, padding="same")
        if spectral_norm:
            conv1 = nn.utils.spectral_norm(conv1)
            conv2 = nn.utils.spectral_norm(conv2)
        super().__init__(activation, conv1, norm, activation, conv2)

    def forward(self, x, cond=None):
        return x + super().forward(x)


class UpSampleBlock(nn.Sequential):
    def __init__(
        self, channels, kernel_size=5, activation=nn.SiLU(), spectral_norm=False
    ):
        upsample = nn.ConvTranspose2d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )
        smooth = nn.Conv2d(channels, channels, kernel_size, padding="same")
        if spectral_norm:
            upsample = nn.utils.spectral_norm(upsample)
            smooth = nn.utils.spectral_norm(smooth)
        super().__init__(upsample, activation, smooth)

    def forward(self, x, cond=None):
        return super().forward(x)


class DownSampleBlock(nn.Sequential):
    def __init__(
        self, channels, kernel_size=5, activation=nn.SiLU(), spectral_norm=False
    ):
        downsample = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)
        smooth = nn.Conv2d(channels, channels, kernel_size, padding="same")
        if spectral_norm:
            downsample = nn.utils.spectral_norm(downsample)
            smooth = nn.utils.spectral_norm(smooth)
        super().__init__(downsample, activation, smooth)

    def forward(self, x, cond=None):
        return super().forward(x)
