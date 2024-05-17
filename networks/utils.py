import torch
from torch import nn


class MLP(nn.Sequential):
    """
    MLP with optional spectral normalization.
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
        spectral_norm=False,
        **kwargs,
    ):
        layers: list[nn.Module] = [nn.Flatten()]
        layer_dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layer = nn.Linear(in_dim, out_dim)
            if spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            layers.append(layer)
            layers.append(activation)
        super().__init__(*layers[:-1])


class CIN(nn.Module):
    """
    Conditional Instance Normalization with optional spectral normalization.
    'A Learned Representation For Artistic Style'
    https://arxiv.org/abs/1610.07629v5
    """

    def __init__(self, channels, condition_dim, spectral_norm=False, **kwargs):
        super().__init__()
        self.mean = nn.Linear(condition_dim, channels)
        self.std = nn.Linear(condition_dim, channels)
        if spectral_norm:
            self.mean = nn.utils.spectral_norm(self.mean)
            self.std = nn.utils.spectral_norm(self.std)

    def forward(self, x, cond):
        mean = self.mean(cond).unsqueeze(-1).unsqueeze(-1)
        std = self.std(cond).unsqueeze(-1).unsqueeze(-1)
        return mean + std * nn.functional.instance_norm(x)


class ConditionalSequential(nn.ModuleList):
    def forward(self, x, cond):
        for module in self:
            x = module(x, cond)
        return x
