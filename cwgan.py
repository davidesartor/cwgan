import torch
from torch import Tensor, nn
from lightning import LightningModule


class CWGAN(LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        critic: nn.Module,
        optimizer="RMSprop",
        lr=1e-5,
        critic_iter=4,
        gradient_penalty: float = 10,
        weight_clip: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["generator", "critic"])
        self.critic = critic
        self.generator = generator

    def forward(self, y):
        return self.generator(y)

    def configure_optimizers(self):
        if self.hparams["optimizer"].lower() == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
            kwargs: dict = dict(lr=self.hparams["lr"])
        elif self.hparams["optimizer"].lower() == "adam":
            optimizer_cls = torch.optim.Adam
            kwargs: dict = dict(lr=self.hparams["lr"], betas=(0.0, 0.9), fused=True)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams['optimizer']}")

        optimizer_for_generator = optimizer_cls(self.generator.parameters(), **kwargs)
        optimizer_for_critic = optimizer_cls(self.critic.parameters(), **kwargs)
        return [optimizer_for_generator, optimizer_for_critic], []

    def optimize_critic(self, y, x_real, x_generated):
        loss = self.critic(y, x_generated).mean() - self.critic(y, x_real).mean()
        self.log("Loss/Critic", loss, prog_bar=True)

        if self.hparams["gradient_penalty"]:
            penalty = self.gradient_penalty(y, x_real, x_generated)
            loss += penalty
            self.log("Loss/Penalty", penalty, prog_bar=True)

        optimizer_for_generator, optimizer = self.optimizers()  # type: ignore
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if self.hparams["weight_clip"] is not None:
            self.weight_clip(self.critic, self.hparams["weight_clip"])

    def optimize_generator(self, y, x_real, x_generated):
        loss = -self.critic(y, x_generated).mean()
        self.log("Loss/Generator", loss, prog_bar=True)

        optimizer, optimizer_for_critic = self.optimizers()  # type: ignore
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

    def training_step(self, batch, batch_idx):
        x_real, y = batch
        x_generated = self.generator(y)

        self.optimize_critic(y, x_real, x_generated.detach())
        if batch_idx % self.hparams["critic_iter"] == 0:
            self.optimize_generator(y, x_real, x_generated)

    def weight_clip(self, critic: nn.Module, clip_value: float = 0.01):
        """
        Enforce Lipschitz contraint following the original WGAN paper:
        'Wasserstein GAN'
        Martin Arjovsky, Soumith Chintala, Léon Bottou
        https://arxiv.org/abs/1701.07875
        """
        with torch.no_grad():
            for param in critic.parameters():
                param.clamp_(-clip_value, clip_value)

    def gradient_penalty(self, y: Tensor, x1: Tensor, x2: Tensor, coeff: float = 10.0):
        """
        Enforce Lipschitz contraint following the WGAN-GP paper:
        'Improved Training of Wasserstein GANs'
        Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville
        https://arxiv.org/abs/1704.00028
        """
        factory_kwargs = {"device": x1.device, "dtype": x1.dtype}
        batch_size, *x_shape = x1.shape

        # interpolate between the two examples
        a = torch.rand(batch_size, **factory_kwargs)
        a = a.view(batch_size, *[1 for _ in x_shape])
        x_interp = (a * x1 + (1 - a) * x2).requires_grad_(True)

        # compute the gradient of the critic with respect to the interpolation
        gradients = torch.autograd.grad(
            outputs=self.critic(y, x_interp),
            inputs=x_interp,
            grad_outputs=torch.ones(batch_size, 1, **factory_kwargs),
            create_graph=True,
            retain_graph=True,
        )[0].reshape(batch_size, -1)

        # penalize deviations from unitary gradient norm
        gradient_norm = torch.linalg.vector_norm(gradients, dim=-1)
        return coeff * nn.functional.mse_loss(
            gradient_norm, torch.ones_like(gradient_norm)
        )
