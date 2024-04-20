from lightning import Trainer
from lightning.pytorch import callbacks, loggers
import torch
import custom_callbacks
from setproctitle import setproctitle
from pl_bolts.datamodules import CIFAR10DataModule
from models.vision import Generator, Critic
from cwgan import CWGAN


class VisionCWGAN(CWGAN):
    def __init__(self, shape, classes, **kwargs):
        super().__init__(
            generator=Generator(shape, classes, **kwargs),
            critic=Critic(shape, classes, **kwargs),
            **kwargs,
        )
        self.save_hyperparameters()

    def validation_step(self, batch, batch_idx):
        classes = torch.arange(self.hparams["classes"], device=self.device)
        generated_imgs = self.generator(
            torch.arange(self.hparams["classes"], device=self.device)
        )
        if isinstance(self.logger, loggers.WandbLogger):
            for i, img in enumerate(generated_imgs):
                self.logger.log_image(f"Generated/Class_{i}", [img])

    def test_step(self, batch, batch_idx):
        real_imgs, classes = batch
        generated_imgs = self.generator(classes)
        if isinstance(self.logger, loggers.WandbLogger):
            self.logger.log_table(
                "Generated vs Real",
                columns=["Class", "Real", "Generated"],
                data=[
                    [f"Class_{i}", real, gen]
                    for i, real, gen in zip(classes[:30], real_imgs, generated_imgs)
                ],
            )


if __name__ == "__main__":
    setproctitle("𐎀𐎍𐎙 𐎊𐎟𐏐𐎢 𐭓𐭕𐭁 𐭆𐭖𐭈𐭌 𐭃𐭉𐭆𐭍𐭊 𐤀𐤋𐤊 𐤄𐤅𐤔𐤋𐤉𐤕")
    torch.set_float32_matmul_precision("medium")

    batch_size = 512
    fminst = CIFAR10DataModule(
        ".",
        batch_size=batch_size,
        val_split=(50000 - (50000 // batch_size) * batch_size + 1) / 50000,
        num_workers=8,
        pin_memory=True,
    )

    trainer = Trainer(
        max_time="00:08:00:00",
        devices="2,",
        callbacks=[
            custom_callbacks.WatchModel(),
            callbacks.RichProgressBar(),
            callbacks.RichModelSummary(),
        ],
        logger=loggers.WandbLogger(project="wgan", log_model=True, tags=["cifar10"]),
    )

    model = VisionCWGAN(
        shape=fminst.dims,
        classes=fminst.num_classes,
        noise_dim=32,
        hidden=64,
        optimizer="adam",
        lr=1e-4,
        critic_iter=2,
        gradient_penalty=None,
        weight_clip=None,
    )

    fminst.prepare_data()
    fminst.setup()
    trainer.fit(model, fminst.train_dataloader(), fminst.val_dataloader())
    trainer.test(model, dataloaders=fminst.test_dataloader())
