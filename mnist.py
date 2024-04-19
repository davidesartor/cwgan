import torch
from lightning import Trainer
from lightning.pytorch import callbacks, loggers
import custom_callbacks
from pl_bolts.datamodules import FashionMNISTDataModule
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
                    for i, real, gen in zip(classes, real_imgs, generated_imgs)
                ],
            )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    fminst = FashionMNISTDataModule(
        ".",
        batch_size=1000,
        val_split=1000 / 60000,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    trainer = Trainer(
        max_time="00:06:00:00",
        devices="2,",
        callbacks=[
            custom_callbacks.WatchModel(),
            callbacks.RichProgressBar(),
            callbacks.RichModelSummary(-1),
        ],
        logger=loggers.WandbLogger(project="wgan", log_model=True, tags=["mnist"]),
    )

    model = VisionCWGAN(
        shape=fminst.dims,
        classes=fminst.num_classes,
        noise_dim=32,
        optimizer="adam",
        lr=3e-4,
        critic_iter=1,
        gradient_penalty=None,
        weight_clip=None,
    )

    fminst.prepare_data()
    fminst.setup()
    trainer.fit(model, fminst.train_dataloader(), fminst.val_dataloader())
    trainer.test(model, dataloaders=fminst.test_dataloader())
