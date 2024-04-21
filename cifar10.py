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
        gen_imgs = self.generator(
            torch.arange(self.hparams["classes"], device=self.device)
        )
        if isinstance(self.logger, loggers.WandbLogger):
            for i, img in enumerate(gen_imgs):
                self.logger.log_image(f"Generated/Class_{i}", [img])

    def test_step(self, batch, batch_idx):
        real_imgs, classes = batch
        gen_imgs = self.generator(classes)
        if isinstance(self.logger, loggers.WandbLogger):
            for i, (c, r, g) in list(enumerate(zip(classes, real_imgs, gen_imgs)))[:30]:
                self.logger.log_image(
                    f"Test/{i}", [r, g], caption=[f"Real", "Generated"]
                )


if __name__ == "__main__":
    setproctitle("i'm just a test, feel free to sudo -pkill me UwU")
    torch.set_float32_matmul_precision("medium")

    batch_size = 1024
    datamodule = CIFAR10DataModule(
        ".",
        batch_size=batch_size,
        val_split=(50000 - (50000 // batch_size) * batch_size + 1) / 50000,
        num_workers=8,
        pin_memory=True,
    )

    trainer = Trainer(
        max_time="00:24:00:00",
        devices="2,",
        callbacks=[
            custom_callbacks.WatchModel(),
            callbacks.RichProgressBar(),
            callbacks.RichModelSummary(),
        ],
        logger=loggers.WandbLogger(project="wgan", log_model=True, tags=["cifar10"]),
    )

    model = VisionCWGAN(
        shape=datamodule.dims,
        classes=datamodule.num_classes,
        optimizer="adam",
        lr=1e-5,
        critic_iter=2,
        gradient_penalty=None,
        weight_clip=None,
    )

    datamodule.prepare_data()
    datamodule.setup()
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
    trainer.test(model, dataloaders=datamodule.test_dataloader())
