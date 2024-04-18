import torch
from lightning.pytorch import callbacks
from lightning.pytorch.cli import LightningCLI
from lisa import LisaDataModule, LisaCWGAN
import custom_callbacks


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    cli = LightningCLI(
        run=False,
        datamodule_class=LisaDataModule,
        model_class=LisaCWGAN,
        save_config_kwargs=dict(overwrite=True),
        trainer_defaults=dict(
            max_time="00:01:00:00",
            callbacks=[
                custom_callbacks.WandBCustomLogs(),
                callbacks.RichProgressBar(),
                callbacks.RichModelSummary(-1),
                callbacks.BatchSizeFinder(max_trials=10),
            ],
            logger=dict(
                class_path="lightning.pytorch.loggers.WandbLogger",
                init_args=dict(project="wgan", log_model=True),
            ),
        ),
    )

    cli.trainer.fit(cli.model, cli.datamodule)
    cli.trainer.test(cli.model, cli.datamodule)
