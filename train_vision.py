from setproctitle import setproctitle
from lightning.pytorch import callbacks, cli
import torch

from vision import nets, datamodules, custom_callbacks


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.dims", "model.shape", apply_on="instantiate")
        parser.link_arguments(
            "data.num_classes", "model.classes", apply_on="instantiate"
        )


if __name__ == "__main__":
    setproctitle("i'm just a test, feel free to sudo -pkill me UwU")
    torch.set_float32_matmul_precision("medium")

    parser = CLI(
        run=False,
        model_class=nets.VisionCWGAN,
        save_config_kwargs=dict(overwrite=True),
        trainer_defaults=dict(
            max_time="00:06:00:00",
            devices="2,",
            callbacks=[
                custom_callbacks.WatchModel(),
                custom_callbacks.LogImages(),
                callbacks.RichProgressBar(),
                callbacks.RichModelSummary(),
            ],
            logger=dict(
                class_path="lightning.pytorch.loggers.WandbLogger",
                init_args=dict(project="wgan", log_model=True),
            ),
        ),
    )

    parser.trainer.fit(parser.model, parser.datamodule)
    parser.trainer.test(parser.model, parser.datamodule)
