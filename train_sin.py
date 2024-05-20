from setproctitle import setproctitle
from lightning.pytorch import callbacks, cli
import torch

from gaps import nets, custom_callbacks, datamodules


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        for argname in ["signal_shape", "parameter_names"]:
            parser.link_arguments(
                f"data.{argname}", f"model.{argname}", apply_on="instantiate"
            )


if __name__ == "__main__":
    setproctitle("i'm just a test, feel free to sudo -pkill me UwU")
    torch.set_float32_matmul_precision("medium")

    parser = CLI(
        run=False,
        model_class=nets.SinCWGAN,
        datamodule_class=datamodules.SinusoidsDatamodule,
        save_config_kwargs=dict(overwrite=True),
        trainer_defaults=dict(
            # max_time="00:00:30:00",
            max_epochs=512,
            devices="2,",
            callbacks=[
                callbacks.ModelCheckpoint(),
                callbacks.RichProgressBar(),
                callbacks.RichModelSummary(),
                custom_callbacks.WatchModel(),
                custom_callbacks.GradientAccumulationScheduler(
                    scheduling={64: 2, 128: 4, 256: 16},
                ),
            ],
            logger=dict(
                class_path="lightning.pytorch.loggers.WandbLogger",
                init_args=dict(project="wgan", log_model=True, tags=["sinusoids"]),
            ),
        ),
    )

    parser.trainer.fit(parser.model, parser.datamodule)
    parser.trainer.test(parser.model, parser.datamodule)
