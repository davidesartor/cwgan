from lightning.pytorch import callbacks, loggers
import torch


class WatchModel(callbacks.Callback):
    def on_fit_start(self, trainer, pl_module):
        trainer.logger.watch(pl_module)  # type: ignore

    def on_fit_end(self, trainer, pl_module):
        trainer.logger.experiment.unwatch(pl_module)  # type: ignore
