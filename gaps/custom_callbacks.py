from lightning.pytorch import callbacks, loggers
import torch


class WatchModel(callbacks.Callback):
    def on_fit_start(self, trainer, pl_module):
        trainer.logger.watch(pl_module)  # type: ignore

    def on_fit_end(self, trainer, pl_module):
        trainer.logger.experiment.unwatch(pl_module)  # type: ignore


class GradientAccumulationScheduler(callbacks.GradientAccumulationScheduler):
    def on_train_start(self, trainer, pl_module) -> None:
        try:
            super().on_train_start(trainer, pl_module)
        except RuntimeError:
            pass

    def on_train_epoch_start(self, trainer, pl_module):
        scheduled_accumulate = self.get_accumulate_grad_batches(trainer.current_epoch)
        pl_module.hparams.update(accumulate_grad_batches=scheduled_accumulate)
