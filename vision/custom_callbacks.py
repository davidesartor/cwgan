from lightning.pytorch import callbacks, loggers
import torch


class WatchModel(callbacks.Callback):
    def on_fit_start(self, trainer, pl_module):
        trainer.logger.watch(pl_module)  # type: ignore

    def on_fit_end(self, trainer, pl_module):
        trainer.logger.experiment.unwatch(pl_module)  # type: ignore


class LogImages(callbacks.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        classes = torch.arange(pl_module.hparams["classes"], device=pl_module.device)
        gen_imgs = pl_module.generator(classes)
        if isinstance(trainer.logger, loggers.WandbLogger):
            for i, img in enumerate(gen_imgs):
                trainer.logger.log_image(f"Generated/Class_{i}", [img])

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        real_imgs, classes = batch
        gen_imgs = pl_module.generator(classes)
        if isinstance(trainer.logger, loggers.WandbLogger):
            for i, (c, r, g) in list(enumerate(zip(classes, real_imgs, gen_imgs)))[:30]:
                trainer.logger.log_image(
                    f"Test/{i}", [r, g], caption=[f"Real", "Generated"]
                )
