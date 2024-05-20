from pl_bolts.datamodules import CIFAR10DataModule, FashionMNISTDataModule
from lightning import LightningDataModule


def leave_one_batch(data_size, batch_size):
    return (data_size - (data_size // batch_size) * batch_size + 1) / data_size


class CIFAR10(CIFAR10DataModule, LightningDataModule):
    def __init__(self, data_dir="datasets/", batch_size=512, val_split=None, **kwargs):
        if val_split is None:
            val_split = leave_one_batch(50000, batch_size)
        super().__init__(
            data_dir=data_dir, val_split=val_split, batch_size=512, **kwargs
        )


class FMNIST(FashionMNISTDataModule, LightningDataModule):
    def __init__(self, data_dir="datasets/", batch_size=256, val_split=None, **kwargs):
        if val_split is None:
            val_split = leave_one_batch(60000, batch_size)
        super().__init__(
            data_dir=data_dir, val_split=val_split, batch_size=512, **kwargs
        )
