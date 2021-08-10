from dataclasses import dataclass

from omegaconf import MISSING


@dataclass()
class DataModuleConfig:
    _target_: str = MISSING


@dataclass()
class Cifar10DataModuleConfig(DataModuleConfig):
    _target_: str = "trainer.factory.datasets.cifar10.Cifar10DataModule"


@dataclass()
class Imagenet1kDataModuleConfig(DataModuleConfig):
    _target_: str = "trainer.factory.datasets.imagenet1k.Imagenet1kDataModule"


# NOTE: If you want to add your datamodule, please implement YourCustomDataModuleConfig class here.
