import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Final, Optional, Tuple

import albumentations as albu
import torchvision
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from trainer.factory.datasets import BaseDataModule, DatasetStats


@dataclass(frozen=True)
class Cifar10Stats(DatasetStats):
    num_classes: int = 10
    input_size: int = 32
    mean: Tuple[float, float, float] = (0.49139968, 0.48215841, 0.44653091)
    std: Tuple[float, float, float] = (0.24703223, 0.24348513, 0.26158784)


class CIAFR10(torchvision.datasets.CIFAR10):
    def __init__(
        self,
        root: pathlib.Path,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super().__init__(
            root=root,
            train=train,
            transform=None,
            target_transform=None,
            download=download,
        )
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, target


class Cifar10DataModule(BaseDataModule):
    """The LightningDataModule for CIFAR-10 dataset.

    Attributes:
        dataset_stats (DatasetStats): The dataclass which holds dataset statistics.
        root (pathlib.Path): The root path which dataset exists. If not, try to download.

    """

    def __init__(self, batch_size: int, num_workers: int, root: pathlib.Path) -> None:
        super().__init__(batch_size, num_workers, root)
        self.dataset_stats: DatasetStats = Cifar10Stats()
        self.root: Final[pathlib.Path] = root / "cifar10"

    def prepare_data(self, *args, **kwargs) -> None:
        """Try to download dataset (DO NOT assign train/val here)."""
        self.root.mkdir(exist_ok=True, parents=True)
        torchvision.datasets.CIFAR10(root=self.root, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.root, train=False, download=True)

    def setup(self, stage=None, *args, **kwargs) -> None:
        """Assign test dataset"""
        self.train_dataset: Dataset = CIAFR10(
            root=self.root,
            train=True,
            transform=self._get_transform(train=True),
            download=True,
        )

        self.val_dataset: Dataset = CIAFR10(
            root=self.root,
            train=False,
            transform=self._get_transform(train=False),
            download=True,
        )

    def _get_transform(
        self,
        train: bool,
        normalize: bool = True,
    ) -> albu.Compose:
        transform = list()

        input_size: Final = self.input_size
        mean: Final = self.dataset_stats.mean
        std: Final = self.dataset_stats.std

        if train:
            transform.extend(
                [
                    albu.augmentations.transforms.HorizontalFlip(p=0.5),
                    albu.augmentations.transforms.PadIfNeeded(
                        min_height=input_size + 4, min_width=input_size + 4
                    ),
                    albu.augmentations.crops.transforms.RandomCrop(
                        height=input_size, width=input_size, p=1.0
                    ),
                ]
            )
        else:
            pass

        if normalize:
            transform.extend(
                [albu.augmentations.transforms.Normalize(mean=mean, std=std)]
            )

        transform.extend([ToTensorV2()])
        return albu.Compose(transform)
