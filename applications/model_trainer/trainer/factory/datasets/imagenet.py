import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Final, Optional, Tuple

import albumentations as albu
import torchvision
from albumentations.pytorch import ToTensorV2

from trainer.factory.datasets import BaseDataModule, DatasetStats


@dataclass(frozen=True)
class Imagenet1kStats(DatasetStats):
    num_classes: int = 1000
    input_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class ImageNet1k(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root: pathlib.Path,
        train: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__(
            root=root / "train" if train else root / "val",
            transform=None,
            target_transform=None,
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


class Imagenet1kDataModule(BaseDataModule):
    """The LightningDataModule for ImageNet-1k dataset.

    Attributes:
        dataset_stats (DatasetStats): The dataclass which holds dataset statistics.
        root (pathlib.Path): The root path which dataset exists.

    """

    def __init__(self, batch_size: int, num_workers: int, root: pathlib.Path) -> None:
        super().__init__(batch_size, num_workers, root)
        self.dataset_stats: DatasetStats = Imagenet1kStats()
        self.root: Final[pathlib.Path] = root / "imagenet1k"

    def prepare_data(self, *args, **kwargs) -> None:
        """check if ImageNet dataset exists (DO NOT assign train/val here)."""
        if not (self.root / "train").exists():
            raise ValueError(
                f"Please download and set ImageNet-1k train data under {self.root}."
            )
        if not (self.root / "val").exists():
            raise ValueError(
                f"Please download and set ImageNet-1k val data under {self.root}."
            )

    def setup(self, stage=None, *args, **kwargs) -> None:
        """Assign test dataset"""
        self.train_dataset = ImageNet1k(
            root=self.root, train=True, transform=self._get_transform(train=True)
        )

        self.val_dataset = ImageNet1k(
            root=self.root, train=False, transform=self._get_transform(train=True)
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
                    albu.augmentations.geometric.resize.Resize(height=256, width=256),
                    albu.augmentations.crops.transforms.RandomCrop(
                        height=input_size, width=input_size, p=1.0
                    ),
                ]
            )
        else:
            transform.extend(
                [
                    albu.augmentations.geometric.resize.Resize(height=256, width=256),
                    albu.augmentations.crops.transforms.CenterCrop(
                        height=input_size, width=input_size
                    ),
                ]
            )

        if normalize:
            transform.extend(
                [albu.augmentations.transforms.Normalize(mean=mean, std=std)]
            )

        transform.extend([ToTensorV2()])
        return albu.Compose(transform)
