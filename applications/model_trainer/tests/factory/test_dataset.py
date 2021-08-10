import pathlib

import torch
import torchvision
from torch.utils.data import DataLoader

from trainer.factory.datasets.cifar10 import CIAFR10


class TestCIFAR10:
    def test(self, transform_cifar10_factory):
        root = pathlib.Path("data/cifar10")
        transform = transform_cifar10_factory()

        dataset = CIAFR10(root=root, train=False, transform=transform)
        batch_size = 16
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=8
        )
        for x, t in loader:
            assert type(x) == torch.Tensor
            assert x.size() == torch.Size([batch_size, 3, 32, 32])
            dirpath = pathlib.Path("outputs/pytest")
            dirpath.mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(x, dirpath / "cifar10.png")
            break
