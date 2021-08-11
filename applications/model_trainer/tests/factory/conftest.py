import albumentations as albu
import pytest
from albumentations.pytorch import ToTensorV2


@pytest.fixture
def transform_cifar10_factory():
    def f():
        mean = (0.49139968, 0.48215841, 0.44653091)
        std = (0.24703223, 0.24348513, 0.26158784)
        return albu.Compose(
            [
                albu.augmentations.transforms.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    return f
