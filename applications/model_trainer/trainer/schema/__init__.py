from .arch import (  # noqa # NOTE: If you want to add your architecture, please add YourCustomArchConfig class in this line.
    ArchConfig,
    Resnet34Config,
    Resnet50Config,
    Resnet56Config,
    Wideresnet40Config,
)
from .datamodule import (  # noqa # NOTE: If you want to add your datamodule, please add YourCustomDataModuleConfig class in this line.
    Cifar10DataModuleConfig,
    DataModuleConfig,
    Imagenet1kDataModuleConfig,
)
from .env import DefaultEnvConfig, EnvConfig  # noqa
from .optimizer import AdamConfig, OptimizerConfig, SgdConfig  # noqa
from .scheduler import (  # noqa
    CosinConfig,
    MultistepConfig,
    PlateauConfig,
    SchedulerConfig,
)
