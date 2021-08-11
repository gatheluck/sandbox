from dataclasses import dataclass
from typing import List

from omegaconf import MISSING


@dataclass
class SchedulerConfig:
    verbose: bool = False


@dataclass
class CosinConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.CosineAnnealingLR"
    last_epoch: int = -1
    T_max: int = MISSING
    eta_min: float = MISSING


@dataclass
class MultistepConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.MultiStepLR"
    last_epoch: int = -1
    milestones: List[int] = MISSING
    gamma: float = MISSING


@dataclass
class PlateauConfig(SchedulerConfig):
    _target_: str = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    mode: str = MISSING
    factor: float = MISSING
    patience: int = MISSING
