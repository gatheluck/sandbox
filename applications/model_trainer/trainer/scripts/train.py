import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Final, List

import comet_ml  # noqa NOTE:this is needed to avoid error. For detail, please check; https://github.com/PyTorchLightning/pytorch-lightning/issues/5829
import hydra
import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers.base import LightningLoggerBase

import trainer.schema as schema
from trainer.factory.lit import SupervisedClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class ModelCheckpointConfig:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    filename: str = "{epoch}-{val_err1:.2f}"
    dirpath: str = "checkpoint"
    mode: str = "min"
    monitor: str = "val_err1"
    save_last: bool = True
    verbose: bool = True


@dataclass(frozen=True)
class EarlyStoppingConfig:
    _target_: str = "pytorch_lightning.callbacks.EarlyStopping"
    monitor: str = "val_err1"
    min_delta: float = 0.0
    patience: int = 10
    mode: str = "min"
    verbose: bool = True


@dataclass
class TrainConfig:
    # grouped configs
    arch: schema.ArchConfig = MISSING
    env: schema.EnvConfig = MISSING
    datamodule: schema.DataModuleConfig = MISSING
    optimizer: schema.OptimizerConfig = MISSING
    scheduler: schema.SchedulerConfig = MISSING
    # ungrouped configs
    epochs: int = MISSING
    batch_size: int = MISSING


cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)
# arch
cs.store(group="arch", name="resnet34", node=schema.Resnet34Config)
cs.store(group="arch", name="resnet50", node=schema.Resnet50Config)
cs.store(group="arch", name="resnet56", node=schema.Resnet56Config)
cs.store(group="arch", name="wideresnet40", node=schema.Wideresnet40Config)
# datamodule
cs.store(group="datamodule", name="cifar10", node=schema.Cifar10DataModuleConfig)
cs.store(group="datamodule", name="imagenet1k", node=schema.Imagenet1kDataModuleConfig)
# env
cs.store(group="env", name="default", node=schema.DefaultEnvConfig)
cs.store(group="env", name="nogpu", node=schema.NogpuEnvConfig)
# optimizer
cs.store(group="optimizer", name="sgd", node=schema.SgdConfig)
cs.store(group="optimizer", name="adam", node=schema.AdamConfig)
# scheduler
cs.store(group="scheduler", name="cosin", node=schema.CosinConfig)
cs.store(group="scheduler", name="multistep", node=schema.MultistepConfig)
cs.store(group="scheduler", name="plateau", node=schema.PlateauConfig)


@hydra.main(config_path="../config", config_name="train")
def train(cfg: TrainConfig) -> None:
    """train classifer.
    Args:
        cfg (TrainConfig): Configs whose type is checked by schema.
    """
    # Make config read only.
    # without this, config values might be changed accidentally.
    OmegaConf.set_readonly(cfg, True)  # type: ignore
    logger.info(OmegaConf.to_yaml(cfg))

    # get original working directory since hydra automatically changes it.
    cwd: Final[pathlib.Path] = pathlib.Path(hydra.utils.get_original_cwd())

    # setup datamodule
    root: Final[pathlib.Path] = cwd / "data"
    datamodule = instantiate(cfg.datamodule, cfg.batch_size, cfg.env.num_workers, root)

    # setup model
    arch = instantiate(cfg.arch)
    model = SupervisedClassifier(
        encoder=arch,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
    )

    # Setup loggers
    # - MLFlow: It is used as local logger.
    # - TensorBoard: It is used as local logger.
    # - CometML: It is used as online logger. The environmental variable "COMET_API_KEY" need to be defined for use.
    loggers: List[LightningLoggerBase] = list()
    loggers.append(pl.loggers.TensorBoardLogger(save_dir="tensorboard"))
    loggers.append(pl.loggers.MLFlowLogger(save_dir="mlflow"))
    try:
        loggers.append(pl.loggers.CometLogger(api_key=os.environ.get("COMET_API_KEY")))
    except pl.utilities.exceptions.MisconfigurationException:
        logger.info(
            "CometML is not available. If you want to use it, please define COMET_API_KEY as an environment variable."
        )

    # Setup callbacks
    # - ModelCheckpoint: It saves checkpoint.
    # - EarlyStopping: It handles early stopping.
    callbacks: List[Callback] = list()
    callbacks.append(instantiate(ModelCheckpointConfig))
    callbacks.append(instantiate(EarlyStoppingConfig))

    # setup trainer
    trainer = pl.Trainer(
        accelerator="ddp",
        benchmark=True,
        callbacks=callbacks,
        deterministic=False,
        gpus=cfg.env.gpus,  # number of gpus to train
        logger=loggers,
        max_epochs=cfg.epochs,
        # min_epochs=cfg.epochs,  # min epochs
        num_nodes=cfg.env.num_nodes,  # number of GPU nodes
        profiler="pytorch",
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
