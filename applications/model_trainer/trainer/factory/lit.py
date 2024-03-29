from typing import Dict, Final, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler

import trainer.common
import trainer.schema as schema


class SupervisedClassifier(pl.LightningModule):
    """Lightning Module for supervised image classfication.

    Attributes:
        encoder (nn.Module): The encoder to extract feature for classification.
        optimizer_cfg (schema.OptimizerConfig): The config for optimizer.
        scheduler_cfg (schema.SchedulerConfig): The config for sheduler.
        criterion (_Loss): The loss used by optimizer.

    """

    def __init__(
        self,
        encoder: nn.Module,
        optimizer_cfg: schema.OptimizerConfig,
        scheduler_cfg: schema.SchedulerConfig,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.criterion: Final[_Loss] = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.encoder(x)

    def training_step(  # type: ignore
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Single training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input tensor and label.
            batch_index (int): An index of the batch.

        Returns:
            torch.Tensor: The loss tensor.
                If multiple nodes are used for training, return type should be Dict[str, torch.Tensor].

        """
        # DO NOT need to send GPUs manually. lightning automatically sends to GPUs.
        x, t = batch

        # forward to encoder
        output = self.encoder(x)
        loss = self.criterion(output, t)

        # save sample input
        if batch_idx == 1:
            torchvision.utils.save_image(x.detach()[:32], "train_img_sample.png")

        # calculate metrics.
        err1, err5 = trainer.common.calc_errors(
            output.detach(), t.detach(), topk=(1, 5)
        )

        # logging loss / top1 err / top5 err
        self.log("train_loss", loss.detach(), on_epoch=True, sync_dist=True)
        self.log("train_err1", err1.detach(), on_epoch=True, sync_dist=True)
        self.log("train_err5", err5.detach(), on_epoch=True, sync_dist=True)
        return loss

    def validation_step(  # type: ignore
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Single validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): The tuple of input tensor and label.
            batch_index (int): The index of batch.

        Returns:
            Dict[torch.Tensor]: The dict of log info.

        """
        x, t = batch  # DO NOT need to send GPU manually.

        # forward to encoder
        output = self.encoder(x)
        loss = self.criterion(output, t)

        # calculate metrics.
        err1, err5 = trainer.common.calc_errors(
            output.detach(), t.detach(), topk=(1, 5)
        )

        # logging loss / top1 err / top5 err
        self.log("val_loss", loss.detach(), on_epoch=True, sync_dist=True)
        self.log("val_err1", err1.detach(), on_epoch=True, sync_dist=True)
        self.log("val_err5", err5.detach(), on_epoch=True, sync_dist=True)

    def configure_optimizers(
        self,
    ) -> Dict[str, Union[torch.optim.Optimizer, Dict[str, Union[str, _LRScheduler]]]]:
        """setup optimzier and scheduler."""
        optimizer = instantiate(self.optimizer_cfg, params=self.encoder.parameters())
        scheduler = {
            "scheduler": instantiate(self.scheduler_cfg, optimizer),
            "monitor": "train_err1",
        }

        return dict(optimizer=optimizer, lr_scheduler=scheduler)
