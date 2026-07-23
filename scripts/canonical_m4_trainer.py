from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER


class DifferentialLRDetectionTrainer(DetectionTrainer):
    """Detection trainer with frozen, stage-specific backbone/head rates."""

    backbone_lr: float = 0.0
    head_lr: float = 0.0

    @staticmethod
    def _split_decay(
        named_parameters: Iterable[tuple[str, nn.Parameter]],
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return decay, no_decay

    def build_optimizer(
        self,
        model: nn.Module,
        name: str = "AdamW",
        lr: float = 0.001,
        momentum: float = 0.9,
        decay: float = 1e-5,
        iterations: float = 1e5,
    ) -> torch.optim.Optimizer:
        del lr, iterations
        if str(name).lower() != "adamw":
            raise RuntimeError("Canonical M4 protocol requires AdamW")
        modules = getattr(model, "model", None)
        if modules is None or not len(modules):
            raise RuntimeError("Unable to locate YOLO detection head")
        head_ids = {id(parameter) for parameter in modules[-1].parameters()}
        backbone_named = []
        head_named = []
        for parameter_name, parameter in model.named_parameters():
            target = head_named if id(parameter) in head_ids else backbone_named
            target.append((parameter_name, parameter))
        backbone_decay, backbone_no_decay = self._split_decay(backbone_named)
        head_decay, head_no_decay = self._split_decay(head_named)
        groups = [
            {
                "params": backbone_decay,
                "lr": float(self.backbone_lr),
                "weight_decay": float(decay),
                "param_group": "backbone_weight",
            },
            {
                "params": backbone_no_decay,
                "lr": float(self.backbone_lr),
                "weight_decay": 0.0,
                "param_group": "backbone_no_decay",
            },
            {
                "params": head_decay,
                "lr": float(self.head_lr),
                "weight_decay": float(decay),
                "param_group": "head_weight",
            },
            {
                "params": head_no_decay,
                "lr": float(self.head_lr),
                "weight_decay": 0.0,
                "param_group": "head_no_decay",
            },
        ]
        if not head_decay and not head_no_decay:
            raise RuntimeError("Canonical M4 detection head parameter group is empty")
        optimizer = torch.optim.AdamW(groups, betas=(float(momentum), 0.999))
        LOGGER.info(
            "Canonical M4 differential AdamW: "
            f"backbone_lr={self.backbone_lr:g}, head_lr={self.head_lr:g}, "
            f"backbone_params={len(backbone_decay) + len(backbone_no_decay)}, "
            f"head_params={len(head_decay) + len(head_no_decay)}"
        )
        return optimizer
