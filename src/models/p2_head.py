from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from src.models.coordinate_attention import register_ultralytics_modules


def build_p2_model(config: Path, *, pretrained: Path | None = None) -> YOLO:
    register_ultralytics_modules()
    model = YOLO(str(config), task="detect")
    if pretrained is not None:
        model.load(str(pretrained))
    return model


def detection_strides(model: YOLO) -> list[int]:
    stride = getattr(model.model, "stride", None)
    if stride is None:
        raise RuntimeError("P2 model did not expose detection strides")
    return [int(value) for value in stride.detach().cpu().tolist()]


def model_complexity(model: YOLO, size: int = 640) -> dict[str, Any]:
    module = model.model
    parameters = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    with torch.no_grad():
        output = module(torch.zeros(1, 3, size, size))
    levels = len(output) if isinstance(output, list) else len(module.stride)
    return {
        "parameters": int(parameters),
        "trainable_parameters": int(trainable),
        "strides": detection_strides(model),
        "detection_levels": int(levels),
    }

