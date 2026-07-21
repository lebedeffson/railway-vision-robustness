from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import Tensor


def spatial_transform(
    images: Tensor,
    *,
    angle_degrees: float = 0.0,
    scale: float = 1.0,
    padding_mode: str = "reflection",
) -> Tensor:
    if images.ndim != 4:
        raise ValueError("Expected BCHW image tensor")
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle) / scale
    sine = math.sin(angle) / scale
    theta = images.new_tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0]]
    ).unsqueeze(0).repeat(images.shape[0], 1, 1)
    grid = functional.affine_grid(theta, images.shape, align_corners=False)
    return functional.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )
