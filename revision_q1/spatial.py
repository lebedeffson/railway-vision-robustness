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


def transform_xywh_boxes(
    boxes: Tensor,
    *,
    angle_degrees: float = 0.0,
    scale: float = 1.0,
) -> Tensor:
    """Transform normalized xywh boxes with the same centered image affine."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Expected normalized xywh boxes")
    if boxes.numel() == 0:
        return boxes.clone()
    centers = boxes[:, :2]
    half = boxes[:, 2:] / 2
    signs = boxes.new_tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    corners = centers[:, None, :] + signs[None, :, :] * half[:, None, :]
    centered = corners * 2 - 1
    # affine_grid maps output to input, so content coordinates use its inverse.
    angle = math.radians(angle_degrees)
    inverse = boxes.new_tensor([
        [math.cos(angle) * scale, math.sin(angle) * scale],
        [-math.sin(angle) * scale, math.cos(angle) * scale],
    ])
    transformed = torch.matmul(centered, inverse.T)
    transformed = ((transformed + 1) / 2).clamp(0.0, 1.0)
    minimum = transformed.amin(dim=1)
    maximum = transformed.amax(dim=1)
    return torch.cat(((minimum + maximum) / 2, maximum - minimum), dim=1)
