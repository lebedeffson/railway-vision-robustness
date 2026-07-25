from __future__ import annotations

import torch
from torch import nn


class HardSwish(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.clamp(value + 3.0, 0.0, 6.0) / 6.0


class CoordinateAttention(nn.Module):
    """Coordinate Attention with separate height and width gates."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, int(channels) // int(reduction))
        self.channels = int(channels)
        self.reduce = nn.Conv2d(self.channels, hidden, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(hidden)
        self.activation = HardSwish()
        self.height_gate = nn.Conv2d(hidden, self.channels, kernel_size=1)
        self.width_gate = nn.Conv2d(hidden, self.channels, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.channels:
            raise ValueError(
                f"CoordinateAttention expected N,{self.channels},H,W; got "
                f"{tuple(value.shape)}"
            )
        height_context = value.mean(dim=3, keepdim=True)
        width_context = value.mean(dim=2, keepdim=True).transpose(2, 3)
        joined = torch.cat((height_context, width_context), dim=2)
        joined = self.activation(self.norm(self.reduce(joined)))
        height, width = value.shape[2:]
        height_features, width_features = torch.split(
            joined, (height, width), dim=2
        )
        width_features = width_features.transpose(2, 3)
        height_weight = self.height_gate(height_features).sigmoid()
        width_weight = self.width_gate(width_features).sigmoid()
        return value * height_weight * width_weight


def register_ultralytics_modules() -> None:
    """Expose the custom layer to Ultralytics' YAML parser."""
    import ultralytics.nn.tasks as tasks

    tasks.CoordinateAttention = CoordinateAttention

