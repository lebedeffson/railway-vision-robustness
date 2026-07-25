from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class LearningRateMultipliers:
    backbone: float
    neck: float
    head: float


PHASES = {
    "T0": LearningRateMultipliers(0.0, 0.0, 1.0),
    "T1": LearningRateMultipliers(0.0, 0.5, 1.0),
    "T2": LearningRateMultipliers(0.1, 0.5, 1.0),
}


def component_for_layer(index: int, total_layers: int) -> str:
    if index < 0 or index >= total_layers:
        raise ValueError("layer index is outside the model")
    if index <= 10:
        return "backbone"
    if index == total_layers - 1:
        return "head"
    return "neck"


def configure_phase(model: nn.Module, phase: str) -> dict[str, int]:
    if phase not in PHASES:
        raise ValueError(f"Unknown gradual-transfer phase: {phase}")
    modules = getattr(model, "model", None)
    if modules is None:
        raise ValueError("YOLO model must expose .model layers")
    multipliers = PHASES[phase]
    counts = {"backbone": 0, "neck": 0, "head": 0}
    for index, module in enumerate(modules):
        component = component_for_layer(index, len(modules))
        enabled = getattr(multipliers, component) > 0
        for parameter in module.parameters():
            parameter.requires_grad = enabled
            counts[component] += int(enabled)
    return counts


def parameter_groups(
    model: nn.Module, *, base_lr: float, phase: str
) -> list[dict[str, object]]:
    if phase not in PHASES:
        raise ValueError(f"Unknown gradual-transfer phase: {phase}")
    modules = getattr(model, "model", None)
    if modules is None:
        raise ValueError("YOLO model must expose .model layers")
    grouped: dict[str, list[nn.Parameter]] = {
        "backbone": [],
        "neck": [],
        "head": [],
    }
    for index, module in enumerate(modules):
        component = component_for_layer(index, len(modules))
        grouped[component].extend(
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    multipliers = PHASES[phase]
    return [
        {
            "name": component,
            "params": parameters,
            "lr": float(base_lr) * getattr(multipliers, component),
        }
        for component, parameters in grouped.items()
        if parameters
    ]


class L2SPAnchor:
    def __init__(self, model: nn.Module, names: Iterable[str] | None = None) -> None:
        allowed = set(names) if names is not None else None
        self.reference = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if allowed is None or name in allowed
        }

    def penalty(self, model: nn.Module, coefficient: float) -> torch.Tensor:
        terms = []
        for name, parameter in model.named_parameters():
            if name in self.reference and parameter.requires_grad:
                reference = self.reference[name].to(
                    device=parameter.device, dtype=parameter.dtype
                )
                terms.append((parameter - reference).square().sum())
        if not terms:
            parameter = next(model.parameters())
            return parameter.new_zeros(())
        return float(coefficient) * torch.stack(terms).sum()


def replay_source(step: int, railway_batches: int = 4) -> str:
    if railway_batches < 1:
        raise ValueError("railway_batches must be positive")
    return "CrowdHuman" if step % (railway_batches + 1) == railway_batches else "OSDaR23"

