from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor


TAU = 1e-8
LAYERS = ("P3", "P4", "P5")
MODES = (
    "existing_normalization",
    "N1_quantile",
    "N2_robust_sigmoid",
    "N3_zscore_sigmoid",
)


def fit_channel_statistics(samples: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    if set(samples) != set(LAYERS):
        raise ValueError(f"Expected separate {LAYERS} samples, got {sorted(samples)}")
    output: dict[str, dict[str, Tensor]] = {}
    for layer in LAYERS:
        values = samples[layer].detach().float().cpu()
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError(f"{layer} samples must have shape channels x observations")
        median = values.median(dim=1).values
        mad = (values - median[:, None]).abs().median(dim=1).values
        output[layer] = {
            "q01": torch.quantile(values, 0.01, dim=1),
            "q05": torch.quantile(values, 0.05, dim=1),
            "q50": median,
            "q95": torch.quantile(values, 0.95, dim=1),
            "q99": torch.quantile(values, 0.99, dim=1),
            "median": median,
            "mad": mad,
            "mean": values.mean(dim=1),
            "std": values.std(dim=1, unbiased=False),
            "observations": torch.tensor(values.shape[1]),
        }
    return output


def _broadcast(value: Tensor, feature: Tensor) -> Tensor:
    if feature.ndim != 4:
        raise ValueError("Feature tensor must be BCHW")
    if value.numel() != feature.shape[1]:
        raise ValueError("Channel statistics do not match feature channels")
    return value.to(feature.device, feature.dtype).view(1, -1, 1, 1)


def membership(feature: Tensor, statistics: dict[str, Tensor], mode: str) -> Tensor:
    if mode == "existing_normalization":
        low, high = statistics["q05"], statistics["q95"]
        result = (feature - _broadcast(low, feature)) / (
            _broadcast(high - low, feature) + TAU
        )
        return result.clamp(0.0, 1.0)
    if mode == "N1_quantile":
        low, high = statistics["q01"], statistics["q99"]
        result = (feature - _broadcast(low, feature)) / (
            _broadcast(high - low, feature) + TAU
        )
        return result.clamp(0.0, 1.0)
    if mode == "N2_robust_sigmoid":
        center = _broadcast(statistics["median"], feature)
        scale = _broadcast(statistics["mad"], feature)
        return torch.sigmoid((feature - center) / (scale + TAU))
    if mode == "N3_zscore_sigmoid":
        center = _broadcast(statistics["mean"], feature)
        scale = _broadcast(statistics["std"], feature)
        return torch.sigmoid((feature - center) / (scale + TAU))
    raise ValueError(f"Unknown normalization mode: {mode}")


def distribution_diagnostics(values: Tensor) -> dict[str, float]:
    flattened = values.detach().float().flatten()
    if flattened.numel() == 0:
        raise ValueError("Cannot diagnose an empty membership tensor")
    return {
        "fraction_below_0.01": float((flattened < 0.01).float().mean()),
        "fraction_above_0.99": float((flattened > 0.99).float().mean()),
        "mean_membership": float(flattened.mean()),
        "std_membership": float(flattened.std(unbiased=False)),
        "q01": float(torch.quantile(flattened, 0.01)),
        "q50": float(torch.quantile(flattened, 0.50)),
        "q99": float(torch.quantile(flattened, 0.99)),
    }


def saturation_warning(diagnostics: dict[str, float], threshold: float = 0.20) -> bool:
    return (
        diagnostics["fraction_below_0.01"] > threshold
        or diagnostics["fraction_above_0.99"] > threshold
    )


def normalized_quality_recovery(
    clean: float, attacked: float, defended: float, epsilon_num: float = TAU
) -> float:
    return (defended - attacked) / (clean - attacked + epsilon_num)
