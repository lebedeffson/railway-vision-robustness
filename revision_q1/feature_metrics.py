from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import Tensor

from revision_q1.normalization import TAU, membership


MAX_RANK_SAMPLES = 65_536


def tnorm(left: Tensor, right: Tensor, operator: str) -> Tensor:
    """Apply the formal fuzzy conjunction pointwise to memberships in [0, 1]."""
    if operator == "product":
        return left * right
    if operator == "godel":
        return torch.minimum(left, right)
    if operator == "lukasiewicz":
        return torch.clamp(left + right - 1.0, min=0.0)
    raise ValueError(operator)


def _cosine(left: Tensor, right: Tensor) -> float:
    denominator = left.norm() * right.norm()
    if denominator <= TAU:
        return 1.0 if left.norm() <= TAU and right.norm() <= TAU else 0.0
    return float(torch.dot(left, right) / (denominator + TAU))


def _pearson(left: Tensor, right: Tensor) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    return _cosine(left_centered, right_centered)


def _rank_sample(left: Tensor, right: Tensor) -> tuple[np.ndarray, np.ndarray]:
    if left.numel() > MAX_RANK_SAMPLES:
        indices = torch.linspace(
            0, left.numel() - 1, MAX_RANK_SAMPLES,
            device=left.device, dtype=torch.float64,
        ).long()
        left, right = left[indices], right[indices]
    return left.detach().float().cpu().numpy(), right.detach().float().cpu().numpy()


def activation_entropy(values: Tensor, bins: int = 64) -> float:
    flattened = values.detach().float().flatten().clamp(0.0, 1.0)
    histogram = torch.histc(flattened, bins=bins, min=0.0, max=1.0)
    probabilities = histogram / histogram.sum().clamp_min(TAU)
    probabilities = probabilities[probabilities > 0]
    if probabilities.numel() < 2:
        return 0.0
    return float(-(probabilities * probabilities.log()).sum() / math.log(bins))


def pair_metrics(
    clean: Tensor,
    other: Tensor,
    statistics: dict[str, Tensor],
    mode: str,
) -> list[dict[str, float]]:
    if clean.shape != other.shape:
        raise ValueError("Feature pairs must have identical shape")
    clean_membership = membership(clean, statistics, mode)
    other_membership = membership(other, statistics, mode)
    rows: list[dict[str, float]] = []
    for image_index in range(clean.shape[0]):
        left = clean_membership[image_index].flatten().float()
        right = other_membership[image_index].flatten().float()
        raw_left = clean[image_index].flatten().float()
        raw_right = other[image_index].flatten().float()
        difference = left - right
        absolute = difference.abs()
        rank_left, rank_right = _rank_sample(left, right)
        entropy_clean = activation_entropy(clean_membership[image_index])
        entropy_other = activation_entropy(other_membership[image_index])
        rows.append({
            "cosine_similarity": _cosine(left, right),
            "cosine_raw": _cosine(raw_left, raw_right),
            "euclidean_distance": float(difference.norm()),
            "normalized_euclidean_distance": float(
                difference.norm() / (left.norm() + TAU)
            ),
            "l1_distance": float(absolute.sum()),
            "mse": float(difference.square().mean()),
            "mae": float(absolute.mean()),
            "pearson_correlation": _pearson(left, right),
            "spearman_correlation": float(
                spearmanr(rank_left, rank_right).statistic
            ),
            "relative_l2": float(difference.norm() / (left.norm() + TAU)),
            "mean_activation_shift": float((left.mean() - right.mean()).abs()),
            "activation_entropy_clean": entropy_clean,
            "activation_entropy_other": entropy_other,
            "entropy_shift": abs(entropy_clean - entropy_other),
            "product": float(tnorm(left, right, "product").mean()),
            "godel": float(tnorm(left, right, "godel").mean()),
            "lukasiewicz": float(tnorm(left, right, "lukasiewicz").mean()),
        })
    return rows


def distance_recovery(before: float, after: float) -> float:
    return math.nan if abs(before) <= TAU else (before - after) / (before + TAU)


def similarity_recovery(before: float, after: float) -> float:
    denominator = 1.0 - before
    return math.nan if abs(denominator) <= TAU else (after - before) / (denominator + TAU)
