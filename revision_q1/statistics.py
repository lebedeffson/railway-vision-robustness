from __future__ import annotations

import math
import warnings
from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold


def group_fold_assignments(groups: Iterable[str], folds: int = 3) -> np.ndarray:
    group_array = np.asarray(list(groups), dtype=object)
    unique = np.unique(group_array)
    if len(unique) < folds:
        raise ValueError("Not enough independent sequence_id groups")
    result = np.full(len(group_array), -1, dtype=int)
    dummy = np.zeros((len(group_array), 1))
    for fold, (_, test_indices) in enumerate(
        GroupKFold(n_splits=folds).split(dummy, groups=group_array)
    ):
        result[test_indices] = fold
    return result


def cluster_sample_plan(
    sequence_ids: Iterable[str], iterations: int, seed: int
) -> list[np.ndarray]:
    values = np.asarray(list(sequence_ids), dtype=object)
    unique = pd.unique(values)
    if len(unique) < 2:
        raise ValueError("Cluster bootstrap requires at least two sequences")
    rng = np.random.default_rng(seed)
    return [rng.choice(unique, size=len(unique), replace=True) for _ in range(iterations)]


def materialize_cluster_sample(sequence_ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    indices = {value: np.flatnonzero(sequence_ids == value) for value in pd.unique(sequence_ids)}
    return np.concatenate([indices[value] for value in selected])


def unique_cluster_samples(
    sequence_ids: Iterable[str], iterations: int, seed: int
) -> list[tuple[np.ndarray, int]]:
    values = np.asarray(list(sequence_ids), dtype=object)
    plans = cluster_sample_plan(values, iterations, seed)
    multiplicities = Counter(tuple(sorted(str(value) for value in plan)) for plan in plans)
    return [
        (
            materialize_cluster_sample(values, np.asarray(selected, dtype=object)),
            count,
        )
        for selected, count in multiplicities.items()
    ]


def correlation(kind: str, left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "spearman":
            return float(spearmanr(left[valid], right[valid]).statistic)
        if kind == "pearson":
            return float(pearsonr(left[valid], right[valid]).statistic)
    raise ValueError(kind)


def paired_cluster_delta_correlation(
    frame: pd.DataFrame,
    target: str,
    left_metric: str,
    right_metric: str,
    *,
    kind: str = "spearman",
    iterations: int = 5000,
    seed: int = 20260720,
) -> dict[str, float | int]:
    required = {"sequence_id", target, left_metric, right_metric}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    sequence_ids = frame["sequence_id"].astype(str).to_numpy()
    target_values = frame[target].to_numpy(float)
    left_values = frame[left_metric].to_numpy(float)
    right_values = frame[right_metric].to_numpy(float)
    observed = abs(correlation(kind, target_values, left_values)) - abs(
        correlation(kind, target_values, right_values)
    )
    samples: list[float] = []
    for sampled, multiplicity in unique_cluster_samples(sequence_ids, iterations, seed):
        delta = abs(correlation(kind, target_values[sampled], left_values[sampled])) - abs(
            correlation(kind, target_values[sampled], right_values[sampled])
        )
        if math.isfinite(delta):
            samples.extend([delta] * multiplicity)
    values = np.asarray(samples)
    if not len(values):
        return {
            "delta_rho": observed, "ci_low": math.nan, "ci_high": math.nan,
            "p_value": 1.0, "bootstrap_iterations": iterations,
            "sequences": int(pd.Series(sequence_ids).nunique()), "frames": len(frame),
        }
    probability_low = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
    probability_high = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
    return {
        "delta_rho": observed,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "p_value": float(min(1.0, 2 * min(probability_low, probability_high))),
        "bootstrap_iterations": iterations,
        "sequences": int(pd.Series(sequence_ids).nunique()),
        "frames": len(frame),
    }


def holm_bonferroni(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    count = len(p)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def benjamini_hochberg(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    count = len(p)
    for rank_from_end, index in enumerate(order[::-1], 1):
        rank = count - rank_from_end + 1
        running = min(running, p[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def floor_effect_mode(map50: float, threshold: float = 0.01) -> str:
    return "recall_f1_false_negatives" if map50 <= threshold else "map_and_detection"


def cluster_mean_interval(
    frame: pd.DataFrame,
    value: str,
    *,
    iterations: int = 5000,
    seed: int = 20260720,
) -> dict[str, float | int]:
    sequence_ids = frame["sequence_id"].astype(str).to_numpy()
    values = frame[value].to_numpy(float)
    samples = []
    for indices, multiplicity in unique_cluster_samples(sequence_ids, iterations, seed):
        samples.extend([float(np.nanmean(values[indices]))] * multiplicity)
    distribution = np.asarray(samples)
    return {
        "estimate": float(np.nanmean(values)),
        "ci_low": float(np.nanpercentile(distribution, 2.5)),
        "ci_high": float(np.nanpercentile(distribution, 97.5)),
        "sequences": int(pd.Series(sequence_ids).nunique()),
        "bootstrap_iterations": iterations,
    }
