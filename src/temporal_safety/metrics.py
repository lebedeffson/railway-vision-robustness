from __future__ import annotations

from typing import Any

import numpy as np


def relative_reduction(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / baseline) if baseline else 0.0


def paired_scene_bootstrap(
    baseline: dict[str, float],
    candidate: dict[str, float],
    iterations: int,
    seed: int,
) -> dict[str, float]:
    scenes = sorted(set(baseline) & set(candidate))
    if not scenes:
        raise ValueError("No paired scenes")
    effects = np.asarray([candidate[scene] - baseline[scene] for scene in scenes])
    generator = np.random.default_rng(seed)
    draws = generator.choice(effects, size=(iterations, len(effects)), replace=True).mean(axis=1)
    return {
        "estimate": float(effects.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "scenes": len(scenes),
        "iterations": iterations,
    }


def gate_checks(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    rules: dict[str, Any],
    maximum_scene_recall_drop: float,
) -> dict[str, bool]:
    false_alarm_ratio = (
        float(candidate["false_alarms_per_minute"])
        / max(float(baseline["false_alarms_per_minute"]), 1e-12)
    )
    return {
        "recall": float(candidate["recall"]) - float(baseline["recall"])
        >= float(rules["absolute_recall_improvement_min"]),
        "FN_per_frame": relative_reduction(
            float(baseline["FN_per_frame"]), float(candidate["FN_per_frame"])
        )
        >= float(rules["relative_FN_reduction_min"]),
        "false_alarms_per_minute": false_alarm_ratio
        <= 1.0 + float(rules["relative_false_alarms_increase_max"]),
        "F1": float(candidate["f1"]) - float(baseline["f1"])
        >= -float(rules["F1_degradation_max"]),
        "per_scene_recall": float(candidate["maximum_scene_recall_degradation"])
        <= maximum_scene_recall_drop,
    }
