from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from revision_q1.analyze import add_endpoints, apply_normalization_aliases, boolean_series
from revision_q1.protocol import load_protocol, output_root
from revision_q1.statistics import (
    cluster_mean_interval,
    cluster_sample_plan,
    correlation,
    materialize_cluster_sample,
    unique_cluster_samples,
)


def paired_summary(
    frame: pd.DataFrame,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    pivot = frame.pivot(index="sequence_id", columns="checkpoint", values=metric).dropna()
    differences = pivot["stage2_best"].to_numpy(float) - pivot["stage1_best"].to_numpy(float)
    sequence_ids = pivot.index.to_numpy(str)
    samples = []
    for selected in cluster_sample_plan(sequence_ids, iterations, seed):
        positions = [int(np.flatnonzero(sequence_ids == value)[0]) for value in selected]
        samples.append(float(np.mean(differences[positions])))
    values = np.asarray(samples)
    return {
        "metric": metric,
        "stage2_minus_stage1": float(differences.mean()),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "probability_stage2_greater": float((np.count_nonzero(values > 0) + 1) / (len(values) + 1)),
        "sequences": len(pivot),
        "bootstrap_iterations": iterations,
    }


def image_quality(matrix: pd.DataFrame, checkpoint: str) -> pd.DataFrame:
    selected = matrix.copy()
    if "selected_best" in selected:
        selected = selected[boolean_series(selected["selected_best"])]
    selected = selected[selected["defense"] == "none"]
    grouped = selected.groupby(["sequence_id", "image_path"], as_index=False).agg({
        "f1_clean": "first", "recall_clean": "first", "fn_clean": "first",
    })
    summary = grouped.groupby("sequence_id", as_index=False).agg({
        "f1_clean": "mean", "recall_clean": "mean", "fn_clean": "mean",
    })
    summary["checkpoint"] = checkpoint
    return summary


def diagnostic_direction(matrix: pd.DataFrame, checkpoint: str, mode: str) -> pd.DataFrame:
    data = add_endpoints(apply_normalization_aliases(matrix, mode))
    if "selected_best" in data:
        data = data[boolean_series(data["selected_best"])]
    rows: list[dict[str, object]] = []
    scopes = {
        "damage": data[data["defense"] == "none"],
        "recovery": data[data["defense"] != "none"],
    }
    for task, scope in scopes.items():
        endpoint = "delta_f1_damage" if task == "damage" else "delta_f1_recovery"
        metrics = (
            ("product", "godel", "lukasiewicz")
            if task == "damage"
            else ("product_recovery", "godel_recovery", "lukasiewicz_recovery")
        )
        aggregate = scope.groupby("sequence_id", as_index=False).agg(
            {endpoint: "mean", **{metric: "mean" for metric in metrics}}
        )
        for metric in metrics:
            target = aggregate[endpoint].to_numpy(float)
            values = aggregate[metric].to_numpy(float)
            sequence_ids = aggregate["sequence_id"].astype(str).to_numpy()
            rho = correlation("spearman", target, values)
            samples = []
            for indices, multiplicity in unique_cluster_samples(
                sequence_ids, 5000, 20260720
            ):
                sampled = correlation("spearman", target[indices], values[indices])
                if math.isfinite(sampled):
                    samples.extend([sampled] * multiplicity)
            rows.append({
                "checkpoint": checkpoint, "task": task, "metric": metric,
                "estimate": rho,
                "ci_low": float(np.percentile(samples, 2.5)) if samples else math.nan,
                "ci_high": float(np.percentile(samples, 97.5)) if samples else math.nan,
                "effect_sign": int(np.sign(rho)) if math.isfinite(rho) else 0,
                "sequences": len(aggregate),
            })
    adaptive = data[(data["attack"] == "pgd") & (data["defense"] == "tnorm")]
    if not adaptive.empty:
        adaptive = adaptive.copy()
        adaptive["adaptive"] = boolean_series(adaptive["adaptive"])
        grouped = adaptive.groupby(["sequence_id", "adaptive"], as_index=False)["f1_defended"].mean()
        pivot = grouped.pivot(index="sequence_id", columns="adaptive", values="f1_defended")
        if False in pivot and True in pivot:
            differences = pivot.dropna().reset_index()
            differences["difference"] = differences[True] - differences[False]
            interval = cluster_mean_interval(
                differences, "difference", iterations=5000, seed=20260720
            )
            rows.append({
                "checkpoint": checkpoint, "task": "adaptive_pgd",
                "metric": "adaptive_minus_nonadaptive_f1",
                "estimate": interval["estimate"], "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
                "effect_sign": int(np.sign(interval["estimate"])),
                "sequences": len(pivot.dropna()),
            })
    object_background = data[["sequence_id", "image_path", "c_atk_object", "c_atk_background"]].copy()
    object_background["difference"] = (
        object_background["c_atk_object"] - object_background["c_atk_background"]
    )
    object_background = object_background.groupby(
        ["sequence_id", "image_path"], as_index=False
    )["difference"].mean()
    interval = cluster_mean_interval(object_background, "difference", iterations=5000, seed=20260720)
    rows.append({
        "checkpoint": checkpoint, "task": "object_background",
        "metric": "c_atk_object_minus_background",
        "estimate": interval["estimate"], "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "effect_sign": int(np.sign(interval["estimate"])),
        "sequences": int(data["sequence_id"].nunique()),
    })
    return pd.DataFrame(rows)


def main() -> None:
    protocol = load_protocol()
    root = output_root(protocol)
    parser = argparse.ArgumentParser(description="Paired Stage 2 versus Stage 1 sensitivity")
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument(
        "--sequence-metrics", type=Path,
        default=root / "raw/checkpoint_sequence_metrics.csv",
    )
    parser.add_argument("--selection", type=Path, default=root / "config/normalization_selection.json")
    parser.add_argument("--output", type=Path, default=root)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    mode = selection["selected_normalization"]
    stage2 = pd.read_csv(args.stage2)
    stage1 = pd.read_csv(args.stage1)
    quality = pd.concat([
        image_quality(stage2, "stage2_best"), image_quality(stage1, "stage1_best")
    ], ignore_index=True)
    if args.sequence_metrics.is_file():
        map_metrics = pd.read_csv(args.sequence_metrics)
        quality = quality.merge(
            map_metrics, on=["sequence_id", "checkpoint"], how="outer"
        )
    rows = []
    for metric in ("f1_clean", "recall_clean", "fn_clean", "map50", "map50_95"):
        if metric in quality and quality.groupby("checkpoint")[metric].count().min() > 0:
            rows.append(paired_summary(
                quality[["sequence_id", "checkpoint", metric]].dropna(), metric,
                int(protocol["bootstrap_iterations"]), int(protocol["random_seed"]),
            ))
    tables = args.output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tables / "07_checkpoint_paired_comparison.csv", index=False)
    sensitivity = pd.concat([
        diagnostic_direction(stage2, "stage2_best", mode),
        diagnostic_direction(stage1, "stage1_best", mode),
    ], ignore_index=True)
    pivot = sensitivity.pivot_table(
        index=["task", "metric"], columns="checkpoint",
        values=["estimate", "ci_low", "ci_high", "effect_sign", "sequences"],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{checkpoint}" for metric, checkpoint in pivot.columns]
    pivot = pivot.reset_index()
    if {"effect_sign_stage1_best", "effect_sign_stage2_best"} <= set(pivot):
        pivot["same_direction"] = (
            pivot["effect_sign_stage1_best"] == pivot["effect_sign_stage2_best"]
        )
        pivot["ci_overlap"] = (
            pivot[["ci_high_stage1_best", "ci_high_stage2_best"]].min(axis=1)
            >= pivot[["ci_low_stage1_best", "ci_low_stage2_best"]].max(axis=1)
        )
    pivot.to_csv(tables / "08_checkpoint_sensitivity.csv", index=False)
    print(json.dumps({
        "normalization": mode, "paired_metrics": len(rows),
        "sensitivity_checks": len(pivot),
    }, indent=2))


if __name__ == "__main__":
    main()
