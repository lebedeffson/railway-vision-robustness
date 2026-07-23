from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from canonical_m4_common import OUTPUT_ROOT, atomic_json, load_protocol
from revision_q1.analyze import boolean_series, run_analysis
from revision_q1.statistics import (
    paired_cluster_delta_correlation,
    unique_cluster_samples,
)


MATRIX = OUTPUT_ROOT / "attacks/canonical_test_matrix.csv"
DESTINATION = OUTPUT_ROOT / "models"


def analysis_protocol(protocol: dict) -> dict:
    damage_d2 = [
        "attack", "epsilon", "actual_L1", "actual_L2", "actual_Linf",
        "cosine", "l1_distance", "normalized_l2", "mse", "mae",
        "pearson", "spearman", "entropy_shift",
    ]
    damage_d3 = damage_d2 + [
        "product", "lukasiewicz", "c_sp_object", "c_dir", "c_atk_object",
    ]
    recovery_r2 = [
        "attack", "epsilon", "defense", "f1_attack",
        "cosine_recovery", "l1_recovery", "normalized_l2_recovery",
        "mse_recovery", "mae_recovery", "pearson_recovery",
        "spearman_recovery", "entropy_recovery",
    ]
    recovery_r3 = recovery_r2 + [
        "product_recovery", "lukasiewicz_recovery", "G_raw", "C_def",
    ]
    return {
        "damage_models": {
            "D2": damage_d2,
            "D3": damage_d3,
            # The shared analysis implementation also emits the legacy D4
            # comparison; it is an explicitly identical non-primary alias.
            "D4": damage_d3,
        },
        "recovery_models": {
            "R2": recovery_r2,
            "R3": recovery_r3,
            "R4": recovery_r3,
        },
        "algorithms": {"primary": ["ridge", "elasticnet"]},
        "statistics": {"group_folds": 5},
        "random_seed": int(protocol["statistics"]["bootstrap_seed"]),
    }


def object_global(data: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    selected = data[
        boolean_series(data["selected_best"])
        & data["defense"].eq("none")
        & ~boolean_series(data["adaptive"])
    ].copy()
    frame = selected.groupby(
        ["grouped_scene_id", "image_path", "attack", "epsilon_px"],
        as_index=False,
    ).agg(
        delta_recall_damage=("delta_recall_damage", "mean"),
        c_atk_object=("c_atk_object", "mean"),
        c_atk_global=("c_atk_global", "mean"),
    )
    result = paired_cluster_delta_correlation(
        frame.rename(columns={"grouped_scene_id": "sequence_id"}),
        "delta_recall_damage",
        "c_atk_object",
        "c_atk_global",
        iterations=iterations,
        seed=seed,
    )
    return pd.DataFrame([{
        "endpoint": "delta_recall_damage",
        "comparison": "C_atk_object_vs_C_atk_global",
        **result,
        "grouped_scenes": int(frame["grouped_scene_id"].nunique()),
        "frames": int(frame["image_path"].nunique()),
    }])


def adaptive_comparison(
    data: pd.DataFrame, iterations: int, seed: int
) -> pd.DataFrame:
    selected = data[
        boolean_series(data["selected_best"])
        & data["attack"].eq("pgd")
        & data["defense"].eq("Product_preprocessing")
    ].copy()
    scene = selected.groupby(
        ["grouped_scene_id", "epsilon_px", "adaptive"], as_index=False
    ).agg(f1=("f1_defended", "mean"), recall=("recall_defended", "mean"))
    rows = []
    for epsilon, scope in scene.groupby("epsilon_px"):
        pivot = scope.pivot(
            index="grouped_scene_id", columns="adaptive", values=["f1", "recall"]
        ).dropna()
        if True not in pivot.columns.levels[1] or False not in pivot.columns.levels[1]:
            continue
        clusters = pivot.index.astype(str).to_numpy()
        for metric in ("f1", "recall"):
            delta = (
                pivot[(metric, True)].to_numpy(float)
                - pivot[(metric, False)].to_numpy(float)
            )
            observed = float(delta.mean())
            samples = []
            for indices, multiplicity in unique_cluster_samples(
                clusters, iterations, seed
            ):
                samples.extend([float(delta[indices].mean())] * multiplicity)
            values = np.asarray(samples)
            lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
            upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
            rows.append({
                "epsilon_px": epsilon,
                "metric": metric,
                "estimate_adaptive_minus_nonadaptive": observed,
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "p_value": min(1.0, 2 * min(lower, upper)),
                "grouped_scenes": len(clusters),
            })
    return pd.DataFrame(rows)


def conclusion(gains: pd.DataFrame, comparison: str) -> dict:
    scope = gains[
        gains["comparison"].eq(comparison)
        & gains["algorithm"].eq("ridge")
        & gains["endpoint"].isin(
            ["delta_f1_damage", "delta_f1_recovery"]
        )
    ].copy()
    primary = scope[scope["metric"].eq("delta_mae")]
    if primary.empty:
        return {"status": "not_estimable"}
    row = primary.iloc[0]
    statistical = (
        float(row["holm_corrected_p"]) < 0.05
        and not (float(row["ci_low"]) <= 0 <= float(row["ci_high"]))
    )
    practical = (
        float(row["relative_mae_reduction"]) >= 5.0
        or bool((scope["metric"].eq("delta_r2") & (scope["estimate"] >= 0.05)).any())
        or bool(
            (
                scope["metric"].eq("delta_spearman")
                & (scope["estimate"].abs() >= 0.05)
            ).any()
        )
    )
    return {
        "status": (
            "positive" if statistical and practical
            else "small_effect" if statistical
            else "negative"
        ),
        "statistically_confirmed": statistical,
        "practical_threshold_met": practical,
        "delta_mae": float(row["estimate"]),
        "relative_mae_reduction_percent":
            float(row["relative_mae_reduction"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "holm_p": float(row["holm_corrected_p"]),
    }


def run() -> dict:
    protocol = load_protocol()
    data = pd.read_csv(MATRIX, low_memory=False)
    mode = str(data["normalization"].dropna().unique()[0])
    iterations = int(protocol["statistics"]["bootstrap_iterations"])
    DESTINATION.mkdir(parents=True, exist_ok=True)
    summary = run_analysis(
        data,
        analysis_protocol(protocol),
        mode,
        iterations,
        DESTINATION,
    )
    tables = DESTINATION / "tables"
    gains = pd.read_csv(tables / "12_multiple_comparison_corrections.csv")
    h3 = object_global(
        data, iterations, int(protocol["statistics"]["bootstrap_seed"])
    )
    h4 = adaptive_comparison(
        data, iterations, int(protocol["statistics"]["bootstrap_seed"])
    )
    h3.to_csv(tables / "10_object_global_comparison.csv", index=False)
    h4.to_csv(tables / "11_adaptive_comparison.csv", index=False)
    outcome = {
        "status": "PASS",
        **summary,
        "D3_vs_D2": conclusion(gains, "D3_vs_D2"),
        "R3_vs_R2": conclusion(gains, "R3_vs_R2"),
        "object_vs_global_rows": len(h3),
        "adaptive_vs_nonadaptive_rows": len(h4),
        "limitations": [
            "five independent test scenes limit statistical power",
            "micro-overfit metrics are not validation or test metrics",
            "JPEG is supplementary non-adaptive only",
        ],
    }
    atomic_json(DESTINATION / "analysis_summary.json", outcome)
    return outcome


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
