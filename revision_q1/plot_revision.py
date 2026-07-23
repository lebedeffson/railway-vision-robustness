from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision_q1.analyze import add_endpoints, apply_normalization_aliases, boolean_series
from revision_q1.protocol import load_protocol, output_root


def save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def errorbar_table(frame: pd.DataFrame, label: str, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, max(4, len(frame) * 0.32)))
    y = np.arange(len(frame))
    values = frame["estimate"].to_numpy(float)
    axis.errorbar(
        values, y,
        xerr=np.maximum(0, np.vstack((
            values - frame["ci_low"], frame["ci_high"] - values
        ))),
        fmt="o", capsize=3,
    )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, frame[label].astype(str))
    axis.set_title(title)
    save(figure, path)


def main() -> None:
    protocol = load_protocol()
    root = output_root(protocol)
    parser = argparse.ArgumentParser(description="Render all frozen Q1 figures from CSV")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=root)
    args = parser.parse_args()
    figures = args.output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    correlations = pd.read_csv(args.output / "tables/03_baseline_metric_correlations.csv")
    scope = correlations[
        (correlations["correlation"] == "spearman") & (correlations["layer"] == "mean")
    ].copy()
    scope["label"] = scope.apply(
        lambda row: (
            f"{row['task']}: {row['metric']} "
            f"[Holm p={row['holm_corrected_p']:.3g}, n={int(row['sequences'])} scenes]"
        ), axis=1,
    )
    errorbar_table(
        scope, "label", "Metric correlations with 95% sequence-cluster CI",
        figures / "02_metric_correlations_with_ci.png",
    )
    deltas = pd.read_csv(args.output / "tables/04_tnorm_vs_baseline_bootstrap.csv")
    scope = deltas[deltas["layer"] == "mean"].copy()
    scope["estimate"] = scope["delta_rho"]
    scope["label"] = scope.apply(
        lambda row: (
            f"{row['tnorm']} - {row['baseline']} "
            f"[Holm p={row['holm_corrected_p']:.3g}, n={int(row['sequences'])}]"
        ), axis=1,
    )
    errorbar_table(
        scope, "label", "Absolute Spearman gain; Holm p shown in CSV",
        figures / "03_tnorm_minus_baseline_delta_rho.png",
    )
    gains = pd.read_csv(args.output / "tables/12_multiple_comparison_corrections.csv")
    for task, filename, title in (
        ("damage", "04_damage_model_incremental_gain.png", "Damage-model gains"),
        ("recovery", "05_recovery_model_incremental_gain.png", "Recovery-model gains"),
    ):
        scope = gains[gains["task"] == task].copy()
        scope["label"] = scope.apply(
            lambda row: (
                f"{row['endpoint']}: {row['algorithm']}: {row['comparison']}: {row['metric']} "
                f"[Holm p={row['holm_corrected_p']:.3g}, n={int(row['sequences'])}]"
            ), axis=1,
        )
        errorbar_table(scope, "label", f"{title}, 95% sequence-cluster CI", figures / filename)
    sensitivity = pd.read_csv(args.output / "tables/08_checkpoint_sensitivity.csv")
    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(sensitivity))
    axis.errorbar(
        x, sensitivity["estimate_stage2_best"],
        yerr=np.maximum(0, np.vstack((
            sensitivity["estimate_stage2_best"] - sensitivity["ci_low_stage2_best"],
            sensitivity["ci_high_stage2_best"] - sensitivity["estimate_stage2_best"],
        ))), fmt="o", capsize=3, label="Stage 2 best",
    )
    axis.errorbar(
        x, sensitivity["estimate_stage1_best"],
        yerr=np.maximum(0, np.vstack((
            sensitivity["estimate_stage1_best"] - sensitivity["ci_low_stage1_best"],
            sensitivity["ci_high_stage1_best"] - sensitivity["estimate_stage1_best"],
        ))), fmt="x", capsize=3, label="Stage 1 best",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, sensitivity["metric"], rotation=45, ha="right")
    axis.set_title("Checkpoint sensitivity: effect direction")
    axis.legend()
    save(figure, figures / "06_checkpoint_sensitivity.png")
    mode = json.loads(
        (args.output / "config/normalization_selection.json").read_text(encoding="utf-8")
    )["selected_normalization"]
    raw = add_endpoints(apply_normalization_aliases(pd.read_csv(args.matrix), mode))
    if "selected_best" in raw:
        raw = raw[boolean_series(raw["selected_best"])]
    recovery = raw[raw["defense"] != "none"].groupby(
        ["sequence_id", "image_path", "defense"], as_index=False
    ).agg(delta_f1_recovery=("delta_f1_recovery", "mean"), g_recovery=("g_recovery", "mean"))
    figure, axis = plt.subplots(figsize=(7, 5))
    for defense, group in recovery.groupby("defense"):
        axis.scatter(group["g_recovery"], group["delta_f1_recovery"], s=8, alpha=.35, label=defense)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_xlabel("Feature G (unclipped)")
    axis.set_ylabel("F1 recovery")
    axis.set_title(f"F1 recovery versus G; {recovery.sequence_id.nunique()} scenes")
    axis.legend(fontsize=7)
    save(figure, figures / "07_f1_recovery_vs_feature_G.png")
    difficulty = pd.read_csv(args.output / "tables/09_scene_difficulty_analysis.csv")
    for field, filename, title in (
        ("object_count_stratum", "08_object_count_stratification.png", "Object-count strata"),
        ("small_object_stratum", "09_small_object_stratification.png", "Small-object strata"),
    ):
        scope = difficulty[difficulty["stratum_type"] == field].dropna(subset=["delta_r2"])
        figure, axis = plt.subplots(figsize=(8, 5))
        for task, group in scope.groupby("task"):
            axis.plot(group["stratum"], group["delta_r2"], "o-", label=task)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title(f"{title}: D3/R3 incremental R2 (exploratory if <5 scenes)")
        axis.tick_params(axis="x", rotation=30)
        axis.legend()
        save(figure, figures / filename)
    spatial = pd.read_csv(args.output / "tables/10_spatial_stress_test.csv")
    figure, axis = plt.subplots(figsize=(9, 5))
    values = spatial["delta_f1"].to_numpy(float)
    axis.bar(
        np.arange(len(spatial)), values,
        yerr=np.maximum(0, np.vstack((
            values - spatial["delta_f1_ci_low"],
            spatial["delta_f1_ci_high"] - values,
        ))), capsize=3,
    )
    axis.set_xticks(np.arange(len(spatial)), spatial["scenario"], rotation=45, ha="right")
    axis.set_title(f"Fixed-affine stress test; n={int(spatial['sequences'].max())} scenes")
    save(figure, figures / "10_spatial_stress_test.png")
    adaptive = raw[(raw["attack"] == "pgd") & (raw["defense"] == "tnorm")].groupby(
        ["epsilon_px", "steps", "adaptive"], as_index=False
    ).agg(f1=("f1_defended", "mean"), sequences=("sequence_id", "nunique"))
    figure, axis = plt.subplots(figsize=(8, 5))
    for adaptive_value, group in adaptive.groupby("adaptive"):
        group = group.sort_values(["steps", "epsilon_px"])
        axis.plot(group["epsilon_px"], group["f1"], "o-", label=f"adaptive={adaptive_value}")
    axis.set_xlabel("epsilon (pixel levels / 255)")
    axis.set_ylabel("Defended F1")
    axis.set_title("Adaptive versus non-adaptive PGD")
    axis.legend()
    save(figure, figures / "11_adaptive_vs_nonadaptive.png")
    required = [
        "01_feature_normalization_by_layer.png", "02_metric_correlations_with_ci.png",
        "03_tnorm_minus_baseline_delta_rho.png", "04_damage_model_incremental_gain.png",
        "05_recovery_model_incremental_gain.png", "06_checkpoint_sensitivity.png",
        "07_f1_recovery_vs_feature_G.png", "08_object_count_stratification.png",
        "09_small_object_stratification.png", "10_spatial_stress_test.png",
        "11_adaptive_vs_nonadaptive.png",
    ]
    missing = [name for name in required if not (figures / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required Q1 figures: {missing}")
    print(json.dumps({"figures": len(required), "missing": missing}, indent=2))


if __name__ == "__main__":
    main()
