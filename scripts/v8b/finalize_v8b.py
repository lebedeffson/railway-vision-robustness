from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, trim_mean, wilcoxon
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs/person_v8b/results/OOF_PREDICTIONS.csv"
DEFAULT_OUTPUT = ROOT / "outputs/person_v8b/final"
MODELS = ("U0", "U1", "U2", "U3")
EXPECTED = {
    "U2_scene_macro_MAE": 1.5954889301566773,
    "U3_scene_macro_MAE": 1.7679771024802038,
    "relative_MAE_reduction_percent": -10.810991481250088,
    "scene_wins": 5,
}
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260726


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def expected_calibration_error(
    actual: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if index == bins - 1
            else probability < edges[index + 1]
        )
        if selected.any():
            value += selected.mean() * abs(
                float(actual[selected].mean()) - float(probability[selected].mean())
            )
    return float(value)


def safe_spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    if (
        len(actual) < 2
        or np.var(actual) <= 0
        or np.var(predicted) <= 0
    ):
        return float("nan")
    return abs(float(spearmanr(actual, predicted).statistic))


def read_oof(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "grouped_scene_id",
        "model",
        "fn_actual",
        "fn_prediction",
        "fn_prediction_elasticnet",
        "recall_actual",
        "recall_prediction",
        "unsafe_actual",
        "unsafe_probability",
        "tp",
        "fp",
        "fn",
        "gt_count",
        "normalization_fit_scenes",
        "prototype_fit_scenes",
        "test_used",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"OOF input is missing columns: {missing}")
    if set(frame["model"]) != set(MODELS):
        raise RuntimeError("OOF input must contain exactly U0/U1/U2/U3")
    if frame["grouped_scene_id"].nunique() != 15:
        raise RuntimeError("OOF input must contain exactly 15 grouped scenes")
    if frame["test_used"].astype(str).str.lower().isin({"true", "1"}).any():
        raise RuntimeError("OOF input reports test access")
    if not frame["normalization_fit_scenes"].eq(14).all():
        raise RuntimeError("Normalization was not fit on 14 train scenes")
    if not frame["prototype_fit_scenes"].eq(14).all():
        raise RuntimeError("Prototypes were not fit on 14 train scenes")
    numeric = frame[
        [
            "fn_actual",
            "fn_prediction",
            "fn_prediction_elasticnet",
            "recall_prediction",
            "unsafe_probability",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("OOF predictions contain NaN/Inf")
    counts = frame.groupby("model").size()
    if counts.nunique() != 1 or int(counts.iloc[0]) != 1085:
        raise RuntimeError(f"Unexpected per-model frame counts: {counts.to_dict()}")
    return frame


def per_scene_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, scene), group in oof.groupby(
        ["model", "grouped_scene_id"], sort=True
    ):
        actual_fn = group["fn_actual"].to_numpy(dtype=float)
        ridge_fn = group["fn_prediction"].to_numpy(dtype=float)
        elastic_fn = group["fn_prediction_elasticnet"].to_numpy(dtype=float)
        recall_actual = pd.to_numeric(
            group["recall_actual"], errors="coerce"
        ).to_numpy(dtype=float)
        recall_mask = np.isfinite(recall_actual)
        unsafe = group["unsafe_actual"].to_numpy(dtype=int)
        probability = group["unsafe_probability"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "grouped_scene_id": scene,
                "frames": len(group),
                "fn_per_frame": float(actual_fn.mean()),
                "ridge_mae_fn": mean_absolute_error(actual_fn, ridge_fn),
                "elasticnet_mae_fn": mean_absolute_error(actual_fn, elastic_fn),
                "ridge_mae_recall": (
                    mean_absolute_error(
                        recall_actual[recall_mask],
                        group["recall_prediction"].to_numpy(dtype=float)[recall_mask],
                    )
                    if recall_mask.any()
                    else np.nan
                ),
                "ridge_r2_fn": (
                    r2_score(actual_fn, ridge_fn)
                    if len(group) > 1 and np.var(actual_fn) > 0
                    else np.nan
                ),
                "elasticnet_r2_fn": (
                    r2_score(actual_fn, elastic_fn)
                    if len(group) > 1 and np.var(actual_fn) > 0
                    else np.nan
                ),
                "ridge_abs_spearman_fn": safe_spearman(actual_fn, ridge_fn),
                "elasticnet_abs_spearman_fn": safe_spearman(actual_fn, elastic_fn),
                "brier": brier_score_loss(unsafe, probability),
                "ece": expected_calibration_error(unsafe, probability),
                "auroc": (
                    roc_auc_score(unsafe, probability)
                    if len(np.unique(unsafe)) == 2
                    else np.nan
                ),
                "auprc": (
                    average_precision_score(unsafe, probability)
                    if len(np.unique(unsafe)) == 2
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_models(scene: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        selected = scene[scene["model"].eq(model)]
        for estimator, prefix in (("Ridge", "ridge"), ("ElasticNet", "elasticnet")):
            mae = selected[f"{prefix}_mae_fn"].to_numpy(dtype=float)
            weights = selected["frames"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "estimator": estimator,
                    "scene_macro_mae_fn": float(np.mean(mae)),
                    "frame_weighted_mae_fn": float(np.average(mae, weights=weights)),
                    "median_scene_mae_fn": float(np.median(mae)),
                    "trimmed_mean_scene_mae_fn": float(trim_mean(mae, 0.1)),
                    "scene_macro_r2_fn": float(
                        np.nanmean(selected[f"{prefix}_r2_fn"].to_numpy(dtype=float))
                    ),
                    "scene_macro_abs_spearman_fn": float(
                        np.nanmean(
                            selected[
                                f"{prefix}_abs_spearman_fn"
                            ].to_numpy(dtype=float)
                        )
                    ),
                    "scene_macro_mae_recall": (
                        float(np.nanmean(selected["ridge_mae_recall"]))
                        if estimator == "Ridge"
                        else np.nan
                    ),
                    "scene_macro_brier": (
                        float(np.mean(selected["brier"]))
                        if estimator == "Ridge"
                        else np.nan
                    ),
                    "scene_macro_ece": (
                        float(np.mean(selected["ece"]))
                        if estimator == "Ridge"
                        else np.nan
                    ),
                    "scene_macro_auroc": (
                        float(np.nanmean(selected["auroc"]))
                        if estimator == "Ridge"
                        else np.nan
                    ),
                    "scene_macro_auprc": (
                        float(np.nanmean(selected["auprc"]))
                        if estimator == "Ridge"
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_samples(values: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_ITERATIONS, len(values))
    )
    return values[indices].mean(axis=1)


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    estimates = bootstrap_samples(values)
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def hierarchy_tables(
    scene: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = (("U1", "U0"), ("U2", "U1"), ("U3", "U2"))
    metrics = {
        "mae_fn": ("ridge_mae_fn", "lower"),
        "mae_recall": ("ridge_mae_recall", "lower"),
        "brier": ("brier", "lower"),
        "ece": ("ece", "lower"),
        "auroc": ("auroc", "higher"),
        "auprc": ("auprc", "higher"),
    }
    rows: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for candidate, reference in pairs:
        left = scene[scene["model"].eq(candidate)].set_index("grouped_scene_id")
        right = scene[scene["model"].eq(reference)].set_index("grouped_scene_id")
        common = sorted(set(left.index) & set(right.index))
        for metric, (column, direction) in metrics.items():
            delta = (
                left.loc[common, column].to_numpy(dtype=float)
                - right.loc[common, column].to_numpy(dtype=float)
            )
            delta = delta[np.isfinite(delta)]
            low, high = bootstrap_ci(delta)
            if len(delta) and np.any(delta != 0):
                p_value = float(
                    wilcoxon(
                        delta,
                        alternative="less" if direction == "lower" else "greater",
                    ).pvalue
                )
            else:
                p_value = 1.0
            raw_p.append(p_value)
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "direction": direction,
                    "delta": float(np.mean(delta)),
                    "relative_change_percent": (
                        float(
                            -np.mean(delta)
                            / max(float(right.loc[common, column].mean()), 1e-12)
                            * 100.0
                        )
                        if direction == "lower"
                        else float("nan")
                    ),
                    "ci95_low": low,
                    "ci95_high": high,
                    "scene_wins": int(
                        (delta < 0).sum()
                        if direction == "lower"
                        else (delta > 0).sum()
                    ),
                    "scene_losses": int(
                        (delta > 0).sum()
                        if direction == "lower"
                        else (delta < 0).sum()
                    ),
                    "scene_ties": int((delta == 0).sum()),
                    "scenes": len(delta),
                    "raw_p": p_value,
                }
            )
    adjusted = holm_adjust(raw_p)
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_p"] = value
        row["holm_pass_0_05"] = value < 0.05
    hierarchy = pd.DataFrame(rows)
    holm = hierarchy[
        [
            "candidate",
            "reference",
            "metric",
            "raw_p",
            "holm_p",
            "holm_pass_0_05",
        ]
    ].copy()
    return hierarchy, holm


def loso_sensitivity(scene: pd.DataFrame) -> pd.DataFrame:
    u2 = scene[scene["model"].eq("U2")].set_index("grouped_scene_id")
    u3 = scene[scene["model"].eq("U3")].set_index("grouped_scene_id")
    scenes = sorted(set(u2.index) & set(u3.index))
    rows = []
    for excluded in scenes:
        kept = [scene_id for scene_id in scenes if scene_id != excluded]
        u2_mae = float(u2.loc[kept, "ridge_mae_fn"].mean())
        u3_mae = float(u3.loc[kept, "ridge_mae_fn"].mean())
        reduction = (u2_mae - u3_mae) / max(u2_mae, 1e-12) * 100.0
        rows.append(
            {
                "excluded_scene": excluded,
                "U2_MAE": u2_mae,
                "U3_MAE": u3_mae,
                "relative_MAE_reduction_percent": reduction,
                "winner": "U3" if u3_mae < u2_mae else "U2" if u2_mae < u3_mae else "TIE",
                "effect_sign_changed_vs_full": reduction > 0,
            }
        )
    return pd.DataFrame(rows)


def risk_coverage(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("U2", "U3"):
        selected = oof[oof["model"].eq(model)]
        for coverage in np.round(np.arange(1.0, 0.0, -0.1), 1):
            per_scene = []
            for _, group in selected.groupby("grouped_scene_id", sort=True):
                count = max(1, int(math.ceil(float(coverage) * len(group))))
                accepted = group.nsmallest(count, "unsafe_probability")
                tp = int(accepted["tp"].sum())
                fn = int(accepted["fn"].sum())
                per_scene.append(
                    {
                        "recall": tp / (tp + fn) if tp + fn else 1.0,
                        "fn_per_frame": fn / len(accepted),
                    }
                )
            rows.append(
                {
                    "model": model,
                    "coverage": float(coverage),
                    "operator_referral_fraction": 1.0 - float(coverage),
                    "scene_macro_accepted_recall": float(
                        np.mean([value["recall"] for value in per_scene])
                    ),
                    "scene_macro_accepted_fn_per_frame": float(
                        np.mean([value["fn_per_frame"] for value in per_scene])
                    ),
                }
            )
    return pd.DataFrame(rows)


def calibration_table(scene: pd.DataFrame) -> pd.DataFrame:
    return (
        scene.groupby("model", sort=False)[["brier", "ece", "auroc", "auprc"]]
        .mean()
        .reset_index()
        .rename(
            columns={
                "brier": "scene_macro_brier",
                "ece": "scene_macro_ece",
                "auroc": "scene_macro_auroc",
                "auprc": "scene_macro_auprc",
            }
        )
    )


def article_tables(
    output: Path,
    aggregate: pd.DataFrame,
    scene: pd.DataFrame,
    calibration: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "development_scenes": 15,
                "development_frames": 1085,
                "target": "person",
                "detector": "frozen B0 YOLO11m",
                "feature_layers": "P3/P4/P5",
                "regions": "global/proposal/background",
                "validation": "nested scene LOSO",
                "test": "SEALED; access_count=0",
            }
        ]
    ).to_csv(tables / "TABLE_1_DATA_PROTOCOL.csv", index=False)
    pd.DataFrame(
        [
            {"model": "U0", "features": "mean/max confidence"},
            {"model": "U1", "features": "U0 + prediction statistics"},
            {"model": "U2", "features": "U1 + standard representation metrics"},
            {"model": "U3", "features": "U2 + Product/Łukasiewicz consistency"},
        ]
    ).to_csv(tables / "TABLE_2_MODEL_HIERARCHY.csv", index=False)
    aggregate[
        aggregate["model"].isin(["U2", "U3"])
        & aggregate["estimator"].eq("Ridge")
    ][
        [
            "model",
            "scene_macro_mae_fn",
            "scene_macro_r2_fn",
            "scene_macro_abs_spearman_fn",
        ]
    ].to_csv(tables / "TABLE_3_PRIMARY_RESULT.csv", index=False)
    comparison = scene[scene["model"].isin(["U2", "U3"])].pivot(
        index=["grouped_scene_id", "frames", "fn_per_frame"],
        columns="model",
        values="ridge_mae_fn",
    ).reset_index()
    comparison.columns.name = None
    comparison = comparison.rename(columns={"U2": "U2_MAE", "U3": "U3_MAE"})
    comparison["delta_U3_minus_U2"] = comparison["U3_MAE"] - comparison["U2_MAE"]
    comparison["winner"] = np.where(
        comparison["delta_U3_minus_U2"] < 0,
        "U3",
        np.where(comparison["delta_U3_minus_U2"] > 0, "U2", "TIE"),
    )
    comparison.to_csv(tables / "TABLE_4_PER_SCENE.csv", index=False)
    calibration.to_csv(tables / "TABLE_5_CALIBRATION.csv", index=False)
    coverage.to_csv(tables / "TABLE_6_RISK_COVERAGE.csv", index=False)


def style_axis(axis: Any) -> None:
    axis.grid(True, alpha=0.25, linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def figures(
    output: Path,
    scene: pd.DataFrame,
    loso: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(12, 3.2))
    axis.axis("off")
    labels = [
        "Frozen B0",
        "P3/P4/P5\nextraction",
        "Train-only\nnormalization",
        "U0 → U1 → U2 → U3",
        "Nested\n15-scene LOSO",
        "Scene bootstrap\nand frozen gate",
    ]
    xs = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels, strict=True)):
        axis.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.5", "fc": "#eef4fb", "ec": "#355c7d"},
        )
        if index < len(labels) - 1:
            axis.annotate(
                "",
                xy=(xs[index + 1] - 0.07, 0.5),
                xytext=(x + 0.07, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#355c7d"},
            )
    axis.set_title("Canonical v8b frozen-detector failure-risk pipeline", pad=18)
    fig.tight_layout()
    fig.savefig(directory / "FIGURE_1_PIPELINE.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    u2 = scene[scene["model"].eq("U2")].set_index("grouped_scene_id")
    u3 = scene[scene["model"].eq("U3")].set_index("grouped_scene_id")
    scenes = sorted(set(u2.index) & set(u3.index))
    fig, axis = plt.subplots(figsize=(9, 7))
    positions = np.arange(len(scenes))
    for position, scene_id in zip(positions, scenes, strict=True):
        left = float(u2.loc[scene_id, "ridge_mae_fn"])
        right = float(u3.loc[scene_id, "ridge_mae_fn"])
        axis.plot([left, right], [position, position], color="#9aa6b2", linewidth=1)
    axis.scatter(
        u2.loc[scenes, "ridge_mae_fn"], positions, label="U2", color="#2b6cb0"
    )
    axis.scatter(
        u3.loc[scenes, "ridge_mae_fn"], positions, label="U3", color="#c53030"
    )
    axis.set_yticks(positions, [str(value) for value in scenes], fontsize=7)
    axis.set_xlabel("MAE FN/frame (lower is better)")
    axis.set_ylabel("Grouped scene")
    axis.legend()
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(directory / "FIGURE_2_PER_SCENE_MAE.png", dpi=220)
    plt.close(fig)

    delta = (
        u3.loc[scenes, "ridge_mae_fn"].to_numpy(dtype=float)
        - u2.loc[scenes, "ridge_mae_fn"].to_numpy(dtype=float)
    )
    boot = bootstrap_samples(delta)
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.hist(boot, bins=50, color="#6b8fb3", alpha=0.85)
    axis.axvline(0, color="#222222", linestyle="--", label="No difference")
    axis.axvline(float(delta.mean()), color="#c53030", label="Observed mean")
    axis.set_xlabel("Bootstrap mean ΔMAE (U3 − U2)")
    axis.set_ylabel("Frequency")
    axis.legend()
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(directory / "FIGURE_3_BOOTSTRAP_DELTA.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 6))
    colors = np.where(
        loso["relative_MAE_reduction_percent"].to_numpy() > 0,
        "#2f855a",
        "#c53030",
    )
    axis.barh(
        np.arange(len(loso)),
        loso["relative_MAE_reduction_percent"],
        color=colors,
    )
    axis.axvline(0, color="#222222", linewidth=1)
    axis.set_yticks(
        np.arange(len(loso)), loso["excluded_scene"].astype(str), fontsize=7
    )
    axis.set_xlabel("Relative MAE reduction after excluding scene (%)")
    axis.set_ylabel("Excluded grouped scene")
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(directory / "FIGURE_4_LOSO_SENSITIVITY.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for model, color in (("U2", "#2b6cb0"), ("U3", "#c53030")):
        selected = coverage[coverage["model"].eq(model)].sort_values("coverage")
        axes[0].plot(
            selected["coverage"],
            selected["scene_macro_accepted_fn_per_frame"],
            marker="o",
            label=model,
            color=color,
        )
        axes[1].plot(
            selected["coverage"],
            selected["scene_macro_accepted_recall"],
            marker="o",
            label=model,
            color=color,
        )
    axes[0].set_ylabel("Accepted FN/frame")
    axes[1].set_ylabel("Accepted detector recall")
    for axis in axes:
        axis.set_xlabel("Coverage")
        axis.legend()
        style_axis(axis)
    fig.tight_layout()
    fig.savefig(directory / "FIGURE_5_RISK_COVERAGE.png", dpi=220)
    plt.close(fig)


def redacted_oof(oof: pd.DataFrame, output: Path) -> None:
    frame = oof.copy()
    if "image_path" in frame:
        frame["image_id"] = frame["image_path"].map(lambda value: Path(value).name)
        frame = frame.drop(columns=["image_path"])
    frame.to_csv(output / "OOF_INPUT_REDACTED.csv", index=False)


def finalize(input_path: Path, output: Path) -> dict[str, Any]:
    oof = read_oof(input_path)
    output.mkdir(parents=True, exist_ok=True)
    scene = per_scene_metrics(oof)
    aggregate = aggregate_models(scene)
    hierarchy, holm = hierarchy_tables(scene)
    loso = loso_sensitivity(scene)
    calibration = calibration_table(scene)
    coverage = risk_coverage(oof)

    scene.to_csv(output / "PER_SCENE_FINAL.csv", index=False)
    aggregate.to_csv(output / "MODEL_COMPARISON.csv", index=False)
    hierarchy.to_csv(output / "BOOTSTRAP_FINAL.csv", index=False)
    loso.to_csv(output / "LOSO_FINAL.csv", index=False)
    calibration.to_csv(output / "CALIBRATION_FINAL.csv", index=False)
    coverage.to_csv(output / "RISK_COVERAGE_FINAL.csv", index=False)
    holm.to_csv(output / "HOLM_FINAL.csv", index=False)
    redacted_oof(oof, output)
    article_tables(output, aggregate, scene, calibration, coverage)
    figures(output, scene, loso, coverage)

    ridge = aggregate[aggregate["estimator"].eq("Ridge")].set_index("model")
    u2_mae = float(ridge.loc["U2", "scene_macro_mae_fn"])
    u3_mae = float(ridge.loc["U3", "scene_macro_mae_fn"])
    relative_reduction = (u2_mae - u3_mae) / u2_mae * 100.0
    primary = hierarchy[
        hierarchy["candidate"].eq("U3")
        & hierarchy["reference"].eq("U2")
        & hierarchy["metric"].eq("mae_fn")
    ].iloc[0]
    checks = {
        "U2_scene_macro_MAE_exact": abs(
            u2_mae - EXPECTED["U2_scene_macro_MAE"]
        )
        <= 1e-9,
        "U3_scene_macro_MAE_exact": abs(
            u3_mae - EXPECTED["U3_scene_macro_MAE"]
        )
        <= 1e-9,
        "relative_reduction_exact": abs(
            relative_reduction - EXPECTED["relative_MAE_reduction_percent"]
        )
        <= 1e-6,
        "scene_wins_exact": int(primary["scene_wins"])
        == EXPECTED["scene_wins"],
        "all_15_scenes": scene["grouped_scene_id"].nunique() == 15,
        "all_1085_frames_per_model": bool(
            oof.groupby("model").size().eq(1085).all()
        ),
        "test_access_count_zero": not oof["test_used"].astype(bool).any(),
        "no_absolute_paths_in_redacted_input": not (
            output / "OOF_INPUT_REDACTED.csv"
        ).read_text(encoding="utf-8").__contains__("/" + "home/"),
        "secondary_endpoints_cannot_rescue": True,
    }
    metrics = {
        "protocol_id": "canonical-v8b-person-failure-risk-v1",
        "status": "DEVELOPMENT_FAIL",
        "scientific_status": "SCIENTIFIC_FAIL",
        "primary_endpoint": "scene_macro_MAE_fn_per_frame",
        "U2_scene_macro_MAE": u2_mae,
        "U3_scene_macro_MAE": u3_mae,
        "delta_MAE_U3_minus_U2": u3_mae - u2_mae,
        "relative_MAE_reduction_percent": relative_reduction,
        "scene_wins": int(primary["scene_wins"]),
        "scene_losses": int(primary["scene_losses"]),
        "scene_ties": int(primary["scene_ties"]),
        "paired_scene_bootstrap_ci95": [
            float(primary["ci95_low"]),
            float(primary["ci95_high"]),
        ],
        "minimum_LOSO_relative_reduction_percent": float(
            loso["relative_MAE_reduction_percent"].min()
        ),
        "test_status": "SEALED",
        "test_access_count": 0,
        "attacks_status": "OUT_OF_SCOPE",
        "secondary_endpoints_can_rescue": False,
        "checks": checks,
    }
    atomic_json(output / "FINAL_METRICS.json", metrics)
    artifact_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"FINALIZATION_AUDIT.json"}
        and "article/" not in path.relative_to(output).as_posix()
        and "bundles/" not in path.relative_to(output).as_posix()
    }
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "source": "saved development OOF predictions only",
        "source_sha256": sha256_file(input_path),
        "checkpoint_read": False,
        "raw_images_read": False,
        "test_read": False,
        "test_access_count": 0,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "checks": checks,
        "artifact_sha256": artifact_hashes,
    }
    atomic_json(output / "FINALIZATION_AUDIT.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"V8b finalization audit failed: {checks}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently finalize canonical v8b from saved OOF rows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    result = finalize(arguments.input.resolve(), arguments.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
