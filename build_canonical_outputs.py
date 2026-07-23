from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finalize_canonical_v2 import h3_object_vs_global, h4_adaptive
from revision_q1.analyze import add_endpoints, apply_normalization_aliases, boolean_series


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/canonical_v2"
RAW = ROOT / "raw/canonical_test.csv"
ANALYSIS = ROOT / "analysis_test/tables"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"


def selected_mode() -> str:
    payload = json.loads(
        (ROOT / "normalization/normalization_selection.json").read_text(encoding="utf-8")
    )
    mode = payload["selected_normalization"]
    if mode not in {"N1_quantile", "N2_robust_sigmoid"}:
        raise RuntimeError(f"Unsupported frozen canonical normalization: {mode}")
    return mode


def save(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def scene_detection_rows(data: pd.DataFrame) -> pd.DataFrame:
    selected = data[boolean_series(data["selected_best"])].copy()
    keys = [
        "sequence_id", "image_path", "attack", "adaptive", "epsilon_px",
        "steps", "seed", "defense",
    ]
    selected = selected.drop_duplicates(keys)
    selected["fn_per_frame"] = selected["fn_defended"]
    return selected.groupby(
        ["sequence_id", "attack", "adaptive", "epsilon_px", "steps", "defense"],
        as_index=False,
    ).agg(
        frames=("image_path", "nunique"),
        precision=("precision_defended", "mean"),
        recall=("recall_defended", "mean"),
        f1=("f1_defended", "mean"),
        f2=("f2", "mean"),
        fn_per_frame=("fn_per_frame", "mean"),
        confidence_drop=("confidence_drop", "mean"),
        iou_shift=("iou_shift", "mean"),
    )


def split_summary() -> pd.DataFrame:
    manifest = pd.read_csv(ROOT / "split/split_v2_manifest.csv")
    scene = "sequence_id" if "sequence_id" in manifest else "grouped_scene_id"
    return manifest.groupby("split", as_index=False).agg(
        grouped_scenes=(scene, "nunique"), frames=("image_path", "nunique")
    )


def model_gain(task: str, comparison: str) -> pd.DataFrame:
    gains = pd.read_csv(ANALYSIS / "12_multiple_comparison_corrections.csv")
    return gains[
        gains["task"].eq(task)
        & gains["comparison"].eq(comparison)
    ].reset_index(drop=True)


def render_gain(frame: pd.DataFrame, title: str, name: str) -> None:
    scope = frame[
        frame["endpoint"].isin(["delta_f1_damage", "delta_f1_recovery"])
        & frame["algorithm"].eq("ridge")
    ].copy()
    x = np.arange(len(scope))
    values = scope["estimate"].to_numpy(float)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.errorbar(
        x, values,
        yerr=np.maximum(0, np.vstack((values - scope["ci_low"], scope["ci_high"] - values))),
        fmt="o", capsize=4,
    )
    labels = [f"{metric}\nHolm p={p:.3g}" for metric, p in zip(
        scope["metric"], scope["holm_corrected_p"], strict=True
    )]
    axis.set_xticks(x, labels)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_title(f"{title}; 95% grouped-scene bootstrap CI")
    save(figure, name)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW, low_memory=False)
    mode = selected_mode()
    canonical = add_endpoints(apply_normalization_aliases(raw, mode))
    canonical["grouped_scene_id"] = canonical["sequence_id"]

    split_summary().to_csv(TABLES / "01_split_v2_summary.csv", index=False)
    conditions = raw[
        ["attack", "adaptive", "epsilon_px", "steps", "step_size", "random_start", "restarts", "seed"]
    ].drop_duplicates().sort_values(["attack", "adaptive", "epsilon_px", "steps", "seed"])
    conditions.to_csv(TABLES / "04_attack_parameters.csv", index=False)

    per_scene = scene_detection_rows(raw)
    macro = per_scene.groupby(
        ["attack", "adaptive", "epsilon_px", "steps", "defense"], as_index=False
    ).agg(
        independent_scenes=("sequence_id", "nunique"), frames=("frames", "sum"),
        precision=("precision", "mean"), recall=("recall", "mean"),
        f1=("f1", "mean"), f2=("f2", "mean"),
        fn_per_frame=("fn_per_frame", "mean"),
        confidence_drop=("confidence_drop", "mean"), iou_shift=("iou_shift", "mean"),
    )
    macro["aggregation"] = "macro_equal_scene_weight"
    macro.to_csv(TABLES / "05_canonical_robustness.csv", index=False)

    chosen = canonical[boolean_series(canonical["selected_best"])].copy()
    attack_columns = [
        "grouped_scene_id", "subsequence_id", "image_path", "attack", "adaptive",
        "epsilon_px", "steps", "seed", "layer", "c_sp_global", "c_sp_object",
        "c_sp_background", "c_dir", "c_atk_global", "c_atk_object", "c_atk_background",
        "delta_f1_damage", "delta_recall_damage", "false_negatives_increase",
    ]
    chosen[attack_columns].to_csv(TABLES / "06_attack_consistency.csv", index=False)
    defense_columns = [
        "grouped_scene_id", "subsequence_id", "image_path", "attack", "adaptive",
        "epsilon_px", "steps", "seed", "defense", "layer", "product", "lukasiewicz",
        "product_recovery", "lukasiewicz_recovery", "p_clean_preservation",
        "a_attacked_similarity", "r_restored_similarity", "g_recovery", "c_def",
        "delta_f1_recovery", "delta_recall_recovery", "false_negatives_reduction",
        "normalized_quality_recovery",
    ]
    chosen[defense_columns].to_csv(TABLES / "07_defense_consistency.csv", index=False)

    damage = model_gain("damage", "D3_vs_D2")
    recovery = model_gain("recovery", "R3_vs_R2")
    damage.to_csv(TABLES / "08_damage_D2_D3.csv", index=False)
    recovery.to_csv(TABLES / "09_recovery_R2_R3.csv", index=False)
    h3 = pd.DataFrame(h3_object_vs_global(canonical))
    h4_payload = h4_adaptive(canonical)
    h3.to_csv(TABLES / "10_object_global_comparison.csv", index=False)
    pd.DataFrame(h4_payload["comparisons"]).to_csv(
        TABLES / "11_adaptive_comparison.csv", index=False
    )
    per_scene.to_csv(TABLES / "12_per_scene_results.csv", index=False)
    shutil.copy2(ANALYSIS / "13_scene_macro_loso.csv", TABLES / "13_loso_results.csv")
    shutil.copy2(ROOT / "latency/latency_summary.csv", TABLES / "14_latency_summary.csv")
    shutil.copy2(
        ROOT / "audit/canonical_nms_audit.csv",
        TABLES / "15_nms_audit.csv",
    )

    training = ROOT / "training/training_curves.png"
    if not training.is_file():
        raise FileNotFoundError(training)
    shutil.copy2(training, FIGURES / "01_training_curves.png")
    shutil.copy2(ROOT / "calibration/precision_recall_curve.png", FIGURES / "02_precision_recall_curve.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (attack, defense), group in macro[~boolean_series(macro["adaptive"])].groupby(["attack", "defense"]):
        ordered = group.sort_values("epsilon_px")
        axes[0].plot(ordered["epsilon_px"], ordered["f1"], "o-", label=f"{attack}/{defense}")
        axes[1].plot(ordered["epsilon_px"], ordered["recall"], "o-", label=f"{attack}/{defense}")
    axes[0].set(xlabel="epsilon (pixel levels / 255)", ylabel="macro F1")
    axes[1].set(xlabel="epsilon (pixel levels / 255)", ylabel="macro Recall")
    axes[1].legend(fontsize=6)
    save(figure, "03_f1_recall_vs_epsilon.png")
    render_gain(damage, "Canonical D3 minus D2", "04_damage_D3_vs_D2.png")
    render_gain(recovery, "Canonical R3 minus R2", "05_recovery_R3_vs_R2.png")

    deltas = pd.read_csv(ANALYSIS / "04_tnorm_vs_baseline_bootstrap.csv")
    delta_scope = deltas[
        deltas["layer"].eq("mean")
        & deltas["tnorm"].isin(["product", "lukasiewicz", "product_recovery", "lukasiewicz_recovery"])
        & deltas["baseline"].isin(["cosine", "normalized_l2", "cosine_recovery", "normalized_l2_recovery"])
    ].copy()
    values = delta_scope["delta_rho"].to_numpy(float)
    figure, axis = plt.subplots(figsize=(10, max(4, len(delta_scope) * .35)))
    y = np.arange(len(delta_scope))
    axis.errorbar(values, y, xerr=np.maximum(0, np.vstack((values-delta_scope["ci_low"], delta_scope["ci_high"]-values))), fmt="o", capsize=3)
    axis.set_yticks(y, delta_scope["tnorm"] + " vs " + delta_scope["baseline"])
    axis.axvline(0, color="black", linewidth=.8)
    axis.set_title("T-norm minus baseline absolute Spearman; 95% CI")
    save(figure, "06_tnorm_vs_baselines.png")

    figure, axis = plt.subplots(figsize=(8, 4.8))
    if len(h3):
        axis.errorbar(h3["delta_rho"], np.arange(len(h3)), xerr=np.maximum(0, np.vstack((h3["delta_rho"]-h3["ci_low"], h3["ci_high"]-h3["delta_rho"]))), fmt="o", capsize=4)
        axis.set_yticks(np.arange(len(h3)), h3["endpoint"])
    axis.axvline(0, color="black", linewidth=.8)
    axis.set_title("Object minus global consistency; 95% CI")
    save(figure, "07_object_vs_global.png")

    adaptive = pd.DataFrame(h4_payload["comparisons"])
    figure, axis = plt.subplots(figsize=(7, 4.5))
    if len(adaptive):
        values = adaptive["estimate"].to_numpy(float)
        axis.errorbar(adaptive["epsilon_px"], values, yerr=np.maximum(0, np.vstack((values-adaptive["ci_low"], adaptive["ci_high"]-values))), fmt="o-", capsize=4)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set(xlabel="epsilon (pixel levels / 255)", ylabel="adaptive - non-adaptive defended F1")
    save(figure, "08_adaptive_vs_nonadaptive.png")

    scene_plot = per_scene.groupby("sequence_id", as_index=False).agg(f1=("f1", "mean"), recall=("recall", "mean"))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(scene_plot))
    axis.bar(x-.18, scene_plot["f1"], width=.36, label="F1")
    axis.bar(x+.18, scene_plot["recall"], width=.36, label="Recall")
    axis.set_xticks(x, scene_plot["sequence_id"], rotation=25, ha="right")
    axis.legend()
    save(figure, "09_per_scene_effects.png")

    latency = pd.read_csv(TABLES / "14_latency_summary.csv")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.scatter(latency["mean_latency_ms"], latency["fps"])
    for row in latency.itertuples(index=False):
        axis.annotate(row.method, (row.mean_latency_ms, row.fps), fontsize=7)
    axis.set(xlabel="mean latency, ms", ylabel="FPS")
    save(figure, "10_latency_tradeoff.png")

    required_tables = [f"{index:02d}_" for index in range(1, 16)]
    required_figures = [f"{index:02d}_" for index in range(1, 11)]
    missing_tables = [prefix for prefix in required_tables if not any(path.name.startswith(prefix) for path in TABLES.glob("*.csv"))]
    missing_figures = [prefix for prefix in required_figures if not any(path.name.startswith(prefix) for path in FIGURES.glob("*.png"))]
    if missing_tables or missing_figures:
        raise RuntimeError(f"Incomplete canonical outputs: tables={missing_tables}, figures={missing_figures}")
    (ROOT / "report/output_build.json").parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "report/output_build.json").write_text(json.dumps({
        "status": "PASS", "tables": 15, "figures": 10,
        "selected_normalization": mode,
        "independent_test_scenes": int(raw["sequence_id"].nunique()),
        "test_frames": int(raw["image_path"].nunique()),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
