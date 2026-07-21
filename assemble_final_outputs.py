from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/final_practice"


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the eight required publication tables")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    scenes = pd.read_csv(require(root / "audit/scene_manifest.csv"))
    clean = pd.read_csv(require(tables / "clean_model_metrics.csv"))
    scenes.insert(0, "record_type", "dataset_scene")
    clean.insert(0, "record_type", "clean_metric")
    pd.concat([scenes, clean], ignore_index=True, sort=False).to_csv(
        tables / "01_dataset_and_clean_model.csv", index=False
    )

    raw = pd.read_csv(require(root / "unified_diagnostics_raw.csv"))
    attack_parameter_columns = [
        "attack", "adaptive", "epsilon", "epsilon_px", "steps", "restart", "seed",
        "step_size", "random_start", "restarts", "perturbation_norm",
        "attack_objective", "lambda_box", "lambda_cls", "lambda_dfl",
    ]
    raw[attack_parameter_columns].drop_duplicates().sort_values(
        ["attack", "adaptive", "epsilon_px", "steps", "restart", "seed"]
    ).to_csv(tables / "02_attack_parameters.csv", index=False)

    utility = pd.read_csv(require(root / "03_clean_utility/clean_utility_summary.csv"))
    latency = pd.read_csv(require(root / "08_latency/latency.csv"))
    latency["defense"] = latency["method"].str.replace("+detector", "", regex=False).replace({"detector": "none"})
    utility.merge(latency, on="defense", how="left").to_csv(
        tables / "03_clean_utility_and_latency.csv", index=False
    )

    selected = raw["selected_best"].astype(str).str.lower().isin(("true", "1"))
    adaptive = raw["adaptive"].astype(str).str.lower().isin(("true", "1"))
    chosen = raw[selected].copy()
    chosen["adaptive"] = adaptive[selected].to_numpy()
    metrics = [
        "f1_attack", "f1_defended", "recall_attack", "recall_defended",
        "false_negatives", "confidence_drop", "iou_shift", "attack_loss",
    ]
    chosen[~chosen["adaptive"]].groupby(
        ["attack", "epsilon_px", "steps", "defense"], as_index=False
    )[metrics].mean().to_csv(tables / "04_non_adaptive_robustness.csv", index=False)

    attack = pd.read_csv(require(root / "raw/attack_consistency.csv"))
    attack_numeric = [
        column for column in (
            "c_sp_global", "c_sp_object", "c_sp_background", "c_dir", "c_dir_object",
            "c_dir_background", "c_atk_global", "c_atk_object", "c_atk_background",
            "path_gradient_cosine_global", "path_gradient_sign_agreement_global",
            "path_gradient_spearman_global", "path_gradient_topk_overlap_global",
        ) if column in attack
    ]
    attack.groupby(["attack", "adaptive", "epsilon_px", "steps"], as_index=False)[attack_numeric].mean().to_csv(
        tables / "05_attack_consistency.csv", index=False
    )

    defense = pd.read_csv(require(root / "raw/defense_consistency.csv"))
    defense_numeric = [
        column for column in (
            "p_clean_preservation", "a_attacked_similarity", "r_restored_similarity",
            "g_recovery", "c_def", "p_godel", "a_godel", "r_godel", "g_godel",
            "c_def_godel", "p_lukasiewicz", "a_lukasiewicz", "r_lukasiewicz",
            "g_lukasiewicz", "c_def_lukasiewicz", "cosine_recovery", "mse_recovery",
            "mae_recovery", "relative_l2_recovery", "mean_shift_recovery",
            "entropy_recovery", "product_recovery", "godel_recovery",
            "lukasiewicz_recovery",
        ) if column in defense
    ]
    defense.groupby(
        ["attack", "adaptive", "epsilon_px", "steps", "defense", "layer"], as_index=False
    )[defense_numeric].mean().to_csv(tables / "06_defense_consistency.csv", index=False)

    statistics = root / "09_statistics"
    models = pd.read_csv(require(statistics / "model_comparison_m0_m4.csv"))
    bootstrap = pd.read_csv(require(statistics / "sequence_bootstrap.csv"))
    models.insert(0, "record_type", "model")
    bootstrap.insert(0, "record_type", "paired_gain")
    pd.concat([models, bootstrap], ignore_index=True, sort=False).to_csv(
        tables / "07_model_comparison_statistics.csv", index=False
    )

    adaptive_table = pd.read_csv(require(statistics / "adaptive_vs_nonadaptive.csv"))
    adaptive_table.to_csv(tables / "08_adaptive_robustness.csv", index=False)

    summary = {
        "status": "PASS",
        "table_count": 8,
        "tables": [str(path) for path in sorted(tables.glob("0[1-8]_*.csv"))],
    }
    (tables / "table_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
