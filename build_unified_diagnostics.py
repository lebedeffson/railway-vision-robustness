from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_final_practice import canonical_path, load_manifest


FEATURES = "outputs/diagnostics/feature_consistency/feature_consistency_test.csv"
DETECTIONS = "outputs/diagnostics/image_detection/image_detection_test.csv"
ATTACK = "outputs/final_practice/attack_consistency_raw.csv"
ATTACK_ADAPTIVE = "outputs/final_practice/attack_consistency_adaptive_raw.csv"
MANIFEST = "data/yolo_osdar23/manifest.csv"
OUTPUT = "outputs/final_practice/unified_diagnostics_raw.csv"


BASE_KEYS = ["image_path", "split", "attack", "epsilon_px", "defense"]


def add_default_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    defaults = {
        "adaptive": False,
        "restart": 0,
        "seed": 42,
        "class_name": "all",
        "perturbation_norm": "linf",
    }
    for name, value in defaults.items():
        if name not in data:
            data[name] = value
    if "steps" not in data:
        data["steps"] = np.where(data.get("attack", "") == "pgd", 20, 1)
    return data


def attach_sequences(data: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    if "sequence_id" in data and not data["sequence_id"].isna().any():
        return data
    rows = load_manifest(manifest_path)
    exact = {canonical_path(row["image_path"]): row["sequence_id"] for row in rows}
    by_name: dict[str, set[str]] = {}
    for row in rows:
        by_name.setdefault(Path(row["image_path"]).name, set()).add(row["sequence_id"])
    unique_names = {name: next(iter(values)) for name, values in by_name.items() if len(values) == 1}
    result = data.copy()
    result["sequence_id"] = result["image_path"].map(
        lambda value: exact.get(canonical_path(str(value)), unique_names.get(Path(str(value)).name))
    )
    if result["sequence_id"].isna().any():
        missing = result.loc[result["sequence_id"].isna(), "image_path"].drop_duplicates().head(5)
        raise RuntimeError(f"Missing sequence_id mapping: {missing.tolist()}")
    return result


def rename_feature_columns(features: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "level": "layer",
        "cos_norm_attack": "cosine",
        "mse_attack": "mse",
        "mae_attack": "mae",
        "relative_l2_attack": "relative_l2",
        "mean_shift_attack": "mean_shift",
        "entropy_change_attack": "entropy",
        "product_attack": "product",
        "godel_attack": "godel",
        "lukas_attack": "lukasiewicz",
        "recovery_cos_norm": "cosine_recovery",
        "recovery_mse": "mse_recovery",
        "recovery_mae": "mae_recovery",
        "recovery_relative_l2": "relative_l2_recovery",
        "recovery_mean_shift": "mean_shift_recovery",
        "recovery_entropy": "entropy_recovery",
        "recovery_product": "product_recovery",
        "recovery_godel": "godel_recovery",
        "recovery_lukas": "lukasiewicz_recovery",
        "p_product": "p_clean_preservation",
        "a_product": "a_attacked_similarity",
        "r_product": "r_restored_similarity",
        "g_product": "g_recovery",
        "c_def_product": "c_def",
    }
    return features.rename(columns=mapping)


def detection_views(detections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detections = add_default_columns(detections)
    clean = detections[(detections["attack"] == "clean") & (detections["defense"] == "none")][
        ["image_path", "split", "f1", "recall"]
    ].rename(columns={"f1": "f1_clean", "recall": "recall_clean"})
    attack = detections[(detections["attack"] != "clean") & (detections["defense"] == "none")][
        ["image_path", "split", "attack", "epsilon_px", "seed", "adaptive", "f1", "recall"]
    ].rename(columns={"f1": "f1_attack", "recall": "recall_attack"})
    defended = detections.rename(columns={
        "f1": "f1_defended", "recall": "recall_defended", "fn": "false_negatives"
    })
    return clean, attack, defended


def build(
    feature_path: Path,
    detection_path: Path,
    manifest_path: Path,
    attack_paths: list[Path],
) -> pd.DataFrame:
    features = attach_sequences(add_default_columns(pd.read_csv(feature_path)), manifest_path)
    features = rename_feature_columns(features)
    detections = add_default_columns(pd.read_csv(detection_path))
    clean, attacked, defended = detection_views(detections)
    join_keys = BASE_KEYS + ["seed", "adaptive"]
    result = features.merge(
        defended[[*join_keys, "f1_defended", "recall_defended", "false_negatives"]],
        on=join_keys, how="left", validate="many_to_one",
    )
    result = result.merge(clean, on=["image_path", "split"], how="left", validate="many_to_one")
    result = result.merge(
        attacked,
        on=["image_path", "split", "attack", "epsilon_px", "seed", "adaptive"],
        how="left", validate="many_to_one",
    )
    available_attack_paths = [path for path in attack_paths if path.is_file()]
    if available_attack_paths:
        attack = pd.concat([pd.read_csv(path) for path in available_attack_paths], ignore_index=True)
        attack_columns = [
            "image_path", "attack", "epsilon_px", "seed", "adaptive",
            "c_sp_global", "c_sp_object", "c_sp_background", "c_dir",
            "c_atk_global", "c_atk_object",
        ]
        result = result.merge(
            attack[attack_columns],
            on=["image_path", "attack", "epsilon_px", "seed", "adaptive"],
            how="left", validate="many_to_one",
        )

    result["epsilon"] = result.get("epsilon", result["epsilon_px"] / 255.0)
    result["damage"] = result["f1_clean"] - result["f1_attack"]
    result["recovery"] = result["f1_defended"] - result["f1_attack"]
    result["confidence_drop"] = result.get("confidence_drop", np.nan)
    result["iou_shift"] = result.get("iou_shift", np.nan)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the single final-practice CSV")
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--detections", type=Path, default=DETECTIONS)
    parser.add_argument("--attack-consistency", type=Path, default=ATTACK)
    parser.add_argument("--attack-adaptive-consistency", type=Path, default=ATTACK_ADAPTIVE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(
        args.features, args.detections, args.manifest,
        [args.attack_consistency, args.attack_adaptive_consistency],
    )
    required = {
        "sequence_id", "image_path", "split", "class_name", "attack", "adaptive",
        "epsilon", "steps", "restart", "seed", "defense", "layer",
        "f1_clean", "f1_attack", "f1_defended", "recall_clean", "recall_attack",
        "recall_defended", "false_negatives", "cosine", "mse", "mae",
        "relative_l2", "mean_shift", "entropy", "product", "godel", "lukasiewicz",
        "p_clean_preservation", "a_attacked_similarity", "r_restored_similarity",
        "g_recovery", "c_def",
    }
    missing = required - set(result)
    if missing:
        raise RuntimeError(f"Unified final schema is incomplete: {sorted(missing)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
