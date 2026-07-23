from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    TEST_MARKER,
    assert_role_allowed,
    atomic_json,
    load_protocol,
    sha256,
)
from canonical_m4_runtime import (
    FrameInput,
    detection,
    feature_sets,
    fgsm,
    frame_labels,
    load_image,
    load_model,
    object_mask,
    pgd,
)
from extract_attack_consistency import consistency_row
from extract_feature_consistency import FeatureHook
from revision_q1.feature_metrics import (
    distance_recovery,
    pair_metrics,
    similarity_recovery,
)


SOURCE_MANIFEST = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"
DESTINATION = OUTPUT_ROOT / "attacks"


def condition_id(
    image_path: str,
    attack: str,
    adaptive: bool,
    epsilon_px: float,
) -> str:
    raw = f"{image_path}|{attack}|{adaptive}|{epsilon_px:.12g}"
    return hashlib.sha256(raw.encode()).hexdigest()


def average_pair(
    clean: torch.Tensor,
    other: torch.Tensor,
    statistics: dict[str, torch.Tensor],
    mode: str,
) -> dict[str, float]:
    rows = pair_metrics(clean, other, statistics, mode)
    return {
        key: float(np.nanmean([row[key] for row in rows]))
        for key in rows[0]
    }


def tnorm(left: float, right: float, name: str) -> float:
    left = min(1.0, max(0.0, left))
    right = min(1.0, max(0.0, right))
    if name == "product":
        return left * right
    if name == "lukasiewicz":
        return max(0.0, left + right - 1.0)
    raise ValueError(name)


def feature_row(
    clean: torch.Tensor,
    attacked: torch.Tensor,
    defended: torch.Tensor,
    filtered_clean: torch.Tensor,
    statistics: dict[str, torch.Tensor],
    mode: str,
) -> dict[str, float]:
    attack = average_pair(clean, attacked, statistics, mode)
    restored = average_pair(clean, defended, statistics, mode)
    preservation = average_pair(clean, filtered_clean, statistics, mode)
    result = {
        "cosine": attack["cosine_similarity"],
        "l1_distance": attack["l1_distance"],
        "normalized_l2": attack["normalized_euclidean_distance"],
        "mse": attack["mse"],
        "mae": attack["mae"],
        "pearson": attack["pearson_correlation"],
        "spearman": attack["spearman_correlation"],
        "entropy_shift": attack["entropy_shift"],
        "product": attack["product"],
        "godel": attack["godel"],
        "lukasiewicz": attack["lukasiewicz"],
        "cosine_recovery": similarity_recovery(
            attack["cosine_similarity"], restored["cosine_similarity"]
        ),
        "l1_recovery": distance_recovery(
            attack["l1_distance"], restored["l1_distance"]
        ),
        "normalized_l2_recovery": distance_recovery(
            attack["normalized_euclidean_distance"],
            restored["normalized_euclidean_distance"],
        ),
        "mse_recovery": distance_recovery(attack["mse"], restored["mse"]),
        "mae_recovery": distance_recovery(attack["mae"], restored["mae"]),
        "pearson_recovery": similarity_recovery(
            attack["pearson_correlation"], restored["pearson_correlation"]
        ),
        "spearman_recovery": similarity_recovery(
            attack["spearman_correlation"], restored["spearman_correlation"]
        ),
        "entropy_recovery": distance_recovery(
            attack["entropy_shift"], restored["entropy_shift"]
        ),
        "product_recovery": similarity_recovery(
            attack["product"], restored["product"]
        ),
        "godel_recovery": similarity_recovery(
            attack["godel"], restored["godel"]
        ),
        "lukasiewicz_recovery": similarity_recovery(
            attack["lukasiewicz"], restored["lukasiewicz"]
        ),
    }
    for operator in ("product", "lukasiewicz"):
        before = attack[operator]
        after = restored[operator]
        gain = similarity_recovery(before, after)
        clipped = (
            math.nan if not math.isfinite(gain) else min(1.0, max(0.0, gain))
        )
        prefix = "" if operator == "product" else "lukasiewicz_"
        result[f"{prefix}P"] = preservation[operator]
        result[f"{prefix}A"] = before
        result[f"{prefix}R"] = after
        result[f"{prefix}G_raw"] = gain
        result[f"{prefix}G_clipped"] = clipped
        result[f"{prefix}C_def"] = (
            math.nan
            if not math.isfinite(clipped)
            else tnorm(preservation[operator], clipped, operator)
        )
    result.update({
        "P": result["P"],
        "A": result["A"],
        "R": result["R"],
        "G_raw": result["G_raw"],
        "G_clipped": result["G_clipped"],
        "C_def": result["C_def"],
        "p_clean_preservation": result["P"],
        "a_attacked_similarity": result["A"],
        "r_restored_similarity": result["R"],
        "g_recovery": result["G_raw"],
        "c_def": result["C_def"],
    })
    return result


def result_norms(result, clean: torch.Tensor) -> dict[str, float]:
    delta = result.adversarial - clean
    return {
        "actual_L1": float(delta.abs().sum()),
        "actual_L2": float(delta.norm()),
        "actual_Linf": float(delta.abs().max()),
        "perturbation_l1": float(delta.abs().sum()),
        "perturbation_l2": float(delta.norm()),
        "perturbation_linf": float(delta.abs().max()),
        "attack_loss_clean": float(result.clean_attack_loss),
        "attack_loss_final": float(result.attack_loss),
        "gradient_l1": float(result.path_gradient.abs().sum()),
        "gradient_l2": float(result.path_gradient.norm()),
        "gradient_linf": float(result.path_gradient.abs().max()),
    }


def matrix_conditions(lock: dict[str, Any]) -> list[tuple[str, bool, float, int]]:
    rows = [
        ("fgsm", False, float(epsilon), 1)
        for epsilon in lock["selected"]["fgsm_epsilon_px"]
    ]
    rows += [
        ("pgd", False, float(epsilon), int(lock["pgd_steps"]))
        for epsilon in lock["selected"]["pgd_epsilon_px"]
    ]
    rows += [
        ("pgd", True, float(epsilon), int(lock["adaptive_pgd_steps"]))
        for epsilon in lock["selected"]["adaptive_pgd_epsilon_px"]
    ]
    return rows


def run_condition(
    *,
    model,
    clean: torch.Tensor,
    labels: list[dict[str, Any]],
    clean_detection: dict[str, Any],
    clean_features: list[torch.Tensor],
    clean_defense_features: dict[str, list[torch.Tensor]],
    frame: FrameInput,
    protocol: dict[str, Any],
    statistics: dict[str, dict[str, torch.Tensor]],
    normalization: str,
    threshold: float,
    checkpoint_sha256: str,
    attack: str,
    adaptive: bool,
    epsilon_px: float,
    steps: int,
    seeds: list[int],
) -> list[dict[str, Any]]:
    selected_seeds = [seeds[0]] if attack == "fgsm" else seeds
    candidates = []
    for seed in selected_seeds:
        result = (
            fgsm(model, clean, labels, protocol, epsilon_px, adaptive)
            if attack == "fgsm"
            else pgd(
                model,
                clean,
                labels,
                protocol,
                epsilon_px,
                steps,
                seed,
                adaptive,
            )
        )
        candidates.append((seed, result))
    best = max(
        range(len(candidates)),
        key=lambda index: candidates[index][1].attack_loss,
    )
    rows = []
    mask = object_mask(
        labels, clean.shape[-2], clean.shape[-1], clean.device
    )
    defenses = (
        ["none", "Product_preprocessing"]
        if adaptive
        else ["none", "Product_preprocessing", "bilateral", "median"]
    )
    for restart, (seed, result) in enumerate(candidates):
        consistency = consistency_row(
            clean[0],
            result,
            mask,
            epsilon_px,
            result.path_gradient[0],
            "path_gradient_",
        )
        hook = FeatureHook(model)
        try:
            attacked_features = feature_sets(
                hook, model, result.adversarial, protocol
            )
            for defense in defenses:
                defended_detection, _predictions, _tiles = detection(
                    model,
                    result.adversarial,
                    labels,
                    protocol,
                    threshold,
                    defense,
                )
                defended_features = (
                    attacked_features
                    if defense == "none"
                    else feature_sets(
                        hook, model, result.adversarial, protocol, defense
                    )
                )
                for layer_index, layer in enumerate(("P3", "P4", "P5")):
                    diagnostics = feature_row(
                        clean_features[layer_index],
                        attacked_features[layer_index],
                        defended_features[layer_index],
                        clean_defense_features[defense][layer_index],
                        statistics[layer],
                        normalization,
                    )
                    aliases = {
                        f"{normalization}_cosine_similarity":
                            diagnostics["cosine"],
                        f"{normalization}_l1_distance":
                            diagnostics["l1_distance"],
                        f"{normalization}_normalized_euclidean_distance":
                            diagnostics["normalized_l2"],
                        f"{normalization}_mse": diagnostics["mse"],
                        f"{normalization}_pearson_correlation":
                            diagnostics["pearson"],
                        f"{normalization}_entropy_shift":
                            diagnostics["entropy_shift"],
                        f"{normalization}_product": diagnostics["product"],
                        f"{normalization}_godel": diagnostics["godel"],
                        f"{normalization}_lukasiewicz":
                            diagnostics["lukasiewicz"],
                        f"{normalization}_cosine_similarity_recovery":
                            diagnostics["cosine_recovery"],
                        f"{normalization}_l1_distance_recovery":
                            diagnostics["l1_recovery"],
                        f"{normalization}_normalized_euclidean_distance_recovery":
                            diagnostics["normalized_l2_recovery"],
                        f"{normalization}_mse_recovery":
                            diagnostics["mse_recovery"],
                        f"{normalization}_pearson_correlation_recovery":
                            diagnostics["pearson_recovery"],
                        f"{normalization}_entropy_shift_recovery":
                            diagnostics["entropy_recovery"],
                        f"{normalization}_product_recovery":
                            diagnostics["product_recovery"],
                        f"{normalization}_godel_recovery":
                            diagnostics["godel_recovery"],
                        f"{normalization}_lukasiewicz_recovery":
                            diagnostics["lukasiewicz_recovery"],
                        f"{normalization}_p_product": diagnostics["P"],
                        f"{normalization}_a_product": diagnostics["A"],
                        f"{normalization}_r_product": diagnostics["R"],
                        f"{normalization}_g_product": diagnostics["G_raw"],
                        f"{normalization}_c_def_product":
                            diagnostics["C_def"],
                    }
                    precision = float(defended_detection["precision"])
                    recall = float(defended_detection["recall"])
                    row = {
                        "protocol_id": protocol["protocol_id"],
                        "checkpoint_sha256": checkpoint_sha256,
                        "grouped_scene_id": frame.grouped_scene_id,
                        "sequence_id": frame.grouped_scene_id,
                        "subsequence_id": frame.subsequence_id,
                        "image_path": str(frame.image_path.resolve()),
                        "split": "test",
                        "attack": attack,
                        "adaptive": adaptive,
                        "epsilon_px": epsilon_px,
                        "epsilon": epsilon_px / 255.0,
                        "steps": steps,
                        "step_size": (
                            epsilon_px / 255.0
                            if attack == "fgsm"
                            else epsilon_px / 255.0 / 4.0
                        ),
                        "random_start": attack == "pgd",
                        "restarts": len(selected_seeds),
                        "restart": restart,
                        "seed": seed,
                        "selected_best": restart == best,
                        "defense": defense,
                        "layer": layer,
                        "normalization": normalization,
                        "threshold": threshold,
                        "precision_clean": clean_detection["precision"],
                        "recall_clean": clean_detection["recall"],
                        "f1_clean": clean_detection["f1"],
                        "fn_clean": clean_detection["fn"],
                        "precision_attack": (
                            defended_detection["precision"]
                            if defense == "none"
                            else math.nan
                        ),
                        "recall_attack": (
                            defended_detection["recall"]
                            if defense == "none"
                            else math.nan
                        ),
                        "f1_attack": (
                            defended_detection["f1"]
                            if defense == "none"
                            else math.nan
                        ),
                        "fn_attack": (
                            defended_detection["fn"]
                            if defense == "none"
                            else math.nan
                        ),
                        "precision_defended": precision,
                        "recall_defended": recall,
                        "f1_defended": defended_detection["f1"],
                        "f2": defended_detection["f2"],
                        "fn_defended": defended_detection["fn"],
                        "precision": precision,
                        "recall": recall,
                        "f1": defended_detection["f1"],
                        "fn": defended_detection["fn"],
                        "confidence_drop": (
                            clean_detection["mean_confidence"]
                            - defended_detection["mean_confidence"]
                        ),
                        "iou_shift": math.nan,
                        "nms_timeout": defended_detection["nms_timeout"],
                        "nms_runtime_ms": defended_detection["nms_runtime_ms"],
                        "nms_output_complete":
                            defended_detection["nms_output_complete"],
                        "predictions_after_nms":
                            defended_detection["predictions_after_nms"],
                        "nms_candidates_before":
                            defended_detection["nms_candidates_before"],
                        "damage": (
                            clean_detection["f1"]
                            - (
                                defended_detection["f1"]
                                if defense == "none"
                                else math.nan
                            )
                        ),
                        "recovery": (
                            defended_detection["f1"]
                            - (
                                defended_detection["f1"]
                                if defense == "none"
                                else math.nan
                            )
                        ),
                        "canonical_metric_family": "canonical_tnorm",
                        "legacy_metric_family": "not_used",
                        **result_norms(result, clean),
                        "c_sp_global":
                            consistency["path_gradient_c_sp_global"],
                        "c_sp_object":
                            consistency["path_gradient_c_sp_object"],
                        "c_sp_background":
                            consistency["path_gradient_c_sp_background"],
                        "c_dir":
                            consistency["path_gradient_c_dir_global"],
                        "c_atk_global":
                            consistency["path_gradient_c_atk_product_global"],
                        "c_atk_object":
                            consistency["path_gradient_c_atk_product_object"],
                        "c_atk_background":
                            consistency["path_gradient_c_atk_product_background"],
                        **diagnostics,
                        **aliases,
                    }
                    rows.append(row)
        finally:
            hook.close()
    # Fill attacked quality and endpoints from the none rows for each restart.
    for restart in range(len(candidates)):
        none = next(
            row
            for row in rows
            if row["restart"] == restart
            and row["defense"] == "none"
            and row["layer"] == "P3"
        )
        for row in rows:
            if row["restart"] != restart:
                continue
            row["precision_attack"] = none["precision_defended"]
            row["recall_attack"] = none["recall_defended"]
            row["f1_attack"] = none["f1_defended"]
            row["fn_attack"] = none["fn_defended"]
            row["damage"] = row["f1_clean"] - row["f1_attack"]
            row["recovery"] = row["f1_defended"] - row["f1_attack"]
            row["delta_recall_damage"] = (
                row["recall_clean"] - row["recall_attack"]
            )
            row["delta_recall_recovery"] = (
                row["recall_defended"] - row["recall_attack"]
            )
            row["false_negatives_increase"] = (
                row["fn_attack"] - row["fn_clean"]
            )
            row["false_negatives_reduction"] = (
                row["fn_attack"] - row["fn_defended"]
            )
            denominator = row["f1_clean"] - row["f1_attack"] + 1e-8
            row["normalized_quality_recovery"] = (
                row["f1_defended"] - row["f1_attack"]
            ) / denominator
    return rows


def run() -> dict[str, Any]:
    assert_role_allowed("attack")
    if not TEST_MARKER.is_file():
        raise RuntimeError("Canonical M4 test is not open")
    protocol = load_protocol()
    gate = json.loads(
        (OUTPUT_ROOT / "validation/quality_gate.json").read_text(encoding="utf-8")
    )
    attack_lock = json.loads(
        (OUTPUT_ROOT / "attack_calibration/attack_protocol_lock.json")
        .read_text(encoding="utf-8")
    )
    normalization_manifest = json.loads(
        (OUTPUT_ROOT / "normalization/normalization_manifest.json")
        .read_text(encoding="utf-8")
    )
    marker = json.loads(TEST_MARKER.read_text(encoding="utf-8"))
    checkpoint = Path(gate["checkpoint"])
    checkpoint_hash = sha256(checkpoint)
    if {
        gate["checkpoint_sha256"],
        attack_lock["checkpoint_sha256"],
        normalization_manifest["checkpoint_sha256"],
        marker["checkpoint_sha256"],
    } != {checkpoint_hash}:
        raise RuntimeError("Canonical M4 frozen checkpoint hashes differ")
    statistics_payload = torch.load(
        OUTPUT_ROOT / "normalization/layer_channel_statistics.pt",
        map_location="cpu",
    )
    statistics = statistics_payload["statistics"]
    normalization = normalization_manifest["selected_normalization"]
    threshold = float(gate["safety_threshold"])
    source = pd.read_csv(SOURCE_MANIFEST)
    source = source[source["split"].eq("test")].sort_values(
        ["grouped_scene_id", "subsequence_id", "frame_id"]
    )
    device = torch.device("cuda:0")
    model = load_model(checkpoint, device)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    cache = DESTINATION / "condition_cache"
    cache.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in attack_lock["seeds"]]
    conditions = matrix_conditions(attack_lock)
    all_rows = []
    for source_row in source.itertuples(index=False):
        frame = FrameInput(
            Path(source_row.output_image),
            Path(source_row.output_label),
            str(source_row.grouped_scene_id),
            str(source_row.subsequence_id),
        )
        clean = load_image(frame.image_path, device)
        labels = frame_labels(frame, protocol)
        clean_detection, _clean_predictions, _clean_tiles = detection(
            model, clean, labels, protocol, threshold
        )
        hook = FeatureHook(model)
        try:
            clean_features = feature_sets(hook, model, clean, protocol)
            clean_defense_features = {
                defense: (
                    clean_features
                    if defense == "none"
                    else feature_sets(hook, model, clean, protocol, defense)
                )
                for defense in (
                    "none", "Product_preprocessing", "bilateral", "median"
                )
            }
        finally:
            hook.close()
        for attack, adaptive, epsilon_px, steps in conditions:
            path = cache / (
                condition_id(
                    str(frame.image_path.resolve()),
                    attack,
                    adaptive,
                    epsilon_px,
                )
                + ".json"
            )
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload["checkpoint_sha256"] != checkpoint_hash:
                    raise RuntimeError("Canonical M4 test cache checkpoint mismatch")
                all_rows.extend(payload["rows"])
                continue
            rows = run_condition(
                model=model,
                clean=clean,
                labels=labels,
                clean_detection=clean_detection,
                clean_features=clean_features,
                clean_defense_features=clean_defense_features,
                frame=frame,
                protocol=protocol,
                statistics=statistics,
                normalization=normalization,
                threshold=threshold,
                checkpoint_sha256=checkpoint_hash,
                attack=attack,
                adaptive=adaptive,
                epsilon_px=epsilon_px,
                steps=steps,
                seeds=seeds,
            )
            expected = (
                (1 if attack == "fgsm" else len(seeds))
                * (2 if adaptive else 4)
                * 3
            )
            if len(rows) != expected:
                raise RuntimeError("Canonical M4 condition row count mismatch")
            atomic_json(path, {
                "status": "PASS",
                "checkpoint_sha256": checkpoint_hash,
                "rows": rows,
            })
            all_rows.extend(rows)
        del clean, clean_features, clean_defense_features
    matrix = pd.DataFrame(all_rows)
    key = [
        "image_path", "attack", "adaptive", "epsilon_px", "steps",
        "restart", "seed", "defense", "layer",
    ]
    if bool(matrix.duplicated(key).any()):
        raise RuntimeError("Canonical M4 test matrix has duplicate keys")
    expected_conditions = len(source) * len(conditions)
    observed_conditions = matrix[
        ["image_path", "attack", "adaptive", "epsilon_px", "steps"]
    ].drop_duplicates()
    if len(observed_conditions) != expected_conditions:
        raise RuntimeError("Canonical M4 test matrix has missing conditions")
    if not bool(
        matrix["actual_Linf"]
        .le(matrix["epsilon_px"].astype(float) / 255.0 + 1e-6)
        .all()
    ):
        raise RuntimeError("Canonical M4 test attack exceeds L-infinity budget")
    output = DESTINATION / "canonical_test_matrix.csv"
    temporary = output.with_suffix(".csv.tmp")
    matrix.to_csv(temporary, index=False)
    os.replace(temporary, output)
    nms = matrix[
        matrix["nms_timeout"].astype(bool)
        | ~matrix["nms_output_complete"].astype(bool)
    ].copy()
    nms.to_csv(DESTINATION / "canonical_nms_audit.csv", index=False)
    summary = {
        "status": "PASS",
        "checkpoint_sha256": checkpoint_hash,
        "test_marker_sha256": sha256(TEST_MARKER),
        "frames": int(matrix["image_path"].nunique()),
        "grouped_scenes": int(matrix["grouped_scene_id"].nunique()),
        "conditions": len(observed_conditions),
        "rows": len(matrix),
        "duplicates": 0,
        "missing_conditions": 0,
        "nms_affected_rows": len(nms),
        "matrix_sha256": sha256(output),
    }
    atomic_json(DESTINATION / "canonical_test_matrix.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
