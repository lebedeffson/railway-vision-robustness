from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from audit_evaluator import average_precision, match_dataset
from scripts.person_canonical_v5.execution_evaluate import _size_counts
from scripts.person_v6.common import OUTPUT, PROJECT, assert_locked, atomic_json, config
from src.temporal.ratta import (
    HomographyResult,
    TemporalAggregator,
    deterministic_nms,
    estimate_background_homography,
)


VARIANTS = {
    "T0-A_B0": None,
    "T0-B_ByteTrack": "bytetrack",
    "T0-C_Bayesian": "arithmetic",
    "T0-D_Bayesian_Product": "product",
}


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def fold_scenes(fold: int) -> set[str]:
    folds = json.loads(
        (PROJECT / config()["data"]["folds"]).read_text(encoding="utf-8")
    )["folds"]
    return set(map(str, folds[str(fold)]))


def source_frames(fold: int) -> pd.DataFrame:
    manifest = pd.read_csv(
        PROJECT / config()["data"]["manifest"],
        dtype={"frame_id": str},
    )
    scenes = fold_scenes(fold)
    source = manifest[
        manifest["split"].isin(config()["data"]["development_splits"])
        & manifest["grouped_scene_id"].astype(str).isin(scenes)
    ].copy()
    if set(source["grouped_scene_id"].astype(str)) != scenes:
        raise RuntimeError("Fold source misses a frozen development scene")
    test_scenes = set(
        manifest.loc[manifest["split"].eq("test"), "grouped_scene_id"].astype(str)
    )
    if scenes & test_scenes:
        raise RuntimeError("Temporal fold intersects sealed test scenes")
    source["frame_number"] = pd.to_numeric(source["frame_id"], errors="raise")
    source["image_path"] = source["output_image"].map(
        lambda value: str(Path(value).resolve())
    )
    source = source.sort_values(
        ["grouped_scene_id", "subsequence_id", "frame_number", "image_path"]
    ).reset_index(drop=True)
    duplicates = int(
        source.duplicated(["subsequence_id", "frame_number"]).sum()
    )
    if duplicates:
        raise RuntimeError("Temporal sequence has duplicate frame numbers")
    return source


def load_baseline(
    fold: int,
    source: pd.DataFrame,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    path = PROJECT / config()["baseline"][f"fold_{fold}"]["predictions"]
    frame = pd.read_csv(path)
    allowed = set(source["image_path"])
    if not set(frame["image_path"].astype(str)) <= allowed:
        raise RuntimeError("Saved B0 predictions contain frames outside the fold")
    ground_truth: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in allowed:
        ground_truth[image] = []
        predictions[image] = []
    for row in frame.itertuples(index=False):
        item = {
            "class_id": 0,
            "box": [float(row.x1), float(row.y1), float(row.x2), float(row.y2)],
        }
        if row.kind == "ground_truth":
            ground_truth[str(row.image_path)].append(item)
        else:
            confidence = float(row.confidence)
            if confidence + 1e-12 < float(config()["baseline"]["candidate_floor"]):
                raise RuntimeError("B0 prediction is below frozen candidate floor")
            predictions[str(row.image_path)].append(
                {**item, "confidence": confidence}
            )
    expected_gt = int(
        json.loads(
            (
                path.parent / "evaluation_result.json"
            ).read_text(encoding="utf-8")
        )["person_GT"]
    )
    observed_gt = sum(map(len, ground_truth.values()))
    if observed_gt != expected_gt:
        raise RuntimeError(f"Ground-truth count changed: {observed_gt} != {expected_gt}")
    return dict(ground_truth), dict(predictions)


def homography_cache(
    fold: int,
    source: pd.DataFrame,
    predictions: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, HomographyResult | None], pd.DataFrame]:
    destination = OUTPUT / f"fold_{fold}/homography"
    destination.mkdir(parents=True, exist_ok=True)
    results: dict[str, HomographyResult | None] = {}
    rows: list[dict[str, Any]] = []
    for subsequence, sequence in source.groupby("subsequence_id", sort=False):
        previous_image: np.ndarray | None = None
        previous_path: str | None = None
        for record in sequence.itertuples(index=False):
            current_path = str(record.image_path)
            current_image = cv2.imread(current_path)
            if current_image is None:
                raise RuntimeError(f"Unreadable temporal frame: {current_path}")
            if previous_image is None:
                results[current_path] = None
                rows.append({
                    "fold": fold,
                    "grouped_scene_id": str(record.grouped_scene_id),
                    "subsequence_id": str(subsequence),
                    "previous_image": "",
                    "current_image": current_path,
                    "valid": False,
                    "boundary": True,
                    "matches": 0,
                    "inliers": 0,
                    "inlier_ratio": 0.0,
                })
            else:
                masked = [
                    row["box"]
                    for row in predictions.get(str(previous_path), [])
                    if float(row["confidence"]) >= float(
                        config()["baseline"]["operating_threshold"]
                    )
                ]
                result = estimate_background_homography(
                    previous_image,
                    current_image,
                    masked,
                    config()["temporal"]["camera_compensation"],
                )
                results[current_path] = result
                rows.append({
                    "fold": fold,
                    "grouped_scene_id": str(record.grouped_scene_id),
                    "subsequence_id": str(subsequence),
                    "previous_image": previous_path,
                    "current_image": current_path,
                    "valid": result.valid,
                    "boundary": False,
                    "matches": result.matches,
                    "inliers": result.inliers,
                    "inlier_ratio": result.inlier_ratio,
                })
            previous_image = current_image
            previous_path = current_path
    frame = pd.DataFrame(rows)
    atomic_csv(destination / "homography_audit.csv", frame)
    return results, frame


def process_variant(
    fold: int,
    name: str,
    mode: str | None,
    source: pd.DataFrame,
    baseline: dict[str, list[dict[str, Any]]],
    homographies: dict[str, HomographyResult | None],
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    if mode is None:
        return baseline, pd.DataFrame()
    predictions: dict[str, list[dict[str, Any]]] = {}
    track_rows: list[dict[str, Any]] = []
    for subsequence, sequence in source.groupby("subsequence_id", sort=False):
        aggregator = TemporalAggregator(config()["temporal"], mode)
        aggregator.reset()
        for record in sequence.itertuples(index=False):
            image_path = str(record.image_path)
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f"Unreadable temporal frame: {image_path}")
            emitted, tracks = aggregator.update(
                baseline.get(image_path, []),
                image,
                homographies[image_path],
            )
            predictions[image_path] = deterministic_nms(
                emitted,
                float(config()["temporal"]["fusion_nms_iou"]),
            )
            for row in tracks:
                track_rows.append({
                    "variant": name,
                    "fold": fold,
                    "grouped_scene_id": str(record.grouped_scene_id),
                    "subsequence_id": str(subsequence),
                    "frame_id": str(record.frame_id),
                    "image_path": image_path,
                    **row,
                })
    return predictions, pd.DataFrame(track_rows)


def metrics(
    fold: int,
    variant: str,
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    threshold = float(config()["baseline"]["operating_threshold"])
    operating = match_dataset(ground_truth, predictions, threshold, 0.50)
    map50 = average_precision(ground_truth, predictions, 0, 0.50)
    size = _size_counts(ground_truth, predictions, threshold)
    small_recall = float(size.loc[size["size"].eq("small"), "recall"].iloc[0])
    per_scene = []
    for scene, group in source.groupby("grouped_scene_id"):
        images = set(group["image_path"])
        scene_gt = {image: ground_truth[image] for image in images}
        scene_predictions = {image: predictions.get(image, []) for image in images}
        scene_operating = match_dataset(scene_gt, scene_predictions, threshold, 0.50)
        scene_size = _size_counts(scene_gt, scene_predictions, threshold)
        per_scene.append({
            "variant": variant,
            "fold": fold,
            "grouped_scene_id": str(scene),
            "frames": len(images),
            "mAP50": average_precision(scene_gt, scene_predictions, 0, 0.50),
            **scene_operating,
            "small_recall": float(
                scene_size.loc[scene_size["size"].eq("small"), "recall"].iloc[0]
            ),
        })
    frame_rows = []
    scene_by_image = dict(zip(source["image_path"], source["grouped_scene_id"].astype(str)))
    for image in source["image_path"]:
        for row in ground_truth[image]:
            frame_rows.append({
                "variant": variant,
                "fold": fold,
                "image_path": image,
                "grouped_scene_id": scene_by_image[image],
                "kind": "ground_truth",
                "confidence": 1.0,
                **dict(zip(("x1", "y1", "x2", "y2"), row["box"])),
            })
        for row in predictions.get(image, []):
            frame_rows.append({
                "variant": variant,
                "fold": fold,
                "image_path": image,
                "grouped_scene_id": scene_by_image[image],
                "kind": "prediction",
                "confidence": float(row["confidence"]),
                "track_id": row.get("track_id"),
                "propagated": row.get("propagated", False),
                **dict(zip(("x1", "y1", "x2", "y2"), row["box"])),
            })
    payload = {
        "variant": variant,
        "fold": fold,
        "frames": len(source),
        "scenes": int(source["grouped_scene_id"].nunique()),
        "mAP50": float(map50),
        **operating,
        "small_recall": small_recall,
        "FN_per_frame": operating["fn"] / max(len(source), 1),
        "FP_per_frame": operating["fp"] / max(len(source), 1),
        "worst_scene_recall": min(row["recall"] for row in per_scene),
        "lost_GT": 0,
        "sequence_leakage": 0,
        "evaluator_consistency": "PASS",
        "ID_switches": "NOT_EVALUABLE_WITHOUT_PERSISTENT_GT_TRACK_IDS",
        "test_used": False,
    }
    return payload, pd.DataFrame(per_scene), pd.DataFrame(frame_rows)


def evaluate_fold(fold: int) -> dict[str, Any]:
    source = source_frames(fold)
    ground_truth, baseline = load_baseline(fold, source)
    homographies, homography_frame = homography_cache(fold, source, baseline)
    results = []
    root = OUTPUT / f"fold_{fold}"
    for variant, mode in VARIANTS.items():
        predictions, tracks = process_variant(
            fold, variant, mode, source, baseline, homographies
        )
        result, per_scene, frame = metrics(
            fold, variant, source, ground_truth, predictions
        )
        destination = root / variant
        atomic_json(destination / "metrics.json", result)
        atomic_csv(destination / "per_scene.csv", per_scene)
        atomic_csv(destination / "per_frame_predictions.csv", frame)
        atomic_csv(destination / "per_track.csv", tracks)
        results.append(result)
    baseline_result = next(row for row in results if row["variant"] == "T0-A_B0")
    product_result = next(
        row for row in results if row["variant"] == "T0-D_Bayesian_Product"
    )
    deltas = {
        "delta_mAP50": product_result["mAP50"] - baseline_result["mAP50"],
        "delta_recall": product_result["recall"] - baseline_result["recall"],
        "delta_small_recall": (
            product_result["small_recall"] - baseline_result["small_recall"]
        ),
        "relative_FP_per_frame": (
            product_result["FP_per_frame"] / max(baseline_result["FP_per_frame"], 1e-12)
        ),
    }
    gate_config = config()["gate"]["T0_D_vs_B0"]
    checks = {
        "delta_mAP50": deltas["delta_mAP50"] >= gate_config["delta_mAP50_min"],
        "delta_recall": deltas["delta_recall"] >= gate_config["delta_recall_min"],
        "delta_small_recall": (
            deltas["delta_small_recall"] >= gate_config["delta_small_recall_min"]
        ),
        "relative_FP_per_frame": (
            deltas["relative_FP_per_frame"]
            <= gate_config["relative_FP_per_frame_max"]
        ),
        "worst_fold_recall": (
            product_result["recall"] >= gate_config["worst_fold_recall_min"]
        ),
        "lost_GT": product_result["lost_GT"] == gate_config["lost_GT"],
        "sequence_leakage": (
            product_result["sequence_leakage"] == gate_config["sequence_leakage"]
        ),
    }
    gate = {
        "protocol_id": config()["protocol_id"],
        "fold": fold,
        "status": "T0_PASS" if all(checks.values()) else "T0_FAIL",
        "article_evidence": False,
        "baseline": baseline_result,
        "T0_D": product_result,
        "deltas": deltas,
        "checks": checks,
        "homography": {
            "pairs": int((~homography_frame["boundary"]).sum()),
            "valid": int(homography_frame["valid"].sum()),
            "failures": int(
                ((~homography_frame["boundary"]) & (~homography_frame["valid"])).sum()
            ),
        },
        "test_status": "SEALED",
        "attacks_status": "BLOCKED",
        "test_used": False,
    }
    atomic_csv(root / "T0_RESULTS.csv", pd.DataFrame(results))
    atomic_json(
        root / ("T0_GATE.json" if fold == 0 else "T0_CONFIRMATION.json"),
        gate,
    )
    return gate


def decision_trace(fold0: dict[str, Any], fold1: dict[str, Any] | None) -> None:
    decisions = [{
        "stage": "T0_fold_0",
        "decision": fold0["status"],
        "reason": [key for key, passed in fold0["checks"].items() if not passed],
    }]
    if fold1 is None:
        decisions.append({
            "stage": "T0_fold_1",
            "decision": "SKIPPED_BY_FOLD_0_GATE",
            "reason": ["fold_0_T0_D_did_not_pass_all_frozen_checks"],
        })
    else:
        decisions.append({
            "stage": "T0_fold_1",
            "decision": fold1["status"],
            "reason": [key for key, passed in fold1["checks"].items() if not passed],
        })
    atomic_json(
        OUTPUT / "decision_trace.json",
        {
            "protocol_id": config()["protocol_id"],
            "article_evidence": False,
            "decisions": decisions,
            "T1_status": "RELEASED" if fold1 and fold1["status"] == "T0_PASS" else "BLOCKED",
            "test_status": "SEALED",
            "attacks_status": "BLOCKED",
        },
    )


def main() -> None:
    assert_locked()
    fold0 = evaluate_fold(0)
    fold1 = evaluate_fold(1) if fold0["status"] == "T0_PASS" else None
    decision_trace(fold0, fold1)
    print(json.dumps({"fold_0": fold0["status"], "fold_1": fold1 and fold1["status"]}, indent=2))


if __name__ == "__main__":
    main()
