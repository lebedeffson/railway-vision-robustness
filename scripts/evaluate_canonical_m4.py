from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from audit_evaluator import average_precision, box_iou, match_dataset
from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    assert_role_allowed,
    atomic_json,
    load_protocol,
    now,
    sha256,
)
from canonical_m4_tiling import (
    frozen_tiles,
    fuse_predictions,
    restore_global_box,
)
from run_micro_view_candidate_v2 import read_yolo_labels
from train_canonical_m4 import training_root


TILING_ROOT = OUTPUT_ROOT / "tiling_audit"
TILE_MANIFEST = TILING_ROOT / "tile_manifest.csv"
SOURCE_MANIFEST = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def output_root(mode: str, seed: int, fold: int | None) -> Path:
    if mode == "cv":
        if fold is None:
            raise ValueError("CV evaluation requires a fold")
        return OUTPUT_ROOT / "scene_cv" / f"fold_{fold}" / f"seed_{seed}" / "evaluation"
    return OUTPUT_ROOT / "validation" / f"seed_{seed}"


def selected_sources(
    mode: str, fold: int | None, protocol: dict[str, Any]
) -> pd.DataFrame:
    source = pd.read_csv(SOURCE_MANIFEST)
    if mode == "official":
        result = source[source["split"].eq("val")].copy()
    else:
        scenes = {
            str(value)
            for value in protocol["scene_cv"]["fold_validation_scenes"][int(fold)]
        }
        result = source[
            source["split"].eq("train")
            & source["grouped_scene_id"].astype(str).isin(scenes)
        ].copy()
    return result.sort_values(
        ["grouped_scene_id", "subsequence_id", "frame_id"]
    ).reset_index(drop=True)


def source_ground_truth(source: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in source.itertuples(index=False):
        path = str(Path(row.output_image).resolve())
        with Image.open(path) as image:
            result[path] = read_yolo_labels(Path(row.output_label), image.width, image.height)
    return result


def infer_frame(
    model: YOLO,
    rows: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tile_lookup = {tile.tile_id: tile for tile in frozen_tiles(protocol)}
    raw: list[dict[str, Any]] = []
    for row in rows.sort_values("tile_id").itertuples(index=False):
        result = model.predict(
            source=str(row.tile_image),
            imgsz=int(protocol["model"]["input_size"]),
            conf=float(protocol["fusion"]["confidence_prefilter"]),
            iou=0.70,
            max_det=int(protocol["fusion"]["per_tile_max_detections"]),
            device=0,
            verbose=False,
        )[0]
        if result.boxes is None:
            continue
        tile = tile_lookup[str(row.tile_id)]
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            raw.append({
                "tile_id": str(row.tile_id),
                "class_id": int(class_id),
                "confidence": float(confidence),
                "box": restore_global_box(list(map(float, box)), tile),
            })
    fused = fuse_predictions(raw, protocol)
    return raw, fused


def infer_all(
    checkpoint: Path,
    source: pd.DataFrame,
    protocol: dict[str, Any],
    destination: Path,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame, float]:
    tile_manifest = pd.read_csv(TILE_MANIFEST)
    tile_manifest["source_image"] = tile_manifest["source_image"].map(
        lambda value: str(Path(value).resolve())
    )
    model = YOLO(str(checkpoint))
    cache = destination / "prediction_cache"
    cache.mkdir(parents=True, exist_ok=True)
    fused_by_image: dict[str, list[dict[str, Any]]] = {}
    raw_rows: list[dict[str, Any]] = []
    total_runtime_ms = 0.0
    for record in source.itertuples(index=False):
        image_path = str(Path(record.output_image).resolve())
        frame_cache = cache / f"{Path(image_path).stem}.json"
        if frame_cache.is_file():
            payload = json.loads(frame_cache.read_text(encoding="utf-8"))
            if payload.get("checkpoint_sha256") != sha256(checkpoint):
                raise RuntimeError(f"Prediction cache checkpoint mismatch: {frame_cache}")
        else:
            frame_tiles = tile_manifest[tile_manifest["source_image"].eq(image_path)]
            if len(frame_tiles) != int(protocol["tiling"]["expected_tiles_per_frame"]):
                raise RuntimeError(f"Missing M4 inference tiles for {image_path}")
            started = time.perf_counter()
            raw, fused = infer_frame(model, frame_tiles, protocol)
            inference_runtime_ms = (time.perf_counter() - started) * 1000
            payload = {
                "source_image": image_path,
                "grouped_scene_id": str(record.grouped_scene_id),
                "checkpoint_sha256": sha256(checkpoint),
                "raw": raw,
                "fused": fused,
                "inference_runtime_ms": inference_runtime_ms,
            }
            atomic_json(frame_cache, payload)
        total_runtime_ms += float(payload.get("inference_runtime_ms", 0.0))
        fused_by_image[image_path] = list(payload["fused"])
        for row in payload["raw"]:
            raw_rows.append({
                "image_path": image_path,
                "grouped_scene_id": str(record.grouped_scene_id),
                "kind": "raw_tile_prediction",
                **{key: value for key, value in row.items() if key != "box"},
                "x1": row["box"][0],
                "y1": row["box"][1],
                "x2": row["box"][2],
                "y2": row["box"][3],
            })
    return fused_by_image, pd.DataFrame(raw_rows), total_runtime_ms


def matched_indices(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    confidence: float,
    iou_threshold: float,
) -> set[int]:
    active = sorted(
        [row for row in predictions if float(row["confidence"]) >= confidence],
        key=lambda row: -float(row["confidence"]),
    )
    matched: set[int] = set()
    for prediction in active:
        available = [
            (index, box_iou(prediction["box"], target["box"]))
            for index, target in enumerate(ground_truth)
            if index not in matched
            and int(prediction["class_id"]) == int(target["class_id"])
        ]
        if available:
            index, overlap = max(available, key=lambda item: item[1])
            if overlap >= iou_threshold:
                matched.add(index)
    return matched


def map_metrics(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> tuple[float, float, dict[int, float]]:
    classes = sorted({
        int(target["class_id"])
        for targets in ground_truth.values()
        for target in targets
    })
    ap50 = {
        class_id: average_precision(ground_truth, predictions, class_id, 0.5)
        for class_id in classes
    }
    finite50 = [value for value in ap50.values() if math.isfinite(value)]
    map50 = float(np.mean(finite50)) if finite50 else 0.0
    all_values = [
        average_precision(ground_truth, predictions, class_id, threshold)
        for class_id in classes
        for threshold in np.arange(0.50, 0.96, 0.05)
    ]
    finite = [value for value in all_values if math.isfinite(value)]
    return map50, float(np.mean(finite)) if finite else 0.0, ap50


def threshold_sweep(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    protocol: dict[str, Any],
) -> pd.DataFrame:
    values = np.arange(
        float(protocol["evaluation"]["confidence_min"]),
        float(protocol["evaluation"]["confidence_max"])
        + float(protocol["evaluation"]["confidence_step"]) / 2,
        float(protocol["evaluation"]["confidence_step"]),
    )
    rows = []
    for confidence in values:
        metrics = match_dataset(
            ground_truth,
            predictions,
            float(confidence),
            float(protocol["evaluation"]["iou_match_threshold"]),
        )
        metrics["f2"] = (
            5 * metrics["precision"] * metrics["recall"]
            / max(4 * metrics["precision"] + metrics["recall"], 1e-12)
        )
        rows.append({"threshold": float(round(confidence, 6)), **metrics})
    return pd.DataFrame(rows)


def select_thresholds(sweep: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    standard = sweep.sort_values(
        ["f1", "recall", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    safety = sweep.sort_values(
        ["f2", "recall", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    return standard, safety


def scoped_rows(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenes = []
    classes = []
    for scene, frame in source.groupby("grouped_scene_id"):
        paths = {str(Path(path).resolve()) for path in frame["output_image"]}
        gt = {path: ground_truth[path] for path in paths}
        pred = {path: predictions.get(path, []) for path in paths}
        map50, map5095, _ = map_metrics(gt, pred)
        metrics = match_dataset(gt, pred, threshold)
        scenes.append({
            "grouped_scene_id": str(scene),
            "frames": len(paths),
            "mAP50": map50,
            "mAP50_95": map5095,
            "fn_per_frame": metrics["fn"] / max(len(paths), 1),
            **metrics,
        })
    for class_id in range(6):
        gt = {
            path: [row for row in rows if int(row["class_id"]) == class_id]
            for path, rows in ground_truth.items()
        }
        pred = {
            path: [row for row in rows if int(row["class_id"]) == class_id]
            for path, rows in predictions.items()
        }
        metrics = match_dataset(gt, pred, threshold)
        ap50 = average_precision(gt, pred, class_id, 0.5)
        classes.append({"class_id": class_id, "AP50": ap50, **metrics})
    return pd.DataFrame(scenes), pd.DataFrame(classes)


def size_and_border_rows(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = {
        name: {"gt": 0, "tp": 0}
        for name in ("small", "medium", "large")
    }
    border_counts = {
        name: {"gt": 0, "tp": 0}
        for name in ("border", "non_border")
    }
    width = float(protocol["dataset"]["image_width"])
    height = float(protocol["dataset"]["image_height"])
    small = float(protocol["evaluation"]["size_area_ratio_thresholds"]["small"])
    medium = float(protocol["evaluation"]["size_area_ratio_thresholds"]["medium"])
    internal_x = {tile.left for tile in frozen_tiles(protocol) if tile.left} | {
        tile.right for tile in frozen_tiles(protocol) if tile.right < width
    }
    internal_y = {tile.top for tile in frozen_tiles(protocol) if tile.top} | {
        tile.bottom for tile in frozen_tiles(protocol) if tile.bottom < height
    }
    for image_path, targets in ground_truth.items():
        matched = matched_indices(
            targets,
            predictions.get(image_path, []),
            threshold,
            float(protocol["evaluation"]["iou_match_threshold"]),
        )
        for index, target in enumerate(targets):
            x1, y1, x2, y2 = map(float, target["box"])
            area = (x2 - x1) * (y2 - y1) / (width * height)
            size = "small" if area < small else ("medium" if area < medium else "large")
            border = any(x1 < value < x2 for value in internal_x) or any(
                y1 < value < y2 for value in internal_y
            )
            counts[size]["gt"] += 1
            counts[size]["tp"] += int(index in matched)
            border_counts["border" if border else "non_border"]["gt"] += 1
            border_counts["border" if border else "non_border"]["tp"] += int(index in matched)
    size_frame = pd.DataFrame([
        {"size": name, **values, "recall": values["tp"] / max(values["gt"], 1)}
        for name, values in counts.items()
    ])
    border_frame = pd.DataFrame([
        {"scope": name, **values, "recall": values["tp"] / max(values["gt"], 1)}
        for name, values in border_counts.items()
    ])
    return size_frame, border_frame


def detections_frame(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    scene_by_image = {
        str(Path(row.output_image).resolve()): str(row.grouped_scene_id)
        for row in source.itertuples(index=False)
    }
    rows = []
    for image_path, targets in ground_truth.items():
        for row in targets:
            rows.append({
                "image_path": image_path,
                "grouped_scene_id": scene_by_image[image_path],
                "kind": "ground_truth",
                "class_id": int(row["class_id"]),
                "confidence": 1.0,
                "x1": row["box"][0], "y1": row["box"][1],
                "x2": row["box"][2], "y2": row["box"][3],
            })
        for row in predictions.get(image_path, []):
            rows.append({
                "image_path": image_path,
                "grouped_scene_id": scene_by_image[image_path],
                "kind": "prediction",
                "class_id": int(row["class_id"]),
                "confidence": float(row["confidence"]),
                "x1": row["box"][0], "y1": row["box"][1],
                "x2": row["box"][2], "y2": row["box"][3],
            })
    return pd.DataFrame(rows)


def reparse_detections(
    frame: pd.DataFrame,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    gt: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        item = {
            "class_id": int(row.class_id),
            "box": [float(row.x1), float(row.y1), float(row.x2), float(row.y2)],
        }
        if row.kind == "ground_truth":
            gt[str(row.image_path)].append(item)
        else:
            predictions[str(row.image_path)].append({
                **item, "confidence": float(row.confidence),
            })
    for path in set(predictions) - set(gt):
        gt[path] = []
    return dict(gt), dict(predictions)


def evaluate(
    mode: str,
    seed: int,
    fold: int | None,
    checkpoint: Path,
) -> dict[str, Any]:
    assert_role_allowed("scene_cv" if mode == "cv" else "validation")
    protocol = load_protocol()
    source = selected_sources(mode, fold, protocol)
    expected = (
        int(protocol["dataset"]["split_frame_counts"]["val"])
        if mode == "official"
        else None
    )
    if expected is not None and len(source) != expected:
        raise RuntimeError("Official validation frame count mismatch")
    destination = output_root(mode, seed, fold)
    destination.mkdir(parents=True, exist_ok=True)
    ground_truth = source_ground_truth(source)
    predictions, raw, total_runtime_ms = infer_all(
        checkpoint, source, protocol, destination
    )
    detections = detections_frame(source, ground_truth, predictions)
    atomic_csv(raw, destination / "raw_tile_predictions.csv")
    atomic_csv(detections, destination / "predictions_and_ground_truth.csv")
    sweep = threshold_sweep(ground_truth, predictions, protocol)
    standard, safety = select_thresholds(sweep)
    atomic_csv(sweep, destination / "threshold_sweep.csv")
    map50, map5095, _ = map_metrics(ground_truth, predictions)
    scene, per_class = scoped_rows(
        source, ground_truth, predictions, float(safety["threshold"])
    )
    per_size, border = size_and_border_rows(
        ground_truth,
        predictions,
        float(safety["threshold"]),
        protocol,
    )
    atomic_csv(scene, destination / "per_scene_metrics.csv")
    atomic_csv(per_class, destination / "per_class_metrics.csv")
    atomic_csv(per_size, destination / "per_size_metrics.csv")
    atomic_csv(border, destination / "border_object_metrics.csv")

    re_gt, re_predictions = reparse_detections(
        pd.read_csv(destination / "predictions_and_ground_truth.csv")
    )
    repeated_map50, repeated_map5095, _ = map_metrics(re_gt, re_predictions)
    repeated = match_dataset(re_gt, re_predictions, float(safety["threshold"]))
    tolerance = float(protocol["evaluation"]["independent_evaluator_tolerance"])
    differences = {
        "mAP50": abs(map50 - repeated_map50),
        "mAP50_95": abs(map5095 - repeated_map5095),
        "precision": abs(float(safety["precision"]) - repeated["precision"]),
        "recall": abs(float(safety["recall"]) - repeated["recall"]),
        "f1": abs(float(safety["f1"]) - repeated["f1"]),
    }
    training_results_path = (
        training_root(mode, seed, fold) / "stage3/results.csv"
    )
    built_in = {}
    if training_results_path.is_file():
        training_results = pd.read_csv(training_results_path)
        last = training_results.iloc[-1]
        built_in = {
            key: float(last[key])
            for key in (
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            )
            if key in last and np.isfinite(last[key])
        }
    consistency = {
        "status": (
            "PASS"
            if max(differences.values()) <= tolerance and bool(built_in)
            else "FAIL"
        ),
        "primary_evaluator": "project_global_tiling_fusion_in_memory",
        "independent_evaluator": "project_global_tiling_fusion_csv_reparse",
        "ultralytics_training_evaluator_scope":
            "tile_level_diagnostic_not_numerically_equivalent_to_global_fusion",
        "ultralytics_training_evaluator_completed": bool(built_in),
        "ultralytics_tile_level_diagnostic": built_in,
        "tolerance": tolerance,
        "absolute_differences": differences,
    }
    atomic_json(destination / "evaluator_consistency.json", consistency)
    if consistency["status"] != "PASS":
        raise RuntimeError("Canonical M4 independent evaluator mismatch")
    selection = {
        "selection_split": "train_cv" if mode == "cv" else "validation",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "standard": standard.to_dict(),
        "safety": safety.to_dict(),
        "test_used": False,
    }
    atomic_json(destination / "threshold_selection.json", selection)
    result = {
        "status": "PASS",
        "finished_at": now(),
        "protocol_id": protocol["protocol_id"],
        "mode": mode,
        "fold": fold,
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "frames": len(source),
        "scenes": int(source["grouped_scene_id"].nunique()),
        "mAP50": map50,
        "mAP50_95": map5095,
        "standard_threshold": float(standard["threshold"]),
        "standard_precision": float(standard["precision"]),
        "standard_recall": float(standard["recall"]),
        "standard_f1": float(standard["f1"]),
        "safety_threshold": float(safety["threshold"]),
        "safety_precision": float(safety["precision"]),
        "safety_recall": float(safety["recall"]),
        "safety_f1": float(safety["f1"]),
        "safety_f2": float(safety["f2"]),
        "safety_fn": int(safety["fn"]),
        "safety_fn_per_frame": float(safety["fn"]) / max(len(source), 1),
        "scene_macro_map50": float(scene["mAP50"].mean()),
        "scene_macro_safety_recall": float(scene["recall"].mean()),
        "scene_safety_recall_std": float(scene["recall"].std(ddof=0)),
        "mean_tiling_inference_latency_ms": total_runtime_ms / max(len(source), 1),
        "small_recall": float(
            per_size.loc[per_size["size"].eq("small"), "recall"].iloc[0]
        ),
        "medium_recall": float(
            per_size.loc[per_size["size"].eq("medium"), "recall"].iloc[0]
        ),
        "large_recall": float(
            per_size.loc[per_size["size"].eq("large"), "recall"].iloc[0]
        ),
        "evaluator_consistency_passed": True,
        "all_expected_frames_present": expected is None or len(source) == expected,
        "no_duplicate_predictions": not detections[
            detections["kind"].eq("prediction")
        ].duplicated(
            subset=["image_path", "class_id", "confidence", "x1", "y1", "x2", "y2"]
        ).any(),
        "no_missing_scene": (
            int(source["grouped_scene_id"].nunique())
            == (2 if mode == "cv" else int(protocol["dataset"]["split_group_counts"]["val"]))
        ),
        "no_nan_or_inf": bool(
            np.isfinite(
                [
                    map50, map5095, standard["f1"], safety["recall"],
                    scene["mAP50"].mean(),
                ]
            ).all()
        ),
        "test_evaluated": False,
    }
    atomic_json(destination / "evaluation_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cv", "official"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if checkpoint is None:
        root = training_root(args.mode, args.seed, args.fold)
        completion = json.loads(
            (root / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
        )
        checkpoint = Path(completion["final_checkpoint"])
    print(json.dumps(evaluate(args.mode, args.seed, args.fold, checkpoint), indent=2))


if __name__ == "__main__":
    main()
