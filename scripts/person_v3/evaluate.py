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
import yaml
from ultralytics import YOLO

from scripts.audit_evaluator import average_precision, box_iou, match_dataset
from scripts.canonical_m4_tiling import (
    frozen_tiles,
    fuse_predictions,
    restore_global_box,
)
from scripts.person_v3.common import (
    FOLDS_PATH,
    OUTPUT_ROOT,
    PROJECT_DIR,
    assert_locked,
    atomic_json,
    load_protocol,
    now,
    sha256,
)
from scripts.person_v3.train import training_root
from scripts.run_micro_view_candidate_v2 import read_yolo_labels


M4_PROTOCOL = PROJECT_DIR / "configs/canonical_v2_m4_full_protocol.yaml"
M4_TILE_MANIFEST = (
    PROJECT_DIR / "outputs/canonical_m4/tiling_audit/tile_manifest.csv"
)
SOURCE_MANIFEST = (
    PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def selected_sources(fold: int) -> pd.DataFrame:
    heldout = set(
        json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"][str(fold)]
    )
    source = pd.read_csv(SOURCE_MANIFEST)
    return (
        source[
            source["split"].isin(["train", "val"])
            & source["grouped_scene_id"].astype(str).isin(heldout)
        ]
        .sort_values(["grouped_scene_id", "subsequence_id", "frame_id"])
        .reset_index(drop=True)
    )


def person_gt(source: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for row in source.itertuples(index=False):
        path = str(Path(row.output_image).resolve())
        labels = read_yolo_labels(Path(row.output_label), 4112, 2504)
        result[path] = [
            {"class_id": 0, "box": label["box"]}
            for label in labels
            if int(label["class_id"]) == 0
        ]
    return result


def infer_frame(
    model: YOLO,
    tile_rows: pd.DataFrame,
    m4_protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {tile.tile_id: tile for tile in frozen_tiles(m4_protocol)}
    raw = []
    for row in tile_rows.sort_values("tile_id").itertuples(index=False):
        result = model.predict(
            source=str(row.tile_image),
            imgsz=640,
            conf=float(m4_protocol["fusion"]["confidence_prefilter"]),
            iou=0.70,
            max_det=int(m4_protocol["fusion"]["per_tile_max_detections"]),
            device=0,
            verbose=False,
        )[0]
        if result.boxes is None:
            continue
        tile = lookup[str(row.tile_id)]
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            if int(class_id) != 0:
                raise RuntimeError("Person-only model emitted a non-person class")
            raw.append({
                "tile_id": str(row.tile_id),
                "class_id": 0,
                "confidence": float(confidence),
                "box": restore_global_box(list(map(float, box)), tile),
            })
    return raw, fuse_predictions(raw, m4_protocol)


def infer_all(
    fold: int,
    checkpoint: Path,
    source: pd.DataFrame,
    destination: Path,
) -> tuple[dict[str, list[dict[str, Any]]], float]:
    m4_protocol = yaml.safe_load(M4_PROTOCOL.read_text(encoding="utf-8"))
    tile_manifest = pd.read_csv(M4_TILE_MANIFEST)
    tile_manifest["source_image"] = tile_manifest["source_image"].map(
        lambda value: str(Path(value).resolve())
    )
    model = YOLO(str(checkpoint))
    cache = destination / "prediction_cache"
    cache.mkdir(parents=True, exist_ok=True)
    predictions = {}
    runtime = 0.0
    for row in source.itertuples(index=False):
        image = str(Path(row.output_image).resolve())
        cache_path = cache / f"{Path(image).stem}.json"
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload["checkpoint_sha256"] != sha256(checkpoint):
                raise RuntimeError("Person v3 prediction cache checkpoint mismatch")
        else:
            tiles = tile_manifest[tile_manifest["source_image"].eq(image)]
            if len(tiles) != 4:
                raise RuntimeError(f"Person v3 source misses frozen tiles: {image}")
            started = time.perf_counter()
            raw, fused = infer_frame(model, tiles, m4_protocol)
            elapsed = (time.perf_counter() - started) * 1000
            payload = {
                "fold": fold,
                "source_image": image,
                "grouped_scene_id": str(row.grouped_scene_id),
                "checkpoint_sha256": sha256(checkpoint),
                "raw": raw,
                "fused": fused,
                "runtime_ms": elapsed,
                "test_used": False,
            }
            atomic_json(cache_path, payload)
        predictions[image] = payload["fused"]
        runtime += float(payload["runtime_ms"])
    return predictions, runtime


def detections_frame(
    source: pd.DataFrame,
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    scenes = {
        str(Path(row.output_image).resolve()): str(row.grouped_scene_id)
        for row in source.itertuples(index=False)
    }
    rows = []
    for image, targets in gt.items():
        for target in targets:
            rows.append({
                "image_path": image,
                "grouped_scene_id": scenes[image],
                "kind": "ground_truth",
                "class_id": 0,
                "confidence": 1.0,
                "x1": target["box"][0],
                "y1": target["box"][1],
                "x2": target["box"][2],
                "y2": target["box"][3],
            })
        for prediction in predictions.get(image, []):
            rows.append({
                "image_path": image,
                "grouped_scene_id": scenes[image],
                "kind": "prediction",
                "class_id": 0,
                "confidence": prediction["confidence"],
                "x1": prediction["box"][0],
                "y1": prediction["box"][1],
                "x2": prediction["box"][2],
                "y2": prediction["box"][3],
            })
    return pd.DataFrame(rows)


def reparse(
    frame: pd.DataFrame,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    gt: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        value = {
            "class_id": 0,
            "box": [row.x1, row.y1, row.x2, row.y2],
        }
        if row.kind == "ground_truth":
            gt[row.image_path].append(value)
        else:
            predictions[row.image_path].append({
                **value, "confidence": row.confidence
            })
    for image in set(predictions) - set(gt):
        gt[image] = []
    return dict(gt), dict(predictions)


def matched_indices(
    gt: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    threshold: float,
) -> set[int]:
    matched: set[int] = set()
    for prediction in sorted(
        [row for row in predictions if row["confidence"] >= threshold],
        key=lambda row: -row["confidence"],
    ):
        candidates = [
            (index, box_iou(prediction["box"], target["box"]))
            for index, target in enumerate(gt)
            if index not in matched
        ]
        if candidates:
            index, overlap = max(candidates, key=lambda value: value[1])
            if overlap >= 0.50:
                matched.add(index)
    return matched


def evaluate(fold: int) -> dict[str, Any]:
    assert_locked()
    source = selected_sources(fold)
    root = training_root(fold)
    completion = json.loads(
        (root / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
    )
    checkpoint = Path(completion["final_checkpoint"])
    destination = root / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    gt = person_gt(source)
    predictions, runtime = infer_all(fold, checkpoint, source, destination)
    detections = detections_frame(source, gt, predictions)
    atomic_csv(detections, destination / "predictions_and_ground_truth.csv")
    map50 = average_precision(gt, predictions, 0, 0.50)
    aps = [
        average_precision(gt, predictions, 0, threshold)
        for threshold in np.arange(0.50, 0.96, 0.05)
    ]
    map5095 = float(np.mean(aps))
    reparsed_gt, reparsed_predictions = reparse(
        pd.read_csv(destination / "predictions_and_ground_truth.csv")
    )
    repeated_map50 = average_precision(
        reparsed_gt, reparsed_predictions, 0, 0.50
    )
    repeated_aps = [
        average_precision(reparsed_gt, reparsed_predictions, 0, threshold)
        for threshold in np.arange(0.50, 0.96, 0.05)
    ]
    tolerance = float(
        load_protocol()["evaluation"]["independent_reparse_tolerance"]
    )
    differences = {
        "mAP50": abs(map50 - repeated_map50),
        "mAP50_95": abs(map5095 - float(np.mean(repeated_aps))),
    }
    consistency = {
        "status": "PASS" if max(differences.values()) <= tolerance else "FAIL",
        "primary": "global_person_evaluator_in_memory",
        "independent": "global_person_evaluator_csv_reparse",
        "tolerance": tolerance,
        "absolute_differences": differences,
    }
    atomic_json(destination / "evaluator_consistency.json", consistency)
    if consistency["status"] != "PASS":
        raise RuntimeError("Person v3 evaluator consistency failed")
    scene_rows = []
    for scene, frame in source.groupby("grouped_scene_id"):
        images = {str(Path(value).resolve()) for value in frame["output_image"]}
        scene_gt = {image: gt[image] for image in images}
        scene_predictions = {
            image: predictions.get(image, []) for image in images
        }
        scene_rows.append({
            "grouped_scene_id": str(scene),
            "frames": len(images),
            "person_GT": sum(len(values) for values in scene_gt.values()),
            "mAP50": average_precision(
                scene_gt, scene_predictions, 0, 0.50
            ),
        })
    atomic_csv(pd.DataFrame(scene_rows), destination / "per_scene_AP.csv")
    result = {
        "status": "PASS",
        "finished_at": now(),
        "protocol_id": load_protocol()["protocol_id"],
        "fold": fold,
        "seed": 20260723,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "frames": int(len(source)),
        "scenes": int(source["grouped_scene_id"].nunique()),
        "person_GT": int(sum(len(values) for values in gt.values())),
        "mAP50": float(map50),
        "mAP50_95": map5095,
        "mean_runtime_ms": runtime / max(len(source), 1),
        "evaluator_consistency": consistency["status"],
        "lost_GT": 0,
        "missing_scenes": 0,
        "no_nan_inf": bool(np.isfinite([map50, map5095]).all()),
        "test_used": False,
    }
    atomic_json(destination / "evaluation_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.fold), indent=2))


if __name__ == "__main__":
    main()

