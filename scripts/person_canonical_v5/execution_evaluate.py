from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.audit_evaluator import average_precision, match_dataset
from scripts.person_canonical_v5.common import atomic_csv, atomic_json, now, sha256
from scripts.person_canonical_v5.execution_common import (
    OUTPUT,
    assert_execution_locked,
    config,
)
from scripts.person_canonical_v5.train_candidates import CANDIDATE_ROOT
from scripts.person_v3.evaluate import (
    detections_frame,
    infer_all,
    person_gt,
    reparse,
    selected_sources,
)
from scripts.person_v5.evaluate import size_recall
from src.models.coordinate_attention import register_ultralytics_modules


def _size_counts(
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> pd.DataFrame:
    from scripts.person_v3.evaluate import matched_indices

    rows = []
    bounds = {"small": (0.0, 0.001), "medium": (0.001, 0.01), "large": (0.01, math.inf)}
    counts = {name: {"GT": 0, "TP": 0} for name in bounds}
    for image, targets in gt.items():
        matched = matched_indices(targets, predictions.get(image, []), threshold)
        for index, target in enumerate(targets):
            x1, y1, x2, y2 = target["box"]
            area = (x2 - x1) * (y2 - y1) / (4112 * 2504)
            bucket = next(
                name for name, (lower, upper) in bounds.items()
                if lower <= area < upper
            )
            counts[bucket]["GT"] += 1
            counts[bucket]["TP"] += int(index in matched)
    for name, values in counts.items():
        rows.append({
            "size": name,
            **values,
            "FN": values["GT"] - values["TP"],
            "recall": values["TP"] / max(values["GT"], 1),
        })
    return pd.DataFrame(rows)


def _metrics(
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
    frames: int,
) -> dict[str, float]:
    operating = match_dataset(gt, predictions, threshold, 0.50)
    aps = [
        average_precision(gt, predictions, 0, float(iou))
        for iou in config()["evaluation"]["AP_IoU"]
    ]
    return {
        "mAP50": float(aps[0]),
        "mAP50_95": float(np.mean(aps)),
        **operating,
        **size_recall(gt, predictions, threshold),
        "FN_per_frame": float(operating["fn"]) / max(frames, 1),
        "FP_per_frame": float(operating["fp"]) / max(frames, 1),
        "number_of_GT": int(sum(len(values) for values in gt.values())),
        "number_of_predictions": int(
            sum(
                row["confidence"] >= threshold
                for values in predictions.values()
                for row in values
            )
        ),
    }


def _evaluate_checkpoint(
    candidate: str,
    fold: int,
    checkpoint: Path,
    destination: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    register_ultralytics_modules()
    source = selected_sources(fold)
    gt = person_gt(source)
    predictions, runtime_ms = infer_all(
        fold, checkpoint, source, destination
    )
    threshold = float(config()["evaluation"]["confidence_threshold"])
    metrics = _metrics(gt, predictions, threshold, len(source))
    frame = detections_frame(source, gt, predictions)
    csv_path = destination / "predictions_and_ground_truth.csv"
    atomic_csv(frame, csv_path)
    reparsed_gt, reparsed_predictions = reparse(pd.read_csv(csv_path))
    for image in gt:
        reparsed_gt.setdefault(image, [])
        reparsed_predictions.setdefault(image, [])
    repeated = _metrics(
        reparsed_gt, reparsed_predictions, threshold, len(source)
    )
    tolerance = float(config()["evaluation"]["evaluator_tolerance"])
    parity_keys = ("mAP50", "mAP50_95", "precision", "recall", "f1")
    differences = {
        key: abs(float(metrics[key]) - float(repeated[key]))
        for key in parity_keys
    }
    consistent = max(differences.values(), default=0.0) <= tolerance
    path_to_scene = {
        str(Path(row.output_image).resolve()): str(row.grouped_scene_id)
        for row in source.itertuples(index=False)
    }
    scene_rows = []
    for scene in sorted(set(path_to_scene.values())):
        images = {
            image for image, value in path_to_scene.items() if value == scene
        }
        scene_gt = {image: gt[image] for image in images}
        scene_predictions = {
            image: predictions.get(image, []) for image in images
        }
        scene_metrics = _metrics(
            scene_gt, scene_predictions, threshold, len(images)
        )
        scene_rows.append({
            "candidate": candidate,
            "fold": fold,
            "grouped_scene_id": scene,
            "frames": len(images),
            **scene_metrics,
        })
    per_scene = pd.DataFrame(scene_rows)
    per_size = _size_counts(gt, predictions, threshold)
    per_size.insert(0, "fold", fold)
    per_size.insert(0, "candidate", candidate)
    confidence = pd.DataFrame(
        {
            "confidence": [
                float(row["confidence"])
                for values in predictions.values()
                for row in values
            ]
        }
    )
    result = {
        "status": "PASS" if consistent else "FAIL",
        "created_at": now(),
        "candidate": candidate,
        "fold": fold,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "threshold": threshold,
        "frames": int(len(source)),
        "scenes": int(source["grouped_scene_id"].nunique()),
        **metrics,
        "mean_runtime_ms": float(runtime_ms) / max(len(source), 1),
        "evaluator_consistency": "PASS" if consistent else "FAIL",
        "evaluator_differences": differences,
        "lost_GT": abs(
            sum(len(values) for values in gt.values())
            - sum(len(values) for values in reparsed_gt.values())
        ),
        "missing_scenes": int(
            source["grouped_scene_id"].nunique() - len(per_scene)
        ),
        "NaN": int(
            any(np.isnan(float(metrics[key])) for key in parity_keys)
        ),
        "Inf": int(
            any(np.isinf(float(metrics[key])) for key in parity_keys)
        ),
        "test_used": False,
    }
    return result, frame, per_scene, per_size, confidence


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["recall"]),
        float(row["small_recall"]),
        float(row["mAP50"]),
        -float(row["FN_per_frame"]),
        -float(row["source_order"]),
    )


def evaluate(candidate: str, fold: int) -> dict[str, Any]:
    assert_execution_locked()
    if fold not in range(5):
        raise RuntimeError("Person-v5 execution only permits folds 0..4")
    root = CANDIDATE_ROOT / candidate / f"fold_{fold}"
    training = json.loads(
        (root / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
    )
    stage2 = root / "stage2"
    candidates = [
        ("stage2_best", stage2 / "weights/best.pt", 0),
        ("stage2_last", stage2 / "weights/last.pt", 1),
    ]
    rows: list[dict[str, Any]] = []
    hashes: dict[str, dict[str, Any]] = {}
    for label, checkpoint, order in candidates:
        if not checkpoint.is_file():
            raise RuntimeError(f"Checkpoint selection input missing: {checkpoint}")
        digest = sha256(checkpoint)
        if digest in hashes:
            rows.append({**hashes[digest], "label": label, "source_order": order})
            continue
        destination = root / "checkpoint_selection" / label
        result, _, _, _, _ = _evaluate_checkpoint(
            candidate, fold, checkpoint, destination
        )
        row = {
            "label": label,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": digest,
            "source_order": order,
            **result,
        }
        hashes[digest] = row
        rows.append(row)
    selected = max(rows, key=_selection_key)
    selected_path = Path(selected["checkpoint"])
    selected_copy = root / "selected.pt"
    if not selected_copy.is_file() or sha256(selected_copy) != selected["checkpoint_sha256"]:
        shutil.copy2(selected_path, selected_copy)
    selection = {
        "status": "PASS",
        "candidate": candidate,
        "fold": fold,
        "created_at": now(),
        "rule": config()["checkpoint_selection"]["lexicographic"],
        "validation_only": True,
        "selected_label": selected["label"],
        "selected_checkpoint": str(selected_copy.resolve()),
        "selected_checkpoint_sha256": sha256(selected_copy),
        "candidates": rows,
        "test_used": False,
    }
    atomic_json(root / "checkpoint_selection.json", selection)
    history = stage2 / "results.csv"
    if history.is_file():
        shutil.copy2(history, root / "training_history.csv")
    destination = root / "evaluation"
    result, frame, per_scene, per_size, confidence = _evaluate_checkpoint(
        candidate, fold, selected_copy, destination
    )
    result["checkpoint_selection"] = selection["selected_label"]
    atomic_csv(frame, destination / "predictions_and_ground_truth.csv")
    atomic_csv(per_scene, destination / "per_scene.csv")
    atomic_csv(per_size, destination / "per_size.csv")
    atomic_csv(confidence, destination / "confidence_distribution.csv")
    atomic_json(destination / "evaluation_result.json", result)
    atomic_json(destination / "evaluator_consistency.json", {
        "status": result["evaluator_consistency"],
        "absolute_differences": result["evaluator_differences"],
    })
    if result["status"] != "PASS" or result["lost_GT"] != 0:
        raise RuntimeError("Candidate independent evaluator failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.candidate, args.fold), indent=2))


if __name__ == "__main__":
    main()
