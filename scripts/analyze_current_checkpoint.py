from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from audit_evaluator import box_iou
from rescue_common import (
    PROJECT_DIR,
    OUTPUT_ROOT,
    assert_test_sealed,
    atomic_json,
    completed_marker,
    load_protocol,
    sha256,
)


OUTPUT = OUTPUT_ROOT / "current_checkpoint"
DATA = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/data.yaml"
MANIFEST = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"


def build_taxonomy(detections: Path, threshold: float) -> pd.DataFrame:
    frame = pd.read_csv(detections)
    output = []
    for image_path, rows in frame.groupby("image_path"):
        gt = rows[rows["kind"] == "ground_truth"].to_dict("records")
        predictions = rows[rows["kind"] == "prediction"].sort_values(
            "confidence", ascending=False
        ).to_dict("records")
        matched_gt: set[int] = set()
        matched_prediction: set[int] = set()
        for prediction_index, prediction in enumerate(predictions):
            if float(prediction["confidence"]) < threshold:
                continue
            candidates = [
                (gt_index, box_iou(
                    [prediction["x1"], prediction["y1"], prediction["x2"], prediction["y2"]],
                    [target["x1"], target["y1"], target["x2"], target["y2"]],
                ))
                for gt_index, target in enumerate(gt)
                if gt_index not in matched_gt
                and int(target["class_id"]) == int(prediction["class_id"])
            ]
            if candidates:
                gt_index, overlap = max(candidates, key=lambda item: item[1])
                if overlap >= 0.5:
                    matched_gt.add(gt_index)
                    matched_prediction.add(prediction_index)
        for gt_index, target in enumerate(gt):
            if gt_index in matched_gt:
                error = "detected"
            else:
                target_box = [target["x1"], target["y1"], target["x2"], target["y2"]]
                overlaps = [
                    (
                        prediction,
                        box_iou(
                            [prediction["x1"], prediction["y1"], prediction["x2"], prediction["y2"]],
                            target_box,
                        ),
                    )
                    for prediction in predictions
                ]
                if any(
                    overlap >= 0.5
                    and int(prediction["class_id"]) == int(target["class_id"])
                    and float(prediction["confidence"]) < threshold
                    for prediction, overlap in overlaps
                ):
                    error = "confidence_below_threshold"
                elif any(
                    overlap >= 0.5
                    and int(prediction["class_id"]) != int(target["class_id"])
                    and float(prediction["confidence"]) >= threshold
                    for prediction, overlap in overlaps
                ):
                    error = "wrong_class"
                elif any(
                    0.1 <= overlap < 0.5
                    and int(prediction["class_id"]) == int(target["class_id"])
                    and float(prediction["confidence"]) >= threshold
                    for prediction, overlap in overlaps
                ):
                    error = "localization_failure"
                else:
                    error = "object_not_detected"
            output.append({
                "image_path": image_path, "kind": "ground_truth",
                "class_id": int(target["class_id"]), "error_type": error,
            })
        for prediction_index, prediction in enumerate(predictions):
            if float(prediction["confidence"]) < threshold or prediction_index in matched_prediction:
                continue
            prediction_box = [
                prediction["x1"], prediction["y1"], prediction["x2"], prediction["y2"]
            ]
            duplicate = any(
                int(target["class_id"]) == int(prediction["class_id"])
                and box_iou(
                    [target["x1"], target["y1"], target["x2"], target["y2"]],
                    prediction_box,
                ) >= 0.5
                for target in gt
            )
            output.append({
                "image_path": image_path, "kind": "prediction",
                "class_id": int(prediction["class_id"]),
                "error_type": "duplicate_prediction" if duplicate else "background_false_positive",
            })
    return pd.DataFrame(output)


def main() -> None:
    assert_test_sealed()
    protocol = load_protocol()
    checkpoint = PROJECT_DIR / protocol["failed_run"]["checkpoint"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = OUTPUT / "baseline_rescue_summary.json"
    if not summary.is_file() or not (OUTPUT / "detections_val.csv").is_file():
        subprocess.run([
            str(PROJECT_DIR / ".venv/bin/python"), "baseline_rescue_audit.py",
            "--model", str(checkpoint), "--data", str(DATA),
            "--manifest", str(MANIFEST), "--output", str(OUTPUT),
            "--splits", "val", "--imgsz", "1280",
            "--small-area", str(protocol["object_sizes"]["small_area_ratio_max"]),
            "--medium-area", str(protocol["object_sizes"]["medium_area_ratio_max"]),
        ], cwd=PROJECT_DIR, check=True)
    thresholds = json.loads((OUTPUT / "threshold_selection.json").read_text(encoding="utf-8"))
    threshold = float(thresholds["standard"]["confidence"])
    taxonomy = build_taxonomy(OUTPUT / "detections_val.csv", threshold)
    taxonomy.to_csv(OUTPUT / "error_taxonomy.csv", index=False)
    counts = Counter(taxonomy["error_type"])
    result = {
        "status": "PASS",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "selection_split": "validation",
        "standard_threshold": threshold,
        "error_taxonomy_counts": dict(counts),
        "per_class_metrics": "ap_by_class.csv and clean_metrics_by_class_and_size.csv",
        "per_scene_metrics": "clean_metrics_per_scene.csv",
        "per_size_metrics": "clean_metrics_by_class_and_size.csv",
        "test_evaluated": False,
    }
    atomic_json(OUTPUT / "summary.json", result)
    completed_marker(
        OUTPUT, inputs=[checkpoint, DATA, MANIFEST],
        outputs=[
            OUTPUT / "summary.json", OUTPUT / "error_taxonomy.csv",
            OUTPUT / "ap_by_class.csv", OUTPUT / "clean_metrics_per_scene.csv",
            OUTPUT / "clean_metrics_by_class_and_size.csv",
            OUTPUT / "threshold_sweep.csv", OUTPUT / "detections_val.csv",
        ],
        extra={"stage": "current_checkpoint_diagnosis", "test_evaluated": False},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
