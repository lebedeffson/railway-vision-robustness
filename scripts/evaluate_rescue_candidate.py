from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_evaluator import average_precision
from rescue_common import (
    PROJECT_DIR,
    assert_test_sealed,
    atomic_json,
    completed_marker,
    load_protocol,
    sha256,
)


def detection_dicts(path: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    frame = pd.read_csv(path)
    ground_truth: defaultdict[str, list[dict]] = defaultdict(list)
    predictions: defaultdict[str, list[dict]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        record = {
            "class_id": int(row.class_id),
            "box": [float(row.x1), float(row.y1), float(row.x2), float(row.y2)],
        }
        if row.kind == "ground_truth":
            ground_truth[str(row.image_path)].append(record)
        else:
            predictions[str(row.image_path)].append({
                **record, "confidence": float(row.confidence),
            })
    for image in set(predictions) - set(ground_truth):
        ground_truth[image] = []
    return dict(ground_truth), dict(predictions)


def scoped_map50(
    ground_truth: dict[str, list[dict]], predictions: dict[str, list[dict]],
    image_paths: set[str],
) -> tuple[float, dict[int, float]]:
    gt = {path: ground_truth.get(path, []) for path in sorted(image_paths)}
    pred = {path: predictions.get(path, []) for path in sorted(image_paths)}
    classes = sorted({
        target["class_id"] for rows in gt.values() for target in rows
    })
    per_class = {
        class_id: average_precision(gt, pred, class_id, 0.5)
        for class_id in classes
    }
    finite = [value for value in per_class.values() if np.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0, per_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_test_sealed()
    protocol = load_protocol()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "baseline_rescue_summary.json"
    if not summary_path.is_file():
        subprocess.run([
            str(PROJECT_DIR / ".venv/bin/python"), "baseline_rescue_audit.py",
            "--model", str(args.checkpoint), "--data", str(args.data),
            "--manifest", str(args.manifest), "--output", str(args.output),
            "--splits", "val", "--imgsz", str(args.imgsz),
            "--small-area", str(protocol["object_sizes"]["small_area_ratio_max"]),
            "--medium-area", str(protocol["object_sizes"]["medium_area_ratio_max"]),
        ], cwd=PROJECT_DIR, check=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("evaluated_splits") != ["val"]:
        raise RuntimeError("Candidate evaluator may use only official validation")
    if summary.get("checkpoint_sha256") != sha256(args.checkpoint):
        raise RuntimeError("Candidate evaluation checkpoint hash mismatch")
    ground_truth, predictions = detection_dicts(args.output / "detections_val.csv")
    manifest = pd.read_csv(args.manifest)
    validation = manifest[manifest["split"] == "val"].copy()
    scene_rows = pd.read_csv(args.output / "clean_metrics_per_scene.csv")
    ap_rows = []
    for scene, rows in validation.groupby("grouped_scene_id"):
        paths = set(rows["output_image"].astype(str))
        map50, per_class = scoped_map50(ground_truth, predictions, paths)
        ap_rows.append({
            "grouped_scene_id": str(scene), "mAP50": map50,
            **{f"AP50_class_{class_id}": value for class_id, value in per_class.items()},
        })
    scene_ap = pd.DataFrame(ap_rows)
    scene_combined = scene_rows.merge(scene_ap, on="grouped_scene_id", how="left")
    scene_combined.to_csv(args.output / "metrics_per_scene_with_ap.csv", index=False)

    metrics = pd.read_csv(args.output / "clean_metrics_train_val_test.csv")
    standard = metrics[
        (metrics["split"] == "val") & (metrics["operating_point"] == "standard")
    ].iloc[0]
    safety = metrics[
        (metrics["split"] == "val") & (metrics["operating_point"] == "safety")
    ].iloc[0]
    safety_scenes = scene_combined[scene_combined["operating_point"] == "safety"]
    details = pd.read_csv(args.output / "clean_metrics_by_class_and_size.csv")
    small = details[
        (details["split"] == "val")
        & (details["operating_point"] == "safety")
        & (details["scope"] == "size")
        & (details["name"] == "small")
    ]
    gate = protocol["quality_gate"]
    result = {
        "status": "PASS",
        "candidate": args.candidate,
        "seed": args.seed,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256(args.data),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "imgsz": args.imgsz,
        "validation_mAP50": float(standard["mAP50"]),
        "validation_mAP50_95": float(standard["mAP50-95"]),
        "validation_standard_recall": float(standard["recall"]),
        "validation_standard_f1": float(standard["f1"]),
        "validation_safety_recall": float(safety["recall"]),
        "validation_safety_f2": float(safety["f2"]),
        "validation_safety_fn_per_frame": float(safety["fn_per_frame"]),
        "validation_small_object_safety_recall": (
            float(small.iloc[0]["recall"]) if len(small) else None
        ),
        "scene_macro_map50": float(scene_ap["mAP50"].mean()),
        "scene_map50_std": float(scene_ap["mAP50"].std(ddof=0)),
        "scene_macro_safety_recall": float(safety_scenes["recall"].mean()),
        "scene_safety_recall_std": float(safety_scenes["recall"].std(ddof=0)),
        "minimum_scene_safety_recall": float(safety_scenes["recall"].min()),
        "map50_passed": float(standard["mAP50"]) >= float(gate["validation_map50_min"]),
        "safety_recall_passed": (
            float(safety["recall"]) >= float(gate["validation_safety_recall_min"])
        ),
        "scene_floor_passed": (
            float(safety_scenes["recall"].min()) >= float(gate["validation_scene_recall_min"])
        ),
        "test_evaluated": False,
    }
    result["quality_gate_passed"] = bool(
        result["map50_passed"] and result["safety_recall_passed"]
        and result["scene_floor_passed"]
    )
    atomic_json(args.output / "candidate_result.json", result)
    completed_marker(
        args.output,
        inputs=[args.checkpoint, args.data, args.manifest],
        outputs=[
            args.output / "candidate_result.json",
            args.output / "threshold_selection.json",
            args.output / "clean_metrics_train_val_test.csv",
            args.output / "metrics_per_scene_with_ap.csv",
            args.output / "detections_val.csv",
        ],
        extra={
            "stage": "candidate_validation_evaluation",
            "candidate": args.candidate, "seed": args.seed,
            "test_evaluated": False,
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
