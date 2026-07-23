from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.audit_evaluator import average_precision
from scripts.person_v3.evaluate import (
    atomic_csv,
    detections_frame,
    infer_all,
    person_gt,
    reparse,
    selected_sources,
)
from scripts.person_v4.common import (
    assert_locked,
    atomic_json,
    load_protocol,
    now,
    sha256,
)
from scripts.person_v4.train import training_root


def evaluate(variant: str, fold: int) -> dict[str, Any]:
    assert_locked()
    source = selected_sources(fold)
    root = training_root(variant, fold)
    completion = json.loads(
        (root / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
    )
    checkpoint = Path(completion["final_checkpoint"])
    destination = root / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    gt = person_gt(source)
    predictions, runtime = infer_all(
        fold, checkpoint, source, destination
    )
    frame_index = source[
        ["grouped_scene_id", "subsequence_id", "output_image"]
    ].copy()
    frame_index["image_path"] = frame_index["output_image"].map(
        lambda value: str(Path(value).resolve())
    )
    frame_index.drop(columns=["output_image"]).to_csv(
        destination / "frame_index.csv", index=False
    )
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
        average_precision(
            reparsed_gt, reparsed_predictions, 0, threshold
        )
        for threshold in np.arange(0.50, 0.96, 0.05)
    ]
    tolerance = float(
        load_protocol()["evaluation"]["independent_evaluator_tolerance"]
    )
    differences = {
        "mAP50": abs(map50 - repeated_map50),
        "mAP50_95": abs(map5095 - float(np.mean(repeated_aps))),
    }
    consistency = {
        "status": "PASS"
        if max(differences.values()) <= tolerance
        else "FAIL",
        "primary": "global_person_evaluator_in_memory",
        "independent": "global_person_evaluator_csv_reparse",
        "tolerance": tolerance,
        "absolute_differences": differences,
    }
    atomic_json(destination / "evaluator_consistency.json", consistency)
    if consistency["status"] != "PASS":
        raise RuntimeError("Person v4 evaluator consistency failed")
    scene_rows = []
    for scene, frame in source.groupby("grouped_scene_id"):
        images = {str(Path(value).resolve()) for value in frame["output_image"]}
        scene_gt = {image: gt[image] for image in images}
        scene_predictions = {
            image: predictions.get(image, []) for image in images
        }
        scene_rows.append(
            {
                "variant": variant,
                "fold": fold,
                "grouped_scene_id": str(scene),
                "frames": len(images),
                "person_GT": sum(len(values) for values in scene_gt.values()),
                "mAP50": average_precision(
                    scene_gt, scene_predictions, 0, 0.50
                ),
            }
        )
    atomic_csv(
        pd.DataFrame(scene_rows), destination / "per_scene_AP.csv"
    )
    expected_scenes = set(source["grouped_scene_id"].astype(str))
    indexed_scenes = set(frame_index["grouped_scene_id"].astype(str))
    result = {
        "status": "PASS",
        "finished_at": now(),
        "protocol_id": load_protocol()["protocol_id"],
        "variant": variant,
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
        "lost_GT": int(
            sum(len(values) for values in gt.values())
            - detections["kind"].eq("ground_truth").sum()
        ),
        "missing_scenes": int(len(expected_scenes - indexed_scenes)),
        "no_nan_inf": bool(np.isfinite([map50, map5095]).all()),
        "test_used": False,
    }
    atomic_json(destination / "evaluation_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A1", "A2", "A3"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.variant, args.fold), indent=2))


if __name__ == "__main__":
    main()
