from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scripts.audit_evaluator import average_precision, match_dataset
from scripts.person_v3.evaluate import (
    detections_frame,
    infer_all,
    matched_indices,
    person_gt,
    reparse,
    selected_sources,
)
from scripts.person_v5.common import (
    OUTPUT,
    PROJECT,
    assert_runtime_locked,
    atomic_json,
    now,
    sha256,
)


RUNTIME = PROJECT / "configs/canonical_v5_person_data_first_runtime.yaml"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def threshold_sweep(
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    config = yaml.safe_load(RUNTIME.read_text(encoding="utf-8"))[
        "evaluation"
    ]
    values = np.arange(
        float(config["confidence_min"]),
        float(config["confidence_max"])
        + float(config["confidence_step"]) / 2,
        float(config["confidence_step"]),
    )
    return pd.DataFrame(
        [
            {
                "threshold": round(float(threshold), 6),
                **match_dataset(gt, predictions, float(threshold), 0.50),
            }
            for threshold in values
        ]
    )


def size_recall(
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> dict[str, float]:
    counts = {
        "small": {"gt": 0, "tp": 0},
        "medium": {"gt": 0, "tp": 0},
        "large": {"gt": 0, "tp": 0},
    }
    for image, targets in gt.items():
        matched = matched_indices(
            targets, predictions.get(image, []), threshold
        )
        for index, target in enumerate(targets):
            x1, y1, x2, y2 = target["box"]
            area = (x2 - x1) * (y2 - y1) / (4112 * 2504)
            bucket = (
                "small"
                if area < 0.001
                else ("medium" if area < 0.01 else "large")
            )
            counts[bucket]["gt"] += 1
            counts[bucket]["tp"] += int(index in matched)
    return {
        f"{bucket}_recall": values["tp"] / max(values["gt"], 1)
        for bucket, values in counts.items()
    }


def evaluate(fold: int) -> dict[str, Any]:
    assert_runtime_locked()
    if fold not in (0, 1):
        raise RuntimeError("V5 triage evaluator only permits folds 0 and 1")
    root = OUTPUT / f"railway/fold_{fold}"
    training = json.loads(
        (root / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
    )
    checkpoint = Path(training["final_checkpoint"])
    destination = root / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    source = selected_sources(fold)
    gt = person_gt(source)
    predictions, runtime_ms = infer_all(
        fold, checkpoint, source, destination
    )
    frame = detections_frame(source, gt, predictions)
    csv_path = destination / "predictions_and_ground_truth.csv"
    atomic_csv(frame, csv_path)
    reparsed_gt, reparsed_predictions = reparse(
        pd.read_csv(csv_path)
    )
    map50 = average_precision(gt, predictions, 0, 0.50)
    map50_reparsed = average_precision(
        reparsed_gt, reparsed_predictions, 0, 0.50
    )
    runtime_config = yaml.safe_load(RUNTIME.read_text(encoding="utf-8"))
    aps = [
        average_precision(gt, predictions, 0, float(threshold))
        for threshold in runtime_config["evaluation"]["AP_IoU"]
    ]
    sweep = threshold_sweep(gt, predictions)
    selected = sweep.sort_values(
        ["f1", "recall", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    threshold = float(selected["threshold"])
    metrics = match_dataset(gt, predictions, threshold, 0.50)
    reparsed_metrics = match_dataset(
        reparsed_gt, reparsed_predictions, threshold, 0.50
    )
    tolerance = float(
        runtime_config["evaluation"]["evaluator_tolerance"]
    )
    consistent = (
        abs(map50 - map50_reparsed) <= tolerance
        and all(
            abs(float(metrics[key]) - float(reparsed_metrics[key]))
            <= tolerance
            for key in ("precision", "recall", "f1")
        )
    )
    scenes = sorted(source["grouped_scene_id"].astype(str).unique())
    per_scene = []
    path_to_scene = {
        str(Path(row.output_image).resolve()): str(row.grouped_scene_id)
        for row in source.itertuples(index=False)
    }
    for scene in scenes:
        scene_images = {
            path for path, value in path_to_scene.items() if value == scene
        }
        scene_gt = {path: gt[path] for path in scene_images}
        scene_predictions = {
            path: predictions.get(path, []) for path in scene_images
        }
        per_scene.append(
            {
                "fold": fold,
                "grouped_scene_id": scene,
                "frames": len(scene_images),
                "mAP50": average_precision(
                    scene_gt, scene_predictions, 0, 0.50
                ),
                **match_dataset(
                    scene_gt, scene_predictions, threshold, 0.50
                ),
            }
        )
    atomic_csv(sweep, destination / "threshold_sweep.csv")
    atomic_csv(pd.DataFrame(per_scene), destination / "per_scene.csv")
    result = {
        "status": "PASS" if consistent else "FAIL",
        "fold": fold,
        "finished_at": now(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "threshold": threshold,
        "mAP50": map50,
        "mAP50_95": float(np.mean(aps)),
        **metrics,
        **size_recall(gt, predictions, threshold),
        "evaluator_consistency": "PASS" if consistent else "FAIL",
        "lost_GT": abs(
            sum(len(values) for values in gt.values())
            - sum(len(values) for values in reparsed_gt.values())
        ),
        "missing_scenes": 0,
        "NaN_Inf": int(
            not np.isfinite(
                [
                    map50,
                    np.mean(aps),
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                ]
            ).all()
        ),
        "runtime_ms": runtime_ms,
        "test_used": False,
    }
    atomic_json(destination / "evaluation_result.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"Canonical v5 fold {fold} evaluation failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.fold), indent=2))


if __name__ == "__main__":
    main()

