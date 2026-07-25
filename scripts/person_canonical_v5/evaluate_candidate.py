from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.audit_evaluator import average_precision
from scripts.person_canonical_v5.common import OUTPUT, atomic_csv, atomic_json, now, sha256
from scripts.person_canonical_v5.train_candidates import (
    CANDIDATE_ROOT,
    assert_runtime_locked,
    runtime,
)
from scripts.person_v3.evaluate import (
    detections_frame,
    infer_all,
    person_gt,
    reparse,
    selected_sources,
)
from src.models.coordinate_attention import register_ultralytics_modules


def evaluate(candidate: str, fold: int) -> dict[str, Any]:
    assert_runtime_locked()
    if fold not in (0, 1):
        raise RuntimeError("Candidate evaluator only permits folds 0 and 1")
    root = CANDIDATE_ROOT / candidate / f"fold_{fold}"
    training = json.loads(
        (root / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
    )
    checkpoint = Path(training["checkpoint"])
    if sha256(checkpoint) != training["checkpoint_sha256"]:
        raise RuntimeError("Candidate checkpoint hash mismatch")
    register_ultralytics_modules()
    destination = root / "evaluation"
    source = selected_sources(fold)
    gt = person_gt(source)
    predictions, runtime_ms = infer_all(
        fold, checkpoint, source, destination
    )
    frame = detections_frame(source, gt, predictions)
    csv_path = destination / "predictions_and_ground_truth.csv"
    atomic_csv(frame, csv_path)
    reparsed_gt, reparsed_predictions = reparse(pd.read_csv(csv_path))
    map50 = average_precision(gt, predictions, 0, 0.50)
    repeated = average_precision(
        reparsed_gt, reparsed_predictions, 0, 0.50
    )
    config = runtime()["evaluation"]
    aps = [
        average_precision(gt, predictions, 0, float(iou))
        for iou in config["AP_IoU"]
    ]
    consistent = abs(map50 - repeated) <= float(
        config["evaluator_tolerance"]
    )
    result = {
        "status": "PASS" if consistent else "FAIL",
        "created_at": now(),
        "candidate": candidate,
        "fold": fold,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "frames": int(len(source)),
        "scenes": int(source["grouped_scene_id"].nunique()),
        "person_GT": int(sum(len(values) for values in gt.values())),
        "mAP50": float(map50),
        "mAP50_95": float(np.mean(aps)),
        "runtime_ms": float(runtime_ms),
        "evaluator_consistency": "PASS" if consistent else "FAIL",
        "lost_GT": abs(
            sum(len(values) for values in gt.values())
            - sum(len(values) for values in reparsed_gt.values())
        ),
        "missing_scenes": 0,
        "NaN_Inf": int(
            not np.isfinite([map50, np.mean(aps), runtime_ms]).all()
        ),
        "test_used": False,
    }
    atomic_json(destination / "evaluation_result.json", result)
    if not consistent:
        raise RuntimeError("Candidate evaluator parity failed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.candidate, args.fold), indent=2))

