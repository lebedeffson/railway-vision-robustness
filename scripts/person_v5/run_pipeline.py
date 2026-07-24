from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scripts.audit_evaluator import match_dataset
from scripts.person_v3.evaluate import reparse
from scripts.person_v5.evaluate import size_recall
from scripts.person_v5.common import (
    OUTPUT,
    PROJECT,
    TEST_MARKER,
    assert_runtime_locked,
    atomic_json,
    now,
)


PYTHON = PROJECT / ".venv/bin/python"
STATUS = OUTPUT / "pipeline_status.json"
RUNTIME = PROJECT / "configs/canonical_v5_person_data_first_runtime.yaml"


def status(stage: str, state: str, **extra: Any) -> None:
    payload = (
        json.loads(STATUS.read_text(encoding="utf-8"))
        if STATUS.is_file()
        else {
            "protocol_id": "canonical-v5-person-data-first-v1",
            "test_opened": False,
            "stages": {},
        }
    )
    payload["updated_at"] = now()
    payload["current_stage"] = stage
    payload["stages"][stage] = {"status": state, **extra}
    atomic_json(STATUS, payload)


def run(script: str, *arguments: str) -> None:
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            f"scripts/person_v5/{script}",
            *arguments,
        ],
        cwd=PROJECT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(PROJECT), str(PROJECT / "scripts")]
            ),
        },
        check=True,
    )


def external_checkpoint() -> Path:
    marker = OUTPUT / "external_pretraining/run/TRAINING_COMPLETE.json"
    if not marker.is_file():
        status("external_pretraining", "running", started_at=now())
        run("train.py", "--stage", "external")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    status(
        "external_pretraining",
        "success",
        finished_at=now(),
        checkpoint_sha256=payload["best_sha256"],
    )
    return Path(payload["best"])


def fold(fold_id: int) -> None:
    result_path = (
        OUTPUT
        / f"railway/fold_{fold_id}/evaluation/evaluation_result.json"
    )
    if not result_path.is_file():
        status(f"D1_fold{fold_id}", "running", started_at=now())
        run("train.py", "--stage", "fold", "--fold", str(fold_id))
        run("evaluate.py", "--fold", str(fold_id))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    status(
        f"D1_fold{fold_id}",
        "success",
        finished_at=now(),
        checkpoint_sha256=result["checkpoint_sha256"],
        mAP50=result["mAP50"],
        recall=result["recall"],
        small_recall=result["small_recall"],
    )


def pooled_threshold(frame: pd.DataFrame) -> pd.Series:
    config = yaml.safe_load(RUNTIME.read_text(encoding="utf-8"))[
        "evaluation"
    ]
    gt, predictions = reparse(frame)
    rows = []
    for threshold in np.arange(
        float(config["confidence_min"]),
        float(config["confidence_max"])
        + float(config["confidence_step"]) / 2,
        float(config["confidence_step"]),
    ):
        rows.append(
            {
                "threshold": round(float(threshold), 6),
                **match_dataset(gt, predictions, float(threshold), 0.50),
            }
        )
    sweep = pd.DataFrame(rows)
    destination = OUTPUT / "triage"
    destination.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(destination / "threshold_sweep.csv", index=False)
    return sweep.sort_values(
        ["f1", "recall", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]


def triage() -> dict[str, Any]:
    frames = []
    results = {}
    for fold_id in (0, 1):
        root = OUTPUT / f"railway/fold_{fold_id}/evaluation"
        frame = pd.read_csv(root / "predictions_and_ground_truth.csv")
        frame["fold"] = fold_id
        frames.append(frame)
        results[fold_id] = json.loads(
            (root / "evaluation_result.json").read_text(encoding="utf-8")
        )
    merged = pd.concat(frames, ignore_index=True)
    selected = pooled_threshold(merged)
    rows = []
    for fold_id, frame in zip((0, 1), frames):
        gt, predictions = reparse(frame)
        metrics = match_dataset(
            gt, predictions, float(selected["threshold"]), 0.50
        )
        rows.append(
            {
                "fold": fold_id,
                "mAP50": results[fold_id]["mAP50"],
                **metrics,
                **size_recall(
                    gt,
                    predictions,
                    float(selected["threshold"]),
                ),
                "evaluator_consistency": results[fold_id][
                    "evaluator_consistency"
                ],
                "lost_GT": results[fold_id]["lost_GT"],
                "missing_scenes": results[fold_id]["missing_scenes"],
                "NaN_Inf": results[fold_id]["NaN_Inf"],
            }
        )
    metrics = pd.DataFrame(rows)
    baseline = pd.read_csv(
        PROJECT / "outputs/person_v3/triage/fold_metrics.csv"
    )
    merged_comparison = metrics.merge(
        baseline[["fold", "mAP50", "recall"]],
        on="fold",
        suffixes=("_D1", "_D0"),
        validate="one_to_one",
    )
    improved = int(
        (
            merged_comparison["mAP50_D1"].gt(
                merged_comparison["mAP50_D0"]
            )
            & merged_comparison["recall_D1"].gt(
                merged_comparison["recall_D0"]
            )
        ).sum()
    )
    protocol = yaml.safe_load(
        (
            PROJECT / "configs/canonical_v5_person_data_first.yaml"
        ).read_text(encoding="utf-8")
    )
    gate = protocol["two_fold_gate"]["require_all"]
    checks = {
        "macro_mAP50": float(metrics["mAP50"].mean()),
        "macro_recall": float(metrics["recall"].mean()),
        "macro_small_recall": float(metrics["small_recall"].mean()),
        "worst_fold_recall": float(metrics["recall"].min()),
        "folds_improved_over_D0": improved,
        "evaluator_consistency": (
            "PASS"
            if metrics["evaluator_consistency"].eq("PASS").all()
            else "FAIL"
        ),
        "lost_GT": int(metrics["lost_GT"].sum()),
        "missing_scenes": int(metrics["missing_scenes"].sum()),
        "NaN_Inf": int(metrics["NaN_Inf"].sum()),
    }
    passed = (
        checks["macro_mAP50"] >= gate["macro_mAP50_min"]
        and checks["macro_recall"] >= gate["macro_recall_min"]
        and checks["macro_small_recall"]
        >= gate["macro_small_recall_min"]
        and checks["worst_fold_recall"]
        >= gate["worst_fold_recall_min"]
        and checks["folds_improved_over_D0"]
        >= gate["folds_improved_over_D0_min"]
        and checks["evaluator_consistency"]
        == gate["evaluator_consistency"]
        and checks["lost_GT"] == gate["lost_GT"]
        and checks["missing_scenes"] == gate["missing_scenes"]
        and checks["NaN_Inf"] == gate["NaN_Inf"]
        and not TEST_MARKER.exists()
    )
    destination = OUTPUT / "triage"
    metrics.to_csv(destination / "fold_metrics.csv", index=False)
    merged_comparison.to_csv(
        destination / "D1_vs_D0_per_fold.csv", index=False
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "protocol_id": protocol["protocol_id"],
        "threshold": float(selected["threshold"]),
        "checks": checks,
        "test_opened": False,
        "attacks_allowed": False,
        "next_action": (
            "run_folds_2_4_unchanged"
            if passed
            else "stop_data_first_v1_and_keep_test_sealed"
        ),
    }
    atomic_json(destination / "two_fold_gate.json", result)
    return result


def main() -> None:
    assert_runtime_locked()
    if TEST_MARKER.exists():
        raise RuntimeError("Railway test marker exists")
    external_checkpoint()
    for fold_id in (0, 1):
        fold(fold_id)
    result = triage()
    status(
        "two_fold_gate",
        "success" if result["status"] == "PASS" else "failed",
        finished_at=now(),
        result=result["status"],
        test_opened=False,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
