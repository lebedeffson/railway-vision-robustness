from __future__ import annotations

import json
import math
import os
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.audit_evaluator import average_precision, match_dataset
from scripts.person_v3.common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    TEST_MARKER,
    assert_locked,
    atomic_json,
    load_protocol,
    now,
    sha256,
)
from scripts.person_v3.evaluate import reparse
from scripts.person_v3.train import training_root


PYTHON = PROJECT_DIR / ".venv/bin/python"
STATUS = OUTPUT_ROOT / "pipeline_status.json"


def status(stage: str, value: str, **extra: Any) -> None:
    payload = (
        json.loads(STATUS.read_text(encoding="utf-8"))
        if STATUS.is_file()
        else {"protocol_id": load_protocol()["protocol_id"], "stages": {}}
    )
    payload["updated_at"] = now()
    payload["current_stage"] = stage
    payload["stages"][stage] = {"status": value, **extra}
    atomic_json(STATUS, payload)


def run(script: str, *arguments: str) -> None:
    subprocess.run(
        [str(PYTHON), "-u", f"scripts/person_v3/{script}", *arguments],
        cwd=PROJECT_DIR,
        env={**os.environ, "PYTHONPATH": str(PROJECT_DIR)},
        check=True,
    )


def run_fold(fold: int) -> dict[str, Any]:
    stage = f"fold_{fold}"
    result_path = training_root(fold) / "evaluation/evaluation_result.json"
    if not result_path.is_file():
        status(stage, "running", started_at=now())
        run("train.py", "--fold", str(fold))
        run("evaluate.py", "--fold", str(fold))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["status"] != "PASS":
        raise RuntimeError(f"Person v3 fold {fold} failed")
    status(
        stage,
        "success",
        finished_at=now(),
        checkpoint_sha256=result["checkpoint_sha256"],
        mAP50=result["mAP50"],
    )
    return result


def combined(
    folds: list[int],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[int, tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]],
]:
    frames = []
    by_fold = {}
    for fold in folds:
        path = (
            training_root(fold)
            / "evaluation/predictions_and_ground_truth.csv"
        )
        frame = pd.read_csv(path)
        frame["fold"] = fold
        frames.append(frame)
        by_fold[fold] = reparse(frame)
    merged = pd.concat(frames, ignore_index=True)
    gt, predictions = reparse(merged)
    return gt, predictions, by_fold


def threshold_sweep(
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    protocol = load_protocol()["evaluation"]
    values = np.arange(
        float(protocol["confidence_min"]),
        float(protocol["confidence_max"])
        + float(protocol["confidence_step"]) / 2,
        float(protocol["confidence_step"]),
    )
    rows = []
    for threshold in values:
        metrics = match_dataset(gt, predictions, float(threshold), 0.50)
        rows.append({"threshold": round(float(threshold), 6), **metrics})
    return pd.DataFrame(rows)


def size_recall(
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> dict[str, float]:
    # Source images are fixed at 4112x2504.
    counts = {
        "small": {"gt": 0, "tp": 0},
        "medium": {"gt": 0, "tp": 0},
        "large": {"gt": 0, "tp": 0},
    }
    from scripts.person_v3.evaluate import matched_indices

    for image, targets in gt.items():
        matched = matched_indices(targets, predictions.get(image, []), threshold)
        for index, target in enumerate(targets):
            x1, y1, x2, y2 = target["box"]
            ratio = (x2 - x1) * (y2 - y1) / (4112 * 2504)
            name = "small" if ratio < 0.001 else (
                "medium" if ratio < 0.01 else "large"
            )
            counts[name]["gt"] += 1
            counts[name]["tp"] += int(index in matched)
    return {
        f"{name}_recall": values["tp"] / max(values["gt"], 1)
        for name, values in counts.items()
    }


def fold_rows(
    folds: list[int],
    by_fold: dict[int, tuple[dict, dict]],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for fold in folds:
        gt, predictions = by_fold[fold]
        metrics = match_dataset(gt, predictions, threshold, 0.50)
        evaluation = json.loads(
            (
                training_root(fold)
                / "evaluation/evaluation_result.json"
            ).read_text(encoding="utf-8")
        )
        rows.append({
            "fold": fold,
            "mAP50": evaluation["mAP50"],
            "mAP50_95": evaluation["mAP50_95"],
            **metrics,
            **size_recall(gt, predictions, threshold),
            "evaluator_consistency": evaluation["evaluator_consistency"],
            "lost_GT": evaluation["lost_GT"],
            "missing_scenes": evaluation["missing_scenes"],
        })
    return pd.DataFrame(rows)


def triage() -> dict[str, Any]:
    folds = [int(value) for value in load_protocol()["triage"]["folds"]]
    gt, predictions, by_fold = combined(folds)
    sweep = threshold_sweep(gt, predictions)
    selected = sweep.sort_values(
        ["f1", "recall", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    rows = fold_rows(folds, by_fold, float(selected["threshold"]))
    destination = OUTPUT_ROOT / "triage"
    destination.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(destination / "threshold_sweep.csv", index=False)
    rows.to_csv(destination / "fold_metrics.csv", index=False)
    macro_map = float(rows["mAP50"].mean())
    macro_recall = float(rows["recall"].mean())
    worst_recall = float(rows["recall"].min())
    macro_small = float(rows["small_recall"].mean())
    config = load_protocol()["triage"]
    technical = bool(
        rows["evaluator_consistency"].eq("PASS").all()
        and rows["lost_GT"].eq(0).all()
        and rows["missing_scenes"].eq(0).all()
        and np.isfinite(
            rows[["mAP50", "recall", "f1", "small_recall"]].to_numpy()
        ).all()
        and not TEST_MARKER.exists()
    )
    hard = (
        macro_map < float(config["hard_fail"]["macro_mAP50_below"])
        or macro_recall < float(config["hard_fail"]["macro_recall_below"])
        or worst_recall < float(config["hard_fail"]["any_fold_recall_below"])
        or not technical
    )
    promising = (
        macro_map >= float(config["promising"]["macro_mAP50_min"])
        and macro_recall >= float(config["promising"]["macro_recall_min"])
        and worst_recall >= float(config["promising"]["every_fold_recall_min"])
        and technical
    )
    borderline = (
        macro_map >= float(config["borderline"]["macro_mAP50_min"])
        and macro_recall >= float(config["borderline"]["macro_recall_min"])
        and worst_recall >= float(
            config["borderline"]["every_fold_recall_min"]
        )
        and macro_small >= float(
            config["borderline"]["macro_small_recall_min"]
        )
        and technical
    )
    triage_status = (
        "hard_fail" if hard
        else ("promising" if promising else ("borderline_continue" if borderline else "hard_fail"))
    )
    result = {
        "status": triage_status,
        "role": "development_two_fold_go_no_go",
        "created_at": now(),
        "folds": folds,
        "pooled_threshold": float(selected["threshold"]),
        "macro_mAP50": macro_map,
        "macro_recall": macro_recall,
        "macro_F1": float(rows["f1"].mean()),
        "macro_small_recall": macro_small,
        "worst_fold_recall": worst_recall,
        "technical_checks_passed": technical,
        "test_opened": False,
        "continue_remaining_folds": triage_status in {
            "promising", "borderline_continue"
        },
    }
    atomic_json(destination / "triage_gate.json", result)
    return result


def oof_gate() -> dict[str, Any]:
    folds = list(range(5))
    gt, predictions, by_fold = combined(folds)
    sweep = threshold_sweep(gt, predictions)
    standard = sweep.sort_values(
        ["f1", "recall", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    safety_candidates = sweep[
        sweep["recall"].ge(0.50) & sweep["precision"].ge(0.30)
    ]
    safety = (
        safety_candidates.sort_values("threshold", ascending=False).iloc[0]
        if not safety_candidates.empty else None
    )
    destination = OUTPUT_ROOT / "OOF"
    destination.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(destination / "threshold_sweep.csv", index=False)
    map50 = average_precision(gt, predictions, 0, 0.50)
    aps = [
        average_precision(gt, predictions, 0, threshold)
        for threshold in np.arange(0.50, 0.96, 0.05)
    ]
    if safety is not None:
        rows = fold_rows(folds, by_fold, float(safety["threshold"]))
    else:
        rows = fold_rows(folds, by_fold, float(standard["threshold"]))
    rows.to_csv(destination / "fold_metrics.csv", index=False)
    config = load_protocol()["OOF_gate"]
    passed = bool(
        safety is not None
        and map50 >= float(config["mAP50_min"])
        and float(safety["recall"]) >= float(config["safety_recall_min"])
        and float(standard["f1"]) >= float(config["standard_F1_min"])
        and float(rows["recall"].min()) >= float(config["worst_fold_recall_min"])
        and rows["evaluator_consistency"].eq("PASS").all()
        and rows["lost_GT"].eq(0).all()
        and rows["missing_scenes"].eq(0).all()
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "created_at": now(),
        "mAP50": float(map50),
        "mAP50_95": float(np.mean(aps)),
        "standard_threshold": float(standard["threshold"]),
        "standard_precision": float(standard["precision"]),
        "standard_recall": float(standard["recall"]),
        "standard_F1": float(standard["f1"]),
        "safety_threshold": (
            None if safety is None else float(safety["threshold"])
        ),
        "safety_precision": (
            None if safety is None else float(safety["precision"])
        ),
        "safety_recall": (
            None if safety is None else float(safety["recall"])
        ),
        "worst_fold_recall": float(rows["recall"].min()),
        "OOF_gate_passed": passed,
        "checkpoint_training_allowed": passed,
        "test_opened": False,
    }
    atomic_json(destination / "OOF_gate.json", result)
    return result


def failure_bundle(stage: str) -> Path:
    destination = OUTPUT_ROOT / f"bundles/TNormFilter_person_v3_{stage}_failed.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        roots = [
            PROJECT_DIR / "configs/canonical_v3_person_safety.yaml",
            OUTPUT_ROOT / "protocol",
            OUTPUT_ROOT / "audit",
            OUTPUT_ROOT / "triage",
            OUTPUT_ROOT / "OOF",
            OUTPUT_ROOT / "pipeline_status.json",
        ]
        for root in roots:
            if root.is_file():
                archive.write(root, root.relative_to(PROJECT_DIR))
            elif root.is_dir():
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(PROJECT_DIR))
    return destination


def main() -> None:
    assert_locked()
    if TEST_MARKER.exists():
        raise RuntimeError("Person v3 pre-test pipeline found test marker")
    for fold in (0, 1):
        run_fold(fold)
    gate = triage()
    status("two_fold_triage", gate["status"], finished_at=now(), **gate)
    if not gate["continue_remaining_folds"]:
        bundle = failure_bundle("triage")
        status(
            "downstream",
            "skipped_by_triage",
            bundle=str(bundle.resolve()),
            bundle_sha256=sha256(bundle),
        )
        return
    for fold in (2, 3, 4):
        run_fold(fold)
    gate = oof_gate()
    status("OOF_gate", gate["status"], finished_at=now(), **gate)
    if not gate["OOF_gate_passed"]:
        bundle = failure_bundle("OOF")
        status(
            "downstream",
            "skipped_by_OOF_gate",
            bundle=str(bundle.resolve()),
            bundle_sha256=sha256(bundle),
        )
        return
    status(
        "final_training",
        "ready",
        note="OOF PASS; final-training runner must use frozen median epoch",
        test_opened=False,
    )


if __name__ == "__main__":
    main()

