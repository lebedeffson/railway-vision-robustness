from __future__ import annotations

import json
import os
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.audit_evaluator import average_precision, box_iou, match_dataset
from scripts.person_v3.evaluate import reparse, selected_sources
from scripts.person_v3.run_pipeline import size_recall
from scripts.person_v4.common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    assert_locked,
    atomic_json,
    load_protocol,
    now,
    sha256,
)
from scripts.person_v4.train import training_root


PYTHON = PROJECT_DIR / ".venv/bin/python"
STATUS = OUTPUT_ROOT / "pipeline_status.json"
VARIANTS = ("A1", "A2", "A3")


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
    pythonpath = os.pathsep.join(
        [str(PROJECT_DIR), str(PROJECT_DIR / "scripts")]
    )
    subprocess.run(
        [str(PYTHON), "-u", f"scripts/person_v4/{script}", *arguments],
        cwd=PROJECT_DIR,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=True,
    )


def run_fold(variant: str, fold: int) -> dict[str, Any]:
    stage = f"{variant}_fold_{fold}"
    root = training_root(variant, fold)
    result_path = root / "evaluation/evaluation_result.json"
    if not result_path.is_file():
        status(stage, "running", started_at=now())
        run("train.py", "--variant", variant, "--fold", str(fold))
        run("evaluate.py", "--variant", variant, "--fold", str(fold))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["status"] != "PASS":
        raise RuntimeError(f"{variant} fold {fold} failed")
    status(
        stage,
        "success",
        finished_at=now(),
        checkpoint_sha256=result["checkpoint_sha256"],
        mAP50=result["mAP50"],
    )
    return result


def detection_path(variant: str, fold: int) -> Path:
    if variant == "A0":
        return (
            PROJECT_DIR
            / f"outputs/person_v3/scene_cv/fold_{fold}/seed_20260723/"
            "evaluation/predictions_and_ground_truth.csv"
        )
    return (
        training_root(variant, fold)
        / "evaluation/predictions_and_ground_truth.csv"
    )


def evaluation_path(variant: str, fold: int) -> Path:
    if variant == "A0":
        return (
            PROJECT_DIR
            / f"outputs/person_v3/scene_cv/fold_{fold}/seed_20260723/"
            "evaluation/evaluation_result.json"
        )
    return training_root(variant, fold) / "evaluation/evaluation_result.json"


def combined(
    variant: str, folds: list[int]
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[int, tuple[dict, dict]],
    pd.DataFrame,
    pd.DataFrame,
]:
    frames = []
    indexes = []
    by_fold = {}
    for fold in folds:
        frame = pd.read_csv(detection_path(variant, fold))
        frame["fold"] = fold
        frames.append(frame)
        by_fold[fold] = reparse(frame)
        if variant == "A0":
            index = selected_sources(fold)[
                ["grouped_scene_id", "subsequence_id", "output_image"]
            ].copy()
            index["image_path"] = index["output_image"].map(
                lambda value: str(Path(value).resolve())
            )
            index.drop(columns=["output_image"], inplace=True)
        else:
            index = pd.read_csv(
                training_root(variant, fold)
                / "evaluation/frame_index.csv"
            )
        index["fold"] = fold
        indexes.append(index)
    merged = pd.concat(frames, ignore_index=True)
    gt, predictions = reparse(merged)
    return gt, predictions, by_fold, merged, pd.concat(
        indexes, ignore_index=True
    )


def threshold_sweep(
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> pd.DataFrame:
    config = load_protocol()["evaluation"]
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


def detection_calibration(
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
    bins: int,
) -> dict[str, float]:
    confidences = []
    correctness = []
    localization = []
    for image, rows in predictions.items():
        targets = gt.get(image, [])
        used: set[int] = set()
        for prediction in sorted(
            [
                value
                for value in rows
                if float(value["confidence"]) >= threshold
            ],
            key=lambda value: -float(value["confidence"]),
        ):
            overlaps = [
                box_iou(prediction["box"], target["box"])
                if index not in used else -1.0
                for index, target in enumerate(targets)
            ]
            best = int(np.argmax(overlaps)) if overlaps else -1
            best_iou = float(overlaps[best]) if best >= 0 else 0.0
            is_correct = best_iou >= 0.50
            if is_correct:
                used.add(best)
            confidences.append(float(prediction["confidence"]))
            correctness.append(float(is_correct))
            localization.append(max(best_iou, 0.0))
    if not confidences:
        return {
            "ECE": 1.0,
            "confidence_quality_spearman": 0.0,
        }
    confidence = np.asarray(confidences)
    correct = np.asarray(correctness)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence < edges[index + 1]
            if index < bins - 1 else confidence <= edges[index + 1]
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(confidence[mask].mean())
                - float(correct[mask].mean())
            )
    confidence_rank = pd.Series(confidence).rank().to_numpy()
    quality_rank = pd.Series(localization).rank().to_numpy()
    correlation = (
        float(np.corrcoef(confidence_rank, quality_rank)[0, 1])
        if np.std(confidence_rank) > 0 and np.std(quality_rank) > 0
        else 0.0
    )
    return {
        "ECE": ece,
        "confidence_quality_spearman": correlation,
    }


def scene_rows(
    merged: pd.DataFrame,
    frame_index: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for scene, index in frame_index.groupby("grouped_scene_id"):
        images = set(index["image_path"].astype(str))
        frame = merged[merged["image_path"].astype(str).isin(images)]
        gt, predictions = reparse(frame)
        rows.append(
            {
                "grouped_scene_id": str(scene),
                "frames": len(images),
                **match_dataset(gt, predictions, threshold, 0.50),
            }
        )
    return pd.DataFrame(rows)


def fold_rows(
    variant: str,
    folds: list[int],
    by_fold: dict[int, tuple[dict, dict]],
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for fold in folds:
        gt, predictions = by_fold[fold]
        evaluation = json.loads(
            evaluation_path(variant, fold).read_text(encoding="utf-8")
        )
        rows.append(
            {
                "variant": variant,
                "fold": fold,
                "mAP50": evaluation["mAP50"],
                "mAP50_95": evaluation["mAP50_95"],
                **match_dataset(gt, predictions, threshold, 0.50),
                **size_recall(gt, predictions, threshold),
                "evaluator_consistency": evaluation[
                    "evaluator_consistency"
                ],
                "lost_GT": evaluation["lost_GT"],
                "missing_scenes": evaluation["missing_scenes"],
            }
        )
    return pd.DataFrame(rows)


def summarize_variant(variant: str, folds: list[int]) -> dict[str, Any]:
    gt, predictions, by_fold, merged, frame_index = combined(variant, folds)
    sweep = threshold_sweep(gt, predictions)
    selected = sweep.sort_values(
        ["f1", "recall", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    threshold = float(selected["threshold"])
    fold_frame = fold_rows(variant, folds, by_fold, threshold)
    scenes = scene_rows(merged, frame_index, threshold)
    destination = OUTPUT_ROOT / f"triage/{variant}"
    destination.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(destination / "threshold_sweep.csv", index=False)
    fold_frame.to_csv(destination / "fold_metrics.csv", index=False)
    scenes.to_csv(destination / "scene_metrics.csv", index=False)
    calibration = detection_calibration(
        gt,
        predictions,
        threshold,
        int(
            load_protocol()["evaluation"]["calibration"][
                "expected_calibration_error_bins"
            ]
        ),
    )
    technical = bool(
        fold_frame["evaluator_consistency"].eq("PASS").all()
        and fold_frame["lost_GT"].eq(0).all()
        and fold_frame["missing_scenes"].eq(0).all()
        and np.isfinite(
            fold_frame[
                ["mAP50", "recall", "f1", "small_recall"]
            ].to_numpy()
        ).all()
    )
    result = {
        "variant": variant,
        "folds": folds,
        "pooled_threshold": threshold,
        "macro_mAP50": float(fold_frame["mAP50"].mean()),
        "macro_recall": float(fold_frame["recall"].mean()),
        "macro_F1": float(fold_frame["f1"].mean()),
        "macro_small_recall": float(
            fold_frame["small_recall"].mean()
        ),
        "worst_fold_recall": float(fold_frame["recall"].min()),
        "pooled_tp": int(selected["tp"]),
        "pooled_fp": int(selected["fp"]),
        "pooled_fn": int(selected["fn"]),
        "scene_recall_std": float(scenes["recall"].std(ddof=0)),
        **calibration,
        "technical_checks_passed": technical,
        "test_opened": False,
    }
    atomic_json(destination / "summary.json", result)
    return result


def choose_candidate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    protocol = load_protocol()
    baseline = next(row for row in summaries if row["variant"] == "A0")
    requirements = protocol["selection"]["require_all"]
    rows = []
    for summary in summaries:
        candidate = dict(summary)
        candidate["delta_macro_mAP50"] = (
            summary["macro_mAP50"] - baseline["macro_mAP50"]
        )
        candidate["delta_macro_recall"] = (
            summary["macro_recall"] - baseline["macro_recall"]
        )
        candidate["delta_macro_small_recall"] = (
            summary["macro_small_recall"]
            - baseline["macro_small_recall"]
        )
        candidate["delta_worst_fold_recall"] = (
            summary["worst_fold_recall"]
            - baseline["worst_fold_recall"]
        )
        candidate["delta_pooled_fp"] = (
            summary["pooled_fp"] - baseline["pooled_fp"]
        )
        candidate["eligible"] = bool(
            summary["variant"] != "A0"
            and summary["technical_checks_passed"]
            and candidate["delta_macro_mAP50"]
            >= float(requirements["absolute_macro_mAP50_gain_min"])
            and candidate["delta_macro_recall"]
            >= float(requirements["absolute_macro_recall_gain_min"])
            and candidate["delta_macro_small_recall"]
            >= float(
                requirements["absolute_macro_small_recall_gain_min"]
            )
            and candidate["delta_worst_fold_recall"] >= 0
            and candidate["delta_pooled_fp"] < 0
        )
        rows.append(candidate)
    frame = pd.DataFrame(rows)
    destination = OUTPUT_ROOT / "triage"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / "variant_comparison.csv", index=False)
    eligible = frame[frame["eligible"]]
    if eligible.empty:
        result = {
            "status": "NO_ELIGIBLE_CANDIDATE",
            "winner": None,
            "remaining_folds_allowed": False,
            "test_opened": False,
            "attacks_allowed": False,
            "hypotheses_H1_H4": "not_tested",
        }
    else:
        winner = eligible.sort_values(
            [
                "worst_fold_recall",
                "macro_small_recall",
                "macro_mAP50",
                "pooled_fp",
            ],
            ascending=[False, False, False, True],
        ).iloc[0]
        result = {
            "status": "ELIGIBLE_CANDIDATE",
            "winner": str(winner["variant"]),
            "remaining_folds_allowed": True,
            "test_opened": False,
            "attacks_allowed": False,
            "hypotheses_H1_H4": "not_tested_before_full_OOF_gate",
        }
    atomic_json(destination / "selection_decision.json", result)
    return result


def oof_gate(variant: str) -> dict[str, Any]:
    folds = list(range(5))
    gt, predictions, by_fold, merged, frame_index = combined(variant, folds)
    sweep = threshold_sweep(gt, predictions)
    standard = sweep.sort_values(
        ["f1", "recall", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    safety_candidates = sweep[
        sweep["recall"].ge(0.50) & sweep["precision"].ge(0.30)
    ]
    safety = (
        safety_candidates.sort_values("threshold", ascending=False).iloc[0]
        if not safety_candidates.empty
        else None
    )
    threshold = float(
        safety["threshold"] if safety is not None else standard["threshold"]
    )
    folds_frame = fold_rows(variant, folds, by_fold, threshold)
    scenes = scene_rows(merged, frame_index, threshold)
    destination = OUTPUT_ROOT / "OOF"
    destination.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(destination / "threshold_sweep.csv", index=False)
    folds_frame.to_csv(destination / "fold_metrics.csv", index=False)
    scenes.to_csv(destination / "scene_metrics.csv", index=False)
    map50 = average_precision(gt, predictions, 0, 0.50)
    aps = [
        average_precision(gt, predictions, 0, value)
        for value in np.arange(0.50, 0.96, 0.05)
    ]
    gate = load_protocol()["continuation"]["eligible_candidate"][
        "OOF_gate"
    ]
    passed = bool(
        safety is not None
        and map50 >= float(gate["mAP50_min"])
        and float(safety["recall"]) >= float(gate["safety_recall_min"])
        and float(standard["f1"]) >= float(gate["F1_min"])
        and float(folds_frame["recall"].min())
        >= float(gate["worst_fold_recall_min"])
        and folds_frame["evaluator_consistency"].eq("PASS").all()
        and folds_frame["lost_GT"].eq(0).all()
        and folds_frame["missing_scenes"].eq(0).all()
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "variant": variant,
        "mAP50": float(map50),
        "mAP50_95": float(np.mean(aps)),
        "standard_threshold": float(standard["threshold"]),
        "standard_F1": float(standard["f1"]),
        "safety_threshold": (
            None if safety is None else float(safety["threshold"])
        ),
        "safety_recall": (
            None if safety is None else float(safety["recall"])
        ),
        "worst_fold_recall": float(folds_frame["recall"].min()),
        "OOF_gate_passed": passed,
        "test_opened": False,
    }
    atomic_json(destination / "OOF_gate.json", result)
    return result


def bundle() -> Path:
    destination = (
        OUTPUT_ROOT / "bundles/TNormFilter_person_v4_DG_triage.zip"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    roots = [
        PROJECT_DIR / "configs/canonical_v4_person_dg_nwd.yaml",
        OUTPUT_ROOT / "protocol",
        OUTPUT_ROOT / "triage",
        OUTPUT_ROOT / "OOF",
        OUTPUT_ROOT / "pipeline_status.json",
    ]
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
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
    summaries = [summarize_variant("A0", [0, 1])]
    status("A0", "registered", metrics=summaries[0])
    for variant in VARIANTS:
        for fold in (0, 1):
            run_fold(variant, fold)
        summary = summarize_variant(variant, [0, 1])
        summaries.append(summary)
        status(f"{variant}_triage", "success", metrics=summary)
    decision = choose_candidate(summaries)
    status("selection", decision["status"], **decision)
    if decision["remaining_folds_allowed"]:
        winner = str(decision["winner"])
        for fold in (2, 3, 4):
            run_fold(winner, fold)
        gate = oof_gate(winner)
        status("OOF_gate", gate["status"], **gate)
        if gate["OOF_gate_passed"]:
            status(
                "final_training",
                "ready",
                winner=winner,
                test_opened=False,
            )
        else:
            status(
                "downstream",
                "blocked_by_OOF_gate",
                test_opened=False,
                attacks="skipped",
            )
    else:
        status(
            "downstream",
            "blocked_by_two_fold_selection",
            test_opened=False,
            attacks="skipped",
        )
    output = bundle()
    status(
        "bundle",
        "success",
        path=str(output.resolve()),
        sha256=sha256(output),
        test_opened=False,
    )


if __name__ == "__main__":
    main()
