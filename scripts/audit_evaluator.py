from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rescue_common import (
    PROJECT_DIR,
    OUTPUT_ROOT,
    assert_test_sealed,
    atomic_json,
    completed_marker,
    ensure_branch,
    load_protocol,
)


OUTPUT = OUTPUT_ROOT / "evaluator"


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-12)


def class_aware_nms(predictions: list[dict[str, Any]], iou_threshold: float = 0.5) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for prediction in sorted(predictions, key=lambda row: -row["confidence"]):
        if any(
            prediction["class_id"] == other["class_id"]
            and box_iou(prediction["box"], other["box"]) > iou_threshold
            for other in kept
        ):
            continue
        kept.append(prediction)
    return kept


def match_dataset(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    confidence: float,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    tp = fp = fn = 0
    for image_id in sorted(ground_truth):
        gt = ground_truth[image_id]
        candidates = sorted(
            [row for row in predictions.get(image_id, []) if row["confidence"] >= confidence],
            key=lambda row: -row["confidence"],
        )
        used: set[int] = set()
        for prediction in candidates:
            available = [
                (index, box_iou(prediction["box"], target["box"]))
                for index, target in enumerate(gt)
                if index not in used and prediction["class_id"] == target["class_id"]
            ]
            if available:
                index, overlap = max(available, key=lambda item: item[1])
                if overlap >= iou_threshold:
                    used.add(index)
                    tp += 1
                    continue
            fp += 1
        fn += len(gt) - len(used)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def average_precision(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    class_id: int,
    iou_threshold: float = 0.5,
) -> float:
    total_gt = sum(
        target["class_id"] == class_id
        for targets in ground_truth.values() for target in targets
    )
    ranked = sorted(
        (
            (image_id, prediction)
            for image_id, rows in predictions.items()
            for prediction in rows if prediction["class_id"] == class_id
        ),
        key=lambda item: -item[1]["confidence"],
    )
    used: defaultdict[str, set[int]] = defaultdict(set)
    true_positives = []
    false_positives = []
    for image_id, prediction in ranked:
        candidates = [
            (index, box_iou(prediction["box"], target["box"]))
            for index, target in enumerate(ground_truth.get(image_id, []))
            if target["class_id"] == class_id and index not in used[image_id]
        ]
        if candidates:
            index, overlap = max(candidates, key=lambda item: item[1])
            matched = overlap >= iou_threshold
        else:
            index, matched = -1, False
        true_positives.append(float(matched))
        false_positives.append(float(not matched))
        if matched:
            used[image_id].add(index)
    if total_gt == 0:
        return math.nan
    tp = np.cumsum(true_positives)
    fp = np.cumsum(false_positives)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))


def golden_case() -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    gt = {
        "perfect_duplicate_wrong_class_localization": [
            {"class_id": 0, "box": [0, 0, 10, 10]},
            {"class_id": 1, "box": [20, 20, 30, 30]},
        ],
        "false_negative": [{"class_id": 0, "box": [0, 0, 10, 10]}],
        "perfect_and_background_fp": [{"class_id": 1, "box": [0, 0, 10, 10]}],
    }
    predictions = {
        "perfect_duplicate_wrong_class_localization": [
            {"class_id": 0, "confidence": 0.90, "box": [0, 0, 10, 10]},
            {"class_id": 0, "confidence": 0.80, "box": [0, 0, 10, 10]},
            {"class_id": 0, "confidence": 0.70, "box": [20, 20, 30, 30]},
            {"class_id": 1, "confidence": 0.60, "box": [20, 20, 25, 25]},
        ],
        "false_negative": [],
        "perfect_and_background_fp": [
            {"class_id": 1, "confidence": 0.95, "box": [0, 0, 10, 10]},
            {"class_id": 1, "confidence": 0.50, "box": [30, 30, 40, 40]},
        ],
    }
    return gt, predictions


def timeout_contract() -> dict[str, Any]:
    return {
        "timeout": True,
        "predictions": [],
        "valid_empty_prediction": False,
        "must_be_excluded_or_rerun": True,
    }


def main() -> None:
    ensure_branch()
    assert_test_sealed()
    protocol = load_protocol()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    gt, raw_predictions = golden_case()
    predictions = {
        image_id: class_aware_nms(rows, 0.5)
        for image_id, rows in raw_predictions.items()
    }
    operating = match_dataset(gt, predictions, confidence=0.5)
    stricter = match_dataset(gt, predictions, confidence=0.85)
    ap = {class_id: average_precision(gt, predictions, class_id) for class_id in (0, 1)}
    map50 = float(np.mean(list(ap.values())))

    expected = {
        "operating": {
            "tp": 2, "fp": 3, "fn": 2, "precision": 0.4,
            "recall": 0.5, "f1": 4 / 9,
        },
        "strict": {
            "tp": 2, "fp": 0, "fn": 2, "precision": 1.0,
            "recall": 0.5, "f1": 2 / 3,
        },
        "ap50_per_class": {"0": 0.5, "1": 0.5},
        "mAP50": 0.5,
    }
    golden_passed = all(
        math.isclose(float(operating[key]), float(value), abs_tol=1e-12)
        for key, value in expected["operating"].items()
    ) and all(
        math.isclose(float(stricter[key]), float(value), abs_tol=1e-12)
        for key, value in expected["strict"].items()
    ) and math.isclose(map50, expected["mAP50"], abs_tol=1e-12)

    width, height, target = 1920, 1080, 1280
    scale = min(target / width, target / height)
    pad_x = (target - width * scale) / 2
    pad_y = (target - height * scale) / 2
    original = [100.5, 50.25, 700.75, 900.0]
    transformed = [
        original[0] * scale + pad_x, original[1] * scale + pad_y,
        original[2] * scale + pad_x, original[3] * scale + pad_y,
    ]
    inverse = [
        (transformed[0] - pad_x) / scale, (transformed[1] - pad_y) / scale,
        (transformed[2] - pad_x) / scale, (transformed[3] - pad_y) / scale,
    ]
    letterbox_error = max(abs(left - right) for left, right in zip(original, inverse))

    built_in = json.loads(
        (PROJECT_DIR / "outputs/canonical_v2/training/checkpoint_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    independent = pd.read_csv(
        PROJECT_DIR / "outputs/canonical_v2/baseline_rescue_v2/clean_metrics_train_val_test.csv"
    )
    val = independent[
        (independent["split"] == "val") & (independent["operating_point"] == "standard")
    ].iloc[0]
    evaluator_map_difference = abs(float(built_in["validation_mAP50"]) - float(val["mAP50"]))
    evaluator_recall_difference = abs(
        float(built_in["validation_recall_at_training_evaluator"])
        - float(val["recall_ap_evaluator"])
    )

    timeout_record = timeout_contract()
    reference = {
        "ground_truth": gt,
        "predictions_before_nms": raw_predictions,
        "predictions_after_nms": predictions,
        "expected": expected,
        "observed": {
            "operating": operating, "strict": stricter,
            "ap50_per_class": {str(key): value for key, value in ap.items()},
            "mAP50": map50,
        },
    }
    atomic_json(OUTPUT / "reference_cases.json", reference)
    audit = {
        "status": "PASS" if (
            golden_passed and letterbox_error <= 1e-9
            and evaluator_map_difference <= 5e-4
            # Ultralytics reports recall at its AP operating convention while
            # the independent export reconstructs it from saved detections.
            # A 0.002 absolute agreement bound is fixed for that convention gap.
            and evaluator_recall_difference <= 2e-3
            and protocol["selection"]["test_usage"] == "forbidden"
        ) else "FAIL",
        "golden_case_passed": golden_passed,
        "map_is_operating_threshold_independent": True,
        "operating_metrics_change_with_threshold": operating != stricter,
        "frame_order_invariant": (
            match_dataset(
                dict(reversed(list(gt.items()))),
                dict(reversed(list(predictions.items()))), 0.5,
            ) == operating
        ),
        "class_aware_matching": True,
        "duplicate_predictions_removed_by_nms": (
            sum(map(len, raw_predictions.values())) - sum(map(len, predictions.values())) == 1
        ),
        "letterbox_roundtrip_max_error_px": letterbox_error,
        "timeout_contract": timeout_record,
        "built_in_vs_independent_map50_difference": evaluator_map_difference,
        "built_in_vs_independent_recall_difference": evaluator_recall_difference,
        "test_predictions_read": False,
    }
    atomic_json(OUTPUT / "evaluator_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Evaluator audit failed: {audit}")
    completed_marker(
        OUTPUT,
        inputs=[
            PROJECT_DIR / "outputs/canonical_v2/training/checkpoint_provenance.json",
            PROJECT_DIR / "outputs/canonical_v2/baseline_rescue_v2/clean_metrics_train_val_test.csv",
        ],
        outputs=[OUTPUT / "reference_cases.json", OUTPUT / "evaluator_audit.json"],
        extra={"stage": "evaluator_audit", "test_predictions_read": False},
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
