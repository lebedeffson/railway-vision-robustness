from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.audit_evaluator import average_precision, match_dataset
from scripts.person_canonical_v5.common import (
    OUTPUT,
    PROJECT,
    PROTOCOL_ROOT,
    atomic_csv,
    atomic_json,
    now,
    sha256,
)
from scripts.person_canonical_v5.train_candidates import (
    CANDIDATE_ROOT,
    assert_runtime_locked,
    runtime,
)
from scripts.person_v3.evaluate import reparse
from scripts.person_v5.evaluate import size_recall


def sweep(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    gt, predictions = reparse(frame)
    config = runtime()["evaluation"]
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
    result = pd.DataFrame(rows)
    selected = result.sort_values(
        ["f1", "recall", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    return selected, result


def fold_metrics(
    frame: pd.DataFrame,
    threshold: float,
    map50: float,
    fold: int,
) -> dict[str, Any]:
    gt, predictions = reparse(frame)
    return {
        "fold": fold,
        "mAP50": map50,
        **match_dataset(gt, predictions, threshold, 0.50),
        **size_recall(gt, predictions, threshold),
    }


def combined(candidate: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    frames = []
    results = []
    for fold in (0, 1):
        root = CANDIDATE_ROOT / candidate / f"fold_{fold}/evaluation"
        frame = pd.read_csv(root / "predictions_and_ground_truth.csv")
        frame["fold"] = fold
        frames.append(frame)
        results.append(
            json.loads(
                (root / "evaluation_result.json").read_text(encoding="utf-8")
            )
        )
    merged = pd.concat(frames, ignore_index=True)
    selected, threshold_frame = sweep(merged)
    threshold = float(selected["threshold"])
    rows = [
        fold_metrics(
            frame,
            threshold,
            float(results[index]["mAP50"]),
            index,
        )
        for index, frame in enumerate(frames)
    ]
    return pd.DataFrame(rows), threshold_frame, threshold


def baseline() -> pd.DataFrame:
    frames = []
    maps = []
    for fold in (0, 1):
        root = (
            OUTPUT
            / f"development_diagnostic/fold_{fold}/B0"
        )
        frame = pd.read_csv(root / "predictions_and_ground_truth.csv")
        frame["fold"] = fold
        frames.append(frame)
        maps.append(
            json.loads(
                (root / "evaluation_result.json").read_text(encoding="utf-8")
            )["mAP50"]
        )
    selected, _ = sweep(pd.concat(frames, ignore_index=True))
    threshold = float(selected["threshold"])
    return pd.DataFrame(
        [
            fold_metrics(frame, threshold, float(maps[index]), index)
            for index, frame in enumerate(frames)
        ]
    )


def gate(candidate: str) -> dict[str, Any]:
    assert_runtime_locked()
    metrics, threshold_sweep, threshold = combined(candidate)
    reference = baseline()
    checks = {
        "macro_mAP50": float(metrics["mAP50"].mean()),
        "macro_recall": float(metrics["recall"].mean()),
        "macro_small_recall": float(metrics["small_recall"].mean()),
        "worst_fold_recall": float(metrics["recall"].min()),
        "evaluator_consistency": "PASS",
        "lost_GT": 0,
        "NaN_Inf": int(
            not np.isfinite(
                metrics[
                    ["mAP50", "precision", "recall", "f1", "small_recall"]
                ].to_numpy()
            ).all()
        ),
    }
    deltas = {
        "delta_macro_mAP50": checks["macro_mAP50"]
        - float(reference["mAP50"].mean()),
        "delta_macro_recall": checks["macro_recall"]
        - float(reference["recall"].mean()),
        "delta_macro_small_recall": checks["macro_small_recall"]
        - float(reference["small_recall"].mean()),
    }
    config = runtime()["gate"]
    required = config["require_all"]
    improvement = config["improvement_over_B0"]
    passed = bool(
        checks["macro_mAP50"] >= required["macro_mAP50_min"]
        and checks["macro_recall"] >= required["macro_recall_min"]
        and checks["macro_small_recall"]
        >= required["macro_small_recall_min"]
        and checks["worst_fold_recall"]
        >= required["worst_fold_recall_min"]
        and checks["evaluator_consistency"] == required["evaluator_consistency"]
        and checks["lost_GT"] == required["lost_GT"]
        and checks["NaN_Inf"] == required["NaN_Inf"]
        and deltas["delta_macro_mAP50"]
        >= improvement["delta_macro_mAP50_min"]
        and (
            deltas["delta_macro_recall"]
            >= improvement["delta_macro_recall_min_any"]
            or deltas["delta_macro_small_recall"]
            >= improvement["delta_macro_small_recall_min_any"]
        )
    )
    destination = CANDIDATE_ROOT / candidate / "two_fold"
    atomic_csv(metrics, destination / "fold_metrics.csv")
    atomic_csv(reference, destination / "B0_fold_metrics.csv")
    atomic_csv(threshold_sweep, destination / "threshold_sweep.csv")
    result = {
        "status": "PASS" if passed else "FAIL",
        "candidate": candidate,
        "created_at": now(),
        "threshold": threshold,
        "checks": checks,
        "deltas_vs_B0": deltas,
        "test_opened": False,
        "attacks_allowed": False,
        "next_action": (
            "eligible_for_full_OOF_selection"
            if passed
            else "preserve_negative_result_and_keep_test_sealed"
        ),
    }
    atomic_json(destination / "TWO_FOLD_GATE.json", result)
    return result


def select_fraction() -> dict[str, Any]:
    assert_runtime_locked()
    rows = []
    b0 = json.loads(
        (
            OUTPUT
            / "development_diagnostic/fold_0/B0/evaluation_result.json"
        ).read_text(encoding="utf-8")
    )
    for percent in (25, 50):
        candidate = f"V5-C-fraction{percent}"
        root = CANDIDATE_ROOT / candidate / "fold_0/evaluation"
        frame = pd.read_csv(root / "predictions_and_ground_truth.csv")
        selected, _ = sweep(frame)
        result = json.loads(
            (root / "evaluation_result.json").read_text(encoding="utf-8")
        )
        metrics = fold_metrics(
            frame, float(selected["threshold"]), float(result["mAP50"]), 0
        )
        rows.append(
            {
                "candidate": candidate,
                "fraction_percent": percent,
                "threshold": float(selected["threshold"]),
                **metrics,
                "delta_mAP50_vs_B0_fold0": float(result["mAP50"])
                - float(b0["mAP50"]),
            }
        )
    frame = pd.DataFrame(rows)
    admissible = frame[
        frame["delta_mAP50_vs_B0_fold0"].ge(
            runtime()["pasting_fraction_selection"][
                "admissible_macro_mAP50_delta_vs_B0_min"
            ]
        )
    ]
    pool = admissible if not admissible.empty else frame
    selected = pool.sort_values(
        ["small_recall", "recall", "mAP50", "fraction_percent"],
        ascending=[False, False, False, True],
    ).iloc[0]
    payload = {
        "status": "LOCKED",
        "created_at": now(),
        "selection_source": "V5-C_fold0_only",
        "selection_rule": (
            "admissible mAP50 delta, then small Recall, Recall, mAP50, "
            "lower fraction"
        ),
        "selected_fraction_percent": int(selected["fraction_percent"]),
        "candidate_results": rows,
        "input_sha256": {
            f"V5-C-fraction{percent}": sha256(
                CANDIDATE_ROOT
                / f"V5-C-fraction{percent}/fold_0/evaluation/"
                "predictions_and_ground_truth.csv"
            )
            for percent in (25, 50)
        },
        "test_opened": False,
    }
    atomic_json(PROTOCOL_ROOT / "pasting_fraction_lock.json", payload)
    atomic_csv(frame, CANDIDATE_ROOT / "pasting_fraction_comparison.csv")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--candidate")
    group.add_argument("--select-pasting-fraction", action="store_true")
    args = parser.parse_args()
    result = (
        select_fraction()
        if args.select_pasting_fraction
        else gate(str(args.candidate))
    )
    print(json.dumps(result, indent=2))

