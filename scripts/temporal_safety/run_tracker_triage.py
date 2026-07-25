from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.temporal_safety.common import (
    OUTPUT,
    assert_locked,
    atomic_csv,
    atomic_json,
    config,
    prediction_path,
    source_frames,
)
from src.temporal_safety import ByteTrackAdapter, OCSortAdapter
from src.temporal_safety.evaluator import (
    evaluate,
    gt_track_metrics,
    maximum_scene_recall_degradation,
    parse_prediction_table,
    run_tracker,
)
from src.temporal_safety.metrics import gate_checks, relative_reduction


TRACKERS = {"bytetrack": ByteTrackAdapter, "ocsort": OCSortAdapter}


def load_role(
    fold: int, role: str
) -> tuple[
    pd.DataFrame,
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    source = source_frames(fold, role)
    table = pd.read_csv(prediction_path(fold, role))
    allowed = set(source["image_path"])
    if role == "train":
        table = table[table["image_path"].astype(str).isin(allowed)].copy()
    ground_truth, predictions = parse_prediction_table(
        table, allowed
    )
    return source, ground_truth, predictions


def evaluate_candidate(
    tracker_name: str,
    parameters: dict[str, Any],
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    raw: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    protocol = config()
    tracker = TRACKERS[tracker_name](parameters, protocol["temporal_logic"])
    started = time.perf_counter()
    predictions, events = run_tracker(
        source,
        raw,
        tracker,
        float(protocol["temporal_logic"]["fusion_nms_iou"]),
    )
    runtime = time.perf_counter() - started
    metrics, per_scene = evaluate(
        source,
        ground_truth,
        predictions,
        float(protocol["detector"]["operating_threshold"]),
        float(protocol["detector"]["iou_threshold"]),
        float(protocol["data"]["expected_nominal_fps"]),
    )
    metrics.update(
        {
            "tracker": tracker_name,
            "parameters": parameters,
            "tracker_runtime_seconds": runtime,
            "tracker_latency_ms_per_frame": 1000.0 * runtime / max(len(source), 1),
            **gt_track_metrics(
                source,
                ground_truth,
                predictions,
                float(protocol["detector"]["operating_threshold"]),
                float(protocol["detector"]["iou_threshold"]),
            ),
        }
    )
    return metrics, per_scene, events


def baseline(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    raw: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    protocol = config()
    metrics, per_scene = evaluate(
        source,
        ground_truth,
        raw,
        float(protocol["detector"]["operating_threshold"]),
        float(protocol["detector"]["iou_threshold"]),
        float(protocol["data"]["expected_nominal_fps"]),
    )
    metrics.update(
        gt_track_metrics(
            source,
            ground_truth,
            raw,
            float(protocol["detector"]["operating_threshold"]),
            float(protocol["detector"]["iou_threshold"]),
        )
    )
    metrics["tracker"] = "frame_detector"
    return metrics, per_scene


def select_on_train() -> dict[str, dict[str, Any]]:
    protocol = config()
    source, ground_truth, raw = load_role(0, "train")
    baseline_metrics, baseline_scene = baseline(source, ground_truth, raw)
    atomic_json(OUTPUT / "selection/train_baseline.json", baseline_metrics)
    rows = []
    selected: dict[str, dict[str, Any]] = {}
    for tracker_name, tracker_class in TRACKERS.items():
        grid = protocol["tracker_grids"][tracker_name]
        if len(grid) > int(protocol["selection"]["maximum_configurations_per_tracker"]):
            raise RuntimeError("Tracker grid exceeds frozen maximum")
        candidates = []
        for index, parameters in enumerate(grid):
            metrics, per_scene, _ = evaluate_candidate(
                tracker_name, parameters, source, ground_truth, raw
            )
            scene_drop = maximum_scene_recall_degradation(
                baseline_scene, per_scene
            )
            metrics["maximum_scene_recall_degradation"] = scene_drop
            false_alarm_ratio = metrics["false_alarms_per_minute"] / max(
                baseline_metrics["false_alarms_per_minute"], 1e-12
            )
            eligible = (
                false_alarm_ratio
                <= 1.0
                + float(
                    protocol["triage_gate"]["relative_false_alarms_increase_max"]
                )
                and metrics["f1"] - baseline_metrics["f1"]
                >= -float(protocol["triage_gate"]["F1_degradation_max"])
                and scene_drop
                <= float(
                    protocol["triage_gate"]["maximum_scene_recall_degradation"]
                )
            )
            metrics.update(
                {
                    "configuration_index": index,
                    "eligible_on_train": eligible,
                    "delta_recall_vs_train_baseline": metrics["recall"]
                    - baseline_metrics["recall"],
                    "relative_FN_reduction_vs_train_baseline": relative_reduction(
                        baseline_metrics["FN_per_frame"], metrics["FN_per_frame"]
                    ),
                    "relative_false_alarm_increase_vs_train_baseline": false_alarm_ratio
                    - 1.0,
                    "delta_F1_vs_train_baseline": metrics["f1"]
                    - baseline_metrics["f1"],
                }
            )
            candidates.append(metrics)
            rows.append(metrics)
        pool = [row for row in candidates if row["eligible_on_train"]]
        if not pool:
            pool = candidates
            selection_status = "NO_CONFIG_MET_TRAIN_CONSTRAINTS"
        else:
            selection_status = "TRAIN_CONSTRAINTS_PASS"
        winner = sorted(
            pool,
            key=lambda row: (
                -float(row["recall"]),
                float(row["FN_per_frame"]),
                -float(row["f1"]),
                float(row["false_alarms_per_minute"]),
                int(row["configuration_index"]),
            ),
        )[0]
        selected[tracker_name] = {
            "tracker": tracker_name,
            "parameters": winner["parameters"],
            "configuration_index": winner["configuration_index"],
            "selection_status": selection_status,
            "selection_data": "fold_0_train_scenes_excluding_folds_0_and_1",
        }
    atomic_csv(OUTPUT / "selection/TRACKER_GRID_RESULTS.csv", pd.DataFrame(rows))
    atomic_json(
        OUTPUT / "selection/SELECTED_TRACKER_CONFIGS.json",
        {
            "selected": selected,
            "heldout_metrics_used": False,
            "test_used": False,
        },
    )
    return selected


def macro(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    lock = assert_locked()
    protocol = config()
    selected = select_on_train()
    baselines: dict[int, dict[str, Any]] = {}
    baseline_scenes = []
    candidate_metrics: dict[str, list[dict[str, Any]]] = {
        name: [] for name in TRACKERS
    }
    candidate_scenes: dict[str, list[pd.DataFrame]] = {
        name: [] for name in TRACKERS
    }
    for fold in (0, 1):
        source, ground_truth, raw = load_role(fold, "heldout")
        base, base_scene = baseline(source, ground_truth, raw)
        base["fold"] = fold
        baselines[fold] = base
        base_scene.insert(0, "fold", fold)
        base_scene.insert(0, "system", "frame_detector")
        baseline_scenes.append(base_scene)
        for tracker_name in TRACKERS:
            metrics, per_scene, events = evaluate_candidate(
                tracker_name,
                selected[tracker_name]["parameters"],
                source,
                ground_truth,
                raw,
            )
            metrics["fold"] = fold
            metrics["maximum_scene_recall_degradation"] = (
                maximum_scene_recall_degradation(base_scene, per_scene)
            )
            candidate_metrics[tracker_name].append(metrics)
            per_scene.insert(0, "fold", fold)
            per_scene.insert(0, "system", tracker_name)
            candidate_scenes[tracker_name].append(per_scene)
            atomic_csv(
                OUTPUT / f"triage/fold_{fold}/{tracker_name}/track_events.csv",
                events,
            )
            atomic_json(
                OUTPUT / f"triage/fold_{fold}/{tracker_name}/metrics.json",
                metrics,
            )
    baseline_macro = {
        key: macro(list(baselines.values()), key)
        for key in (
            "recall",
            "FN_per_frame",
            "false_alarms_per_minute",
            "f1",
            "precision",
            "mAP50",
            "track_recall",
        )
    }
    all_baseline_scene = pd.concat(baseline_scenes, ignore_index=True)
    decisions = []
    for tracker_name, rows in candidate_metrics.items():
        candidate_macro = {
            key: macro(rows, key)
            for key in (
                "recall",
                "FN_per_frame",
                "false_alarms_per_minute",
                "f1",
                "precision",
                "mAP50",
                "track_recall",
            )
        }
        scenes = pd.concat(candidate_scenes[tracker_name], ignore_index=True)
        merged = all_baseline_scene.merge(
            scenes,
            on=["fold", "grouped_scene_id"],
            suffixes=("_baseline", "_candidate"),
        )
        maximum_drop = float(
            (merged["recall_baseline"] - merged["recall_candidate"]).max()
        )
        candidate_macro["maximum_scene_recall_degradation"] = maximum_drop
        checks = gate_checks(
            baseline_macro,
            candidate_macro,
            protocol["triage_gate"],
            float(protocol["triage_gate"]["maximum_scene_recall_degradation"]),
        )
        decisions.append(
            {
                "tracker": tracker_name,
                "status": "TRIAGE_PASS" if all(checks.values()) else "TRIAGE_FAIL",
                "checks": checks,
                "baseline_macro": baseline_macro,
                "candidate_macro": candidate_macro,
                "deltas": {
                    "recall": candidate_macro["recall"]
                    - baseline_macro["recall"],
                    "relative_FN_reduction": relative_reduction(
                        baseline_macro["FN_per_frame"],
                        candidate_macro["FN_per_frame"],
                    ),
                    "relative_false_alarm_increase": candidate_macro[
                        "false_alarms_per_minute"
                    ]
                    / max(baseline_macro["false_alarms_per_minute"], 1e-12)
                    - 1.0,
                    "F1": candidate_macro["f1"] - baseline_macro["f1"],
                },
            }
        )
    passing = [row for row in decisions if row["status"] == "TRIAGE_PASS"]
    winner = None
    if passing:
        winner = sorted(
            passing,
            key=lambda row: (
                -row["candidate_macro"]["recall"],
                row["candidate_macro"]["FN_per_frame"],
                -row["candidate_macro"]["f1"],
                row["candidate_macro"]["false_alarms_per_minute"],
            ),
        )[0]["tracker"]
    status = "TRIAGE_PASS" if winner else "DEVELOPMENT_FAIL"
    payload = {
        "protocol_id": protocol["protocol_id"],
        "status": status,
        "winner": winner,
        "decisions": decisions,
        "selected_parameters": selected,
        "implementation_commit": lock["implementation_commit"],
        "test_status": "SEALED",
        "test_access_count": 0,
        "attacks": "OUT_OF_SCOPE",
    }
    atomic_json(OUTPUT / "triage/TRIAGE_GATE.json", payload)
    atomic_csv(
        OUTPUT / "triage/PER_SCENE_RESULTS.csv",
        pd.concat(
            [
                *baseline_scenes,
                *[
                    frame
                    for frames in candidate_scenes.values()
                    for frame in frames
                ],
            ],
            ignore_index=True,
        ),
    )
    atomic_json(
        OUTPUT / "decision_trace.json",
        {
            "decisions": [
                {
                    "stage": "tracker_parameter_selection",
                    "decision": "FROZEN_FROM_NINE_SUPPORT_SCENES_EXCLUDING_FOLDS_0_AND_1",
                },
                {
                    "stage": "two_fold_triage",
                    "decision": status,
                    "winner": winner,
                },
                {
                    "stage": "full_development",
                    "decision": "ALLOWED" if winner else "BLOCKED_BY_TRIAGE_FAIL",
                },
                {
                    "stage": "test",
                    "decision": "BLOCKED_UNTIL_FULL_DEVELOPMENT_PASS",
                },
            ],
            "test_access_count": 0,
        },
    )


if __name__ == "__main__":
    main()
