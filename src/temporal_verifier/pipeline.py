from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.temporal.ratta import box_iou, deterministic_nms
from src.temporal_safety import ByteTrackAdapter, OCSortAdapter
from src.temporal_safety.evaluator import evaluate, parse_prediction_table
from src.temporal_safety.metrics import paired_scene_bootstrap, relative_reduction
from src.temporal_verifier.features import FEATURE_NAMES


TRACKERS = {"ocsort": OCSortAdapter, "bytetrack": ByteTrackAdapter}


def load_role(
    source: pd.DataFrame,
    prediction_csv: str,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    table = pd.read_csv(prediction_csv)
    allowed = set(source["image_path"].astype(str))
    table = table[table["image_path"].astype(str).isin(allowed)].copy()
    return parse_prediction_table(table, allowed)


def maximum_iou(box: list[float], targets: list[dict[str, Any]]) -> float:
    return max((box_iou(box, row["box"]) for row in targets), default=0.0)


def _raw_confidence(
    box: list[float], raw: list[dict[str, Any]]
) -> float:
    matches = [
        (box_iou(box, row["box"]), float(row["confidence"])) for row in raw
    ]
    if not matches:
        return 0.0
    overlap, confidence = max(matches)
    return confidence if overlap >= 0.90 else 0.0


def collect_tracks(
    tracker_name: str,
    role: str,
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    raw: dict[str, list[dict[str, Any]]],
    parameters: dict[str, Any],
    temporal_logic: dict[str, Any],
    nms_iou: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations: list[dict[str, Any]] = []
    additions: list[dict[str, Any]] = []
    tracker_class = TRACKERS[tracker_name]
    for subsequence, sequence in source.groupby("subsequence_id", sort=False):
        tracker = tracker_class(parameters, temporal_logic)
        tracker.reset(str(subsequence))
        for record in sequence.sort_values("frame_number").itertuples(index=False):
            image_path = str(record.image_path)
            emitted, events = tracker.update(
                raw.get(image_path, []), int(record.width), int(record.height)
            )
            emitted = deterministic_nms(emitted, nms_iou)
            emitted_by_track = {
                int(row["track_id"]): row for row in emitted
                if "track_id" in row
            }
            state_by_track = {state.track_id: state for state in tracker.tracks}
            frame_rows: list[dict[str, Any]] = []
            for event in events:
                track_id = int(event["track_id"])
                state = state_by_track.get(track_id)
                output = emitted_by_track.get(track_id)
                if state is None and output is None:
                    continue
                box = (
                    list(map(float, state.box))
                    if state is not None
                    else list(map(float, output["box"]))
                )
                is_hit = event["event"] in {"matched", "created"}
                detector_confidence = (
                    _raw_confidence(box, raw.get(image_path, []))
                    if is_hit else 0.0
                )
                track_key = f"{tracker_name}::{role}::{subsequence}::{track_id}"
                frame_rows.append({
                    "tracker": tracker_name,
                    "role": role,
                    "track_key": track_key,
                    "track_id": track_id,
                    "grouped_scene_id": str(record.grouped_scene_id),
                    "subsequence_id": str(subsequence),
                    "frame_number": int(record.frame_number),
                    "image_path": image_path,
                    "is_detector_hit": is_hit,
                    "interpolated": event["event"] == "missing",
                    "missing_streak": int(event["missing_streak"]),
                    "detector_confidence": detector_confidence,
                    "tracker_score": float(
                        output["confidence"]
                        if output is not None
                        else (
                            state.accumulated_score if state is not None else 0.0
                        )
                    ),
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "image_width": int(record.width),
                    "image_height": int(record.height),
                    "gt_iou": maximum_iou(
                        box, ground_truth.get(image_path, [])
                    ),
                    "duplicate_overlap": False,
                })
            for left, row in enumerate(frame_rows):
                left_box = [row[key] for key in ("x1", "y1", "x2", "y2")]
                for right, other in enumerate(frame_rows):
                    if left == right:
                        continue
                    right_box = [
                        other[key] for key in ("x1", "y1", "x2", "y2")
                    ]
                    if box_iou(left_box, right_box) >= 0.70:
                        row["duplicate_overlap"] = True
                        break
            observations.extend(frame_rows)
            for row in emitted:
                if row.get("source") == "detector":
                    continue
                additions.append({
                    "tracker": tracker_name,
                    "role": role,
                    "track_key": (
                        f"{tracker_name}::{role}::{subsequence}::"
                        f"{int(row['track_id'])}"
                    ),
                    "track_id": int(row["track_id"]),
                    "grouped_scene_id": str(record.grouped_scene_id),
                    "subsequence_id": str(subsequence),
                    "frame_number": int(record.frame_number),
                    "image_path": image_path,
                    "confidence": float(row["confidence"]),
                    "source": str(row["source"]),
                    "interpolated": bool(row["interpolated"]),
                    "x1": float(row["box"][0]),
                    "y1": float(row["box"][1]),
                    "x2": float(row["box"][2]),
                    "y2": float(row["box"][3]),
                })
    observation_frame = pd.DataFrame(observations)
    addition_frame = pd.DataFrame(additions)
    if observation_frame.empty:
        raise RuntimeError("Tracker produced no observations")
    return observation_frame, addition_frame


def accepted_predictions(
    raw: dict[str, list[dict[str, Any]]],
    additions: pd.DataFrame,
    accepted: set[str],
    nms_iou: float,
) -> dict[str, list[dict[str, Any]]]:
    predictions = {
        image: [dict(row) for row in rows] for image, rows in raw.items()
    }
    if not additions.empty:
        selected = additions[additions["track_key"].astype(str).isin(accepted)]
        for row in selected.itertuples(index=False):
            predictions.setdefault(str(row.image_path), []).append({
                "class_id": 0,
                "box": [
                    float(row.x1),
                    float(row.y1),
                    float(row.x2),
                    float(row.y2),
                ],
                "confidence": float(row.confidence),
                "track_key": str(row.track_key),
                "source": str(row.source),
                "interpolated": bool(row.interpolated),
            })
    return {
        image: deterministic_nms(rows, nms_iou)
        for image, rows in predictions.items()
    }


def evaluate_system(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
    iou: float,
    fps: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    return evaluate(source, ground_truth, predictions, threshold, iou, fps)


def system_deltas(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float]:
    return {
        "recall": float(candidate["recall"]) - float(baseline["recall"]),
        "relative_FN_reduction": relative_reduction(
            float(baseline["FN_per_frame"]), float(candidate["FN_per_frame"])
        ),
        "relative_false_alarm_increase": (
            float(candidate["false_alarms_per_minute"])
            / max(float(baseline["false_alarms_per_minute"]), 1e-12)
            - 1.0
        ),
        "F1": float(candidate["f1"]) - float(baseline["f1"]),
    }


def threshold_eligible(
    deltas: dict[str, float], recall_min: float, fn_min: float
) -> bool:
    return (
        deltas["recall"] >= recall_min
        and deltas["relative_FN_reduction"] >= fn_min
    )


@dataclass
class NestedLogistic:
    model: Pipeline
    calibrator: LogisticRegression
    oof_probabilities: np.ndarray
    labelled_index: np.ndarray


def _model(settings: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                penalty="l2",
                C=float(settings["C"]),
                class_weight=settings["class_weight"],
                max_iter=int(settings["max_iter"]),
                random_state=int(settings["seed"]),
            ),
        ),
    ])


def fit_nested_logistic(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    settings: dict[str, Any],
    maximum_folds: int,
) -> NestedLogistic:
    table = features.merge(
        labels[["track_key", "target"]], on="track_key", validate="one_to_one"
    )
    labelled = table["target"].to_numpy(dtype=int) >= 0
    labelled_index = np.where(labelled)[0]
    x = table[list(FEATURE_NAMES)].to_numpy(dtype=float)
    y = table["target"].to_numpy(dtype=int)
    groups = table["grouped_scene_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups[labelled])
    splits = min(maximum_folds, len(unique_groups))
    if splits < 2:
        raise RuntimeError("Insufficient support scenes for nested verifier CV")
    raw = np.full(len(table), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=splits)
    for train_local, validation_local in splitter.split(
        x[labelled], y[labelled], groups[labelled]
    ):
        train_index = labelled_index[train_local]
        validation_index = labelled_index[validation_local]
        if len(np.unique(y[train_index])) < 2:
            raise RuntimeError("Inner train fold lacks a verifier class")
        model = _model(settings)
        model.fit(x[train_index], y[train_index])
        raw[validation_index] = model.decision_function(x[validation_index])
    if not np.isfinite(raw[labelled]).all():
        raise RuntimeError("Nested OOF verifier scores contain NaN/Inf")
    calibrator = LogisticRegression(
        penalty="l2", C=1.0, max_iter=1000, random_state=int(settings["seed"])
    )
    calibrator.fit(raw[labelled].reshape(-1, 1), y[labelled])
    calibrated = np.full(len(table), np.nan, dtype=float)
    calibrated[labelled] = calibrator.predict_proba(
        raw[labelled].reshape(-1, 1)
    )[:, 1]
    final = _model(settings)
    final.fit(x[labelled], y[labelled])
    return NestedLogistic(final, calibrator, calibrated, labelled_index)


def predict_nested_logistic(
    fitted: NestedLogistic, features: pd.DataFrame
) -> np.ndarray:
    raw = fitted.model.decision_function(
        features[list(FEATURE_NAMES)].to_numpy(dtype=float)
    )
    return fitted.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def calibration_summary(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    fraction, predicted = calibration_curve(
        labels, probabilities, n_bins=10, strategy="quantile"
    )
    return {
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "bins": [
            {"predicted": float(left), "observed": float(right)}
            for left, right in zip(predicted, fraction)
        ],
    }


def two_fold_gate(
    baseline_metrics: list[dict[str, Any]],
    candidate_metrics: list[dict[str, Any]],
    baseline_scenes: pd.DataFrame,
    candidate_scenes: pd.DataFrame,
    rules: dict[str, Any],
) -> dict[str, Any]:
    keys = ("recall", "FN_per_frame", "false_alarms_per_minute", "f1")
    baseline_macro = {
        key: float(np.mean([row[key] for row in baseline_metrics])) for key in keys
    }
    candidate_macro = {
        key: float(np.mean([row[key] for row in candidate_metrics])) for key in keys
    }
    delta = system_deltas(baseline_macro, candidate_macro)
    merged = baseline_scenes.merge(
        candidate_scenes,
        on=["fold", "grouped_scene_id"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    recall_effect = (
        merged["recall_candidate"] - merged["recall_baseline"]
    )
    fn_effect = (
        merged["FN_per_frame_candidate"] - merged["FN_per_frame_baseline"]
    )
    improved = (recall_effect > 1e-12) | (fn_effect < -1e-12)
    bootstrap = paired_scene_bootstrap(
        dict(zip(merged["grouped_scene_id"], merged["recall_baseline"])),
        dict(zip(merged["grouped_scene_id"], merged["recall_candidate"])),
        int(rules["bootstrap_iterations"]),
        int(rules["bootstrap_seed"]),
    )
    checks = {
        "recall": delta["recall"]
        >= float(rules["absolute_macro_recall_improvement_min"]),
        "FN_per_frame": delta["relative_FN_reduction"]
        >= float(rules["relative_macro_FN_reduction_min"]),
        "false_alarms": delta["relative_false_alarm_increase"]
        <= float(rules["relative_false_alarms_increase_max"]),
        "F1": delta["F1"] >= -float(rules["F1_degradation_max"]),
        "worst_scene_recall": float(recall_effect.min()) >= -1e-12,
        "improved_scene_fraction": float(improved.mean())
        >= float(rules["improved_scene_fraction_min"]),
        "bootstrap_recall": float(bootstrap["ci_low"]) > 0,
        "evaluator": all(
            row["evaluator_consistency"] == rules["evaluator_consistency"]
            for row in candidate_metrics
        ),
        "lost_frames": all(
            row["lost_frames"] == int(rules["lost_frames"])
            for row in candidate_metrics
        ),
        "NaN_Inf": all(
            row["NaN_Inf"] == int(rules["NaN_Inf"])
            for row in candidate_metrics
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "baseline_macro": baseline_macro,
        "candidate_macro": candidate_macro,
        "deltas": delta,
        "checks": checks,
        "failed_conditions": [key for key, value in checks.items() if not value],
        "improved_scene_fraction": float(improved.mean()),
        "worst_scene_recall_delta": float(recall_effect.min()),
        "paired_scene_bootstrap_recall": bootstrap,
    }
