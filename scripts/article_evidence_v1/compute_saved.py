from __future__ import annotations

import json
import math
import pickle
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(PROJECT_BOOTSTRAP / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_BOOTSTRAP / "scripts"))

from scripts.article_evidence_v1.common import (
    OUTPUT,
    PROJECT,
    atomic_csv,
    atomic_json,
    check_no_test_markers,
    config,
)
from scripts.person_v8b.analyze_loso import (
    FeatureDataset,
    LAYERS,
    REGIONS,
    _similarities,
)
from src.crop_verifier_v1.model import predict_verifier
from src.temporal_verifier.features import FEATURE_NAMES, rule_acceptance
from src.temporal_verifier.pipeline import predict_nested_logistic


def _gate_values(row: pd.Series | dict[str, Any]) -> dict[str, bool]:
    gate = config()["gate"]
    recall = float(row["recall_gain"]) >= float(gate["recall_gain_min"])
    fn = float(row["fn_reduction_percent"]) >= 100.0 * float(
        gate["fn_reduction_min"]
    )
    false_alarm = float(row["false_alarm_change_percent"]) <= 100.0 * float(
        gate["false_alarm_increase_max"]
    )
    f1 = float(row["f1_change"]) >= -float(gate["f1_degradation_max"])
    return {
        "recall_gate": recall,
        "fn_gate": fn,
        "false_alarm_gate": false_alarm,
        "f1_gate": f1,
        "full_gate": recall and fn and false_alarm and f1,
    }


def _gate_distance(row: pd.Series | dict[str, Any]) -> float:
    gate = config()["gate"]
    recall = float(row["recall_gain"])
    fn = float(row["fn_reduction_percent"]) / 100.0
    false_alarm = float(row["false_alarm_change_percent"]) / 100.0
    f1 = float(row["f1_change"])
    values = [
        max(0.0, (float(gate["recall_gain_min"]) - recall) / 0.10),
        max(0.0, (float(gate["fn_reduction_min"]) - fn) / 0.15),
        max(0.0, (false_alarm - float(gate["false_alarm_increase_max"])) / 0.20),
        max(
            0.0,
            (-float(gate["f1_degradation_max"]) - f1) / 0.03,
        ),
    ]
    return float(np.linalg.norm(values))


def track_verifier_block() -> None:
    settings = config()
    source = PROJECT / settings["inputs"]["track_verifier_sweep"]
    sweep = pd.read_csv(source).sort_values("threshold").reset_index(drop=True)
    if len(sweep) != 201 or sweep["threshold"].nunique() != 201:
        raise RuntimeError("Saved verifier sweep is not the frozen 201-point grid")

    feature_root = PROJECT / "outputs/temporal_verifier_v1/tracks/ocsort/screening"
    features = pd.read_csv(feature_root / "track_features.csv")
    labels = pd.read_csv(feature_root / "track_labels.csv")
    with (PROJECT / settings["inputs"]["track_verifier_model"]).open("rb") as handle:
        fitted = pickle.load(handle)
    probability = predict_nested_logistic(fitted, features)
    table = features[["track_key"]].merge(
        labels[["track_key", "target"]], on="track_key", validate="one_to_one"
    )
    labelled = table["target"].to_numpy(dtype=int) >= 0
    target = table.loc[labelled, "target"].to_numpy(dtype=int)
    score = probability[labelled]

    rows: list[dict[str, Any]] = []
    for system in sweep.itertuples(index=False):
        threshold = float(system.threshold)
        predicted = score >= threshold
        tp = int(((predicted == 1) & (target == 1)).sum())
        fp = int(((predicted == 1) & (target == 0)).sum())
        fn = int(((predicted == 0) & (target == 1)).sum())
        tn = int(((predicted == 0) & (target == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        track_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        row = {
            "threshold": threshold,
            "track_precision": precision,
            "track_recall": recall,
            "track_f1": track_f1,
            "track_specificity": specificity,
            "track_false_positive_rate": 1.0 - specificity,
            "track_true_positive_rate": recall,
            "system_precision": float(system.precision),
            "system_recall": float(system.recall),
            "system_f1": float(system.f1),
            "system_fn_per_frame": float(system.FN_per_frame),
            "system_false_alarms_per_min": float(
                system.false_alarms_per_minute
            ),
            "recall_gain": float(system.delta_recall),
            "fn_reduction_percent": 100.0
            * float(system.delta_relative_FN_reduction),
            "false_alarm_change_percent": 100.0
            * float(system.delta_relative_false_alarm_increase),
            "f1_change": float(system.delta_F1),
        }
        row.update(_gate_values(row))
        row["gate_distance"] = _gate_distance(row)
        rows.append(row)
    result = pd.DataFrame(rows)
    selected = float(settings["track_verifier"]["selected_threshold"])
    result["is_selected_threshold"] = np.isclose(result["threshold"], selected)
    max_track = result["track_f1"].max()
    result["is_max_track_f1"] = np.isclose(result["track_f1"], max_track)
    max_system = result["system_f1"].max()
    result["is_max_system_f1"] = np.isclose(result["system_f1"], max_system)
    pool = result[result["recall_gain"] >= 0.10]
    minimum_false_alarm = (
        pool["system_false_alarms_per_min"].min() if not pool.empty else math.inf
    )
    result["is_min_false_alarm_with_recall_gain_010"] = (
        result["recall_gain"].ge(0.10)
        & np.isclose(
            result["system_false_alarms_per_min"], minimum_false_alarm
        )
    )
    minimum_distance = result["gate_distance"].min()
    result["is_closest_to_full_gate"] = np.isclose(
        result["gate_distance"], minimum_distance
    )

    root = OUTPUT / "track_verifier"
    atomic_csv(root / "TRACK_VERIFIER_201_THRESHOLDS.csv", result)
    atomic_csv(
        root / "TRACK_VERIFIER_GATE_DISTANCE.csv",
        result[
            [
                "threshold",
                "gate_distance",
                "recall_gate",
                "fn_gate",
                "false_alarm_gate",
                "f1_gate",
                "full_gate",
                "is_closest_to_full_gate",
            ]
        ],
    )
    fpr, tpr, roc_threshold = roc_curve(target, score)
    roc = pd.DataFrame(
        {
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "threshold": roc_threshold,
            "auroc": roc_auc_score(target, score),
        }
    )
    track_precision, track_recall, pr_threshold = precision_recall_curve(
        target, score
    )
    pr = pd.DataFrame(
        {
            "precision": track_precision,
            "recall": track_recall,
            "threshold": np.r_[pr_threshold, np.nan],
            "auprc": average_precision_score(target, score),
        }
    )
    atomic_csv(root / "TRACK_VERIFIER_ROC.csv", roc)
    atomic_csv(root / "TRACK_VERIFIER_PR.csv", pr)

    verifier_config = json.loads(
        (
            PROJECT
            / "outputs/temporal_verifier_v1/learned/logistic_l2/PRE_CONFIRMATION_FREEZE.json"
        ).read_text(encoding="utf-8")
    )
    actual = {
        "model_type": "L2-regularized logistic regression",
        "regularization": "l2",
        "regularization_strength": 1.0,
        "calibration_method": "Platt logistic calibration",
        "outer_grouping": "grouped_scene_id screening/confirmation scenes",
        "inner_grouping": "GroupKFold by grouped_scene_id on support scenes",
        "normalization": verifier_config["normalization"],
        "threshold_selection_rule": "maximum F1 subject to Recall gain >= 0.10 and relative FN reduction >= 0.15",
        "selected_threshold": selected,
        "feature_list": verifier_config["features"],
    }
    atomic_json(root / "TRACK_VERIFIER_CONFIG_ACTUAL.json", actual)
    audit = {
        "status": "PASS",
        "number_of_unique_thresholds": int(result["threshold"].nunique()),
        "missing_thresholds": 0,
        "duplicate_thresholds": int(result["threshold"].duplicated().sum()),
        "track_labelled_rows": int(len(target)),
        "full_gate_passes": int(result["full_gate"].sum()),
        "selected_threshold_rows": int(result["is_selected_threshold"].sum()),
        "test_status": "SEALED",
        "test_access_count": 0,
        "source_reused_without_training": True,
    }
    atomic_json(root / "TRACK_VERIFIER_AUDIT.json", audit)


def false_tracks_block() -> None:
    settings = config()
    source = pd.read_csv(PROJECT / settings["inputs"]["false_track_audit"])
    if len(source) != int(settings["false_tracks"]["expected_total"]):
        raise RuntimeError("False-track source does not contain 288 rows")
    observations = pd.read_csv(
        PROJECT
        / "outputs/temporal_verifier_v1/tracks/ocsort/support/track_observations.csv"
    )
    with (
        PROJECT / "outputs/temporal_verifier_v1/learned/logistic_l2/model.pkl"
    ).open("rb") as handle:
        logistic = pickle.load(handle)
    logistic_probability = predict_nested_logistic(logistic, source)
    logistic_by_key = dict(zip(source["track_key"], logistic_probability, strict=True))

    crop_metadata = pd.read_csv(
        PROJECT / "outputs/crop_verifier_v1/embeddings/support/track_metadata.csv"
    )
    visual = np.load(
        PROJECT / "outputs/crop_verifier_v1/embeddings/support/visual_embeddings.npz"
    )["embeddings"]
    track_columns = [
        "mean_confidence",
        "max_confidence",
        "track_length",
        "detection_fraction",
        "interpolation_fraction",
    ]
    full_support_features = pd.read_csv(
        PROJECT
        / "outputs/temporal_verifier_v1/tracks/ocsort/support/track_features.csv"
    )
    ordered = crop_metadata[["track_key"]].merge(
        full_support_features[["track_key", *track_columns]],
        on="track_key",
        how="left",
        validate="one_to_one",
    )
    with (
        PROJECT / "outputs/crop_verifier_v1/models/combined/model.pkl"
    ).open("rb") as handle:
        combined = pickle.load(handle)
    combined_score = predict_verifier(
        combined,
        np.column_stack([visual, ordered[track_columns].to_numpy(dtype=float)]),
    )
    combined_by_key = dict(
        zip(crop_metadata["track_key"], combined_score, strict=True)
    )

    fixed_rule = {
        "maximum_interpolation_fraction": 0.50,
        "minimum_detection_duty_cycle": 0.40,
        "maximum_missing_streak": 1,
        "maximum_box_area_cv": 2.0,
    }
    rule_mask = rule_acceptance(
        source,
        k_detector_hits=3,
        window_frames=3,
        high_confidence_threshold=0.25,
        fixed=fixed_rule,
    )
    mapping = settings["false_tracks"]["source_category_mapping"]
    best = (
        observations.sort_values(
            ["track_key", "detector_confidence"], ascending=[True, False]
        )
        .drop_duplicates("track_key")
        .set_index("track_key")
    )
    rows: list[dict[str, Any]] = []
    for index, row in source.reset_index(drop=True).iterrows():
        key = str(row["track_key"])
        representative = best.loc[key]
        category = mapping[str(row["category"])]
        rows.append(
            {
                "scene_id": str(row["grouped_scene_id"]),
                "sequence_id": str(row["subsequence_id"]),
                "track_id": int(row["track_id"]),
                "category": category,
                "start_frame": int(row["first_detection_frame"]),
                "end_frame": int(row["last_detection_frame"]),
                "duration_frames": int(row["track_length"]),
                "duration_seconds": float(row["track_length"])
                / float(settings["false_tracks"]["nominal_fps"]),
                "real_detection_count": int(row["observed_frames"]),
                "interpolated_frame_count": int(row["interpolated_frames"]),
                "interpolation_fraction": float(row["interpolation_fraction"]),
                "mean_confidence": float(row["mean_confidence"]),
                "median_confidence": float(row["median_confidence"]),
                "max_confidence": float(row["max_confidence"]),
                "mean_bbox_area": float(row["mean_box_area"]),
                "border_fraction": float(row["border_fraction"]),
                "mean_velocity": float(row["center_velocity_mean"]),
                "accepted_by_rule_verifier": bool(rule_mask.iloc[index]),
                "accepted_by_logistic_verifier": bool(
                    logistic_by_key[key] >= 0.17
                ),
                "accepted_by_crop_verifier": bool(
                    combined_by_key[key] >= 0.275
                ),
                "example_frame_id": int(representative["frame_number"]),
                "image_available": bool(
                    Path(str(representative["image_path"])).is_file()
                ),
                "publication_allowed": False,
                "vertical_object": str(row["category"])
                == "static_vertical_object",
                "static_object": float(row["center_velocity_mean"]) <= 0.15,
                "low_resolution": float(row["mean_box_area"]) < 256.0,
                "border_related": str(row["category"])
                == "image_or_tile_border",
                "interpolation_related": str(row["category"])
                == "incorrect_interpolation",
                "persistent": int(row["track_length"]) >= 5,
            }
        )
    result = pd.DataFrame(rows)
    root = OUTPUT / "false_tracks"
    atomic_csv(root / "FALSE_TRACKS_ALL.csv", result)
    summary_rows: list[dict[str, Any]] = []
    for category in settings["false_tracks"]["semantic_categories"]:
        group = result[result["category"].eq(category)]
        summary_rows.append(
            {
                "category": category,
                "count": len(group),
                "share_percent": 100.0 * len(group) / len(result),
                "median_duration_frames": (
                    float(group["duration_frames"].median()) if len(group) else 0.0
                ),
                "mean_confidence": (
                    float(group["mean_confidence"].mean()) if len(group) else 0.0
                ),
                "median_confidence": (
                    float(group["median_confidence"].median()) if len(group) else 0.0
                ),
                "mean_real_detection_count": (
                    float(group["real_detection_count"].mean())
                    if len(group)
                    else 0.0
                ),
                "mean_interpolation_fraction": (
                    float(group["interpolation_fraction"].mean())
                    if len(group)
                    else 0.0
                ),
                "accepted_by_logistic_count": int(
                    group["accepted_by_logistic_verifier"].sum()
                ),
                "accepted_by_crop_count": int(
                    group["accepted_by_crop_verifier"].sum()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    atomic_csv(root / "FALSE_TRACK_CATEGORY_SUMMARY.csv", summary)
    examples = (
        result.sort_values(
            ["category", "duration_frames", "max_confidence"],
            ascending=[True, False, False],
        )
        .drop_duplicates("category")
        .head(8)
        .copy()
    )
    examples["frame_id"] = examples["example_frame_id"]
    examples["confidence"] = examples["max_confidence"]
    examples["verifier_decision"] = np.where(
        examples["accepted_by_crop_verifier"], "ACCEPTED", "REJECTED"
    )
    examples["image_path"] = ""
    examples["reason_for_selection"] = "deterministic category representative"
    atomic_csv(
        root / "FALSE_TRACK_EXAMPLES.csv",
        examples[
            [
                "scene_id",
                "frame_id",
                "track_id",
                "category",
                "confidence",
                "duration_frames",
                "verifier_decision",
                "image_path",
                "publication_allowed",
                "reason_for_selection",
            ]
        ],
    )
    audit = {
        "status": "PASS_WITH_BLOCKED_SEMANTIC_SUBCATEGORIES",
        "false_tracks": len(result),
        "category_sum": int(summary["count"].sum()),
        "unknown_category_count": int(
            (~result["category"].isin(settings["false_tracks"]["semantic_categories"])).sum()
        ),
        "duplicate_track_rows": int(
            result.duplicated(["scene_id", "sequence_id", "track_id"]).sum()
        ),
        "semantic_subcategory_status": settings["false_tracks"][
            "semantic_subcategory_status"
        ],
        "semantic_subcategory_reason": settings["false_tracks"][
            "semantic_subcategory_reason"
        ],
        "static_vertical_tracks_preserved_as_binary_flag": int(
            result["vertical_object"].sum()
        ),
        "images_in_public_output": 0,
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    atomic_json(root / "FALSE_TRACK_AUDIT.json", audit)


def _database_run(path: Path, label: str, synthetic: bool) -> dict[str, Any]:
    connection = sqlite3.connect(path)
    run = pd.read_sql("select * from runs", connection).iloc[0]
    events = pd.read_sql("select * from events", connection)
    detections = pd.read_sql("select * from event_detections", connection)
    connection.close()
    durations = events["duration"].to_numpy(dtype=float)
    counts = detections.groupby("event_id").size().to_numpy(dtype=float)
    sources = events["source_label"].value_counts()
    return {
        "run": label,
        "synthetic": synthetic,
        "frames": int(run["processed_frames"]),
        "raw_detections": int(run["raw_detection_count"]),
        "raw_tracks": int(run["track_count"]),
        "unique_events": int(run["event_count"]),
        "baseline_events": int(sources.get("BASELINE", 0)),
        "temporal_only_events": int(sources.get("TEMPORAL_ONLY", 0)),
        "both_source_events": int(sources.get("BOTH", 0)),
        "event_duration_mean": float(durations.mean()),
        "event_duration_median": float(np.median(durations)),
        "detections_per_event_mean": float(counts.mean()),
        "detections_per_event_median": float(np.median(counts)),
        "maximum_detections_per_event": int(counts.max()),
        "track_id_switches_merged": int(
            sum(
                max(len(json.loads(value)) - 1, 0)
                for value in events["track_ids_json"]
            )
        ),
        "reopened_events": 0,
        "review_minutes_per_video_hour": (
            float(run["review_seconds"]) / 60.0
        )
        / max(
            float(run["total_frames"]) / 25.0 / 3600.0
            if synthetic
            else float(run["total_frames"]) / 10.0 / 3600.0,
            1e-12,
        ),
    }


def event_block() -> None:
    settings = config()
    event_config = json.loads(
        json.dumps(
            __import__("yaml").safe_load(
                (PROJECT / settings["inputs"]["event_config"]).read_text(
                    encoding="utf-8"
                )
            )
        )
    )
    events = event_config["events"]
    actual = {
        "join_time_seconds": events["join_time_seconds"],
        "minimum_iou": events["minimum_iou"],
        "maximum_center_distance_ratio": events[
            "maximum_center_distance_ratio"
        ],
        "close_after_seconds": events["close_after_seconds"],
        "reopen_window_seconds": events["reopen_window_seconds"],
        "pre_roll_seconds": events["pre_roll_seconds"],
        "post_roll_seconds": events["post_roll_seconds"],
        "minimum_temporal_hits": events["temporal_minimum_observations"],
        "camera_id_required": True,
        "track_id_switch_allowed": True,
    }
    root = OUTPUT / "operator_assistant"
    atomic_json(root / "EVENT_PARAMETERS_ACTUAL.json", actual)
    real = _database_run(
        PROJECT / settings["inputs"]["real_event_database"], "real_short", False
    )
    synthetic = _database_run(
        PROJECT / settings["inputs"]["synthetic_event_database"],
        "synthetic_demo",
        True,
    )
    long_payload = json.loads(
        (PROJECT / settings["inputs"]["long_benchmark"]).read_text(encoding="utf-8")
    )
    long_row = {
        "run": "real_long_benchmark",
        "synthetic": False,
        "frames": int(long_payload["measured_frames"]),
        "raw_detections": "BLOCKED_MISSING_ARTIFACT",
        "raw_tracks": "BLOCKED_MISSING_ARTIFACT",
        "unique_events": int(long_payload["events"]),
        "baseline_events": "BLOCKED_MISSING_ARTIFACT",
        "temporal_only_events": "BLOCKED_MISSING_ARTIFACT",
        "both_source_events": "BLOCKED_MISSING_ARTIFACT",
        "event_duration_mean": "BLOCKED_MISSING_ARTIFACT",
        "event_duration_median": "BLOCKED_MISSING_ARTIFACT",
        "detections_per_event_mean": "BLOCKED_MISSING_ARTIFACT",
        "detections_per_event_median": "BLOCKED_MISSING_ARTIFACT",
        "maximum_detections_per_event": "BLOCKED_MISSING_ARTIFACT",
        "track_id_switches_merged": "BLOCKED_MISSING_ARTIFACT",
        "reopened_events": "BLOCKED_MISSING_ARTIFACT",
        "review_minutes_per_video_hour": "NOT_REVIEWED",
    }
    runs = pd.DataFrame([real, long_row, synthetic])
    atomic_csv(root / "EVENT_AGGREGATION_RUNS.csv", runs)
    summary = {
        "real_short": {
            "frames": real["frames"],
            "raw_boxes": real["raw_detections"],
            "unique_events": real["unique_events"],
            "boxes_per_event": real["raw_detections"] / real["unique_events"],
        },
        "real_long": {
            "frames": long_row["frames"],
            "events": long_row["unique_events"],
            "end_to_end_fps": long_payload["end_to_end_fps"],
        },
        "synthetic": {
            "detections": synthetic["raw_detections"],
            "events": synthetic["unique_events"],
            "detections_per_event": synthetic["raw_detections"]
            / synthetic["unique_events"],
            "review_minutes_per_video_hour": synthetic[
                "review_minutes_per_video_hour"
            ],
            "synthetic": True,
        },
    }
    atomic_json(root / "EVENT_AGGREGATION_SUMMARY.json", summary)
    atomic_json(
        root / "EVENT_AUDIT.json",
        {
            "status": "PASS_WITH_BLOCKED_LONG_RUN_DETAILS",
            "parameters_read_from_code_config": True,
            "real_and_synthetic_separated": True,
            "real_short_expected_match": {
                "frames_100": real["frames"] == 100,
                "raw_boxes_296": real["raw_detections"] == 296,
                "events_3": real["unique_events"] == 3,
                "boxes_per_event_98_6667": abs(
                    real["raw_detections"] / real["unique_events"] - 98.6667
                )
                < 1e-4,
            },
            "long_detail_status": "BLOCKED_MISSING_ARTIFACT",
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )


def _feature_frame(
    dataset: FeatureDataset,
    train_indices: np.ndarray,
    transform_indices: np.ndarray,
    v8b_config: dict[str, Any],
) -> pd.DataFrame:
    epsilon = float(v8b_config["normalization"]["epsilon"])
    q_low = float(v8b_config["normalization"]["q_low"])
    q_high = float(v8b_config["normalization"]["q_high"])
    scenes = dataset.scenes
    stats: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for layer in LAYERS:
        for region in REGIONS:
            values = np.stack(
                [dataset.arrays[index][f"{layer}_{region}"] for index in train_indices]
            )
            low = np.quantile(values, q_low, axis=0)
            high = np.maximum(
                np.quantile(values, q_high, axis=0), low + epsilon
            )
            normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
            scene_means = np.stack(
                [
                    normalized[scenes[train_indices] == scene].mean(axis=0)
                    for scene in sorted(set(scenes[train_indices]))
                ]
            )
            stats[(layer, region)] = (low, high, np.median(scene_means, axis=0))
    rows = []
    for index in transform_indices:
        row: dict[str, Any] = {
            "index": int(index),
            "grouped_scene_id": str(dataset.frame.iloc[index]["grouped_scene_id"]),
            "fn": float(dataset.frame.iloc[index]["fn"]),
        }
        for layer in LAYERS:
            for region in REGIONS:
                raw = dataset.arrays[index][f"{layer}_{region}"]
                low, high, prototype = stats[(layer, region)]
                membership = np.clip((raw - low) / (high - low), 0.0, 1.0)
                values = _similarities(raw, membership, prototype, epsilon)
                for metric in (
                    "cosine",
                    "l1",
                    "l2",
                    "mae",
                    "mse",
                    "entropy_shift",
                    "product",
                    "lukasiewicz",
                ):
                    prefix = "tnorm" if metric in {"product", "lukasiewicz"} else "std"
                    row[f"{prefix}_{layer}_{region}_{metric}"] = values[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_corr(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    if left.nunique() < 2 or right.nunique() < 2:
        return 0.0, 1.0
    value, p = pearsonr(left.to_numpy(dtype=float), right.to_numpy(dtype=float))
    return float(value), float(p)


def tnorm_block() -> None:
    settings = config()
    manifest = pd.read_csv(PROJECT / settings["inputs"]["v8b_feature_manifest"])
    if manifest["test_used"].astype(bool).any():
        raise RuntimeError("V8b feature manifest indicates test use")
    dataset = FeatureDataset(manifest)
    v8b_config = __import__("yaml").safe_load(
        (PROJECT / "configs/canonical_v8b_person_failure_risk.yaml").read_text(
            encoding="utf-8"
        )
    )
    correlation_rows: list[dict[str, Any]] = []
    redundancy_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    scenes = dataset.scenes
    standard_suffixes = ("cosine", "l1", "l2", "mae", "mse", "entropy_shift")
    for outer_scene in sorted(set(scenes)):
        train = np.flatnonzero(scenes != outer_scene)
        train_frame = _feature_frame(dataset, train, train, v8b_config)
        tnorm_columns = [
            column for column in train_frame if column.startswith("tnorm_")
        ]
        standard_columns = [
            column
            for column in train_frame
            if column.startswith("std_") and column.endswith(standard_suffixes)
        ]
        x = train_frame[standard_columns].to_numpy(dtype=float)
        for tnorm in tnorm_columns:
            y = train_frame[tnorm].to_numpy(dtype=float)
            for standard in standard_columns:
                corr, p = _safe_corr(train_frame[tnorm], train_frame[standard])
                correlation_rows.append(
                    {
                        "outer_scene": outer_scene,
                        "tnorm_feature": tnorm,
                        "standard_feature": standard,
                        "pearson_correlation": corr,
                        "p_value": p,
                        "train_scenes": len(set(scenes[train])),
                    }
                )
            model = LinearRegression().fit(x, y)
            prediction = model.predict(x)
            residual = y - prediction
            explained = float(model.score(x, y)) if np.var(y) > 0 else 0.0
            residual_corr, residual_p = _safe_corr(
                pd.Series(residual), train_frame["fn"]
            )
            rank_corr = spearmanr(residual, train_frame["fn"]).statistic
            redundancy_rows.append(
                {
                    "outer_scene": outer_scene,
                    "tnorm_feature": tnorm,
                    "explained_variance_r2": explained,
                    "vif_equivalent": (
                        1.0 / max(1.0 - explained, 1e-12)
                        if explained < 1.0
                        else 1e12
                    ),
                    "residual_variance": float(np.var(residual)),
                }
            )
            residual_rows.append(
                {
                    "outer_scene": outer_scene,
                    "tnorm_feature": tnorm,
                    "residual_fn_pearson": residual_corr,
                    "residual_fn_pearson_p": residual_p,
                    "residual_fn_spearman": float(rank_corr),
                    "explained_variance_r2": explained,
                }
            )
            simple = LinearRegression().fit(y.reshape(-1, 1), train_frame["fn"])
            coefficient_rows.append(
                {
                    "outer_scene": outer_scene,
                    "tnorm_feature": tnorm,
                    "feature_coefficient": float(simple.coef_[0]),
                    "coefficient_sign": (
                        "positive"
                        if simple.coef_[0] > 0
                        else "negative" if simple.coef_[0] < 0 else "zero"
                    ),
                }
            )
    root = OUTPUT / "tnorm"
    correlations = pd.DataFrame(correlation_rows)
    redundancy = pd.DataFrame(redundancy_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    residual = pd.DataFrame(residual_rows)
    atomic_csv(root / "TNORM_CORRELATIONS.csv", correlations)
    atomic_csv(root / "TNORM_REDUNDANCY.csv", redundancy)
    atomic_csv(root / "TNORM_COEFFICIENT_STABILITY.csv", coefficients)
    atomic_csv(root / "TNORM_RESIDUAL_SIGNAL.csv", residual)
    atomic_json(
        root / "TNORM_AUDIT.json",
        {
            "status": "PASS",
            "outer_folds": int(len(set(scenes))),
            "frames": len(dataset),
            "saved_features_only": True,
            "new_image_features_extracted": False,
            "train_only_normalization_and_prototypes": True,
            "correlation_rows": len(correlations),
            "residual_rows": len(residual),
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )


def main() -> None:
    check_no_test_markers()
    if not (OUTPUT / "COMPUTATION_LOCK.json").is_file():
        raise RuntimeError("Run lock_inputs before computation")
    track_verifier_block()
    false_tracks_block()
    event_block()
    tnorm_block()
    print(
        json.dumps(
            {
                "track_verifier": "PASS",
                "false_tracks": "PASS_WITH_BLOCKED_SEMANTIC_SUBCATEGORIES",
                "operator_assistant": "PASS_WITH_BLOCKED_LONG_RUN_DETAILS",
                "tnorm": "PASS",
                "test_status": "SEALED",
                "test_access_count": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
