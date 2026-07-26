from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "track_length",
    "observed_frames",
    "interpolated_frames",
    "detection_fraction",
    "maximum_missing_streak",
    "mean_confidence",
    "median_confidence",
    "max_confidence",
    "minimum_confidence",
    "confidence_std",
    "confidence_slope",
    "high_confidence_hits",
    "low_confidence_hits",
    "mean_box_area",
    "box_area_std",
    "box_area_cv",
    "aspect_ratio_mean",
    "aspect_ratio_std",
    "center_velocity_mean",
    "center_velocity_std",
    "acceleration_mean",
    "motion_residual",
    "border_fraction",
    "track_score",
    "frames_with_confidence_above_0_1",
    "frames_with_confidence_above_0_25",
    "frames_with_confidence_above_0_5",
    "maximum_consecutive_detector_hits",
    "max_hits_window_3",
    "max_hits_window_5",
    "interpolation_fraction",
    "duplicate_overlap_fraction",
)


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0])


def _maximum_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _max_hits_in_window(values: np.ndarray, window: int) -> int:
    if len(values) <= window:
        return int(values.sum())
    cumulative = np.concatenate([[0], np.cumsum(values.astype(int))])
    return int(np.max(cumulative[window:] - cumulative[:-window]))


def build_track_features(observations: pd.DataFrame) -> pd.DataFrame:
    required = {
        "track_key",
        "grouped_scene_id",
        "subsequence_id",
        "frame_number",
        "is_detector_hit",
        "interpolated",
        "detector_confidence",
        "tracker_score",
        "x1",
        "y1",
        "x2",
        "y2",
        "image_width",
        "image_height",
        "duplicate_overlap",
    }
    missing = required - set(observations)
    if missing:
        raise ValueError(f"Track observations miss columns: {sorted(missing)}")
    if "gt_iou" in FEATURE_NAMES:
        raise RuntimeError("Ground truth leaked into verifier features")
    rows: list[dict[str, Any]] = []
    for track_key, group in observations.groupby("track_key", sort=True):
        group = group.sort_values("frame_number").reset_index(drop=True)
        detected = group["is_detector_hit"].astype(bool).to_numpy()
        interpolated = group["interpolated"].astype(bool).to_numpy()
        confidences = group.loc[detected, "detector_confidence"].to_numpy(dtype=float)
        if confidences.size == 0:
            confidences = np.asarray([0.0])
        width = np.maximum(
            group["x2"].to_numpy(dtype=float)
            - group["x1"].to_numpy(dtype=float),
            1e-6,
        )
        height = np.maximum(
            group["y2"].to_numpy(dtype=float)
            - group["y1"].to_numpy(dtype=float),
            1e-6,
        )
        area = width * height
        cx = (
            group["x1"].to_numpy(dtype=float)
            + group["x2"].to_numpy(dtype=float)
        ) / 2
        cy = (
            group["y1"].to_numpy(dtype=float)
            + group["y2"].to_numpy(dtype=float)
        ) / 2
        scale = np.maximum(np.sqrt(area[:-1]), 4.0)
        if len(group) > 1:
            velocity = np.hypot(np.diff(cx), np.diff(cy)) / scale
        else:
            velocity = np.asarray([0.0])
        acceleration = (
            np.abs(np.diff(velocity)) if len(velocity) > 1 else np.asarray([0.0])
        )
        border_distance = np.minimum.reduce([
            group["x1"].to_numpy(dtype=float),
            group["y1"].to_numpy(dtype=float),
            group["image_width"].to_numpy(dtype=float)
            - group["x2"].to_numpy(dtype=float),
            group["image_height"].to_numpy(dtype=float)
            - group["y2"].to_numpy(dtype=float),
        ])
        frame_min = int(group["frame_number"].min())
        frame_max = int(group["frame_number"].max())
        track_length = frame_max - frame_min + 1
        observed = int(detected.sum())
        interpolation_count = int(interpolated.sum())
        area_mean = float(np.mean(area))
        row = {
            "track_key": str(track_key),
            "scene_id": str(group["grouped_scene_id"].iloc[0]),
            "grouped_scene_id": str(group["grouped_scene_id"].iloc[0]),
            "subsequence_id": str(group["subsequence_id"].iloc[0]),
            "track_id": int(group["track_id"].iloc[0]),
            "track_length": int(track_length),
            "observed_frames": observed,
            "interpolated_frames": interpolation_count,
            "detection_fraction": observed / max(track_length, 1),
            "maximum_missing_streak": int(group["missing_streak"].max()),
            "mean_confidence": float(np.mean(confidences)),
            "median_confidence": float(np.median(confidences)),
            "max_confidence": float(np.max(confidences)),
            "minimum_confidence": float(np.min(confidences)),
            "confidence_std": float(np.std(confidences)),
            "confidence_slope": _slope(confidences),
            "high_confidence_hits": int(np.sum(confidences >= 0.07)),
            "low_confidence_hits": int(np.sum(confidences < 0.07)),
            "mean_box_area": area_mean,
            "box_area_std": float(np.std(area)),
            "box_area_cv": float(np.std(area) / max(area_mean, 1e-6)),
            "aspect_ratio_mean": float(np.mean(width / height)),
            "aspect_ratio_std": float(np.std(width / height)),
            "center_velocity_mean": float(np.mean(velocity)),
            "center_velocity_std": float(np.std(velocity)),
            "acceleration_mean": float(np.mean(acceleration)),
            "motion_residual": float(np.mean(np.abs(velocity - np.median(velocity)))),
            "border_fraction": float(np.mean(border_distance <= np.sqrt(area))),
            "first_detection_frame": int(
                group.loc[group["is_detector_hit"].astype(bool), "frame_number"].min()
            ) if observed else -1,
            "last_detection_frame": int(
                group.loc[group["is_detector_hit"].astype(bool), "frame_number"].max()
            ) if observed else -1,
            "track_score": float(group["tracker_score"].max()),
            "frames_with_confidence_above_0_1": int(np.sum(confidences >= 0.10)),
            "frames_with_confidence_above_0_25": int(np.sum(confidences >= 0.25)),
            "frames_with_confidence_above_0_5": int(np.sum(confidences >= 0.50)),
            "maximum_consecutive_detector_hits": _maximum_run(detected),
            "max_hits_window_3": _max_hits_in_window(detected, 3),
            "max_hits_window_5": _max_hits_in_window(detected, 5),
            "interpolation_fraction": interpolation_count / max(track_length, 1),
            "duplicate_overlap_fraction": float(
                group["duplicate_overlap"].astype(bool).mean()
            ),
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=["track_key", "grouped_scene_id", *FEATURE_NAMES]
        )
    matrix = result[list(FEATURE_NAMES)].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Track feature table contains NaN/Inf")
    return result


def label_tracks(
    observations: pd.DataFrame,
    minimum_iou: float,
    minimum_matched_detector_frames: int,
    minimum_matched_fraction: float,
    maximum_negative_iou: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for track_key, group in observations.groupby("track_key", sort=True):
        detected = group[group["is_detector_hit"].astype(bool)]
        matched = int((detected["gt_iou"] >= minimum_iou).sum())
        fraction = matched / max(len(detected), 1)
        maximum = float(group["gt_iou"].max())
        if (
            matched >= minimum_matched_detector_frames
            and fraction >= minimum_matched_fraction
        ):
            label, target = "positive", 1
        elif maximum < maximum_negative_iou:
            label, target = "negative", 0
        else:
            label, target = "ambiguous", -1
        rows.append({
            "track_key": str(track_key),
            "label": label,
            "target": target,
            "matched_detector_frames": matched,
            "detector_frames": int(len(detected)),
            "matched_fraction": fraction,
            "maximum_gt_iou": maximum,
        })
    return pd.DataFrame(rows)


def rule_acceptance(
    features: pd.DataFrame,
    k_detector_hits: int,
    window_frames: int,
    high_confidence_threshold: float,
    fixed: dict[str, float],
) -> pd.Series:
    hits_column = f"max_hits_window_{window_frames}"
    return (
        (features[hits_column] >= k_detector_hits)
        & (features["max_confidence"] >= high_confidence_threshold)
        & (features["track_length"] >= k_detector_hits)
        & (
            features["interpolation_fraction"]
            <= float(fixed["maximum_interpolation_fraction"])
        )
        & (
            features["detection_fraction"]
            >= float(fixed["minimum_detection_duty_cycle"])
        )
        & (
            features["maximum_missing_streak"]
            <= int(fixed["maximum_missing_streak"])
        )
        & (features["box_area_cv"] <= float(fixed["maximum_box_area_cv"]))
    )
