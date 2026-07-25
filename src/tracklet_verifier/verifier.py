from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin


FEATURE_NAMES = (
    "confidence_mean",
    "confidence_median",
    "confidence_max",
    "confidence_q10",
    "confidence_q25",
    "confidence_q75",
    "confidence_slope",
    "count_above_0_01",
    "count_above_0_03",
    "count_above_0_05",
    "track_length",
    "hits",
    "gaps",
    "max_consecutive_hits",
    "detection_fraction",
    "velocity_variance",
    "acceleration_variance",
    "compensated_iou_mean",
    "compensated_iou_median",
    "center_motion_residual",
    "scale_variation",
    "aspect_ratio_variation",
    "border_fraction",
    "box_area_mean",
    "appearance_cosine_mean",
    "appearance_cosine_min",
    "embedding_variance",
    "product_reliability",
    "lukasiewicz_reliability",
    "membership_min",
    "membership_mean",
    "membership_variance",
)

MONOTONIC_DIRECTIONS = np.asarray(
    [
        1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, -1, 1, 1, -1,
        -1, 1, 1, -1, -1, -1, -1, 0, 1, 1, -1, 1, 1, 1, 1, -1,
    ],
    dtype=np.int8,
)


def _safe(values: pd.Series | np.ndarray, default: float = 0.0) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    finite = result[np.isfinite(result)]
    replacement = float(np.median(finite)) if finite.size else default
    return np.nan_to_num(result, nan=replacement, posinf=replacement, neginf=replacement)


def _maximum_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def _variation(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    return float(np.std(values) / max(abs(np.mean(values)), 1e-6))


def build_tracklet_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate causal observation rows into one deterministic row per tracklet."""
    required = {
        "track_key", "grouped_scene_id", "frame_order", "detector_confidence",
        "is_detection", "x1", "y1", "x2", "y2", "image_width", "image_height",
        "mu_motion", "mu_appearance", "mu_scale", "mu_border",
        "mu_compensated_iou",
    }
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Tracklet observations miss columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    embedding_columns = sorted(
        [column for column in observations if column.startswith("embedding_")],
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    for track_key, group in observations.groupby("track_key", sort=True):
        group = group.sort_values("frame_order").reset_index(drop=True)
        detected = group["is_detection"].astype(bool).to_numpy()
        confidence = group.loc[detected, "detector_confidence"].astype(float).to_numpy()
        confidence = confidence[np.isfinite(confidence)]
        if confidence.size == 0:
            confidence = np.asarray([0.0])
        width = np.maximum(group["x2"].to_numpy() - group["x1"].to_numpy(), 1e-6)
        height = np.maximum(group["y2"].to_numpy() - group["y1"].to_numpy(), 1e-6)
        area = width * height
        cx = (group["x1"].to_numpy() + group["x2"].to_numpy()) / 2.0
        cy = (group["y1"].to_numpy() + group["y2"].to_numpy()) / 2.0
        scale = np.sqrt(area)
        if len(group) > 1:
            dx = np.diff(cx) / np.maximum(scale[:-1], 4.0)
            dy = np.diff(cy) / np.maximum(scale[:-1], 4.0)
            velocity = np.hypot(dx, dy)
        else:
            velocity = np.asarray([0.0])
        acceleration = np.diff(velocity) if len(velocity) > 1 else np.asarray([0.0])
        border_distance = np.minimum.reduce(
            [
                group["x1"].to_numpy(),
                group["y1"].to_numpy(),
                group["image_width"].to_numpy() - group["x2"].to_numpy(),
                group["image_height"].to_numpy() - group["y2"].to_numpy(),
            ]
        )
        border_fraction = float(np.mean(border_distance < np.maximum(scale, 4.0)))
        member_matrix = group[
            ["mu_motion", "mu_appearance", "mu_scale", "mu_border"]
        ].to_numpy(dtype=float)
        valid_members = member_matrix[np.isfinite(member_matrix).all(axis=1)]
        if valid_members.size == 0:
            valid_members = np.zeros((1, 4), dtype=float)
        persistence = float(np.mean(detected))
        fuzzy = np.column_stack(
            [valid_members, np.full(len(valid_members), persistence)]
        )
        product = np.prod(fuzzy, axis=1)
        lukasiewicz = np.maximum(0.0, np.sum(fuzzy, axis=1) - (fuzzy.shape[1] - 1))
        appearance = valid_members[:, 1]
        compensated = _safe(group["mu_compensated_iou"], 0.0)
        center_residual = 1.0 - _safe(group["mu_motion"], 0.0)
        embedding_variance = 0.0
        if embedding_columns:
            embeddings = group[embedding_columns].to_numpy(dtype=float)
            if len(embeddings) > 1:
                embedding_variance = float(np.mean(np.var(embeddings, axis=0)))
        row = {
            "track_key": str(track_key),
            "grouped_scene_id": str(group["grouped_scene_id"].iloc[0]),
            "subsequence_id": str(group["subsequence_id"].iloc[0]),
            "confidence_mean": float(np.mean(confidence)),
            "confidence_median": float(np.median(confidence)),
            "confidence_max": float(np.max(confidence)),
            "confidence_q10": float(np.quantile(confidence, 0.10)),
            "confidence_q25": float(np.quantile(confidence, 0.25)),
            "confidence_q75": float(np.quantile(confidence, 0.75)),
            "confidence_slope": _slope(confidence),
            "count_above_0_01": int(np.sum(confidence >= 0.01)),
            "count_above_0_03": int(np.sum(confidence >= 0.03)),
            "count_above_0_05": int(np.sum(confidence >= 0.05)),
            "track_length": int(len(group)),
            "hits": int(np.sum(detected)),
            "gaps": int(np.sum(~detected)),
            "max_consecutive_hits": _maximum_run(detected),
            "detection_fraction": persistence,
            "velocity_variance": float(np.var(velocity)),
            "acceleration_variance": float(np.var(acceleration)),
            "compensated_iou_mean": float(np.mean(compensated)),
            "compensated_iou_median": float(np.median(compensated)),
            "center_motion_residual": float(np.mean(center_residual)),
            "scale_variation": _variation(area),
            "aspect_ratio_variation": _variation(width / height),
            "border_fraction": border_fraction,
            "box_area_mean": float(np.mean(area)),
            "appearance_cosine_mean": float(np.mean(appearance)),
            "appearance_cosine_min": float(np.min(appearance)),
            "embedding_variance": embedding_variance,
            "product_reliability": float(np.mean(product)),
            "lukasiewicz_reliability": float(np.mean(lukasiewicz)),
            "membership_min": float(np.min(fuzzy)),
            "membership_mean": float(np.mean(fuzzy)),
            "membership_variance": float(np.var(fuzzy)),
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["track_key", "grouped_scene_id", *FEATURE_NAMES])
    if not np.isfinite(result[list(FEATURE_NAMES)].to_numpy(dtype=float)).all():
        raise RuntimeError("Tracklet feature extraction produced NaN/Inf")
    return result


def label_tracklets(
    observations: pd.DataFrame,
    minimum_iou: float = 0.50,
    minimum_matched_frames: int = 2,
    minimum_matched_fraction: float = 0.50,
    negative_iou: float = 0.30,
) -> pd.DataFrame:
    rows = []
    for track_key, group in observations.groupby("track_key", sort=True):
        detected = group[group["is_detection"].astype(bool)]
        denominator = max(len(detected), 1)
        positive_frames = int((detected["gt_iou"] >= minimum_iou).sum())
        fraction = positive_frames / denominator
        maximum = float(group["gt_iou"].max())
        if positive_frames >= minimum_matched_frames and fraction >= minimum_matched_fraction:
            label = "positive"
            target = 1
        elif maximum < negative_iou:
            label = "negative"
            target = 0
        else:
            label = "ambiguous"
            target = -1
        rows.append(
            {
                "track_key": str(track_key),
                "label": label,
                "target": target,
                "positive_frames": positive_frames,
                "detected_frames": int(len(detected)),
                "matched_fraction": fraction,
                "maximum_gt_iou": maximum,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class MonotoneRankLogistic(BaseEstimator, ClassifierMixin):
    directions: np.ndarray
    epochs: int = 400
    learning_rate: float = 0.03
    weight_decay: float = 1e-4
    ranking_weight: float = 0.25
    maximum_pairs_per_epoch: int = 4096
    seed: int = 20260725

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MonotoneRankLogistic":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError("MonotoneRankLogistic requires both classes")
        self.classes_ = np.asarray([0, 1])
        self.median_ = np.median(x, axis=0).astype(np.float32)
        q25, q75 = np.quantile(x, [0.25, 0.75], axis=0)
        self.scale_ = np.maximum(q75 - q25, 1e-6).astype(np.float32)
        standardized = (x - self.median_) / self.scale_
        tensor_x = torch.tensor(standardized, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32)
        directions = torch.tensor(self.directions.astype(np.float32))
        generator = torch.Generator().manual_seed(self.seed)
        raw = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float32))
        bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
        optimizer = torch.optim.AdamW(
            [raw, bias], lr=self.learning_rate, weight_decay=self.weight_decay
        )
        positive = torch.where(tensor_y == 1)[0]
        negative = torch.where(tensor_y == 0)[0]
        positive_weight = torch.tensor(
            max(float(len(negative)) / max(len(positive), 1), 1.0)
        )
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            signed = torch.where(
                directions == 0,
                raw,
                directions * torch.nn.functional.softplus(raw),
            )
            logits = tensor_x @ signed + bias
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, tensor_y, pos_weight=positive_weight
            )
            pair_count = min(
                self.maximum_pairs_per_epoch,
                max(len(positive), len(negative)),
            )
            pos_index = positive[
                torch.randint(len(positive), (pair_count,), generator=generator)
            ]
            neg_index = negative[
                torch.randint(len(negative), (pair_count,), generator=generator)
            ]
            ranking = torch.nn.functional.softplus(
                -(logits[pos_index] - logits[neg_index])
            ).mean()
            loss = bce + self.ranking_weight * ranking
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            signed = torch.where(
                directions == 0,
                raw,
                directions * torch.nn.functional.softplus(raw),
            )
            self.coef_ = signed.numpy().astype(np.float64)
            self.intercept_ = float(bias)
        if not np.isfinite(self.coef_).all() or not np.isfinite(self.intercept_):
            raise RuntimeError("Monotone logistic training produced NaN/Inf")
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        standardized = (
            np.asarray(x, dtype=np.float64) - self.median_
        ) / self.scale_
        return standardized @ self.coef_ + self.intercept_

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = np.clip(self.decision_function(x), -60.0, 60.0)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])

