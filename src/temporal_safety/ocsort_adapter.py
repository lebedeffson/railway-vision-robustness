from __future__ import annotations

from typing import Any

import numpy as np

from src.temporal.ratta import box_center, box_iou

from .tracker_base import CausalTracker, TrackState


class OCSortAdapter(CausalTracker):
    """Observation-centric adapter with constant-velocity association."""

    name = "ocsort"

    def predicted_box(self, track: TrackState) -> list[float]:
        if track.previous_box is None:
            return list(track.box)
        current = np.asarray(track.box, dtype=float)
        previous = np.asarray(track.previous_box, dtype=float)
        return (current + (current - previous)).tolist()

    def association_cost(
        self, track: TrackState, predicted: list[float], detection: dict[str, Any]
    ) -> float:
        overlap_cost = 1.0 - box_iou(predicted, detection["box"])
        if track.previous_box is None:
            return overlap_cost
        old = np.asarray(box_center(track.previous_box), dtype=float)
        current = np.asarray(box_center(track.box), dtype=float)
        candidate = np.asarray(box_center(detection["box"]), dtype=float)
        velocity = current - old
        observation = candidate - current
        denominator = float(np.linalg.norm(velocity) * np.linalg.norm(observation))
        direction_cost = 0.0 if denominator <= 1e-12 else 0.5 * (
            1.0 - float(np.clip(np.dot(velocity, observation) / denominator, -1.0, 1.0))
        )
        weight = float(self.parameters.get("direction_weight", 0.0))
        return (1.0 - weight) * overlap_cost + weight * direction_cost

