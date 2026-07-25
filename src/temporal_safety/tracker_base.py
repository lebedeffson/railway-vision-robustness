from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.temporal.ratta import box_iou


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, width))
    y1 = float(np.clip(y1, 0, height))
    x2 = float(np.clip(x2, x1, width))
    y2 = float(np.clip(y2, y1, height))
    return [x1, y1, x2, y2]


@dataclass
class TrackState:
    track_id: int
    box: list[float]
    previous_box: list[float] | None
    hits: int
    age: int
    missing_streak: int
    confidence_history: list[float] = field(default_factory=list)
    recent_hits: list[int] = field(default_factory=list)
    accumulated_score: float = 0.0
    confirmed: bool = False
    source_subsequence: str = ""


class CausalTracker:
    """Small auditable ByteTrack/OC-SORT-style tracker.

    It consumes only detector boxes and scores. Ground truth is never accepted by
    the inference API. The two concrete adapters differ only in motion prediction
    and association cost.
    """

    name = "base"

    def __init__(self, parameters: dict[str, Any], logic: dict[str, Any]) -> None:
        self.parameters = parameters
        self.logic = logic
        self.tracks: list[TrackState] = []
        self.next_track_id = 1

    def reset(self, subsequence_id: str = "") -> None:
        self.tracks = []
        self.next_track_id = 1
        self.subsequence_id = subsequence_id

    def predicted_box(self, track: TrackState) -> list[float]:
        return list(track.box)

    def association_cost(
        self, track: TrackState, predicted: list[float], detection: dict[str, Any]
    ) -> float:
        return 1.0 - box_iou(predicted, detection["box"])

    def _associate(
        self,
        track_indices: list[int],
        detection_indices: list[int],
        detections: list[dict[str, Any]],
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        if not track_indices or not detection_indices:
            return [], set(track_indices), set(detection_indices)
        predicted = {index: self.predicted_box(self.tracks[index]) for index in track_indices}
        costs = np.asarray(
            [
                [
                    self.association_cost(
                        self.tracks[track_index],
                        predicted[track_index],
                        detections[detection_index],
                    )
                    for detection_index in detection_indices
                ]
                for track_index in track_indices
            ],
            dtype=float,
        )
        left, right = linear_sum_assignment(costs)
        maximum = 1.0 - float(self.parameters["match_iou_threshold"])
        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(track_indices)
        unmatched_detections = set(detection_indices)
        for row, column in zip(left, right):
            if costs[row, column] <= maximum:
                track_index = track_indices[int(row)]
                detection_index = detection_indices[int(column)]
                matches.append((track_index, detection_index))
                unmatched_tracks.discard(track_index)
                unmatched_detections.discard(detection_index)
        return matches, unmatched_tracks, unmatched_detections

    def _update_track(self, track: TrackState, detection: dict[str, Any]) -> None:
        confidence = float(detection["confidence"])
        track.previous_box = list(track.box)
        track.box = list(detection["box"])
        track.hits += 1
        track.age += 1
        track.missing_streak = 0
        track.confidence_history.append(confidence)
        window = int(self.logic["confirmation_window"])
        track.recent_hits = [*track.recent_hits, 1][-window:]
        track.accumulated_score = (
            float(self.logic["accumulated_score_decay"]) * track.accumulated_score
            + confidence
        )
        track.confirmed = track.confirmed or (
            confidence >= float(self.parameters["high_conf_threshold"])
            or sum(track.recent_hits) >= int(self.parameters["minimum_confirmed_length"])
            or track.accumulated_score >= float(self.logic["accumulated_score_threshold"])
        )

    def _create_track(self, detection: dict[str, Any]) -> TrackState:
        confidence = float(detection["confidence"])
        track = TrackState(
            track_id=self.next_track_id,
            box=list(detection["box"]),
            previous_box=None,
            hits=1,
            age=1,
            missing_streak=0,
            confidence_history=[confidence],
            recent_hits=[1],
            accumulated_score=confidence,
            confirmed=confidence >= float(self.parameters["high_conf_threshold"]),
            source_subsequence=self.subsequence_id,
        )
        self.next_track_id += 1
        self.tracks.append(track)
        return track

    def _emit_detection(
        self, track: TrackState, detection: dict[str, Any]
    ) -> dict[str, Any] | None:
        raw = float(detection["confidence"])
        operating = float(self.parameters["high_conf_threshold"])
        if raw < operating and not track.confirmed:
            return None
        score = raw
        if track.confirmed and raw < operating:
            score = min(0.999, operating + track.accumulated_score)
        return {
            "box": list(detection["box"]),
            "confidence": float(score),
            "raw_confidence": raw,
            "class_id": 0,
            "track_id": track.track_id,
            "source": "detector" if raw >= operating else "temporal_confirmation",
            "interpolated": False,
        }

    def _emit_interpolation(
        self, track: TrackState, width: int, height: int
    ) -> dict[str, Any] | None:
        if not track.confirmed:
            return None
        if track.missing_streak > int(self.logic["maximum_interpolation_gap"]):
            return None
        if track.missing_streak > int(self.parameters["maximum_lost_frames"]):
            return None
        predicted = clip_box(self.predicted_box(track), width, height)
        if predicted[2] <= predicted[0] or predicted[3] <= predicted[1]:
            return None
        uncertainty = track.missing_streak / max(
            int(self.parameters["maximum_lost_frames"]), 1
        )
        if uncertainty > float(self.logic["interpolation_uncertainty_max"]):
            return None
        confidence = max(track.confidence_history[-1], float(self.parameters["high_conf_threshold"]))
        confidence *= float(self.logic["interpolation_score_decay"]) ** track.missing_streak
        return {
            "box": predicted,
            "confidence": float(confidence),
            "raw_confidence": 0.0,
            "class_id": 0,
            "track_id": track.track_id,
            "source": "temporal_interpolation",
            "interpolated": True,
        }

    def update(
        self, detections: list[dict[str, Any]], width: int, height: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        filtered = [
            row
            for row in detections
            if float(row["confidence"]) >= float(self.parameters["low_conf_threshold"])
        ]
        high = [
            index
            for index, row in enumerate(filtered)
            if float(row["confidence"]) >= float(self.parameters["high_conf_threshold"])
        ]
        low = [index for index in range(len(filtered)) if index not in set(high)]
        all_tracks = list(range(len(self.tracks)))
        first, unmatched_tracks, unmatched_high = self._associate(
            all_tracks, high, filtered
        )
        second, unmatched_tracks, unmatched_low = self._associate(
            sorted(unmatched_tracks), low, filtered
        )
        matches = [*first, *second]
        emitted: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            self._update_track(track, filtered[detection_index])
            output = self._emit_detection(track, filtered[detection_index])
            if output is not None:
                emitted.append(output)
            events.append(
                {
                    "track_id": track.track_id,
                    "event": "matched",
                    "confirmed": track.confirmed,
                    "hits": track.hits,
                    "missing_streak": track.missing_streak,
                }
            )
        unmatched_detection_indices = sorted(unmatched_high | unmatched_low)
        for detection_index in unmatched_detection_indices:
            detection = filtered[detection_index]
            if float(detection["confidence"]) < float(
                self.parameters["new_track_threshold"]
            ):
                continue
            track = self._create_track(detection)
            output = self._emit_detection(track, detection)
            if output is not None:
                emitted.append(output)
            events.append(
                {
                    "track_id": track.track_id,
                    "event": "created",
                    "confirmed": track.confirmed,
                    "hits": track.hits,
                    "missing_streak": 0,
                }
            )
        retained: list[TrackState] = []
        for index, track in enumerate(self.tracks):
            if index not in unmatched_tracks:
                retained.append(track)
                continue
            track.age += 1
            track.missing_streak += 1
            window = int(self.logic["confirmation_window"])
            track.recent_hits = [*track.recent_hits, 0][-window:]
            track.accumulated_score *= float(self.logic["missing_score_decay"])
            interpolation = self._emit_interpolation(track, width, height)
            if interpolation is not None:
                emitted.append(interpolation)
            events.append(
                {
                    "track_id": track.track_id,
                    "event": "missing",
                    "confirmed": track.confirmed,
                    "hits": track.hits,
                    "missing_streak": track.missing_streak,
                }
            )
            if track.missing_streak <= int(self.parameters["track_buffer"]):
                retained.append(track)
        self.tracks = retained
        return emitted, events

