from __future__ import annotations

import math
from typing import Any

from .models import EventDetection, ReviewEvent


def box_iou(a: list[float], b: list[float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    return intersection / max(area_a + area_b - intersection, 1e-12)


def center_distance_ratio(
    a: list[float], b: list[float], width: int, height: int
) -> float:
    center_a = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    center_b = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    distance = math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    return distance / max(math.hypot(width, height), 1e-12)


class EventAggregator:
    """Stateful spatial-temporal deduplication without autonomous alarm output."""

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        video_id: str,
        camera_id: str,
        frame_width: int,
        frame_height: int,
    ) -> None:
        self.settings = settings
        self.video_id = str(video_id)
        self.camera_id = str(camera_id)
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.events: dict[str, ReviewEvent] = {}
        self._counter = 0

    def _new_event(self, detection: EventDetection) -> ReviewEvent:
        self._counter += 1
        event = ReviewEvent(
            event_id=f"E{self._counter:06d}",
            video_id=self.video_id,
            camera_id=self.camera_id,
            start_time=detection.timestamp,
            end_time=detection.timestamp,
            representative_box=list(detection.box),
        )
        event.add(detection)
        self.events[event.event_id] = event
        return event

    def _spatial_match(self, event: ReviewEvent, detection: EventDetection) -> bool:
        return (
            box_iou(event.representative_box, detection.box)
            >= float(self.settings["minimum_iou"])
            or center_distance_ratio(
                event.representative_box,
                detection.box,
                self.frame_width,
                self.frame_height,
            )
            <= float(self.settings["maximum_center_distance_ratio"])
        )

    def _match(
        self, detection: EventDetection, allowed_states: set[str]
    ) -> ReviewEvent | None:
        candidates: list[tuple[float, float, ReviewEvent]] = []
        for event in self.events.values():
            if event.state not in allowed_states or event.camera_id != self.camera_id:
                continue
            time_gap = detection.timestamp - event.end_time
            if event.state == "CLOSED":
                if not (0.0 <= time_gap <= float(self.settings["reopen_window_seconds"])):
                    continue
            elif not (0.0 <= time_gap <= float(self.settings["join_time_seconds"])):
                continue
            if not self._spatial_match(event, detection):
                continue
            candidates.append(
                (
                    box_iou(event.representative_box, detection.box),
                    -center_distance_ratio(
                        event.representative_box,
                        detection.box,
                        self.frame_width,
                        self.frame_height,
                    ),
                    event,
                )
            )
        return max(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None

    def _promote_if_ready(self, event: ReviewEvent) -> None:
        has_baseline = any(
            item.source in {"BASELINE", "BOTH"} for item in event.detections
        )
        distinct_temporal_frames = {
            item.frame_number
            for item in event.detections
            if item.source in {"TEMPORAL_ONLY", "BOTH"}
        }
        frozen_confirmation = any(item.confirmed for item in event.detections)
        if (
            has_baseline
            or frozen_confirmation
            or len(distinct_temporal_frames)
            >= int(self.settings["temporal_minimum_observations"])
        ):
            event.state = "ACTIVE"

    def observe(
        self,
        frame_number: int,
        timestamp: float,
        candidates: list[dict[str, Any]],
    ) -> list[ReviewEvent]:
        touched: list[ReviewEvent] = []
        for candidate in candidates:
            detection = EventDetection.from_candidate(
                frame_number, timestamp, candidate
            )
            event = self._match(detection, {"ACTIVE", "PENDING"})
            if event is None:
                event = self._match(detection, {"CLOSED"})
                if event is not None:
                    event.state = "ACTIVE"
            if event is None:
                event = self._new_event(detection)
            else:
                event.add(detection)
            self._promote_if_ready(event)
            touched.append(event)
        self.advance(timestamp)
        return touched

    def advance(self, timestamp: float) -> list[ReviewEvent]:
        closed: list[ReviewEvent] = []
        close_after = float(self.settings.get("close_after_seconds", 3.0))
        for event in self.events.values():
            if event.state == "ACTIVE" and timestamp - event.end_time > close_after:
                event.state = "CLOSED"
                closed.append(event)
            elif event.state == "PENDING" and timestamp - event.end_time > close_after:
                event.state = "REJECTED_UNCONFIRMED"
        return closed

    def finalize(self) -> list[ReviewEvent]:
        for event in self.events.values():
            if event.state == "ACTIVE":
                event.state = "CLOSED"
            elif event.state == "PENDING":
                event.state = "REJECTED_UNCONFIRMED"
        return self.accepted_events()

    def accepted_events(self) -> list[ReviewEvent]:
        return sorted(
            (
                event
                for event in self.events.values()
                if event.state in {"ACTIVE", "CLOSED"}
            ),
            key=lambda event: (event.start_time, event.event_id),
        )
