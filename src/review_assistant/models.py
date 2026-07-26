from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EventDetection:
    frame_number: int
    timestamp: float
    box: list[float]
    confidence: float
    source: str
    track_id: int | None = None
    interpolated: bool = False
    confirmed: bool = False
    motion: float = 0.0

    @classmethod
    def from_candidate(
        cls, frame_number: int, timestamp: float, candidate: dict[str, Any]
    ) -> "EventDetection":
        return cls(
            frame_number=int(frame_number),
            timestamp=float(timestamp),
            box=[float(value) for value in candidate["box"]],
            confidence=float(candidate.get("confidence", 0.0)),
            source=str(candidate.get("review_source", candidate.get("source", "BASELINE"))),
            track_id=(
                int(candidate["track_id"])
                if candidate.get("track_id") is not None
                else None
            ),
            interpolated=bool(candidate.get("interpolated", False)),
            confirmed=bool(candidate.get("confirmed", False)),
            motion=float(candidate.get("motion", 0.0)),
        )


@dataclass
class ReviewEvent:
    event_id: str
    video_id: str
    camera_id: str
    start_time: float
    end_time: float
    state: str = "PENDING"
    detections: list[EventDetection] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    track_ids: set[int] = field(default_factory=set)
    maximum_confidence: float = 0.0
    representative_box: list[float] = field(default_factory=lambda: [0.0] * 4)
    representative_motion: float = 0.0
    review_status: str = "PENDING"
    operator_comment: str = ""
    priority: int = 100
    muted: bool = False

    def add(self, detection: EventDetection) -> None:
        self.detections.append(detection)
        self.start_time = min(self.start_time, detection.timestamp)
        self.end_time = max(self.end_time, detection.timestamp)
        self.sources.add(detection.source)
        if detection.track_id is not None:
            self.track_ids.add(detection.track_id)
        if detection.confidence >= self.maximum_confidence:
            self.maximum_confidence = detection.confidence
            self.representative_box = list(detection.box)
            self.representative_motion = detection.motion

    @property
    def source_label(self) -> str:
        baseline = any(source in {"BASELINE", "BOTH"} for source in self.sources)
        temporal = any(source in {"TEMPORAL_ONLY", "BOTH"} for source in self.sources)
        if baseline and temporal:
            return "BOTH"
        return "BASELINE" if baseline else "TEMPORAL_ONLY"

    @property
    def real_detection_count(self) -> int:
        return sum(not item.interpolated for item in self.detections)

    @property
    def interpolated_count(self) -> int:
        return sum(item.interpolated for item in self.detections)

    @property
    def duration(self) -> float:
        return max(self.end_time - self.start_time, 0.0)

    def to_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "video_id": self.video_id,
            "camera_id": self.camera_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "sources": sorted(self.sources),
            "source_label": self.source_label,
            "track_ids": sorted(self.track_ids),
            "maximum_confidence": self.maximum_confidence,
            "real_detection_count": self.real_detection_count,
            "interpolated_count": self.interpolated_count,
            "spatial_region": self.representative_box,
            "representative_motion": self.representative_motion,
            "review_status": self.review_status,
            "operator_comment": self.operator_comment,
            "priority": self.priority,
            "muted": self.muted,
            "state": self.state,
        }

    def detections_as_records(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.detections]
