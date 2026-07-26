from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .event_aggregator_v2 import EvidenceEventAggregator


@dataclass
class PersonEpisode:
    person_episode_id: str
    source_track_ids: set[int] = field(default_factory=set)
    observations: list[dict[str, Any]] = field(default_factory=list)
    merge_reasons: list[str] = field(default_factory=list)
    final_status: str = "PENDING_REVIEW"

    def add(self, row: dict[str, Any]) -> None:
        self.observations.append(dict(row))
        if row.get("track_id") is not None and int(row["track_id"]) >= 0:
            self.source_track_ids.add(int(row["track_id"]))

    @property
    def first_frame(self) -> int:
        return min(int(row["video_frame_id"]) for row in self.observations)

    @property
    def last_frame(self) -> int:
        return max(int(row["video_frame_id"]) for row in self.observations)

    @property
    def candidate_ids(self) -> list[str]:
        return [str(row["candidate_id"]) for row in self.observations]

    def signature(self) -> str:
        payload = json.dumps(
            {
                "id": self.person_episode_id,
                "tracks": sorted(self.source_track_ids),
                "candidates": self.candidate_ids,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class HazardEvent:
    hazard_event_id: str
    camera_id: str
    start_time: float
    end_time: float
    person_episode_ids: list[str]
    rejected_candidate_ids: list[str]
    candidate_ids: list[str]

    def signature(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def build_person_episodes(rows: list[dict[str, Any]]) -> list[PersonEpisode]:
    """Preserve every accepted observation in one auditable child episode."""
    episodes: dict[str, PersonEpisode] = {}
    for row in sorted(
        rows, key=lambda item: (int(item["video_frame_id"]), str(item["candidate_id"]))
    ):
        track_id = int(row.get("track_id", -1))
        identity = (
            f"track:{track_id}"
            if track_id >= 0
            else f"candidate:{row['candidate_id']}"
        )
        episode = episodes.setdefault(identity, PersonEpisode(identity))
        episode.add(row)
    before = sum(len(episode.observations) for episode in episodes.values())
    if before != len(rows):
        raise RuntimeError("PersonEpisode conservation failure")
    return sorted(episodes.values(), key=lambda item: item.person_episode_id)


def build_hazard_events(
    rows: list[dict[str, Any]],
    person_episodes: list[PersonEpisode],
    settings: dict[str, Any],
    *,
    camera_id: str,
    width: int,
    height: int,
) -> list[HazardEvent]:
    aggregator = EvidenceEventAggregator(
        settings,
        video_id=camera_id,
        camera_id=camera_id,
        frame_width=width,
        frame_height=height,
    )
    by_candidate = {
        candidate_id: episode.person_episode_id
        for episode in person_episodes
        for candidate_id in episode.candidate_ids
    }
    for frame_id in sorted({int(row["video_frame_id"]) for row in rows}):
        frame_rows = [
            row for row in rows if int(row["video_frame_id"]) == frame_id
        ]
        timestamp = float(frame_rows[0]["timestamp"])
        candidates = [
            {
                "candidate_id": row["candidate_id"],
                "box": [
                    row["bbox_x1"],
                    row["bbox_y1"],
                    row["bbox_x2"],
                    row["bbox_y2"],
                ],
                "confidence": row["output_confidence"],
                "source": row["review_source"],
                "track_id": (
                    int(row["track_id"]) if int(row["track_id"]) >= 0 else None
                ),
                "interpolated": bool(row["is_interpolated"]),
                "confirmed": bool(row["confirmed"]),
                "processing_status": row.get("processing_status", "NORMAL"),
            }
            for row in frame_rows
        ]
        aggregator.observe(frame_id, timestamp, candidates)
    output = []
    represented: set[str] = set()
    for event in aggregator.finalize():
        candidate_ids = [item.candidate_id for item in event.detections]
        children = sorted({by_candidate[item] for item in candidate_ids})
        represented.update(children)
        output.append(
            HazardEvent(
                hazard_event_id=event.event_id,
                camera_id=camera_id,
                start_time=event.start_time,
                end_time=event.end_time,
                person_episode_ids=children,
                rejected_candidate_ids=[],
                candidate_ids=candidate_ids,
            )
        )
    expected = {episode.person_episode_id for episode in person_episodes}
    if represented != expected:
        missing = sorted(expected - represented)
        raise RuntimeError(f"Hazard grouping lost PersonEpisodes: {missing}")
    return output
