from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrackFragment:
    fragment_id: str
    scene_id: str
    track_id: int
    observations: list[dict[str, Any]]
    appearance: np.ndarray | None = None
    appearance_samples: np.ndarray | None = None
    appearance_spread: float = 0.0
    depth_median: float = 0.0
    depth_slope: float = 0.0
    depth_sequence: np.ndarray | None = None
    gt_episode_id: str = ""

    @property
    def first_frame(self) -> int:
        return min(int(row["video_frame_id"]) for row in self.observations)

    @property
    def last_frame(self) -> int:
        return max(int(row["video_frame_id"]) for row in self.observations)

    @property
    def candidate_ids(self) -> list[str]:
        return [str(row["candidate_id"]) for row in self.observations]

    @property
    def real_count(self) -> int:
        return sum(not bool(row["is_interpolated"]) for row in self.observations)

    @property
    def interpolated_count(self) -> int:
        return len(self.observations) - self.real_count

    def signature(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "fragment_id": self.fragment_id,
                    "candidate_ids": self.candidate_ids,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()


def _bottom_center(row: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            (float(row["bbox_x1"]) + float(row["bbox_x2"])) / 2,
            float(row["bbox_y2"]),
        ]
    )


def split_fragments(
    rows: list[dict[str, Any]], settings: dict[str, Any], width: int, height: int
) -> list[TrackFragment]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    untracked = -1
    for row in sorted(rows, key=lambda x: (int(x["video_frame_id"]), str(x["candidate_id"]))):
        track = int(row.get("track_id", -1))
        if track < 0:
            track = untracked
            untracked -= 1
        grouped.setdefault(track, []).append(row)
    output = []
    diagonal = max(float(np.hypot(width, height)), 1.0)
    for track, observations in sorted(grouped.items()):
        chunks: list[list[dict[str, Any]]] = [[]]
        interpolation_streak = 0
        for row in observations:
            split = False
            if chunks[-1]:
                previous = chunks[-1][-1]
                gap = int(row["video_frame_id"]) - int(previous["video_frame_id"])
                center_jump = np.linalg.norm(_bottom_center(row) - _bottom_center(previous)) / diagonal
                area = max((row["bbox_x2"] - row["bbox_x1"]) * (row["bbox_y2"] - row["bbox_y1"]), 1)
                previous_area = max((previous["bbox_x2"] - previous["bbox_x1"]) * (previous["bbox_y2"] - previous["bbox_y1"]), 1)
                area_jump = abs(float(np.log(area / previous_area)))
                previous_confidence = max(float(previous["output_confidence"]), 1e-9)
                confidence_drop = (
                    float(row["output_confidence"]) / previous_confidence
                    < float(settings["confidence_drop_ratio"])
                )
                interpolation_streak = interpolation_streak + 1 if row["is_interpolated"] else 0
                split = (
                    gap > int(settings["maximum_frame_gap"])
                    or center_jump > float(settings["maximum_center_jump_diagonal"])
                    or area_jump > float(settings["maximum_log_area_jump"])
                    or confidence_drop
                    or interpolation_streak > int(settings["maximum_interpolation_streak"])
                )
            if split:
                chunks.append([])
                interpolation_streak = int(bool(row["is_interpolated"]))
            chunks[-1].append(row)
        for index, chunk in enumerate(chunks):
            output.append(
                TrackFragment(
                    fragment_id=f"{chunk[0]['scene_id']}:T{track}:F{index:03d}",
                    scene_id=str(chunk[0]["scene_id"]),
                    track_id=track,
                    observations=chunk,
                )
            )
    if sum(len(fragment.observations) for fragment in output) != len(rows):
        raise RuntimeError("Fragment conservation failure")
    return output


PAIR_FEATURES = [
    "appearance_distance",
    "motion_error",
    "depth_difference",
    "temporal_gap",
    "size_discontinuity",
    "confidence_discontinuity",
    "overlap_conflict",
]


def pair_features(left: TrackFragment, right: TrackFragment) -> np.ndarray:
    first, second = (left, right) if left.first_frame <= right.first_frame else (right, left)
    a = first.observations[-1]
    b = second.observations[0]
    gap = max(second.first_frame - first.last_frame, 0)
    centers = [_bottom_center(row) for row in first.observations]
    velocity = centers[-1] - centers[-2] if len(centers) > 1 else np.zeros(2)
    predicted = centers[-1] + velocity * gap
    diagonal = max(float(np.hypot(4112, 2504)), 1)
    motion_error = float(np.linalg.norm(predicted - _bottom_center(b)) / diagonal)
    area_a = max((a["bbox_x2"] - a["bbox_x1"]) * (a["bbox_y2"] - a["bbox_y1"]), 1)
    area_b = max((b["bbox_x2"] - b["bbox_x1"]) * (b["bbox_y2"] - b["bbox_y1"]), 1)
    appearance = 1.0
    if left.appearance is not None and right.appearance is not None:
        appearance = 1 - float(
            np.dot(left.appearance, right.appearance)
            / max(np.linalg.norm(left.appearance) * np.linalg.norm(right.appearance), 1e-12)
        )
    simultaneous = not (left.last_frame < right.first_frame or right.last_frame < left.first_frame)
    return np.asarray(
        [
            appearance,
            motion_error,
            abs(left.depth_median - right.depth_median),
            gap / 30.0,
            abs(float(np.log(area_a / area_b))),
            abs(float(a["output_confidence"]) - float(b["output_confidence"])),
            float(simultaneous),
        ],
        dtype=np.float64,
    )


def complete_link_clusters(
    fragments: list[TrackFragment],
    pair_scores: dict[tuple[str, str], tuple[float, float]],
    safe_threshold: float,
    cannot_threshold: float,
) -> tuple[list[list[str]], int]:
    clusters: dict[int, set[str]] = {
        index: {fragment.fragment_id} for index, fragment in enumerate(fragments)
    }
    membership = {
        fragment.fragment_id: index for index, fragment in enumerate(fragments)
    }
    abstentions = 0
    lookup = {fragment.fragment_id: fragment for fragment in fragments}

    def score(a: str, b: str) -> tuple[float, float]:
        return pair_scores.get(tuple(sorted((a, b))), (0.0, 1.0))

    edges = []
    for left, right in itertools.combinations(fragments, 2):
        mu, sigma = score(left.fragment_id, right.fragment_id)
        lower, upper = mu - 1.96 * sigma, mu + 1.96 * sigma
        if lower >= safe_threshold:
            edges.append((lower, left.fragment_id, right.fragment_id))
        elif upper > cannot_threshold:
            abstentions += 1
    for _, left, right in sorted(edges, reverse=True):
        left_key = membership[left]
        right_key = membership[right]
        if left_key == right_key:
            continue
        left_cluster = clusters[left_key]
        right_cluster = clusters[right_key]
        compatible = True
        for a, b in itertools.product(left_cluster, right_cluster):
            fa, fb = lookup[a], lookup[b]
            mu, sigma = score(a, b)
            simultaneous = not (
                fa.last_frame < fb.first_frame or fb.last_frame < fa.first_frame
            )
            if fa.scene_id != fb.scene_id or simultaneous or mu - 1.96 * sigma < safe_threshold:
                compatible = False
                break
        if compatible:
            left_cluster.update(right_cluster)
            for fragment_id in right_cluster:
                membership[fragment_id] = left_key
            del clusters[right_key]
    return [sorted(cluster) for cluster in clusters.values()], abstentions
