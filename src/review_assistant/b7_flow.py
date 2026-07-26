from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import networkx as nx
import numpy as np

from src.review_assistant.b6_association import (
    PAIR_FEATURES,
    TrackFragment,
    pair_features,
)


B7_FEATURES = PAIR_FEATURES + [
    "direction_discontinuity",
    "velocity_discontinuity",
]


@dataclass(frozen=True)
class CandidateEdge:
    left_id: str
    right_id: str
    raw_features: tuple[float, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.left_id, self.right_id


def _centers(fragment: TrackFragment) -> np.ndarray:
    return np.asarray(
        [
            [
                (float(row["bbox_x1"]) + float(row["bbox_x2"])) / 2.0,
                float(row["bbox_y2"]),
            ]
            for row in fragment.observations
        ],
        dtype=np.float64,
    )


def _velocity(fragment: TrackFragment, tail: bool) -> np.ndarray:
    centers = _centers(fragment)
    if len(centers) < 2:
        return np.zeros(2, dtype=np.float64)
    if tail:
        return centers[-1] - centers[-2]
    return centers[1] - centers[0]


def extended_pair_features(
    left: TrackFragment, right: TrackFragment
) -> np.ndarray:
    base = pair_features(left, right)
    first, second = (
        (left, right) if left.first_frame <= right.first_frame else (right, left)
    )
    velocity_left = _velocity(first, tail=True)
    velocity_right = _velocity(second, tail=False)
    norm_left = float(np.linalg.norm(velocity_left))
    norm_right = float(np.linalg.norm(velocity_right))
    if norm_left > 1e-9 and norm_right > 1e-9:
        cosine = float(
            np.dot(velocity_left, velocity_right) / (norm_left * norm_right)
        )
        direction = 1.0 - float(np.clip(cosine, -1.0, 1.0))
        velocity = abs(math.log((norm_left + 1e-6) / (norm_right + 1e-6)))
    else:
        direction = 0.0
        velocity = 0.0
    return np.concatenate([base, np.asarray([direction, velocity])])


def candidate_edges(
    fragments: list[TrackFragment], settings: dict[str, float]
) -> list[CandidateEdge]:
    output = []
    for left, right in itertools.permutations(fragments, 2):
        if left.scene_id != right.scene_id:
            continue
        if right.first_frame <= left.last_frame:
            continue
        gap = right.first_frame - left.last_frame
        if gap > int(settings["maximum_gap_frames"]):
            continue
        features = extended_pair_features(left, right)
        if features[1] > float(settings["maximum_motion_error"]):
            continue
        if features[4] > float(settings["maximum_size_discontinuity"]):
            continue
        if features[2] > float(settings["maximum_depth_difference"]):
            continue
        output.append(
            CandidateEdge(
                left.fragment_id,
                right.fragment_id,
                tuple(map(float, features)),
            )
        )
    return sorted(output, key=lambda edge: edge.key)


def robust_scene_normalization(
    edges: list[CandidateEdge], epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray([edge.raw_features for edge in edges], dtype=np.float64)
    if not len(values):
        width = len(B7_FEATURES)
        return np.zeros((0, width)), np.zeros(width), np.ones(width)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    scale = 1.4826 * mad + epsilon
    normalized = np.clip((values - median) / scale, -8.0, 8.0)
    return normalized, median, scale


@lru_cache(maxsize=16)
def _path_triplets(
    edge_keys: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], tuple[tuple[tuple[str, str], tuple[str, str]], ...]]:
    outgoing: dict[str, list[str]] = {}
    edge_set = set(edge_keys)
    for left, right in edge_set:
        outgoing.setdefault(left, []).append(right)
    output = {}
    for left, right in edge_set:
        candidates = [
            middle
            for middle in outgoing.get(left, [])
            if (middle, right) in edge_set
        ]
        output[(left, right)] = tuple(
            ((left, middle), (middle, right)) for middle in candidates
        )
    return output


def path_consistency_penalties(
    edges: list[CandidateEdge], probabilities: dict[tuple[str, str], float]
) -> dict[tuple[str, str], float]:
    edge_keys = tuple(edge.key for edge in edges)
    triplets = _path_triplets(edge_keys)
    penalties = {}
    for key in edge_keys:
        paths = triplets[key]
        if not paths:
            penalties[key] = 0.0
            continue
        direct = probabilities[key]
        penalties[key] = min(
            abs(
                direct
                - probabilities[first_edge] * probabilities[second_edge]
            )
            for first_edge, second_edge in paths
        )
    return penalties


def flow_solution(
    fragments: list[TrackFragment],
    edges: list[CandidateEdge],
    probabilities: dict[tuple[str, str], float],
    uncertainty: dict[tuple[str, str], float],
    normalized_features: dict[tuple[str, str], np.ndarray],
    settings: dict[str, float],
    allowed_edges: set[tuple[str, str]] | None = None,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    path_penalties = path_consistency_penalties(edges, probabilities)
    graph = nx.Graph()
    edge_audit = []
    for edge in edges:
        if allowed_edges is not None and edge.key not in allowed_edges:
            continue
        probability = float(
            np.clip(probabilities.get(edge.key, 0.0), 1e-8, 1.0 - 1e-8)
        )
        normalized = normalized_features[edge.key]
        physical_penalty = float(
            np.mean(np.maximum(normalized[[1, 2, 4, 7, 8]], 0.0))
        )
        benefit = (
            math.log(probability / (1.0 - probability))
            - float(settings["lambda_uncertainty"])
            * float(uncertainty.get(edge.key, 0.0))
            - float(settings["lambda_motion_depth"])
            * physical_penalty
            - float(settings["lambda_path"])
            * path_penalties.get(edge.key, 0.0)
        )
        selected_candidate = benefit > 0.0
        edge_audit.append(
            {
                "left_fragment_id": edge.left_id,
                "right_fragment_id": edge.right_id,
                "probability": probability,
                "uncertainty": float(uncertainty.get(edge.key, 0.0)),
                "physical_penalty": physical_penalty,
                "path_penalty": path_penalties.get(edge.key, 0.0),
                "benefit": benefit,
                "positive_benefit": selected_candidate,
            }
        )
        if selected_candidate:
            graph.add_edge(
                f"out:{edge.left_id}",
                f"in:{edge.right_id}",
                weight=benefit,
                key=edge.key,
            )
    matching = nx.algorithms.matching.max_weight_matching(
        graph, maxcardinality=False, weight="weight"
    )
    links: dict[str, str] = {}
    selected_keys = set()
    for node_a, node_b in matching:
        if node_a.startswith("out:"):
            left, right = node_a[4:], node_b[3:]
        else:
            left, right = node_b[4:], node_a[3:]
        links[left] = right
        selected_keys.add((left, right))
    predecessors = {right: left for left, right in links.items()}
    fragment_ids = [fragment.fragment_id for fragment in fragments]
    paths = []
    visited = set()
    for fragment_id in fragment_ids:
        if fragment_id in predecessors:
            continue
        path = []
        current = fragment_id
        while current and current not in visited:
            path.append(current)
            visited.add(current)
            current = links.get(current, "")
        paths.append(path)
    paths.extend([[fragment_id] for fragment_id in fragment_ids if fragment_id not in visited])
    for row in edge_audit:
        row["selected"] = (
            row["left_fragment_id"],
            row["right_fragment_id"],
        ) in selected_keys
    return sorted((sorted(path) for path in paths), key=lambda path: path[0]), edge_audit


def clustering_signature(clusters: list[list[str]]) -> str:
    return hashlib.sha256(
        repr(sorted(sorted(cluster) for cluster in clusters)).encode()
    ).hexdigest()
