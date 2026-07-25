from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def clip_probability(value: float, epsilon: float = 1e-6) -> float:
    return float(np.clip(value, epsilon, 1.0 - epsilon))


def logit(value: float) -> float:
    value = clip_probability(value)
    return float(np.log(value / (1.0 - value)))


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0))))


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return intersection / max(box_area(left) + box_area(right) - intersection, 1e-12)


def warp_box(box: list[float], homography: np.ndarray) -> list[float]:
    corners = np.asarray(
        [[[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]],
        dtype=np.float32,
    )
    warped = cv2.perspectiveTransform(corners, homography)[0]
    return [
        float(warped[:, 0].min()),
        float(warped[:, 1].min()),
        float(warped[:, 0].max()),
        float(warped[:, 1].max()),
    ]


def roi_embedding(image: np.ndarray, box: list[float], color_bins: int = 8) -> np.ndarray:
    height, width = image.shape[:2]
    x1 = int(np.clip(np.floor(box[0]), 0, width - 1))
    y1 = int(np.clip(np.floor(box[1]), 0, height - 1))
    x2 = int(np.clip(np.ceil(box[2]), x1 + 1, width))
    y2 = int(np.clip(np.ceil(box[3]), y1 + 1, height))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(color_bins * 3 + 8, dtype=np.float32)
    features = []
    for channel in range(3):
        histogram = cv2.calcHist([crop], [channel], None, [color_bins], [0, 256]).ravel()
        features.extend(histogram.tolist())
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    angles = (np.arctan2(gy, gx) + np.pi) % np.pi
    magnitudes = np.hypot(gx, gy)
    gradient, _ = np.histogram(
        angles, bins=8, range=(0.0, np.pi), weights=magnitudes
    )
    vector = np.asarray([*features, *gradient.tolist()], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


@dataclass(frozen=True)
class HomographyResult:
    matrix: np.ndarray
    valid: bool
    matches: int
    inliers: int
    inlier_ratio: float


def estimate_background_homography(
    previous: np.ndarray,
    current: np.ndarray,
    masked_boxes: list[list[float]],
    config: dict[str, Any],
) -> HomographyResult:
    full_height, full_width = previous.shape[:2]
    maximum_width = int(config["maximum_width"])
    scale = min(1.0, maximum_width / full_width)
    size = (int(round(full_width * scale)), int(round(full_height * scale)))
    previous_gray = cv2.resize(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), size)
    current_gray = cv2.resize(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), size)
    mask = np.full(previous_gray.shape, 255, dtype=np.uint8)
    for box in masked_boxes:
        x1, y1, x2, y2 = [int(round(value * scale)) for value in box]
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)
    detector = cv2.ORB_create(nfeatures=int(config["orb_features"]))
    left_points, left_desc = detector.detectAndCompute(previous_gray, mask)
    right_points, right_desc = detector.detectAndCompute(current_gray, None)
    identity = np.eye(3, dtype=np.float64)
    if left_desc is None or right_desc is None:
        return HomographyResult(identity, False, 0, 0, 0.0)
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(left_desc, right_desc, k=2)
    accepted = [
        first for first, second in pairs
        if first.distance < float(config["ratio_test"]) * second.distance
    ]
    if len(accepted) < int(config["minimum_matches"]):
        return HomographyResult(identity, False, len(accepted), 0, 0.0)
    source = np.float32([left_points[item.queryIdx].pt for item in accepted])
    target = np.float32([right_points[item.trainIdx].pt for item in accepted])
    matrix, inlier_mask = cv2.findHomography(
        source,
        target,
        cv2.RANSAC,
        float(config["ransac_threshold_px"]),
    )
    if matrix is None or inlier_mask is None:
        return HomographyResult(identity, False, len(accepted), 0, 0.0)
    inliers = int(inlier_mask.ravel().sum())
    ratio = inliers / max(len(accepted), 1)
    scaling = np.asarray([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=float)
    full_matrix = np.linalg.inv(scaling) @ matrix @ scaling
    valid = (
        inliers >= int(config["minimum_inliers"])
        and ratio >= float(config["minimum_inlier_ratio"])
        and np.isfinite(full_matrix).all()
    )
    return HomographyResult(
        full_matrix if valid else identity,
        valid,
        len(accepted),
        inliers,
        ratio,
    )


@dataclass
class Track:
    track_id: int
    box: list[float]
    embedding: np.ndarray
    confidence_history: list[float]
    posterior: float
    hits: int = 1
    gap: int = 0
    age: int = 1
    last_reliability: float = 0.0


def reliability_memberships(
    predicted_box: list[float],
    detection_box: list[float],
    previous_embedding: np.ndarray,
    current_embedding: np.ndarray,
    image_shape: tuple[int, int],
    config: dict[str, Any],
) -> dict[str, float]:
    px, py = box_center(predicted_box)
    dx, dy = box_center(detection_box)
    diagonal = max(np.sqrt(box_area(predicted_box)), 4.0)
    normalized_motion = np.hypot(dx - px, dy - py) / diagonal
    motion = float(
        np.exp(-0.5 * (normalized_motion / float(config["motion_sigma"])) ** 2)
    )
    appearance = float(np.clip((1.0 + np.dot(previous_embedding, current_embedding)) / 2.0, 0, 1))
    scale = float(
        np.exp(
            -abs(np.log(max(box_area(detection_box), 1.0) / max(box_area(predicted_box), 1.0)))
            / float(config["scale_sigma"])
        )
    )
    height, width = image_shape
    margin = min(
        detection_box[0],
        detection_box[1],
        width - detection_box[2],
        height - detection_box[3],
    )
    border = float(np.clip(margin / max(np.sqrt(box_area(detection_box)), 4.0), 0, 1))
    return {
        "motion": motion,
        "appearance": appearance,
        "scale": scale,
        "border": border,
        "compensated_iou": box_iou(predicted_box, detection_box),
    }


def association_cost(memberships: dict[str, float], weights: dict[str, float]) -> float:
    return float(
        weights["motion"] * (1.0 - memberships["motion"])
        + weights["compensated_iou"] * (1.0 - memberships["compensated_iou"])
        + weights["appearance"] * (1.0 - memberships["appearance"])
        + weights["scale"] * (1.0 - memberships["scale"])
    )


def product_reliability(memberships: dict[str, float]) -> float:
    return float(np.prod([memberships[key] for key in ("motion", "appearance", "scale", "border")]))


def arithmetic_reliability(memberships: dict[str, float]) -> float:
    return float(np.mean([memberships[key] for key in ("motion", "appearance", "scale", "border")]))


class TemporalAggregator:
    def __init__(self, config: dict[str, Any], mode: str) -> None:
        self.config = config
        self.mode = mode
        self.tracks: list[Track] = []
        self.next_track_id = 1

    def reset(self) -> None:
        self.tracks = []
        self.next_track_id = 1

    def _posterior(self, track: Track, confidence: float, reliability: float) -> float:
        bayesian = self.config["bayesian"]
        predicted = (
            float(bayesian["survival_probability"]) * track.posterior
            + float(bayesian["birth_probability"]) * (1.0 - track.posterior)
        )
        evidence = logit(confidence) + float(bayesian["temporal_gain"]) * reliability * (
            logit(confidence) - logit(float(bayesian["candidate_prior"]))
        )
        return min(float(bayesian["maximum_posterior"]), sigmoid(logit(predicted) + evidence))

    def update(
        self,
        detections: list[dict[str, Any]],
        image: np.ndarray,
        homography: HomographyResult | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        association = self.config["association"]
        embeddings = [roi_embedding(image, row["box"]) for row in detections]
        predicted_boxes = [
            warp_box(track.box, homography.matrix)
            if homography is not None and homography.valid
            else track.box
            for track in self.tracks
        ]
        pairs: list[tuple[int, int, dict[str, float], float]] = []
        if self.tracks and detections and homography is not None and homography.valid:
            costs = np.full((len(self.tracks), len(detections)), 1e6, dtype=float)
            memberships: dict[tuple[int, int], dict[str, float]] = {}
            for left, track in enumerate(self.tracks):
                for right, detection in enumerate(detections):
                    member = reliability_memberships(
                        predicted_boxes[left],
                        detection["box"],
                        track.embedding,
                        embeddings[right],
                        image.shape[:2],
                        association,
                    )
                    memberships[(left, right)] = member
                    costs[left, right] = association_cost(member, association["weights"])
            rows, columns = linear_sum_assignment(costs)
            for left, right in zip(rows.tolist(), columns.tolist()):
                if costs[left, right] <= float(association["maximum_cost"]):
                    pairs.append((left, right, memberships[(left, right)], float(costs[left, right])))
        matched_tracks = {left for left, _, _, _ in pairs}
        matched_detections = {right for _, right, _, _ in pairs}
        track_rows: list[dict[str, Any]] = []
        emitted: list[dict[str, Any]] = []
        for left, right, member, cost in pairs:
            track = self.tracks[left]
            detection = detections[right]
            reliability = (
                product_reliability(member)
                if self.mode == "product"
                else arithmetic_reliability(member)
            )
            reliable_product = not (
                self.mode == "product"
                and reliability < float(self.config["bayesian"]["minimum_reliability"])
            )
            track.box = list(map(float, detection["box"]))
            track.embedding = embeddings[right]
            track.confidence_history = (
                track.confidence_history + [float(detection["confidence"])]
            )[-5:]
            track.hits += 1
            track.gap = 0
            track.age += 1
            track.last_reliability = reliability
            if self.mode == "bytetrack":
                accumulated = 1.0 - float(
                    np.prod([1.0 - clip_probability(value) for value in track.confidence_history])
                )
                track.posterior = max(float(detection["confidence"]), accumulated)
            elif not reliable_product:
                track.posterior = float(detection["confidence"])
            else:
                track.posterior = self._posterior(track, float(detection["confidence"]), reliability)
            score = max(float(detection["confidence"]), track.posterior)
            emitted.append({**detection, "confidence": score, "track_id": track.track_id, "propagated": False})
            track_rows.append({
                "track_id": track.track_id,
                "matched": True,
                "posterior": track.posterior,
                "reliability": reliability,
                "association_cost": cost,
                "hits": track.hits,
                "gap": track.gap,
                **{f"mu_{key}": value for key, value in member.items()},
            })
        survivors: list[Track] = []
        for index, track in enumerate(self.tracks):
            if index in matched_tracks:
                survivors.append(track)
                continue
            track.gap += 1
            track.age += 1
            if self.mode != "bytetrack":
                bayesian = self.config["bayesian"]
                track.posterior = (
                    float(bayesian["survival_probability"]) * track.posterior
                    + float(bayesian["birth_probability"]) * (1.0 - track.posterior)
                )
            else:
                track.posterior *= float(self.config["bytetrack"]["confirmed_score_decay"])
            if (
                homography is not None
                and homography.valid
                and track.hits >= int(self.config["minimum_confirmed_hits"])
                and track.gap <= int(self.config["maximum_gap_frames"])
            ):
                track.box = predicted_boxes[index]
                emitted.append({
                    "class_id": 0,
                    "box": track.box,
                    "confidence": track.posterior,
                    "track_id": track.track_id,
                    "propagated": True,
                })
                survivors.append(track)
        self.tracks = survivors
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            confidence = float(detection["confidence"])
            posterior = confidence
            track = Track(
                track_id=self.next_track_id,
                box=list(map(float, detection["box"])),
                embedding=embeddings[index],
                confidence_history=[confidence],
                posterior=posterior,
            )
            self.next_track_id += 1
            self.tracks.append(track)
            emitted.append({**detection, "track_id": track.track_id, "propagated": False})
        return emitted, track_rows


def deterministic_nms(predictions: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(
        predictions,
        key=lambda value: (-float(value["confidence"]), *map(float, value["box"])),
    ):
        if any(box_iou(row["box"], other["box"]) > threshold for other in kept):
            continue
        kept.append(row)
    return kept
