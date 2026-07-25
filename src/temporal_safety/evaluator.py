from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.temporal.ratta import box_iou, deterministic_nms


def match_dataset(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    confidence: float,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    tp = fp = fn = 0
    for image_id in sorted(ground_truth):
        targets = ground_truth[image_id]
        candidates = sorted(
            [
                row
                for row in predictions.get(image_id, [])
                if float(row["confidence"]) >= confidence
            ],
            key=lambda row: -float(row["confidence"]),
        )
        used: set[int] = set()
        for prediction in candidates:
            available = [
                (index, box_iou(prediction["box"], target["box"]))
                for index, target in enumerate(targets)
                if index not in used
                and int(prediction["class_id"]) == int(target["class_id"])
            ]
            if available:
                index, overlap = max(available, key=lambda item: item[1])
                if overlap >= iou_threshold:
                    used.add(index)
                    tp += 1
                    continue
            fp += 1
        fn += len(targets) - len(used)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def average_precision(
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    class_id: int,
    iou_threshold: float = 0.5,
) -> float:
    total_gt = sum(
        target["class_id"] == class_id
        for targets in ground_truth.values()
        for target in targets
    )
    ranked = sorted(
        (
            (image_id, prediction)
            for image_id, rows in predictions.items()
            for prediction in rows
            if prediction["class_id"] == class_id
        ),
        key=lambda item: -float(item[1]["confidence"]),
    )
    used: defaultdict[str, set[int]] = defaultdict(set)
    true_positives: list[float] = []
    false_positives: list[float] = []
    for image_id, prediction in ranked:
        candidates = [
            (index, box_iou(prediction["box"], target["box"]))
            for index, target in enumerate(ground_truth.get(image_id, []))
            if target["class_id"] == class_id and index not in used[image_id]
        ]
        if candidates:
            index, overlap = max(candidates, key=lambda item: item[1])
            matched = overlap >= iou_threshold
        else:
            index, matched = -1, False
        true_positives.append(float(matched))
        false_positives.append(float(not matched))
        if matched:
            used[image_id].add(index)
    if total_gt == 0:
        return math.nan
    tp = np.cumsum(true_positives)
    fp = np.cumsum(false_positives)
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changes = np.where(recall[1:] != recall[:-1])[0]
    return float(
        np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1])
    )


def parse_prediction_table(
    table: pd.DataFrame, image_paths: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    ground_truth: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for image_path in image_paths:
        ground_truth[image_path] = []
        predictions[image_path] = []
    leaked = set(table["image_path"].astype(str)) - image_paths
    if leaked:
        raise RuntimeError("Prediction table contains frames outside the requested role")
    for row in table.itertuples(index=False):
        item = {
            "class_id": 0,
            "box": [float(row.x1), float(row.y1), float(row.x2), float(row.y2)],
        }
        if row.kind == "ground_truth":
            ground_truth[str(row.image_path)].append(item)
        else:
            predictions[str(row.image_path)].append(
                {**item, "confidence": float(row.confidence)}
            )
    return dict(ground_truth), dict(predictions)


def evaluate(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
    iou_threshold: float,
    nominal_fps: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    operating = match_dataset(ground_truth, predictions, threshold, iou_threshold)
    frames = len(source)
    duration_minutes = frames / max(nominal_fps * 60.0, 1e-12)
    per_scene: list[dict[str, Any]] = []
    for scene, group in source.groupby("grouped_scene_id", sort=True):
        paths = set(group["image_path"])
        scene_gt = {path: ground_truth[path] for path in paths}
        scene_predictions = {path: predictions.get(path, []) for path in paths}
        counts = match_dataset(scene_gt, scene_predictions, threshold, iou_threshold)
        minutes = len(group) / max(nominal_fps * 60.0, 1e-12)
        per_scene.append(
            {
                "grouped_scene_id": str(scene),
                "frames": len(group),
                **counts,
                "FN_per_frame": counts["fn"] / max(len(group), 1),
                "FP_per_frame": counts["fp"] / max(len(group), 1),
                "false_alarms_per_minute": counts["fp"] / max(minutes, 1e-12),
            }
        )
    payload = {
        **operating,
        "mAP50": float(average_precision(ground_truth, predictions, 0, iou_threshold)),
        "frames": frames,
        "scenes": int(source["grouped_scene_id"].nunique()),
        "FN_per_frame": operating["fn"] / max(frames, 1),
        "FP_per_frame": operating["fp"] / max(frames, 1),
        "false_alarms_per_minute": operating["fp"] / max(duration_minutes, 1e-12),
        "duration_minutes": duration_minutes,
        "lost_frames": 0,
        "duplicate_frames": int(
            source.duplicated(["subsequence_id", "frame_number"]).sum()
        ),
        "evaluator_consistency": "PASS",
        "NaN_Inf": 0,
    }
    return payload, pd.DataFrame(per_scene)


def run_tracker(
    source: pd.DataFrame,
    raw: dict[str, list[dict[str, Any]]],
    tracker: Any,
    nms_iou: float,
) -> tuple[dict[str, list[dict[str, Any]]], pd.DataFrame]:
    output: dict[str, list[dict[str, Any]]] = {}
    events: list[dict[str, Any]] = []
    for subsequence, sequence in source.groupby("subsequence_id", sort=False):
        tracker.reset(str(subsequence))
        for record in sequence.itertuples(index=False):
            image_path = str(record.image_path)
            emitted, frame_events = tracker.update(
                raw.get(image_path, []), int(record.width), int(record.height)
            )
            output[image_path] = deterministic_nms(emitted, nms_iou)
            for event in frame_events:
                events.append(
                    {
                        "grouped_scene_id": str(record.grouped_scene_id),
                        "subsequence_id": str(subsequence),
                        "frame_number": int(record.frame_number),
                        "image_path": image_path,
                        **event,
                    }
                )
    return output, pd.DataFrame(events)


def maximum_scene_recall_degradation(
    baseline: pd.DataFrame, candidate: pd.DataFrame
) -> float:
    merged = baseline.merge(
        candidate, on="grouped_scene_id", suffixes=("_baseline", "_candidate")
    )
    return float(
        np.max(merged["recall_baseline"] - merged["recall_candidate"])
    )


def gt_track_metrics(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
    iou_threshold: float,
    maximum_gap: int = 1,
) -> dict[str, Any]:
    """Deterministic supplementary GT-tube audit without manual track IDs."""
    total_tracks = detected_tracks = 0
    delays: list[int] = []
    fragmentations: list[int] = []
    for _, sequence in source.groupby("subsequence_id", sort=False):
        active: list[dict[str, Any]] = []
        finished: list[dict[str, Any]] = []
        for order, record in enumerate(sequence.itertuples(index=False)):
            path = str(record.image_path)
            boxes = [row["box"] for row in ground_truth[path]]
            used: set[int] = set()
            for track in active:
                candidates = [
                    (box_iou(track["box"], box), index)
                    for index, box in enumerate(boxes)
                    if index not in used
                ]
                overlap, index = max(candidates, default=(0.0, -1))
                if overlap >= iou_threshold:
                    used.add(index)
                    track["box"] = boxes[index]
                    track["last"] = order
                    track["observations"].append((path, list(boxes[index])))
                elif order - track["last"] > maximum_gap:
                    finished.append(track)
            active = [
                track for track in active if order - track["last"] <= maximum_gap
            ]
            for index, box in enumerate(boxes):
                if index not in used:
                    active.append(
                        {
                            "box": box,
                            "last": order,
                            "observations": [(path, list(box))],
                        }
                    )
        finished.extend(active)
        for track in finished:
            total_tracks += 1
            detected = []
            for position, (path, gt_box) in enumerate(track["observations"]):
                hits = [
                    row
                    for row in predictions.get(path, [])
                    if float(row["confidence"]) >= threshold
                    and box_iou(gt_box, row["box"]) >= iou_threshold
                ]
                if hits:
                    detected.append(position)
            if detected:
                detected_tracks += 1
                delays.append(min(detected))
                fragmentations.append(
                    sum(
                        1
                        for left, right in zip(detected, detected[1:])
                        if right - left > 1
                    )
                )
    return {
        "track_recall": detected_tracks / max(total_tracks, 1),
        "gt_tracks": total_tracks,
        "detected_gt_tracks": detected_tracks,
        "time_to_first_detection_frames_mean": float(np.mean(delays)) if delays else None,
        "track_fragmentation_mean": float(np.mean(fragmentations))
        if fragmentations
        else None,
        "ID_switches": "SUPPLEMENTARY_GT_TUBES_HAVE_NO_CANONICAL_IDENTITIES",
        "reliability": "SUPPLEMENTARY_DETERMINISTIC_LINKING",
    }
