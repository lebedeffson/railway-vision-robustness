from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.operator_assistant_evidence_v2.common import (
    OUTPUT,
    PROJECT,
    assert_test_sealed,
    atomic_csv,
    atomic_json,
    config,
    sha256,
)
from src.review_assistant.event_aggregator import box_iou
from src.review_assistant.event_aggregator_v2 import EvidenceEventAggregator
from src.review_assistant.models import ReviewEvent
from src.review_assistant.processor import merge_review_candidates
from src.review_assistant.processor_v2 import FailSafeReviewProcessor


EventAggregator = EvidenceEventAggregator
ReviewProcessor = FailSafeReviewProcessor


TRACE_PATH = OUTPUT / "PRE_AGGREGATION_OBSERVATIONS.parquet"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE_NOT_GIT"


def _settings(
    base: dict[str, Any],
    *,
    join: float | None = None,
    iou: float | None = None,
    center: float | None = None,
    reopen: float | None = None,
) -> dict[str, Any]:
    output = dict(base)
    if join is not None:
        output["join_time_seconds"] = join
        output["close_after_seconds"] = join
    if iou is not None:
        output["minimum_iou"] = iou
    if center is not None:
        output["maximum_center_distance_ratio"] = center
    if reopen is not None:
        output["reopen_window_seconds"] = reopen
    return output


def _event_signature(events: list[ReviewEvent]) -> list[dict[str, Any]]:
    return [
        {
            "event_id": event.event_id,
            "start_time": round(event.start_time, 9),
            "end_time": round(event.end_time, 9),
            "source_label": event.source_label,
            "processing_status": event.processing_status,
            "candidate_ids": [
                item.candidate_id for item in event.detections
            ],
        }
        for event in events
    ]


def _candidate_from_trace(row: Any) -> dict[str, Any]:
    return {
        "candidate_id": str(row.candidate_id),
        "box": [
            float(row.bbox_x1),
            float(row.bbox_y1),
            float(row.bbox_x2),
            float(row.bbox_y2),
        ],
        "confidence": float(row.output_confidence),
        "raw_confidence": float(row.detector_confidence),
        "review_source": str(row.review_source),
        "source": str(row.final_source),
        "track_id": int(row.track_id) if int(row.track_id) >= 0 else None,
        "interpolated": bool(row.is_interpolated),
        "confirmed": bool(row.confirmed),
        "processing_status": str(row.processing_status),
    }


def _replay(
    trace: pd.DataFrame,
    parameters: dict[str, Any],
    *,
    width: int,
    height: int,
    camera_id: str,
) -> list[ReviewEvent]:
    aggregator = EventAggregator(
        parameters,
        video_id="DEVELOPMENT_EVENT_BENCHMARK_V2",
        camera_id=camera_id,
        frame_width=width,
        frame_height=height,
    )
    sent = trace[trace["sent_to_aggregator"].astype(bool)].sort_values(
        ["video_frame_id", "aggregator_order"]
    )
    for frame_number, group in sent.groupby("video_frame_id", sort=True):
        timestamp = float(group["timestamp"].iloc[0])
        aggregator.observe(
            int(frame_number),
            timestamp,
            [_candidate_from_trace(row) for row in group.itertuples(index=False)],
        )
    return aggregator.finalize()


def _direct_trace() -> tuple[pd.DataFrame, list[ReviewEvent], dict[str, int]]:
    settings = config()
    benchmark = settings["benchmark"]
    manifest = pd.read_csv(
        OUTPUT / "development_event_benchmark_v2_manifest.csv"
    )
    video_path = OUTPUT / "development_event_benchmark_v2.mp4"
    processor = ReviewProcessor()
    detector, temporal = processor._models("combined_queue")
    aggregator = EventAggregator(
        settings["events"]["baseline"],
        video_id="DEVELOPMENT_EVENT_BENCHMARK_V2",
        camera_id=benchmark["source_sequence_id"],
        frame_width=int(benchmark["width"]),
        frame_height=int(benchmark["height"]),
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open development benchmark v2")
    trace_rows: list[dict[str, Any]] = []
    raw_total = tracker_total = verifier_accepted = verifier_rejected = 0
    aggregator_total = 0
    frame_number = -1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_number += 1
        source = manifest.iloc[frame_number]
        timestamp = frame_number / float(benchmark["fps"])
        candidates = detector.candidates(frame)
        raw_total += len(candidates)
        records: dict[str, dict[str, Any]] = {}
        for ordinal, candidate in enumerate(candidates):
            candidate_id = f"C-{frame_number:06d}-{ordinal:04d}"
            candidate["candidate_id"] = candidate_id
            candidate["processing_status"] = "NORMAL"
            box = candidate["box"]
            records[candidate_id] = {
                "run_id": "DEVELOPMENT_EVENT_BENCHMARK_V2",
                "candidate_id": candidate_id,
                "camera_id": benchmark["source_sequence_id"],
                "scene_id": source["scene_id"],
                "source_frame_id": int(source["source_frame_id"]),
                "video_frame_id": frame_number,
                "timestamp": timestamp,
                "bbox_x1": float(box[0]),
                "bbox_y1": float(box[1]),
                "bbox_x2": float(box[2]),
                "bbox_y2": float(box[3]),
                "detector_confidence": float(candidate["confidence"]),
                "output_confidence": float(candidate["confidence"]),
                "detector_threshold": float(
                    processor.config["detector"]["candidate_floor"]
                ),
                "tracker_name": "ocsort",
                "track_id": -1,
                "is_interpolated": False,
                "verifier_score": -1.0,
                "verifier_score_available": False,
                "verifier_decision": "NOT_EVALUATED",
                "processing_status": "NORMAL",
                "rejection_reason": "TRACKER_NOT_EMITTED",
                "review_source": "",
                "final_source": "detector",
                "confirmed": bool(candidate.get("confirmed", False)),
                "tracker_emitted": False,
                "sent_to_aggregator": False,
                "aggregator_order": -1,
                "event_id": "",
            }
        baseline = detector.standard(candidates)
        temporal_rows, _ = temporal.update(frame, candidates)
        tracker_total += len(temporal.last_trace_rows)
        for temporal_trace in temporal.last_trace_rows:
            candidate_id = str(temporal_trace["candidate_id"])
            if candidate_id not in records:
                box = temporal_trace["box"]
                records[candidate_id] = {
                    "run_id": "DEVELOPMENT_EVENT_BENCHMARK_V2",
                    "candidate_id": candidate_id,
                    "camera_id": benchmark["source_sequence_id"],
                    "scene_id": source["scene_id"],
                    "source_frame_id": int(source["source_frame_id"]),
                    "video_frame_id": frame_number,
                    "timestamp": timestamp,
                    "bbox_x1": float(box[0]),
                    "bbox_y1": float(box[1]),
                    "bbox_x2": float(box[2]),
                    "bbox_y2": float(box[3]),
                    "detector_confidence": float(
                        temporal_trace.get("raw_confidence", 0.0)
                    ),
                    "output_confidence": float(
                        temporal_trace.get("confidence", 0.0)
                    ),
                    "detector_threshold": float(
                        processor.config["detector"]["candidate_floor"]
                    ),
                    "tracker_name": "ocsort",
                    "track_id": int(temporal_trace["track_id"]),
                    "is_interpolated": bool(
                        temporal_trace.get("interpolated", False)
                    ),
                    "verifier_score": -1.0,
                    "verifier_score_available": False,
                    "verifier_decision": "NOT_EVALUATED",
                    "processing_status": "NORMAL",
                    "rejection_reason": "",
                    "review_source": "",
                    "final_source": str(temporal_trace.get("source", "")),
                    "confirmed": False,
                    "tracker_emitted": True,
                    "sent_to_aggregator": False,
                    "aggregator_order": -1,
                    "event_id": "",
                }
            record = records[candidate_id]
            record.update(
                {
                    "output_confidence": float(
                        temporal_trace.get("confidence", 0.0)
                    ),
                    "track_id": int(temporal_trace["track_id"]),
                    "is_interpolated": bool(
                        temporal_trace.get("interpolated", False)
                    ),
                    "verifier_score": (
                        float(temporal_trace["verifier_score"])
                        if temporal_trace["verifier_score"] is not None
                        else -1.0
                    ),
                    "verifier_score_available": (
                        temporal_trace["verifier_score"] is not None
                    ),
                    "verifier_decision": str(
                        temporal_trace["verifier_decision"]
                    ),
                    "processing_status": str(
                        temporal_trace["processing_status"]
                    ),
                    "final_source": str(temporal_trace.get("source", "")),
                    "tracker_emitted": True,
                    "rejection_reason": (
                        "" if temporal_trace["accepted"] else "VERIFIER_REJECTED"
                    ),
                }
            )
            if temporal_trace["accepted"]:
                verifier_accepted += 1
            else:
                verifier_rejected += 1
        review_rows = merge_review_candidates(baseline, temporal_rows)
        touched = aggregator.observe(frame_number, timestamp, review_rows)
        if len(touched) != len(review_rows):
            raise RuntimeError("Aggregator touched/event input cardinality mismatch")
        aggregator_total += len(review_rows)
        for order, (row, event) in enumerate(zip(review_rows, touched, strict=True)):
            candidate_id = str(row["candidate_id"])
            record = records[candidate_id]
            record.update(
                {
                    "output_confidence": float(row.get("confidence", 0.0)),
                    "review_source": str(row["review_source"]),
                    "final_source": str(row.get("source", "")),
                    "track_id": (
                        int(row["track_id"])
                        if row.get("track_id") is not None
                        else -1
                    ),
                    "confirmed": bool(row.get("confirmed", False)),
                    "processing_status": str(
                        row.get("processing_status", "NORMAL")
                    ),
                    "sent_to_aggregator": True,
                    "aggregator_order": order,
                    "event_id": event.event_id,
                    "rejection_reason": "",
                }
            )
        trace_rows.extend(records.values())
    capture.release()
    if frame_number + 1 != int(benchmark["frames"]):
        raise RuntimeError(f"Expected 100 video frames, got {frame_number + 1}")
    events = aggregator.finalize()
    event_observations = sum(len(event.detections) for event in events)
    counts = {
        "raw_detector_boxes": raw_total,
        "tracker_observations": tracker_total,
        "verifier_accepted": verifier_accepted,
        "verifier_rejected": verifier_rejected,
        "aggregator_input": aggregator_total,
        "event_observations": event_observations,
        "unique_events": len(events),
    }
    trace = pd.DataFrame(trace_rows).sort_values(
        ["video_frame_id", "candidate_id"]
    )
    trace.to_parquet(TRACE_PATH, index=False)
    atomic_csv(
        OUTPUT / "PIPELINE_STAGE_COUNTS.csv",
        pd.DataFrame(
            [{"stage": key, "count": value} for key, value in counts.items()]
        ),
    )
    return trace, events, counts


def _match_event_episodes(
    event: ReviewEvent,
    gt_boxes: pd.DataFrame,
    threshold: float,
) -> set[str]:
    episodes: set[str] = set()
    for frame_number, detections in itertools.groupby(
        sorted(event.detections, key=lambda item: item.frame_number),
        key=lambda item: item.frame_number,
    ):
        detection_list = list(detections)
        ground_truth = gt_boxes[
            gt_boxes["video_frame_id"] == int(frame_number)
        ]
        if ground_truth.empty:
            continue
        gt_rows = list(ground_truth.itertuples(index=False))
        scores = np.asarray(
            [
                [
                    box_iou(
                        detection.box,
                        [
                            float(gt.bbox_x1),
                            float(gt.bbox_y1),
                            float(gt.bbox_x2),
                            float(gt.bbox_y2),
                        ],
                    )
                    for gt in gt_rows
                ]
                for detection in detection_list
            ],
            dtype=float,
        )
        left, right = linear_sum_assignment(1.0 - scores)
        for row, column in zip(left, right, strict=True):
            if scores[row, column] >= threshold:
                episodes.add(str(gt_rows[int(column)].episode_id))
    return episodes


def _evaluate_events(
    events: list[ReviewEvent],
    gt_boxes: pd.DataFrame,
    episodes: pd.DataFrame,
    parameters: dict[str, Any],
    *,
    duration_seconds: float,
    fps: float,
    configuration_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    threshold = float(config()["frozen_pipeline"]["gt_match_iou"])
    event_episode_sets: dict[str, set[str]] = {
        event.event_id: _match_event_episodes(event, gt_boxes, threshold)
        for event in events
    }
    related_events = {
        event_id for event_id, values in event_episode_sets.items() if values
    }
    represented = set().union(*event_episode_sets.values()) if events else set()
    false_events = [
        event for event in events if not event_episode_sets[event.event_id]
    ]
    false_merges = [
        event
        for event in events
        if len(event_episode_sets[event.event_id]) > 1
    ]
    open_latencies: list[float] = []
    close_latencies: list[float] = []
    durations = [event.duration for event in events]
    for event in events:
        distinct_temporal: set[int] = set()
        activation_time = event.start_time
        for detection in sorted(
            event.detections, key=lambda item: (item.timestamp, item.frame_number)
        ):
            if detection.source in {"BASELINE", "BOTH"} or detection.confirmed:
                activation_time = detection.timestamp
                break
            if detection.source in {"TEMPORAL_ONLY", "BOTH"}:
                distinct_temporal.add(detection.frame_number)
            if len(distinct_temporal) >= int(
                parameters["temporal_minimum_observations"]
            ):
                activation_time = detection.timestamp
                break
        open_latencies.append(max(activation_time - event.start_time, 0.0))
        close_time = min(
            event.end_time + float(parameters["close_after_seconds"]),
            duration_seconds,
        )
        close_latencies.append(max(close_time - event.end_time, 0.0))
    per_episode: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        episode_id = str(episode.episode_id)
        linked = [
            event
            for event in events
            if episode_id in event_episode_sets[event.event_id]
        ]
        gt_frames = set(
            gt_boxes.loc[
                gt_boxes["episode_id"] == episode_id, "video_frame_id"
            ].astype(int)
        )
        covered_frames: set[int] = set()
        for event in linked:
            clip_start = max(
                event.start_time - float(parameters["pre_roll_seconds"]), 0.0
            )
            clip_end = min(
                event.end_time + float(parameters["post_roll_seconds"]),
                duration_seconds,
            )
            covered_frames |= {
                frame
                for frame in gt_frames
                if clip_start <= frame / fps <= clip_end
            }
        per_episode.append(
            {
                "configuration_id": configuration_id,
                "episode_id": episode_id,
                "episode_start_frame": int(episode.start_frame),
                "episode_end_frame": int(episode.end_frame),
                "episode_frames": len(gt_frames),
                "linked_events": len(linked),
                "represented": bool(linked),
                "covered_frames": len(covered_frames),
                "episode_frame_coverage": (
                    len(covered_frames) / len(gt_frames) if gt_frames else 0.0
                ),
            }
        )
    episode_frame_coverage = (
        statistics.fmean(
            row["episode_frame_coverage"] for row in per_episode
        )
        if per_episode
        else 0.0
    )
    event_rows: list[dict[str, Any]] = []
    for event in events:
        values = sorted(event_episode_sets[event.event_id])
        event_rows.append(
            {
                "configuration_id": configuration_id,
                "event_id": event.event_id,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "detections": len(event.detections),
                "matched_episode_count": len(values),
                "matched_episode_ids": json.dumps(values),
                "is_false_event": not values,
                "is_false_merge": len(values) > 1,
            }
        )
    row = {
        "configuration_id": configuration_id,
        "join_time_seconds": float(parameters["join_time_seconds"]),
        "minimum_iou": float(parameters["minimum_iou"]),
        "maximum_center_distance_ratio": float(
            parameters["maximum_center_distance_ratio"]
        ),
        "reopen_window_seconds": float(parameters["reopen_window_seconds"]),
        "raw_observations": sum(len(event.detections) for event in events),
        "unique_events": len(events),
        "observations_per_event": (
            sum(len(event.detections) for event in events) / len(events)
            if events
            else 0.0
        ),
        "events_per_video_hour": len(events)
        * 3600.0
        / max(duration_seconds, 1e-12),
        "event_duration_mean": (
            statistics.fmean(durations) if durations else 0.0
        ),
        "event_duration_median": (
            statistics.median(durations) if durations else 0.0
        ),
        "event_opening_latency_p50": (
            statistics.median(open_latencies) if open_latencies else 0.0
        ),
        "event_opening_latency_p95": (
            float(np.percentile(open_latencies, 95)) if open_latencies else 0.0
        ),
        "event_opening_latency_p99": (
            float(np.percentile(open_latencies, 99)) if open_latencies else 0.0
        ),
        "event_closing_latency_p50": (
            statistics.median(close_latencies) if close_latencies else 0.0
        ),
        "event_closing_latency_p95": (
            float(np.percentile(close_latencies, 95)) if close_latencies else 0.0
        ),
        "event_closing_latency_p99": (
            float(np.percentile(close_latencies, 99)) if close_latencies else 0.0
        ),
        "true_episode_recall": len(represented) / max(len(episodes), 1),
        "false_events_per_minute": len(false_events)
        * 60.0
        / max(duration_seconds, 1e-12),
        "fragmentation": len(related_events) / max(len(episodes), 1),
        "false_merge_rate": len(false_merges) / max(len(events), 1),
        "episode_frame_coverage": episode_frame_coverage,
        "reopened_events": sum(event.reopen_count for event in events),
        "merged_track_id_switches": sum(
            max(len(event.track_ids) - 1, 0) for event in events
        ),
        "baseline_configuration": (
            float(parameters["join_time_seconds"]) == 3
            and float(parameters["minimum_iou"]) == 0.2
            and float(parameters["maximum_center_distance_ratio"]) == 0.1
            and float(parameters["reopen_window_seconds"]) == 30
        ),
    }
    return row, per_episode, event_rows


def _sensitivity(trace: pd.DataFrame) -> None:
    settings = config()
    benchmark = settings["benchmark"]
    base = settings["events"]["baseline"]
    grid = settings["events"]["sensitivity"]
    gt_boxes = pd.read_csv(OUTPUT / "GT_PERSON_BOXES.csv")
    episodes = pd.read_csv(OUTPUT / "GT_PERSON_EPISODES.csv")
    rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for join, iou, center, reopen in itertools.product(
        grid["join_time_seconds"],
        grid["minimum_iou"],
        grid["maximum_center_distance_ratio"],
        grid["reopen_window_seconds"],
    ):
        parameters = _settings(
            base,
            join=float(join),
            iou=float(iou),
            center=float(center),
            reopen=float(reopen),
        )
        identifier = f"J{join}_I{iou:.2f}_C{center:.2f}_R{reopen}"
        events = _replay(
            trace,
            parameters,
            width=int(benchmark["width"]),
            height=int(benchmark["height"]),
            camera_id=benchmark["source_sequence_id"],
        )
        row, per_episode, event_rows = _evaluate_events(
            events,
            gt_boxes,
            episodes,
            parameters,
            duration_seconds=float(benchmark["frames"])
            / float(benchmark["fps"]),
            fps=float(benchmark["fps"]),
            configuration_id=identifier,
        )
        rows.append(row)
        episode_rows.extend(per_episode)
        errors.extend(
            item
            for item in event_rows
            if item["is_false_event"] or item["is_false_merge"]
        )
    frame = pd.DataFrame(rows)
    if len(frame) != int(grid["expected_configurations"]):
        raise RuntimeError(f"Incomplete sensitivity grid: {len(frame)}")
    atomic_csv(OUTPUT / "EVENT_SENSITIVITY_FULL.csv", frame)
    atomic_csv(
        OUTPUT / "EVENT_SENSITIVITY_PER_EPISODE.csv",
        pd.DataFrame(episode_rows),
    )
    atomic_csv(OUTPUT / "EVENT_ERROR_CASES.csv", pd.DataFrame(errors))


def run_short() -> None:
    assert_test_sealed()
    lock = OUTPUT / "EVENT_PROTOCOL_LOCK.json"
    if not lock.is_file():
        raise RuntimeError("EVENT_PROTOCOL_LOCK.json must exist before inference")
    trace, direct_events, counts = _direct_trace()
    settings = config()
    benchmark = settings["benchmark"]
    replay_events = _replay(
        trace,
        settings["events"]["baseline"],
        width=int(benchmark["width"]),
        height=int(benchmark["height"]),
        camera_id=benchmark["source_sequence_id"],
    )
    direct_signature = _event_signature(direct_events)
    replay_signature = _event_signature(replay_events)
    exact = direct_signature == replay_signature
    atomic_json(
        OUTPUT / "EVENT_BASELINE_REPLAY.json",
        {
            "direct_unique_events": len(direct_events),
            "replay_unique_events": len(replay_events),
            "direct_raw_detector_boxes": counts["raw_detector_boxes"],
            "exact_event_signature_match": exact,
            "direct_signature": direct_signature,
            "replay_signature": replay_signature,
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    if not exact:
        raise RuntimeError("Direct/replay event signatures differ")
    _sensitivity(trace)
    lock_payload = json.loads(lock.read_text(encoding="utf-8"))
    atomic_json(
        OUTPUT / "INPUT_PROVENANCE.json",
        {
            "git_commit": _git_commit(),
            "event_protocol_lock_sha256": sha256(lock),
            "benchmark_video_sha256": lock_payload["inputs"][
                "benchmark_video"
            ]["sha256"],
            "benchmark_manifest_sha256": lock_payload["inputs"][
                "benchmark_manifest"
            ]["sha256"],
            "annotation_sha256": lock_payload["inputs"]["source_annotation"][
                "sha256"
            ],
            "detector_checkpoint_sha256": lock_payload["inputs"][
                "detector_checkpoint"
            ]["sha256"],
            "verifier_model_sha256": lock_payload["inputs"][
                "verifier_model"
            ]["sha256"],
            "trace_sha256": sha256(TRACE_PATH),
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    print(
        json.dumps(
            {
                **counts,
                "direct_replay_match": exact,
                "sensitivity_configurations": 108,
            },
            sort_keys=True,
        )
    )


def _build_long_video(path: Path, total_frames: int) -> None:
    source = OUTPUT / "development_event_benchmark_v2.mp4"
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create long benchmark video")
    written = 0
    try:
        while written < total_frames:
            capture = cv2.VideoCapture(str(source))
            while written < total_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
            capture.release()
    finally:
        writer.release()
    if written != total_frames:
        raise RuntimeError(f"Long benchmark frame count mismatch: {written}")


def _stage_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = [
        "video_decode_ms",
        "detector_inference_ms",
        "tracking_ms",
        "track_verification_ms",
        "event_aggregation_ms",
        "full_pipeline_ms",
    ]
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        rows.append(
            {
                "stage": column.removesuffix("_ms"),
                "count": len(values),
                "mean_ms": float(np.mean(values)),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "p99_ms": float(np.percentile(values, 99)),
                "max_ms": float(np.max(values)),
                "std_ms": float(np.std(values)),
            }
        )
    return pd.DataFrame(rows)


def run_runtime() -> None:
    assert_test_sealed()
    settings = config()
    runtime = settings["runtime"]
    benchmark = settings["benchmark"]
    long_video = OUTPUT / "development_event_benchmark_v2_long_1100.mp4"
    if not long_video.is_file():
        _build_long_video(long_video, int(runtime["repeat_benchmark_frames"]))
    processor = ReviewProcessor()
    detector, temporal = processor._models("combined_queue")
    capture = cv2.VideoCapture(str(long_video))
    aggregator = EventAggregator(
        settings["events"]["baseline"],
        video_id="DEVELOPMENT_EVENT_BENCHMARK_V2_RUNTIME",
        camera_id=benchmark["source_sequence_id"],
        frame_width=int(benchmark["width"]),
        frame_height=int(benchmark["height"]),
    )
    rows: list[dict[str, Any]] = []
    warmup = int(runtime["warmup_frames"])
    measured = int(runtime["measured_frames"])
    for frame_number in range(warmup + measured):
        full_started = time.perf_counter()
        decode_started = time.perf_counter()
        ok, frame = capture.read()
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        if not ok:
            break
        detector_started = time.perf_counter()
        candidates = detector.candidates(frame)
        detector_ms = (time.perf_counter() - detector_started) * 1000.0
        for ordinal, candidate in enumerate(candidates):
            candidate["candidate_id"] = (
                f"L-{frame_number:06d}-{ordinal:04d}"
            )
        baseline = detector.standard(candidates)
        temporal_rows, _ = temporal.update(frame, candidates)
        review_rows = merge_review_candidates(baseline, temporal_rows)
        aggregate_started = time.perf_counter()
        if frame_number >= warmup:
            aggregator.observe(
                frame_number - warmup,
                (frame_number - warmup) / float(benchmark["fps"]),
                review_rows,
            )
        aggregate_ms = (time.perf_counter() - aggregate_started) * 1000.0
        full_ms = (time.perf_counter() - full_started) * 1000.0
        if frame_number >= warmup:
            rows.append(
                {
                    "measured_frame_id": frame_number - warmup,
                    "source_repeat_frame_id": frame_number % 100,
                    "video_decode_ms": decode_ms,
                    "detector_inference_ms": detector_ms,
                    "tracking_ms": float(temporal.last_tracker_ms),
                    "track_verification_ms": float(temporal.last_verifier_ms),
                    "event_aggregation_ms": aggregate_ms,
                    "full_pipeline_ms": full_ms,
                    "raw_detections": len(candidates),
                    "tracker_observations": len(temporal.last_trace_rows),
                    "verifier_accepted": sum(
                        bool(row["accepted"])
                        for row in temporal.last_trace_rows
                    ),
                    "verifier_rejected": sum(
                        not bool(row["accepted"])
                        for row in temporal.last_trace_rows
                    ),
                    "aggregator_observations": len(review_rows),
                    "active_event_queue": len(aggregator.accepted_events()),
                }
            )
    capture.release()
    frame = pd.DataFrame(rows)
    if len(frame) != measured:
        raise RuntimeError(f"Expected {measured} measured frames, got {len(frame)}")
    events = aggregator.finalize()
    atomic_csv(OUTPUT / "EVENT_RUNTIME_PER_FRAME.csv", frame)
    summary = _stage_summary(frame)
    atomic_csv(OUTPUT / "EVENT_RUNTIME_SUMMARY.csv", summary)
    atomic_json(
        OUTPUT / "EVENT_RUNTIME_FINAL.json",
        {
            "warmup_frames": warmup,
            "measured_frames": measured,
            "raw_detections": int(frame["raw_detections"].sum()),
            "tracker_observations": int(frame["tracker_observations"].sum()),
            "verifier_accepted": int(frame["verifier_accepted"].sum()),
            "verifier_rejected": int(frame["verifier_rejected"].sum()),
            "aggregator_observations": int(
                frame["aggregator_observations"].sum()
            ),
            "unique_events": len(events),
            "event_observations": sum(
                len(event.detections) for event in events
            ),
            "baseline_events": sum(
                event.source_label == "BASELINE" for event in events
            ),
            "temporal_only_events": sum(
                event.source_label == "TEMPORAL_ONLY" for event in events
            ),
            "combined_events": sum(
                event.source_label == "BOTH" for event in events
            ),
            "detections_per_event": (
                sum(len(event.detections) for event in events)
                / max(len(events), 1)
            ),
            "event_duration_mean": (
                statistics.fmean(event.duration for event in events)
                if events
                else 0.0
            ),
            "event_duration_median": (
                statistics.median(event.duration for event in events)
                if events
                else 0.0
            ),
            "track_switches_merged": sum(
                max(len(event.track_ids) - 1, 0) for event in events
            ),
            "reopened_events": sum(event.reopen_count for event in events),
            "end_to_end_fps": 1000.0
            / max(
                float(
                    summary.loc[
                        summary["stage"] == "full_pipeline", "mean_ms"
                    ].iloc[0]
                ),
                1e-12,
            ),
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    print(
        json.dumps(
            {
                "measured_frames": len(frame),
                "events": len(events),
                "end_to_end_fps": 1000.0
                / float(frame["full_pipeline_ms"].mean()),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("short", "runtime", "all"), default="all"
    )
    args = parser.parse_args()
    if args.phase in {"short", "all"}:
        run_short()
    if args.phase in {"runtime", "all"}:
        run_runtime()


if __name__ == "__main__":
    main()
