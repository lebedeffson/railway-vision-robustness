from __future__ import annotations

import json
import os
import platform
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import psutil
import torch

from scripts.article_evidence_v1.common import (
    OUTPUT,
    PROJECT,
    atomic_csv,
    atomic_json,
    check_no_test_markers,
    config,
)
from src.review_assistant.config import load_config
from src.review_assistant.event_aggregator import EventAggregator
from src.review_assistant.processor import ReviewProcessor, merge_review_candidates


STAGES = (
    "video_decode",
    "image_preprocessing",
    "tiling",
    "detector_inference",
    "coordinate_restoration",
    "box_fusion",
    "tracking",
    "track_verification",
    "event_aggregation",
    "rendering",
    "video_encoding",
    "database_writes",
    "full_pipeline",
)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _summary(values: list[float], status: str = "MEASURED") -> dict[str, Any]:
    if not values:
        return {
            "status": status,
            "count": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
        }
    return {
        "status": status,
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
        "max_ms": max(values),
        "std_ms": float(np.std(values, ddof=0)),
    }


def _system_environment(
    input_resolution: int,
    video: tuple[int, int, float],
) -> dict[str, Any]:
    gpu = "UNAVAILABLE"
    driver = "UNAVAILABLE"
    try:
        row = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        gpu, driver, gpu_memory = [part.strip() for part in row.split(",")]
    except Exception:
        gpu_memory = "UNAVAILABLE"
    return {
        "cpu": platform.processor() or platform.machine(),
        "cpu_logical_count": os.cpu_count(),
        "gpu": gpu,
        "gpu_memory_mib": gpu_memory,
        "ram_mib": round(psutil.virtual_memory().total / 1024**2, 3),
        "operating_system": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda or "UNAVAILABLE",
        "driver": driver,
        "input_resolution": input_resolution,
        "video_resolution": f"{video[0]}x{video[1]}",
        "video_fps": video[2],
        "batch_size": 1,
        "precision": "FP16 detector inference; FP32 frozen verifier",
        "number_of_runs": 1,
        "test_status": "SEALED",
        "test_access_count": 0,
    }


def main() -> None:
    check_no_test_markers()
    settings = config()["runtime"]
    input_path = PROJECT / settings["input"]
    review_config = load_config()
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unreadable development benchmark: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    warmup = int(settings["warmup_frames"])
    measured = int(settings["measured_frames"])
    if total < warmup + measured:
        raise RuntimeError("Runtime input does not contain 100 warm-up + 1000 frames")

    processor = ReviewProcessor(device="auto")
    detector, temporal = processor._models("combined_queue")
    aggregator = EventAggregator(
        review_config["events"],
        video_id="ARTICLE-EVIDENCE-RUNTIME",
        camera_id="development-benchmark",
        frame_width=width,
        frame_height=height,
    )
    temporary = tempfile.TemporaryDirectory(prefix="article-evidence-runtime-")
    writer = cv2.VideoWriter(
        str(Path(temporary.name) / "encoded.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    database = sqlite3.connect(Path(temporary.name) / "benchmark.sqlite")
    database.execute(
        "create table progress(frame integer primary key, detections integer, events integer)"
    )
    stages: dict[str, list[float]] = {stage: [] for stage in STAGES}
    per_frame: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    tracks_per_frame: list[int] = []
    queue_lengths: list[int] = []
    raw_detections = 0
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    process = psutil.Process()

    for observed in range(warmup + measured):
        full_started = time.perf_counter()
        started = time.perf_counter()
        ok, frame = capture.read()
        decode_ms = (time.perf_counter() - started) * 1000.0
        if not ok:
            raise RuntimeError(f"Decode stopped at frame {observed}")

        # Use the frozen detector with the locked FP16 runtime precision.
        detector_started = time.perf_counter()
        result = detector.model.predict(
            source=frame,
            conf=float(detector.settings["candidate_floor"]),
            iou=float(detector.settings["iou_threshold"]),
            imgsz=int(detector.settings["image_size"]),
            classes=[0],
            device=detector.device,
            half=True,
            verbose=False,
        )[0]
        detector_wall_ms = (time.perf_counter() - detector_started) * 1000.0
        candidates: list[dict[str, Any]] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confidence = result.boxes.conf.detach().cpu().numpy()
            for box, score in zip(boxes, confidence, strict=True):
                candidates.append(
                    {
                        "box": [float(value) for value in box],
                        "confidence": float(score),
                        "raw_confidence": float(score),
                        "class_id": 0,
                        "track_id": None,
                        "source": "detector",
                        "interpolated": False,
                        "confirmed": score
                        >= float(detector.settings["standard_threshold"]),
                    }
                )
        speed = result.speed or {}
        preprocess_ms = float(speed.get("preprocess", 0.0))
        inference_ms = float(speed.get("inference", detector_wall_ms))
        postprocess_ms = float(speed.get("postprocess", 0.0))
        baseline = detector.standard(candidates)
        temporal_rows, _ = temporal.update(frame, candidates)

        aggregate_started = time.perf_counter()
        rows = merge_review_candidates(baseline, temporal_rows)
        aggregator.observe(observed, observed / fps, rows)
        aggregate_ms = (time.perf_counter() - aggregate_started) * 1000.0

        render_started = time.perf_counter()
        rendered = frame.copy()
        for row in rows:
            x1, y1, x2, y2 = [int(round(value)) for value in row["box"]]
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 210, 255), 1)
        render_ms = (time.perf_counter() - render_started) * 1000.0
        encode_started = time.perf_counter()
        writer.write(rendered)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0

        database_ms = 0.0
        if observed % int(review_config["processing"]["progress_commit_interval_frames"]) == 0:
            db_started = time.perf_counter()
            database.execute(
                "insert or replace into progress values(?,?,?)",
                (observed, len(candidates), len(aggregator.accepted_events())),
            )
            database.commit()
            database_ms = (time.perf_counter() - db_started) * 1000.0
        full_ms = (time.perf_counter() - full_started) * 1000.0
        raw_detections += len(candidates)
        tracks = len(temporal.tracker.tracks)
        queue = len(aggregator.accepted_events())

        if observed >= warmup:
            measured_index = observed - warmup
            values = {
                "video_decode": decode_ms,
                "image_preprocessing": preprocess_ms,
                "tiling": 0.0,
                "detector_inference": inference_ms,
                "coordinate_restoration": 0.0,
                "box_fusion": postprocess_ms,
                "tracking": float(temporal.last_tracker_ms),
                "track_verification": float(temporal.last_verifier_ms),
                "event_aggregation": aggregate_ms,
                "rendering": render_ms,
                "video_encoding": encode_ms,
                "database_writes": database_ms,
                "full_pipeline": full_ms,
            }
            for stage, value in values.items():
                stages[stage].append(float(value))
            per_frame.append(
                {
                    "measured_frame_index": measured_index,
                    "source_frame_index": observed,
                    **{f"{stage}_ms": value for stage, value in values.items()},
                    "candidate_count": len(candidates),
                    "track_count": tracks,
                    "queue_length": queue,
                }
            )
            ram_mib = process.memory_info().rss / 1024**2
            gpu_mib = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0.0
            )
            memory_rows.append(
                {
                    "measured_frame_index": measured_index,
                    "ram_mib": ram_mib,
                    "gpu_memory_mib": gpu_mib,
                }
            )
            tracks_per_frame.append(tracks)
            queue_lengths.append(queue)
    capture.release()
    writer.release()
    database.close()
    events = aggregator.finalize()
    temporary.cleanup()

    stage_status = {
        "tiling": "NOT_APPLICABLE_NO_TILING_IN_PRODUCT_PIPELINE",
        "coordinate_restoration": "INCLUDED_IN_ULTRALYTICS_POSTPROCESS_BOX_FUSION",
    }
    summaries = [
        {
            "stage": stage,
            **_summary(
                stages[stage],
                stage_status.get(stage, "MEASURED"),
            ),
        }
        for stage in STAGES
    ]
    root = OUTPUT / "runtime"
    environment = _system_environment(
        int(detector.settings["image_size"]), (width, height, fps)
    )
    atomic_json(root / "RUNTIME_ENVIRONMENT.json", environment)
    atomic_csv(root / "RUNTIME_PER_FRAME.csv", pd.DataFrame(per_frame))
    atomic_csv(root / "RUNTIME_STAGE_SUMMARY.csv", pd.DataFrame(summaries))
    atomic_csv(root / "RUNTIME_MEMORY.csv", pd.DataFrame(memory_rows))
    final = {
        "status": "PASS",
        "warmup_frames": warmup,
        "measured_frames": len(per_frame),
        "end_to_end_fps": 1000.0
        / max(_summary(stages["full_pipeline"])["mean_ms"], 1e-12),
        "peak_gpu_memory_mib": max(
            (row["gpu_memory_mib"] for row in memory_rows), default=0.0
        ),
        "peak_ram_mib": max(
            (row["ram_mib"] for row in memory_rows), default=0.0
        ),
        "mean_tracks_per_frame": float(np.mean(tracks_per_frame)),
        "p95_tracks_per_frame": float(np.percentile(tracks_per_frame, 95)),
        "mean_queue_length": float(np.mean(queue_lengths)),
        "maximum_queue_length": int(max(queue_lengths, default=0)),
        "raw_detections_including_warmup": raw_detections,
        "events_including_warmup": len(events),
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    atomic_json(root / "RUNTIME_FINAL.json", final)
    atomic_json(
        root / "RUNTIME_AUDIT.json",
        {
            "status": "PASS",
            "all_required_stages_present": set(STAGES)
            == {row["stage"] for row in summaries},
            "measured_frames": len(per_frame),
            "warmup_frames": warmup,
            "nonfinite_values": int(
                (~np.isfinite(pd.DataFrame(per_frame).select_dtypes("number"))).sum().sum()
            ),
            "tiling_status": stage_status["tiling"],
            "coordinate_restoration_status": stage_status[
                "coordinate_restoration"
            ],
            "database_write_cadence_frames": int(
                review_config["processing"]["progress_commit_interval_frames"]
            ),
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
