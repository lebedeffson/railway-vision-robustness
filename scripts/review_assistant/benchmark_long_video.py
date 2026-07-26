from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import psutil
import torch

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.review_assistant.config import load_config
from src.review_assistant.event_aggregator import EventAggregator
from src.review_assistant.processor import ReviewProcessor, merge_review_candidates


def _summary(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    ordered = sorted(samples)
    p95 = ordered[min(int(round((len(ordered) - 1) * 0.95)), len(ordered) - 1)]
    p99 = ordered[min(int(round((len(ordered) - 1) * 0.99)), len(ordered) - 1)]
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--verifier", type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--encode", action="store_true")
    args = parser.parse_args()
    config = load_config()
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"Could not open benchmark video: {args.input}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    warmup = int(config["benchmark"]["warmup_frames"])
    requested = int(args.frames)
    duration = total / max(fps, 1e-9)
    if total - warmup < int(config["benchmark"]["minimum_frames"]) and duration < (
        float(config["benchmark"]["minimum_video_minutes"]) * 60.0
    ):
        raise SystemExit(
            "Long benchmark requires at least 1000 post-warmup frames "
            "or a video duration of at least 10 minutes."
        )
    measured_frames = min(requested, max(total - warmup, 0))
    if measured_frames <= 0:
        raise SystemExit("No post-warmup frames available.")
    processor = ReviewProcessor(
        checkpoint=args.checkpoint,
        verifier=args.verifier,
        encoder=args.encoder,
        device=args.device,
    )
    detector, temporal = processor._models("combined_queue")
    aggregator = EventAggregator(
        config["events"],
        video_id="BENCHMARK",
        camera_id="benchmark-camera",
        frame_width=width,
        frame_height=height,
    )
    timings: dict[str, list[float]] = {
        key: []
        for key in (
            "decode",
            "detector",
            "tracker",
            "verifier",
            "event_aggregation",
            "video_encoding",
            "end_to_end",
        )
    }
    writer = None
    temporary = tempfile.TemporaryDirectory(prefix="review-benchmark-")
    if args.encode:
        writer = cv2.VideoWriter(
            str(Path(temporary.name) / "encoded.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    peak_ram = 0
    queue_lengths: list[int] = []
    frame_number = -1
    observed = 0
    started = time.perf_counter()
    previous_tick = started
    while observed < warmup + measured_frames:
        decode_started = time.perf_counter()
        ok, frame = capture.read()
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        if not ok:
            break
        frame_number += 1
        frame_started = time.perf_counter()
        detector_started = time.perf_counter()
        candidates = detector.candidates(frame)
        detector_ms = (time.perf_counter() - detector_started) * 1000.0
        baseline = detector.standard(candidates)
        temporal_rows, _ = temporal.update(frame, candidates)
        aggregate_started = time.perf_counter()
        rows = merge_review_candidates(baseline, temporal_rows)
        aggregator.observe(frame_number, frame_number / fps, rows)
        queue_length = len(aggregator.accepted_events())
        aggregate_ms = (time.perf_counter() - aggregate_started) * 1000.0
        encode_started = time.perf_counter()
        if writer is not None:
            writer.write(frame)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        end_to_end_ms = (time.perf_counter() - frame_started) * 1000.0 + decode_ms
        if observed >= warmup:
            timings["decode"].append(decode_ms)
            timings["detector"].append(detector_ms)
            timings["tracker"].append(float(temporal.last_tracker_ms))
            timings["verifier"].append(float(temporal.last_verifier_ms))
            timings["event_aggregation"].append(aggregate_ms)
            timings["video_encoding"].append(encode_ms)
            timings["end_to_end"].append(end_to_end_ms)
            queue_lengths.append(queue_length)
        observed += 1
        peak_ram = max(peak_ram, int(psutil.Process().memory_info().rss))
        previous_tick = time.perf_counter()
    capture.release()
    if writer is not None:
        writer.release()
    temporary.cleanup()
    elapsed = time.perf_counter() - started
    actual_measured = len(timings["end_to_end"])
    result = {
        "status": "PASS" if actual_measured >= measured_frames else "INCOMPLETE",
        "input_name": args.input.name,
        "input_duration_seconds": duration,
        "warmup_frames": warmup,
        "measured_frames": actual_measured,
        "requirement": "1000 post-warmup frames or at least 10 minutes",
        "decode_fps": 1000.0 / max(_summary(timings["decode"])["mean_ms"], 1e-9),
        "detector_fps": 1000.0
        / max(_summary(timings["detector"])["mean_ms"], 1e-9),
        "tracker_fps": 1000.0
        / max(_summary(timings["tracker"])["mean_ms"], 1e-9),
        "verifier_fps": 1000.0
        / max(_summary(timings["verifier"])["mean_ms"], 1e-9),
        "end_to_end_fps": 1000.0
        / max(_summary(timings["end_to_end"])["mean_ms"], 1e-9),
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "peak_ram_bytes": peak_ram,
        "events": len(aggregator.finalize()),
        "mean_queue_length": (
            statistics.fmean(queue_lengths) if queue_lengths else 0.0
        ),
        "maximum_queue_length": max(queue_lengths, default=0),
        "full_wall_seconds_including_warmup": elapsed,
        "stages": {key: _summary(value) for key, value in timings.items()},
        "real_time_claim": False,
        "offline_processing_supported": True,
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "LONG_VIDEO_BENCHMARK.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output / "LONG_VIDEO_BENCHMARK.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer_csv = csv.DictWriter(
            handle,
            fieldnames=["stage", "mean_ms", "median_ms", "p95_ms"],
        )
        writer_csv.writeheader()
        for stage, values in result["stages"].items():
            writer_csv.writerow({"stage": stage, **values})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
