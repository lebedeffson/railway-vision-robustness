from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.article_evidence_v1.common import atomic_csv, atomic_json


CONFIG_PATH = PROJECT / "configs/event_engineering_evidence_v1.yaml"
OUTPUT = PROJECT / "outputs/article_evidence_v1/scaling"
FIGURES = PROJECT / "outputs/article_evidence_v1/figures"


def _run_group(streams: int, config: dict[str, Any]) -> dict[str, Any]:
    group_root = OUTPUT / "workers" / f"{streams}_streams"
    group_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[str], Path]] = []
    command_base = [
        str(Path(sys.executable)),
        str(PROJECT / "scripts/review_assistant/benchmark_long_video.py"),
        "--input",
        str(PROJECT / config["inputs"]["long_benchmark_video"]),
        "--frames",
        str(config["scaling"]["measured_frames_per_stream"]),
    ]
    started = time.perf_counter()
    for index in range(streams):
        worker_output = group_root / f"worker_{index + 1}"
        command = command_base + ["--output", str(worker_output)]
        processes.append(
            (
                subprocess.Popen(
                    command,
                    cwd=PROJECT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                ),
                worker_output,
            )
        )
    timeout = float(config["scaling"]["timeout_seconds"])
    worker_payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    deadline = time.monotonic() + timeout
    for process, worker_output in processes:
        remaining = max(deadline - time.monotonic(), 1.0)
        try:
            stdout, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            errors.append("worker timeout")
        if process.returncode != 0:
            tail = " ".join(stderr.strip().splitlines()[-3:])
            errors.append(f"worker exit={process.returncode}: {tail[:300]}")
            continue
        result_path = worker_output / "LONG_VIDEO_BENCHMARK.json"
        if not result_path.is_file():
            errors.append("worker result missing")
            continue
        worker_payloads.append(json.loads(result_path.read_text(encoding="utf-8")))
    wall = time.perf_counter() - started
    expected = int(config["scaling"]["measured_frames_per_stream"])
    complete = (
        len(worker_payloads) == streams
        and all(item.get("status") == "PASS" for item in worker_payloads)
        and all(int(item.get("measured_frames", 0)) == expected for item in worker_payloads)
    )
    measured_total = sum(
        int(item.get("measured_frames", 0)) for item in worker_payloads
    )
    p95_values = [
        float(item["stages"]["end_to_end"]["p95_ms"]) for item in worker_payloads
    ]
    status = "PASS" if complete else "BLOCKED_RESOURCE_OR_RUNTIME_LIMIT"
    return {
        "streams": streams,
        "status": status,
        "completed_workers": len(worker_payloads),
        "measured_frames_total": measured_total,
        "wall_seconds_including_warmup": wall,
        "total_fps_conservative": measured_total / max(wall, 1e-12),
        "fps_per_camera_conservative": (
            measured_total / max(wall, 1e-12) / streams
        ),
        "p95_latency_ms_worst_camera": max(p95_values, default=0.0),
        "peak_ram_mib_sum_worker_peaks": sum(
            float(item.get("peak_ram_bytes", 0)) / (1024**2)
            for item in worker_payloads
        ),
        "peak_gpu_memory_mib_sum_worker_peaks": sum(
            float(item.get("peak_gpu_memory_bytes", 0)) / (1024**2)
            for item in worker_payloads
        ),
        "maximum_queue_length": max(
            (int(item.get("maximum_queue_length", 0)) for item in worker_payloads),
            default=0,
        ),
        "memory_measurement": "sum_of_per_worker_peak_allocations",
        "errors": " | ".join(errors),
    }


def _plot(frame: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    passed = frame[frame["status"].astype(str).str.startswith("PASS")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].bar(
        passed["streams"].astype(str),
        passed["total_fps_conservative"],
        color="#1f4e79",
    )
    axes[0].set(xlabel="Parallel video streams", ylabel="Aggregate FPS")
    axes[1].bar(
        passed["streams"].astype(str),
        passed["p95_latency_ms_worst_camera"],
        color="#8c2d04",
    )
    axes[1].set(xlabel="Parallel video streams", ylabel="Worst-camera P95, ms")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            FIGURES / f"FIG_10_STREAM_SCALING.{suffix}",
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def _safe_finalize_from_existing(config: dict[str, Any]) -> pd.DataFrame:
    runtime_root = PROJECT / "outputs/article_evidence_v1/runtime"
    final = json.loads(
        (runtime_root / "RUNTIME_FINAL.json").read_text(encoding="utf-8")
    )
    stages = pd.read_csv(runtime_root / "RUNTIME_STAGE_SUMMARY.csv")
    full = stages.loc[stages["stage"] == "full_pipeline"].iloc[0]
    rows: list[dict[str, Any]] = [
        {
            "streams": 1,
            "status": "PASS_EXISTING_VERIFIED_RUN",
            "completed_workers": 1,
            "measured_frames_total": int(final["measured_frames"]),
            "wall_seconds_including_warmup": "NOT_STORED",
            "total_fps_conservative": float(final["end_to_end_fps"]),
            "fps_per_camera_conservative": float(final["end_to_end_fps"]),
            "p95_latency_ms_worst_camera": float(full["p95_ms"]),
            "peak_ram_mib_sum_worker_peaks": float(final["peak_ram_mib"]),
            "peak_gpu_memory_mib_sum_worker_peaks": float(
                final["peak_gpu_memory_mib"]
            ),
            "maximum_queue_length": int(final["maximum_queue_length"]),
            "memory_measurement": "single_process_measured_peak",
            "errors": "",
            "evidence_source": "article_evidence_v1/runtime",
        }
    ]
    for streams in (2, 4):
        rows.append(
            {
                "streams": streams,
                "status": "BLOCKED_RESOURCE_LIMIT",
                "completed_workers": 0,
                "measured_frames_total": 0,
                "wall_seconds_including_warmup": "BLOCKED_RESOURCE_LIMIT",
                "total_fps_conservative": "BLOCKED_RESOURCE_LIMIT",
                "fps_per_camera_conservative": "BLOCKED_RESOURCE_LIMIT",
                "p95_latency_ms_worst_camera": "BLOCKED_RESOURCE_LIMIT",
                "peak_ram_mib_sum_worker_peaks": "BLOCKED_RESOURCE_LIMIT",
                "peak_gpu_memory_mib_sum_worker_peaks": "BLOCKED_RESOURCE_LIMIT",
                "maximum_queue_length": "BLOCKED_RESOURCE_LIMIT",
                "memory_measurement": "NOT_MEASURED",
                "errors": (
                    "Execution intentionally stopped after host instability; "
                    "no numeric value imputed."
                ),
                "evidence_source": "RESOURCE_SAFETY_STOP",
            }
        )
    frame = pd.DataFrame(rows)
    atomic_csv(OUTPUT / "SCALING_RESULTS.csv", frame)
    atomic_json(
        OUTPUT / "SCALING_AUDIT.json",
        {
            "status": "PARTIAL_RESOURCE_SAFETY_STOP",
            "locked_streams": config["scaling"]["streams"],
            "one_stream_source": "existing verified 1000-frame post-warmup run",
            "blocked_streams": [2, 4],
            "blocked_reason": (
                "The interactive host became unstable. Parallel GPU loads were "
                "not continued; missing metrics remain BLOCKED_RESOURCE_LIMIT."
            ),
            "warmup_frames_one_stream": 100,
            "measured_frames_one_stream": 1000,
            "real_time_claim": False,
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    _plot(frame)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--streams",
        nargs="*",
        type=int,
        help="Optional locked subset for resumable execution.",
    )
    parser.add_argument(
        "--safe-finalize",
        action="store_true",
        help="Reuse the verified one-stream run and block unsafe parallel loads.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.safe_finalize:
        frame = _safe_finalize_from_existing(config)
        print(frame.to_json(orient="records"))
        return
    locked_streams = [int(value) for value in config["scaling"]["streams"]]
    requested = args.streams or locked_streams
    if any(value not in locked_streams for value in requested):
        raise SystemExit("Requested stream count is outside the frozen grid")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    existing_path = OUTPUT / "SCALING_RESULTS.csv"
    existing = (
        pd.read_csv(existing_path).to_dict("records")
        if existing_path.is_file()
        else []
    )
    by_stream = {int(row["streams"]): row for row in existing}
    for streams in requested:
        by_stream[streams] = _run_group(streams, config)
        atomic_csv(
            existing_path,
            pd.DataFrame(
                [by_stream[value] for value in sorted(by_stream)]
            ),
        )
    frame = pd.DataFrame([by_stream[value] for value in sorted(by_stream)])
    complete_grid = set(frame["streams"].astype(int)) == set(locked_streams)
    atomic_json(
        OUTPUT / "SCALING_AUDIT.json",
        {
            "status": (
                "PASS"
                if complete_grid and frame["status"].eq("PASS").all()
                else "PARTIAL_OR_RESOURCE_BLOCKED"
            ),
            "locked_streams": locked_streams,
            "completed_stream_rows": sorted(frame["streams"].astype(int).tolist()),
            "warmup_frames_per_stream": config["scaling"]["warmup_frames"],
            "measured_frames_per_stream": config["scaling"][
                "measured_frames_per_stream"
            ],
            "input": Path(config["inputs"]["long_benchmark_video"]).name,
            "real_time_claim": False,
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    if complete_grid:
        _plot(frame)
    print(frame.to_json(orient="records"))


if __name__ == "__main__":
    main()
