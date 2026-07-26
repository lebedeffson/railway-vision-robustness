from __future__ import annotations

import statistics
import time
import math
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import psutil


class RuntimeMetrics:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.frames = 0
        self.peak_cpu_memory_bytes = 0

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        yield
        self.samples[stage].append((time.perf_counter() - started) * 1000.0)

    def frame_complete(self) -> None:
        self.frames += 1
        self.peak_cpu_memory_bytes = max(
            self.peak_cpu_memory_bytes,
            int(psutil.Process().memory_info().rss),
        )

    def summary(self, peak_gpu_memory_bytes: int = 0) -> dict[str, object]:
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        total = self.samples.get("total", [])

        def value(values: list[float], quantile: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            index = min(math.ceil((len(ordered) - 1) * quantile), len(ordered) - 1)
            return float(ordered[index])

        return {
            "frames": self.frames,
            "elapsed_seconds": elapsed,
            "fps": self.frames / elapsed,
            "mean_latency_ms": statistics.fmean(total) if total else 0.0,
            "median_latency_ms": statistics.median(total) if total else 0.0,
            "p95_latency_ms": value(total, 0.95),
            "peak_gpu_memory_bytes": int(peak_gpu_memory_bytes),
            "peak_cpu_memory_bytes": self.peak_cpu_memory_bytes,
            "stage_mean_ms": {
                key: statistics.fmean(values) if values else 0.0
                for key, values in sorted(self.samples.items())
                if key != "total"
            },
        }
