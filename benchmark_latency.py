from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from extract_feature_consistency import FeatureHook, defend, loader, metrics, to_device


PROJECT_DIR = Path(__file__).resolve().parent
MODEL = PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/weights/best.pt"
DATA = PROJECT_DIR / "data/yolo_osdar23/data.yaml"
STATS = PROJECT_DIR / "outputs/diagnostics/feature_consistency/feature_normalization_val.pt"
OUTPUT = PROJECT_DIR / "outputs/final_practice/08_latency"


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    operation, device: torch.device, warmup: int, repetitions: int
) -> tuple[list[float], float]:
    with torch.no_grad():
        for _ in range(warmup):
            operation()
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        values = []
        for _ in range(repetitions):
            synchronize(device)
            start = time.perf_counter_ns()
            operation()
            synchronize(device)
            values.append((time.perf_counter_ns() - start) / 1_000_000.0)
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else float("nan")
    )
    return values, peak_memory_mb


def summary(
    name: str,
    values: list[float],
    batch: int,
    detector_mean: float,
    peak_memory_mb: float,
) -> dict[str, object]:
    mean = statistics.fmean(values)
    return {
        "method": name,
        "runs": len(values),
        "batch_size": batch,
        "mean_latency_ms": mean,
        "median_latency_ms": statistics.median(values),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "fps": 1000.0 * batch / mean,
        "additional_latency_ms": mean - detector_mean,
        "gpu_peak_memory_mb": peak_memory_mb,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector/filter/diagnostic latency benchmark")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--stats", type=Path, default=STATS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    model = YOLO(str(args.model)).model.to(device).float().eval()
    batch = to_device(next(iter(loader(args.data, "test", args.imgsz, args.batch, 0, device.type == "cuda"))), device)
    images = batch["img"]
    hook = FeatureHook(model)
    stats_payload = torch.load(args.stats, map_location="cpu")["stats"]
    reference = hook.extract(model, images)

    operations = {"detector": lambda: model(images)}
    for defense_name in ("tnorm", "bilateral", "gaussian", "jpeg", "median"):
        operations[f"{defense_name}+detector"] = (
            lambda name=defense_name: model(defend(images, name))
        )

    def diagnostic_operation() -> None:
        current = hook.extract(model, images)
        for index, level in enumerate(("P3", "P4", "P5")):
            metrics(reference[index], current[index], stats_payload[level])

    operations["diagnostics+detector"] = diagnostic_operation
    raw: dict[str, list[float]] = {}
    peak_memory: dict[str, float] = {}
    try:
        for name, operation in operations.items():
            raw[name], peak_memory[name] = benchmark(
                operation, device, args.warmup, args.repetitions
            )
    finally:
        hook.close()

    detector_mean = statistics.fmean(raw["detector"])
    rows = [
        summary(name, values, args.batch, detector_mean, peak_memory[name])
        for name, values in raw.items()
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "latency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "imgsz": args.imgsz,
        "batch_size": args.batch,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
