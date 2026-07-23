from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    assert_role_allowed,
    load_protocol,
)
from canonical_m4_runtime import (
    FrameInput,
    detection,
    feature_sets,
    frame_labels,
    load_image,
    load_model,
)
from extract_feature_consistency import FeatureHook
from revision_q1.feature_metrics import pair_metrics


MANIFEST = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"
DESTINATION = OUTPUT_ROOT / "latency"


def run() -> dict:
    assert_role_allowed("test")
    protocol = load_protocol()
    gate = json.loads(
        (OUTPUT_ROOT / "validation/quality_gate.json").read_text(encoding="utf-8")
    )
    normalization = torch.load(
        OUTPUT_ROOT / "normalization/layer_channel_statistics.pt",
        map_location="cpu",
    )
    mode = json.loads(
        (OUTPUT_ROOT / "normalization/normalization_manifest.json")
        .read_text(encoding="utf-8")
    )["selected_normalization"]
    row = (
        pd.read_csv(MANIFEST)
        .query("split == 'test'")
        .sort_values(["grouped_scene_id", "subsequence_id", "frame_id"])
        .iloc[0]
    )
    frame = FrameInput(
        Path(row.output_image),
        Path(row.output_label),
        str(row.grouped_scene_id),
        str(row.subsequence_id),
    )
    device = torch.device("cuda:0")
    model = load_model(Path(gate["checkpoint"]), device)
    labels = frame_labels(frame, protocol)
    threshold = float(gate["safety_threshold"])
    methods = (
        "M4_tiling_YOLO",
        "Product_plus_M4_tiling_YOLO",
        "diagnostics_plus_M4_tiling_YOLO",
        "full_Product_diagnostic_M4_pipeline",
    )
    warmup = int(protocol["latency"]["warmup_runs"])
    measured = int(protocol["latency"]["measurement_runs"])
    rows = []
    for method in methods:
        values = []
        torch.cuda.reset_peak_memory_stats()
        for index in range(warmup + measured):
            torch.cuda.synchronize()
            started = time.perf_counter()
            image = load_image(frame.image_path, device)
            defense = (
                "Product_preprocessing"
                if method in {
                    "Product_plus_M4_tiling_YOLO",
                    "full_Product_diagnostic_M4_pipeline",
                }
                else "none"
            )
            detection(model, image, labels, protocol, threshold, defense)
            if method in {
                "diagnostics_plus_M4_tiling_YOLO",
                "full_Product_diagnostic_M4_pipeline",
            }:
                hook = FeatureHook(model)
                try:
                    clean = feature_sets(hook, model, image, protocol)
                    other = feature_sets(
                        hook, model, image, protocol, defense
                    )
                    for layer_index, layer in enumerate(("P3", "P4", "P5")):
                        pair_metrics(
                            clean[layer_index],
                            other[layer_index],
                            normalization["statistics"][layer],
                            mode,
                        )
                finally:
                    hook.close()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            if index >= warmup:
                values.append(elapsed)
        rows.append({
            "method": method,
            "mean_latency_ms": float(statistics.mean(values)),
            "median_latency_ms": float(statistics.median(values)),
            "p95_latency_ms": float(np.percentile(values, 95)),
            "fps": 1000.0 / float(statistics.mean(values)),
            "peak_gpu_memory_mb":
                torch.cuda.max_memory_allocated() / (1024 ** 2),
            "tiles_per_frame": int(protocol["tiling"]["expected_tiles_per_frame"]),
            "batch_size": int(protocol["latency"]["batch_size"]),
            "warmup_runs": warmup,
            "measurement_runs": measured,
        })
    DESTINATION.mkdir(parents=True, exist_ok=True)
    output = DESTINATION / "latency_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    summary = {"status": "PASS", "methods": len(rows), "output": str(output.resolve())}
    (DESTINATION / "latency_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
