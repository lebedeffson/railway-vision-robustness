from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from ultralytics import YOLO

from extract_feature_consistency import FeatureHook, loader, to_device
from revision_q1.normalization import (
    LAYERS,
    MODES,
    distribution_diagnostics,
    fit_channel_statistics,
    membership,
    saturation_warning,
)
from revision_q1.protocol import assert_split_action, load_protocol, output_root


PROJECT_DIR = Path(__file__).resolve().parents[1]


def sha256_lines(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sample_features(
    model: torch.nn.Module,
    hook: FeatureHook,
    data_loader,
    device: torch.device,
    positions_per_image: int,
    seed: int,
) -> tuple[dict[str, Tensor], list[str]]:
    samples: dict[str, list[Tensor]] = {layer: [] for layer in LAYERS}
    image_paths: list[str] = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for raw_batch in tqdm(data_loader, desc="revision normalization clean val"):
        batch = to_device(raw_batch, device)
        image_paths.extend(str(path) for path in batch["im_file"])
        features = hook.extract(model, batch["img"])
        for layer, feature in zip(LAYERS, features, strict=True):
            cpu = feature.detach().float().cpu()
            batch_size, channels, height, width = cpu.shape
            flattened = cpu.view(batch_size, channels, height * width)
            count = min(positions_per_image, height * width)
            for image_index in range(batch_size):
                indices = torch.randperm(height * width, generator=generator)[:count]
                samples[layer].append(flattened[image_index, :, indices])
    return ({layer: torch.cat(values, dim=1) for layer, values in samples.items()}, image_paths)


def write_diagnostics(
    samples: dict[str, Tensor],
    statistics: dict[str, dict[str, Tensor]],
    output: Path,
    saturation_threshold: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    figure, axes = plt.subplots(len(LAYERS), len(MODES), figsize=(16, 9), sharex=True)
    for layer_index, layer in enumerate(LAYERS):
        feature = samples[layer].unsqueeze(0).unsqueeze(2)
        for mode_index, mode in enumerate(MODES):
            normalized = membership(feature, statistics[layer], mode)
            diagnostics = distribution_diagnostics(normalized)
            rows.append({
                "split": "val",
                "input_state": "clean",
                "layer": layer,
                "normalization": mode,
                **diagnostics,
                "saturation_warning": saturation_warning(
                    diagnostics, saturation_threshold
                ),
                "validation_cv_mae": np.nan,
                "validation_cv_r2": np.nan,
                "selected": False,
            })
            values = normalized.flatten().numpy()
            axes[layer_index, mode_index].hist(values, bins=50, range=(0, 1), density=True)
            axes[layer_index, mode_index].set_title(f"{layer} {mode}", fontsize=9)
    figure.tight_layout()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "01_feature_normalization_by_layer.png", dpi=180)
    figure.savefig(figures / "normalization_distributions.png", dpi=180)
    plt.close(figure)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "02_normalization_ablation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    protocol = load_protocol()
    parser = argparse.ArgumentParser(description="Fit P3/P4/P5 channel normalization on clean validation")
    parser.add_argument("--protocol", type=Path, default=PROJECT_DIR / "config/revision_q1_protocol.yaml")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / protocol["primary_checkpoint"])
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data/yolo_osdar23/data.yaml")
    parser.add_argument("--output", type=Path, default=output_root(protocol))
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    assert_split_action("fit_normalization", "val")
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    yolo = YOLO(str(args.model))
    model = yolo.model.to(device).float().eval()
    hook = FeatureHook(model)
    normalization = protocol["normalization"]
    try:
        samples, image_paths = sample_features(
            model,
            hook,
            loader(args.data, "val", args.imgsz, 1, args.workers, device.type == "cuda"),
            device,
            int(normalization["sample_positions_per_image"]),
            int(protocol["random_seed"]),
        )
    finally:
        hook.close()
    statistics = fit_channel_statistics(samples)
    destination = args.output / "normalization"
    destination.mkdir(parents=True, exist_ok=True)
    torch.save({
        "statistics": statistics,
        "fit_split": "val",
        "fit_inputs": "clean_only",
        "per_layer": True,
        "per_channel": True,
        "image_count": len(image_paths),
        "image_path_hash": sha256_lines(image_paths),
        "protocol_id": protocol["protocol_id"],
    }, destination / "layer_channel_statistics.pt")
    rows = write_diagnostics(
        samples, statistics, args.output,
        float(normalization["saturation_warning_fraction"]),
    )
    metadata = {
        "status": "PASS",
        "fit_split": "val",
        "fit_inputs": "clean_only",
        "image_count": len(image_paths),
        "image_path_hash": sha256_lines(image_paths),
        "layers": list(LAYERS),
        "normalizations": list(MODES),
        "warnings": sum(bool(row["saturation_warning"]) for row in rows),
    }
    (destination / "normalization_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
