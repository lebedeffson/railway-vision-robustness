from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from audit_final_practice import canonical_path
from evaluate_image_level_detection import DEFAULT_CONFIDENCE
from extract_attack_consistency import consistency_row, fgsm, object_masks, sequence_lookup
from extract_feature_consistency import FeatureHook, loader, to_device
from revision_q1.feature_metrics import pair_metrics
from revision_q1.protocol import load_protocol, output_root
from revision_q1.spatial import spatial_transform, transform_xywh_boxes
from revision_q1.statistics import cluster_mean_interval
from run_final_matrix import detection_for_image


PROJECT_DIR = Path(__file__).resolve().parents[1]


def transformed_batch(batch: dict, angle: float, scale: float) -> dict:
    result = dict(batch)
    result["img"] = spatial_transform(batch["img"], angle_degrees=angle, scale=scale)
    result["bboxes"] = transform_xywh_boxes(
        batch["bboxes"], angle_degrees=angle, scale=scale
    )
    return result


def conditions(protocol: dict) -> list[tuple[str, float, float, bool]]:
    spatial = protocol["spatial_stress"]
    rows = []
    for angle in spatial["rotations_degrees"]:
        rows.extend([
            ("rotation_only", float(angle), 1.0, False),
            ("rotation_fgsm", float(angle), 1.0, True),
        ])
    for scale in spatial["scales"]:
        rows.extend([
            ("scale_only", 0.0, float(scale), False),
            ("scale_fgsm", 0.0, float(scale), True),
        ])
    return rows


def main() -> None:
    protocol = load_protocol()
    root = output_root(protocol)
    parser = argparse.ArgumentParser(description="Controlled fixed-affine Q1 stress test")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / protocol["primary_checkpoint"])
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data/yolo_osdar23/data.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data/yolo_osdar23/manifest.csv")
    parser.add_argument("--stats", type=Path, default=root / "normalization/layer_channel_statistics.pt")
    parser.add_argument("--selection", type=Path, default=root / "config/normalization_selection.json")
    parser.add_argument("--output", type=Path, default=root)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    raw_path = args.output / "raw/spatial_stress_test.csv"
    if raw_path.is_file():
        print(f"Spatial stress test already complete: {raw_path}")
        return
    mode = json.loads(args.selection.read_text(encoding="utf-8"))["selected_normalization"]
    statistics = torch.load(args.stats, map_location="cpu")["statistics"]
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    yolo = YOLO(str(args.model))
    model = yolo.model.to(device).float().eval()
    overrides = model.args if isinstance(model.args, dict) else vars(model.args)
    model.args = get_cfg(overrides=overrides)
    model.criterion = None
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hook = FeatureHook(model)
    exact, named = sequence_lookup(args.manifest)
    rows: list[dict[str, object]] = []
    epsilon = float(protocol["spatial_stress"]["fgsm_epsilon_px"])
    try:
        for index, raw_batch in enumerate(tqdm(
            loader(args.data, "test", args.imgsz, 1, 0, device.type == "cuda"),
            desc="Q1 spatial stress",
        )):
            if args.max_images is not None and index >= args.max_images:
                break
            batch = to_device(raw_batch, device)
            image_path = str(batch["im_file"][0])
            sequence_id = exact.get(canonical_path(image_path), named.get(Path(image_path).name))
            clean_features = hook.extract(model, batch["img"])
            clean_detection = detection_for_image(
                model, batch, batch["img"], DEFAULT_CONFIDENCE
            )
            for scenario, angle, scale, attacked in conditions(protocol):
                spatial_batch = transformed_batch(batch, angle, scale)
                attack_values: dict[str, float] = {}
                if attacked:
                    with torch.enable_grad():
                        result = fgsm(model, spatial_batch, epsilon, False)
                    final_images = result.adversarial
                    mask = object_masks(
                        spatial_batch, final_images.shape[-2], final_images.shape[-1]
                    )[0]
                    attack_values = consistency_row(
                        spatial_batch["img"][0], result, mask, epsilon,
                        result.path_gradient[0], "path_gradient_",
                    )
                else:
                    final_images = spatial_batch["img"]
                final_features = hook.extract(model, final_images)
                detection = detection_for_image(
                    model, spatial_batch, final_images, DEFAULT_CONFIDENCE
                )
                for layer_index, layer in enumerate(("P3", "P4", "P5")):
                    values = pair_metrics(
                        clean_features[layer_index], final_features[layer_index],
                        statistics[layer], mode,
                    )[0]
                    rows.append({
                        "sequence_id": sequence_id, "image_path": image_path,
                        "scenario": scenario, "angle_degrees": angle, "scale": scale,
                        "fgsm_epsilon_px": epsilon if attacked else 0.0,
                        "layer": layer, "normalization": mode,
                        "f1_clean": clean_detection["f1"], "f1_stressed": detection["f1"],
                        "recall_clean": clean_detection["recall"],
                        "recall_stressed": detection["recall"],
                        "fn_clean": clean_detection["fn"], "fn_stressed": detection["fn"],
                        "delta_f1": detection["f1"] - clean_detection["f1"],
                        "cosine": values["cosine_similarity"],
                        "product": values["product"], "godel": values["godel"],
                        "lukasiewicz": values["lukasiewicz"],
                        "c_atk_object": attack_values.get(
                            "path_gradient_c_atk_product_object", math.nan
                        ),
                    })
    finally:
        hook.close()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)
    per_image = raw.groupby(
        ["sequence_id", "image_path", "scenario", "angle_degrees", "scale"],
        as_index=False,
    ).agg({
        "f1_clean": "first", "f1_stressed": "first", "delta_f1": "first",
        "recall_stressed": "first", "product": "mean", "godel": "mean",
        "lukasiewicz": "mean", "c_atk_object": "mean",
    })
    summary_rows = []
    for keys, group in per_image.groupby(["scenario", "angle_degrees", "scale"]):
        interval = cluster_mean_interval(
            group, "delta_f1", iterations=int(protocol["bootstrap_iterations"]),
            seed=int(protocol["random_seed"]),
        )
        summary_rows.append({
            "scenario": keys[0], "angle_degrees": keys[1], "scale": keys[2],
            "f1_clean": group["f1_clean"].mean(),
            "f1_stressed": group["f1_stressed"].mean(),
            "delta_f1": interval["estimate"],
            "delta_f1_ci_low": interval["ci_low"],
            "delta_f1_ci_high": interval["ci_high"],
            "recall_stressed": group["recall_stressed"].mean(),
            "product": group["product"].mean(), "godel": group["godel"].mean(),
            "lukasiewicz": group["lukasiewicz"].mean(),
            "c_atk_object": group["c_atk_object"].mean(),
            "frames": group["image_path"].nunique(),
            "sequences": interval["sequences"],
            "bootstrap_iterations": interval["bootstrap_iterations"],
        })
    summary = pd.DataFrame(summary_rows)
    args.output.joinpath("tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "tables/10_spatial_stress_test.csv", index=False)
    print(json.dumps({"rows": len(raw), "conditions": len(summary), "normalization": mode}, indent=2))


if __name__ == "__main__":
    main()
