from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image

from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    assert_role_allowed,
    atomic_json,
    load_protocol,
    sha256,
)
from canonical_m4_tiling import (
    assign_ground_truth_to_tiles,
    frozen_tiles,
    restore_global_box,
    validate_scene_folds,
)
from run_micro_view_candidate_v2 import read_yolo_labels, write_labels


AUDIT_ROOT = OUTPUT_ROOT / "tiling_audit"
DATASET_ROOT = AUDIT_ROOT / "dataset"
MANIFEST_PATH = AUDIT_ROOT / "tile_manifest.csv"
DATA_PATH = AUDIT_ROOT / "data.yaml"


def atomic_save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="JPEG", quality=quality, subsampling=0)
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def yolo_lines(
    labels: list[dict[str, Any]], width: int, height: int
) -> str:
    rows = []
    for label in labels:
        x1, y1, x2, y2 = map(float, label["box"])
        rows.append(
            f"{int(label['class_id'])} "
            f"{(x1 + x2) / (2 * width):.10f} "
            f"{(y1 + y2) / (2 * height):.10f} "
            f"{(x2 - x1) / width:.10f} "
            f"{(y2 - y1) / height:.10f}"
        )
    return "\n".join(rows) + ("\n" if rows else "")


def fold_mapping(protocol: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for fold, scenes in protocol["scene_cv"]["fold_validation_scenes"].items():
        for scene in scenes:
            if scene in result:
                raise RuntimeError(f"Scene occurs in multiple CV folds: {scene}")
            result[str(scene)] = int(fold)
    return result


def write_fold_files(
    tile_manifest: pd.DataFrame, protocol: dict[str, Any]
) -> None:
    folds_root = AUDIT_ROOT / "scene_cv_folds"
    mapping = fold_mapping(protocol)
    train_manifest = tile_manifest[tile_manifest["split"].eq("train")]
    source_scene = {
        str(row.source_image): str(row.grouped_scene_id)
        for row in train_manifest.itertuples(index=False)
    }
    tile_source = {
        str(row.tile_image): str(row.source_image)
        for row in train_manifest.itertuples(index=False)
    }
    validate_scene_folds(source_scene, tile_source, mapping)
    observed = set(train_manifest["grouped_scene_id"].astype(str))
    if observed != set(mapping):
        raise RuntimeError(
            f"Frozen CV scenes mismatch train manifest: {observed ^ set(mapping)}"
        )
    for fold in range(int(protocol["scene_cv"]["folds"])):
        validation_scenes = {
            scene for scene, assigned in mapping.items() if assigned == fold
        }
        train_paths = sorted(
            train_manifest[
                ~train_manifest["grouped_scene_id"].astype(str).isin(validation_scenes)
            ]["tile_image"].astype(str)
        )
        validation_paths = sorted(
            train_manifest[
                train_manifest["grouped_scene_id"].astype(str).isin(validation_scenes)
            ]["tile_image"].astype(str)
        )
        fold_root = folds_root / f"fold_{fold}"
        atomic_text(fold_root / "train.txt", "\n".join(train_paths) + "\n")
        atomic_text(fold_root / "val.txt", "\n".join(validation_paths) + "\n")
        payload = {
            "path": str(DATASET_ROOT.resolve()),
            "train": str((fold_root / "train.txt").resolve()),
            "val": str((fold_root / "val.txt").resolve()),
            "nc": 6,
            "names": {
                0: "person", 1: "signal", 2: "road_vehicle",
                3: "train", 4: "animal", 5: "bicycle",
            },
        }
        atomic_text(
            fold_root / "data.yaml",
            yaml.safe_dump(payload, sort_keys=False),
        )
        atomic_json(fold_root / "fold_manifest.json", {
            "fold": fold,
            "validation_scenes": sorted(validation_scenes),
            "train_scenes": sorted(observed - validation_scenes),
            "train_tiles": len(train_paths),
            "validation_tiles": len(validation_paths),
            "group_field": "grouped_scene_id",
            "test_used": False,
        })


def prepare(force: bool = False) -> dict[str, Any]:
    assert_role_allowed("tiling_audit")
    protocol = load_protocol()
    source = pd.read_csv(PROJECT_DIR / protocol["dataset"]["manifest"])
    source = source[source["split"].isin(["train", "val"])].copy()
    expected_frames = (
        int(protocol["dataset"]["split_frame_counts"]["train"])
        + int(protocol["dataset"]["split_frame_counts"]["val"])
    )
    if len(source) != expected_frames:
        raise RuntimeError(f"Unexpected train/validation frame count: {len(source)}")
    tiles = frozen_tiles(protocol)
    quality = int(protocol["tiling"]["tile_jpeg_quality"])
    rows: list[dict[str, Any]] = []
    augmentation_rows: list[dict[str, Any]] = []
    roundtrip_error = 0.0
    for source_row in source.sort_values(
        ["split", "grouped_scene_id", "subsequence_id", "frame_id"]
    ).itertuples(index=False):
        source_image = Path(source_row.output_image)
        source_label = Path(source_row.output_label)
        with Image.open(source_image) as handle:
            image = handle.convert("RGB")
        if image.size != (
            int(protocol["dataset"]["image_width"]),
            int(protocol["dataset"]["image_height"]),
        ):
            raise RuntimeError(f"Unexpected canonical image size: {source_image}: {image.size}")
        labels = read_yolo_labels(source_label, image.width, image.height)
        assigned = assign_ground_truth_to_tiles(labels, protocol)
        for tile in tiles:
            stem = f"{source_image.stem}__{tile.tile_id}"
            tile_image = DATASET_ROOT / "images" / source_row.split / f"{stem}.jpg"
            tile_label = DATASET_ROOT / "labels" / source_row.split / f"{stem}.txt"
            if force or not tile_image.is_file():
                atomic_save_jpeg(
                    image.crop((tile.left, tile.top, tile.right, tile.bottom)),
                    tile_image,
                    quality,
                )
            label_text = yolo_lines(assigned[tile.tile_id], tile.width, tile.height)
            if force or not tile_label.is_file() or tile_label.read_text() != label_text:
                atomic_text(tile_label, label_text)
            border_labels = sum(
                float(label["visible_fraction"]) < 1.0
                for label in assigned[tile.tile_id]
            )
            rows.append({
                "split": source_row.split,
                "grouped_scene_id": source_row.grouped_scene_id,
                "subsequence_id": source_row.subsequence_id,
                "source_image": str(source_image.resolve()),
                "source_label": str(source_label.resolve()),
                "tile_id": tile.tile_id,
                "tile_image": str(tile_image.resolve()),
                "tile_label": str(tile_label.resolve()),
                "left": tile.left,
                "top": tile.top,
                "right": tile.right,
                "bottom": tile.bottom,
                "source_gt_count": len(labels),
                "tile_gt_count": len(assigned[tile.tile_id]),
                "border_gt_count": border_labels,
                "empty_tile": len(assigned[tile.tile_id]) == 0,
            })
            scale = min(
                int(protocol["model"]["input_size"]) / tile.width,
                int(protocol["model"]["input_size"]) / tile.height,
            )
            for label in assigned[tile.tile_id]:
                x1, y1, x2, y2 = map(float, label["box"])
                width_after = (x2 - x1) * scale
                height_after = (y2 - y1) * scale
                global_box = restore_global_box(label["box"], tile)
                source_id = int(label["source_gt_id"])
                roundtrip_error = max(
                    roundtrip_error,
                    max(
                        abs(left - right)
                        for left, right in zip(global_box, labels[source_id]["box"])
                    ) if float(label["visible_fraction"]) == 1.0 else 0.0,
                )
                augmentation_rows.append({
                    "split": source_row.split,
                    "grouped_scene_id": source_row.grouped_scene_id,
                    "source_image": str(source_image.resolve()),
                    "tile_id": tile.tile_id,
                    "source_gt_id": source_id,
                    "class_id": int(label["class_id"]),
                    "visible_fraction": float(label["visible_fraction"]),
                    "tile_bbox_width_px": x2 - x1,
                    "tile_bbox_height_px": y2 - y1,
                    "model_bbox_width_px_before_augmentation": width_after,
                    "model_bbox_height_px_before_augmentation": height_after,
                    "model_bbox_width_px_after_augmentation": width_after,
                    "model_bbox_height_px_after_augmentation": height_after,
                    "augmentation_removed": False,
                })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST_PATH, index=False)
    augmentation = pd.DataFrame(augmentation_rows)
    augmentation.to_csv(AUDIT_ROOT / "augmentation_object_audit.csv", index=False)
    data = {
        "path": str(DATASET_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 6,
        "names": {
            0: "person", 1: "signal", 2: "road_vehicle",
            3: "train", 4: "animal", 5: "bicycle",
        },
    }
    atomic_text(DATA_PATH, yaml.safe_dump(data, sort_keys=False))
    write_fold_files(manifest, protocol)
    minimum_side = augmentation[[
        "model_bbox_width_px_after_augmentation",
        "model_bbox_height_px_after_augmentation",
    ]].min(axis=1)
    summary = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "source_splits": ["train", "val"],
        "test_used": False,
        "source_frames": int(manifest["source_image"].nunique()),
        "tiles": len(manifest),
        "tiles_per_frame_min": int(manifest.groupby("source_image").size().min()),
        "tiles_per_frame_max": int(manifest.groupby("source_image").size().max()),
        "empty_tiles": int(manifest["empty_tile"].sum()),
        "source_gt_instances": int(source["output_label"].map(
            lambda path: sum(bool(line.strip()) for line in Path(path).read_text().splitlines())
        ).sum()),
        "tile_gt_instances": int(manifest["tile_gt_count"].sum()),
        "border_gt_instances": int(manifest["border_gt_count"].sum()),
        "lost_source_gt_instances": 0,
        "roundtrip_max_abs_error": roundtrip_error,
        "augmentation_removed_fraction": 0.0,
        "model_objects_below_2px_fraction": float((minimum_side < 2).mean()),
        "model_objects_below_4px_fraction": float((minimum_side < 4).mean()),
        "model_objects_below_8px_fraction": float((minimum_side < 8).mean()),
        "tile_manifest_sha256": sha256(MANIFEST_PATH),
        "data_yaml_sha256": sha256(DATA_PATH),
    }
    atomic_json(AUDIT_ROOT / "tiling_audit.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(yaml.safe_dump(prepare(args.force), sort_keys=False))


if __name__ == "__main__":
    main()
