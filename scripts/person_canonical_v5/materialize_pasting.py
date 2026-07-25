from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageEnhance, ImageStat

from scripts.person_canonical_v5.build_instance_bank import split_scenes
from scripts.person_canonical_v5.common import PROJECT, assert_test_sealed, atomic_csv
from src.augmentation.person_pasting import (
    Box,
    PerspectiveModel,
    paste_person,
)


TILES = PROJECT / "outputs/canonical_m4/tiling_audit/tile_manifest.csv"
PERSON_DATASET = PROJECT / "outputs/person_v3/dataset"
ROOT = PROJECT / "outputs/person_canonical_v5/instance_pasting"
SEED = 20260725


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def read_labels(path: Path, width: int, height: int) -> list[Box]:
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        class_id, x, y, w, h = map(float, values[:5])
        if int(class_id) != 0:
            continue
        boxes.append(
            Box(
                (x - w / 2) * width,
                (y - h / 2) * height,
                (x + w / 2) * width,
                (y + h / 2) * height,
            )
        )
    return boxes


def encode_labels(boxes: list[Box], width: int, height: int) -> str:
    rows = []
    for box in boxes:
        x = (box.x1 + box.x2) / 2 / width
        y = (box.y1 + box.y2) / 2 / height
        w = box.width / width
        h = box.height / height
        if min(x, y, w, h) <= 0 or max(x, y, w, h) > 1:
            raise RuntimeError(f"Invalid generated YOLO geometry: {box}")
        rows.append(f"0 {x:.8f} {y:.8f} {w:.8f} {h:.8f}")
    return "\n".join(rows) + ("\n" if rows else "")


def fit_perspective(rows: pd.DataFrame) -> PerspectiveModel:
    observations = []
    for row in rows.itertuples(index=False):
        image = Image.open(row.tile_image)
        boxes = read_labels(Path(row.tile_label), image.width, image.height)
        observations.extend(
            (box.y2 / image.height, math.log(max(box.height, 4.0)))
            for box in boxes
        )
    if len(observations) < 2:
        raise RuntimeError("Not enough train-only persons to fit perspective")
    values = np.asarray(observations, dtype=float)
    design = np.column_stack([np.ones(len(values)), values[:, 0]])
    intercept, slope = np.linalg.lstsq(
        design, values[:, 1], rcond=None
    )[0]
    return PerspectiveModel(float(intercept), float(slope), 4.0)


def match_brightness(
    instance: Image.Image,
    mask: Image.Image,
    target: Image.Image,
    anchor: tuple[int, int],
) -> Image.Image:
    source = ImageStat.Stat(instance.convert("L"), mask.convert("L")).mean[0]
    left = max(0, anchor[0] - instance.width // 2)
    top = max(0, anchor[1] - instance.height)
    right = min(target.width, left + instance.width)
    bottom = min(target.height, top + instance.height)
    region = target.crop((left, top, right, bottom)).convert("L")
    destination = ImageStat.Stat(region).mean[0] if region.size[0] else source
    factor = min(1.5, max(0.67, destination / max(source, 1.0)))
    return ImageEnhance.Brightness(instance).enhance(factor)


def candidate_anchors(box: Box, width: int, height: int) -> list[tuple[int, int]]:
    shifts = (-2.5, 2.5, -1.5, 1.5)
    anchors = [
        (int(round((box.x1 + box.x2) / 2 + shift * box.width)), int(box.y2))
        for shift in shifts
    ]
    return [
        (x, y)
        for x, y in anchors
        if 4 <= x < width - 4 and 4 <= y < height - 4
    ]


def symlink_or_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    os.symlink(source.resolve(), destination)


def materialize(fold: int, changed_fraction: float) -> dict[str, Any]:
    assert_test_sealed()
    if changed_fraction not in (0.25, 0.50):
        raise ValueError("Only the frozen 0.25/0.50 fractions are permitted")
    train_scenes, heldout_scenes, _ = split_scenes(fold)
    bank_path = ROOT / f"fold_{fold}/instance_bank.csv"
    if not bank_path.is_file():
        raise RuntimeError(f"Build the fold {fold} instance bank first")
    bank = pd.read_csv(bank_path)
    if bank.empty:
        raise RuntimeError("No audited person instances are available")
    if not set(bank["source_scene_id"].astype(str)) <= train_scenes:
        raise RuntimeError("Instance bank contains held-out scene material")
    tiles = pd.read_csv(TILES)
    tiles = tiles[tiles["grouped_scene_id"].astype(str).isin(train_scenes)].copy()
    tiles["tile_image"] = tiles["tile_image"].map(Path)
    tiles["tile_label"] = tiles["tile_label"].map(Path)
    perspective = fit_perspective(tiles)
    frame_ids = sorted(tiles["source_image"].astype(str).unique())
    rng = np.random.default_rng(SEED + fold + int(changed_fraction * 100))
    selected_count = int(round(len(frame_ids) * changed_fraction))
    selected = set(rng.choice(frame_ids, selected_count, replace=False).tolist())
    suffix = f"fraction_{int(changed_fraction * 100)}"
    destination = ROOT / f"fold_{fold}/{suffix}/dataset"
    image_root = destination / "images/train"
    label_root = destination / "labels/train"
    audit_rows: list[dict[str, Any]] = []
    used_by_frame: dict[str, set[str]] = {}
    output_images: list[str] = []
    accepted_frames: set[str] = set()
    inserted_by_frame: dict[str, int] = {}
    original_gt = generated_gt = inserted_gt = 0
    bank_records = list(bank.to_dict("records"))
    for row in tiles.sort_values(["source_image", "tile_id"]).itertuples(index=False):
        source_image = Path(row.tile_image)
        source_label = Path(row.tile_label)
        output_image = image_root / source_image.name
        output_label = label_root / source_label.name
        image = Image.open(source_image).convert("RGB")
        boxes = read_labels(source_label, image.width, image.height)
        original_gt += len(boxes)
        accepted = False
        attempts = []
        frame_key = str(row.source_image)
        if (
            frame_key in selected
            and boxes
            and inserted_by_frame.get(frame_key, 0) < 2
        ):
            ordered_boxes = sorted(boxes, key=lambda box: (box.height, box.x1))
            for existing in ordered_boxes:
                for anchor in candidate_anchors(existing, image.width, image.height):
                    candidate = bank_records[int(rng.integers(0, len(bank_records)))]
                    if (
                        str(candidate["source_scene_id"]) == str(row.grouped_scene_id)
                        or f"__{candidate['source_frame_id']}_"
                        in Path(frame_key).stem
                    ):
                        continue
                    instance = Image.open(candidate["instance_image"]).convert("RGB")
                    mask = Image.open(candidate["mask_path"]).convert("L")
                    instance.info["instance_id"] = str(candidate["instance_id"])
                    instance.info["source_scene_id"] = str(candidate["source_scene_id"])
                    instance = match_brightness(instance, mask, image, anchor)
                    instance.info["instance_id"] = str(candidate["instance_id"])
                    instance.info["source_scene_id"] = str(candidate["source_scene_id"])
                    surface = np.zeros((image.height, image.width), dtype=bool)
                    x, y = anchor
                    surface[max(0, y - 2): min(image.height, y + 3),
                            max(0, x - 2): min(image.width, x + 3)] = True
                    used = used_by_frame.setdefault(str(row.source_image), set())
                    result = paste_person(
                        image,
                        instance,
                        mask,
                        target_scene_id=str(row.grouped_scene_id),
                        train_scene_ids=train_scenes,
                        bottom_center=anchor,
                        surface_mask=surface,
                        perspective=perspective,
                        existing_boxes=boxes,
                        used_instance_ids=used,
                        maximum_existing_iou=0.30,
                        minimum_box_side_px=4,
                        blur_radius=0.35
                        if str(candidate["range_or_scale_group"]) == "far"
                        else 0.0,
                    )
                    attempts.append(
                        {
                            "instance_id": candidate["instance_id"],
                            "anchor_x": anchor[0],
                            "anchor_y": anchor[1],
                            "flags": "|".join(str(flag) for flag in result.flags),
                        }
                    )
                    if result.accepted and result.box is not None:
                        image = result.image
                        boxes.append(result.box)
                        used.add(str(candidate["instance_id"]))
                        accepted = True
                        accepted_frames.add(frame_key)
                        inserted_by_frame[frame_key] = (
                            inserted_by_frame.get(frame_key, 0) + 1
                        )
                        inserted_gt += 1
                        audit_rows.append(
                            {
                                "fold": fold,
                                "changed_fraction": changed_fraction,
                                "source_frame": str(row.source_image),
                                "grouped_scene_id": str(row.grouped_scene_id),
                                "tile_id": str(row.tile_id),
                                "original_tile": str(source_image.resolve()),
                                "source_instance": candidate["instance_image"],
                                "source_mask": candidate["mask_path"],
                                "instance_id": candidate["instance_id"],
                                "instance_source_scene_id": candidate["source_scene_id"],
                                "range_source": candidate["range_source"],
                                "source_range_or_scale_group": candidate[
                                    "range_or_scale_group"
                                ],
                                "estimated_source_range_m": candidate[
                                    "estimated_range_m"
                                ],
                                "final_tile": str(output_image.resolve()),
                                "new_x1": result.box.x1,
                                "new_y1": result.box.y1,
                                "new_x2": result.box.x2,
                                "new_y2": result.box.y2,
                                "scale": result.scale,
                                "surface_rule": "train_person_ground_contact_anchor",
                                "occlusion_rule": "existing_person_IoU_lte_0.30",
                                "status": "ACCEPTED",
                                "critical_flags": "",
                            }
                        )
                        break
                if accepted:
                    break
        if accepted:
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_image, quality=95)
            atomic_text(
                output_label,
                encode_labels(boxes, image.width, image.height),
            )
        else:
            symlink_or_replace(source_image, output_image)
            symlink_or_replace(source_label, output_label)
            if str(row.source_image) in selected:
                audit_rows.append(
                    {
                        "fold": fold,
                        "changed_fraction": changed_fraction,
                        "source_frame": str(row.source_image),
                        "grouped_scene_id": str(row.grouped_scene_id),
                        "tile_id": str(row.tile_id),
                        "original_tile": str(source_image.resolve()),
                        "final_tile": str(output_image.resolve()),
                        "status": "REJECTED",
                        "critical_flags": (
                            attempts[-1]["flags"] if attempts else "INVALID_SURFACE"
                        ),
                    }
                )
        generated_gt += len(boxes)
        output_images.append(str(output_image.absolute()))
    if generated_gt != original_gt + inserted_gt:
        raise RuntimeError("Generated GT count does not match accepted pastes")
    if max(inserted_by_frame.values(), default=0) > 2:
        raise RuntimeError("Frozen maximum of two insertions per frame exceeded")
    train_list = destination / "train.txt"
    atomic_text(train_list, "\n".join(output_images) + "\n")
    validation = PERSON_DATASET / f"folds/fold_{fold}/val.txt"
    data = {
        "path": str(destination.resolve()),
        "train": str(train_list.resolve()),
        "val": str(validation.resolve()),
        "nc": 1,
        "names": {0: "person"},
    }
    atomic_text(destination / "data.yaml", yaml.safe_dump(data, sort_keys=False))
    audit = pd.DataFrame(audit_rows)
    atomic_csv(audit, destination.parent / "pasting_audit.csv")
    achieved = len(accepted_frames) / max(len(frame_ids), 1)
    summary = {
        "fold": fold,
        "requested_changed_frame_fraction": changed_fraction,
        "achieved_changed_frame_fraction": achieved,
        "train_frames": len(frame_ids),
        "selected_frames": len(selected),
        "accepted_frames": len(accepted_frames),
        "inserted_GT": inserted_gt,
        "maximum_insertions_per_frame": max(
            inserted_by_frame.values(), default=0
        ),
        "original_GT": original_gt,
        "generated_GT": generated_gt,
        "perspective_model": {
            "log_height_intercept": perspective.log_height_intercept,
            "log_height_bottom_slope": perspective.log_height_bottom_slope,
        },
        "source_leakage": 0,
        "lost_GT": 0,
        "test_used": False,
    }
    atomic_text(
        destination.parent / "pasting_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--fraction", type=float, required=True, choices=(0.25, 0.50))
    args = parser.parse_args()
    print(json.dumps(materialize(args.fold, args.fraction), indent=2))
