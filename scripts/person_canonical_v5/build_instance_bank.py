from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import cv2
from PIL import Image

from scripts.person_canonical_v5.common import PROJECT, assert_test_sealed, atomic_csv
from src.augmentation.person_pasting import (
    Box,
    InstanceRecord,
    grabcut_mask,
    mask_quality,
    source_instance_id,
    validate_instance_bank,
)
from src.data.openlabel_person_geometry import iter_person_geometry


MANIFEST = PROJECT / "data/yolo_osdar23_rescue_v1/manifest.csv"
FOLDS = PROJECT / "outputs/person_v3/protocol/folds.json"
ROOT = PROJECT / "outputs/person_canonical_v5/instance_pasting"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def split_scenes(fold: int) -> tuple[set[str], set[str], set[str]]:
    manifest = pd.read_csv(MANIFEST)
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))["folds"]
    heldout = set(map(str, folds[str(fold)]))
    development = set(
        manifest.loc[
            manifest["split"].isin(["train", "val"]), "grouped_scene_id"
        ].astype(str)
    )
    test = set(
        manifest.loc[manifest["split"].eq("test"), "grouped_scene_id"].astype(str)
    )
    train = development - heldout
    if train & heldout or train & test or heldout & test:
        raise RuntimeError("Scene leakage in person-v5 instance-bank split")
    return train, heldout, test


def raw_label_path(subsequence_id: str) -> Path:
    return (
        PROJECT
        / "data/raw"
        / subsequence_id
        / f"{subsequence_id}_labels.json"
    )


def expanded_box(box: Box, width: int, height: int, margin: float = 0.15) -> Box:
    horizontal = box.width * margin
    vertical = box.height * margin
    return Box(
        box.x1 - horizontal,
        box.y1 - vertical,
        box.x2 + horizontal,
        box.y2 + vertical,
    ).clipped(width, height)


def crop_box(image: Image.Image, box: Box) -> Image.Image:
    return image.crop(
        (
            int(math.floor(box.x1)),
            int(math.floor(box.y1)),
            int(math.ceil(box.x2)),
            int(math.ceil(box.y2)),
        )
    )


def process_subsequence(
    subsequence: str,
    subset: pd.DataFrame,
    image_root: Path,
    mask_root: Path,
) -> tuple[list[InstanceRecord], list[dict[str, Any]]]:
    records: list[InstanceRecord] = []
    audit: list[dict[str, Any]] = []
    label_path = raw_label_path(str(subsequence))
    if not label_path.is_file():
        raise RuntimeError(f"Missing OpenLABEL document: {label_path}")
    frame_lookup = {
        str(row.frame_id): row for row in subset.itertuples(index=False)
    }
    current_source: Path | None = None
    current_image: Image.Image | None = None
    for geometry in iter_person_geometry(
        label_path, image_width=4112, image_height=2504
    ):
        row = frame_lookup.get(geometry.frame_id)
        if row is None:
            continue
        source = Path(row.source_image)
        instance_id = source_instance_id(
            str(row.grouped_scene_id), str(row.frame_id), geometry.box
        )
        status = "REJECTED"
        reason = ""
        image_path = image_root / f"{instance_id}.png"
        mask_path = mask_root / f"{instance_id}.png"
        try:
            if current_source != source:
                current_source = source
                current_image = Image.open(source).convert("RGB")
            assert current_image is not None
            image = current_image
            region = expanded_box(
                geometry.box, image.width, image.height
            )
            cropped = crop_box(image, region)
            relative = Box(
                geometry.box.x1 - region.x1,
                geometry.box.y1 - region.y1,
                geometry.box.x2 - region.x1,
                geometry.box.y2 - region.y1,
            )
            mask = grabcut_mask(cropped, relative)
            passed, quality = mask_quality(
                mask,
                minimum_foreground_fraction=0.10,
                maximum_foreground_fraction=0.85,
                maximum_border_fraction=0.25,
            )
            if not passed:
                reason = "MASK_HALO"
            elif geometry.visibility < 0.50:
                reason = "EXCESSIVE_OCCLUSION"
            else:
                if not image_path.is_file():
                    cropped.save(image_path)
                if not mask_path.is_file():
                    Image.fromarray(mask).save(mask_path)
                status = "PASS"
                records.append(
                    InstanceRecord(
                        instance_id=instance_id,
                        source_scene_id=str(row.grouped_scene_id),
                        source_frame_id=str(row.frame_id),
                        image_path=image_path,
                        mask_path=mask_path,
                        box=geometry.box,
                        original_width=image.width,
                        original_height=image.height,
                        estimated_range_m=geometry.assignment.distance_m,
                        visibility=geometry.visibility,
                        occlusion=geometry.occlusion,
                        quality_status=status,
                    )
                )
            audit.append(
                {
                    "instance_id": instance_id,
                    "source_scene_id": str(row.grouped_scene_id),
                    "source_subsequence_id": str(row.subsequence_id),
                    "source_frame_id": str(row.frame_id),
                    "object_uuid": geometry.object_uuid,
                    "source_image": str(source.resolve()),
                    "instance_image": str(image_path.resolve()) if status == "PASS" else "",
                    "mask_path": str(mask_path.resolve()) if status == "PASS" else "",
                    "bbox_x1": geometry.box.x1,
                    "bbox_y1": geometry.box.y1,
                    "bbox_x2": geometry.box.x2,
                    "bbox_y2": geometry.box.y2,
                    "range_source": geometry.assignment.source,
                    "range_or_scale_group": geometry.assignment.group,
                    "estimated_range_m": geometry.assignment.distance_m,
                    "visibility": geometry.visibility,
                    "occlusion": geometry.occlusion,
                    "mask_foreground_fraction": quality["foreground_fraction"],
                    "mask_border_fraction": quality["border_fraction"],
                    "quality_status": status,
                    "rejection_reason": reason,
                }
            )
        except (OSError, ValueError, cv2.error) as error:
            audit.append(
                {
                    "instance_id": instance_id,
                    "source_scene_id": str(row.grouped_scene_id),
                    "source_subsequence_id": str(row.subsequence_id),
                    "source_frame_id": str(row.frame_id),
                    "object_uuid": geometry.object_uuid,
                    "source_image": str(source.resolve()),
                    "quality_status": status,
                    "rejection_reason": f"PREPARATION_ERROR:{error}",
                }
            )
    return records, audit


def build(fold: int, workers: int = 4) -> dict[str, Any]:
    assert_test_sealed()
    train_scenes, heldout_scenes, test_scenes = split_scenes(fold)
    manifest = pd.read_csv(MANIFEST)
    rows = manifest[
        manifest["split"].isin(["train", "val"])
        & manifest["grouped_scene_id"].astype(str).isin(train_scenes)
    ].copy()
    destination = ROOT / f"fold_{fold}"
    image_root = destination / "instances/images"
    mask_root = destination / "instances/masks"
    image_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    records: list[InstanceRecord] = []
    audit: list[dict[str, Any]] = []
    groups = [
        (str(subsequence), subset.copy())
        for subsequence, subset in rows.groupby("subsequence_id", sort=True)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [
            pool.submit(
                process_subsequence,
                subsequence,
                subset,
                image_root,
                mask_root,
            )
            for subsequence, subset in groups
        ]
        for future in pending:
            partial_records, partial_audit = future.result()
            records.extend(partial_records)
            audit.extend(partial_audit)
    validate_instance_bank(
        records,
        train_scene_ids=train_scenes,
        heldout_scene_ids=heldout_scenes,
        test_scene_ids=test_scenes,
    )
    frame = pd.DataFrame(audit)
    atomic_csv(frame, destination / "instance_bank_audit.csv")
    accepted = frame[frame["quality_status"].eq("PASS")].copy()
    atomic_csv(accepted, destination / "instance_bank.csv")
    summary = {
        "fold": fold,
        "train_scenes": sorted(train_scenes),
        "heldout_scenes": sorted(heldout_scenes),
        "test_scene_count": len(test_scenes),
        "candidates": int(len(frame)),
        "accepted": int(len(accepted)),
        "rejected": int(len(frame) - len(accepted)),
        "verified_lidar_geometry": int(
            accepted["range_source"].eq("verified_lidar_geometry").sum()
        ),
        "bbox_area_scale_fallback": int(
            accepted["range_source"].eq("bbox_area_scale_fallback").sum()
        ),
        "source_leakage": 0,
        "test_used": False,
    }
    atomic_json(destination / "instance_bank_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    print(json.dumps(build(arguments.fold, arguments.workers), indent=2))
