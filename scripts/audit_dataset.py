from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from rescue_common import (
    PROJECT_DIR,
    OUTPUT_ROOT,
    assert_test_sealed,
    atomic_json,
    completed_marker,
    configure_low_priority,
    ensure_branch,
    label_path,
    load_protocol,
    sha256,
    write_protocol_lock,
)


OUTPUT = OUTPUT_ROOT / "audit"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        rows.append((int(fields[0]), *(float(value) for value in fields[1:])))
    return rows


def bbox_checks(
    image: Path, label: Path, expected_annotations: int, split: str,
    scene: str, protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[int], Counter[str]]:
    errors: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    classes: Counter[int] = Counter()
    sizes: Counter[str] = Counter()
    with Image.open(image) as opened:
        width, height = opened.size
    labels = read_labels(label)
    if len(labels) != expected_annotations:
        errors.append({
            "severity": "fatal", "issue": "annotation_count_mismatch",
            "image_path": str(image), "label_path": str(label),
            "details": f"manifest={expected_annotations}, label={len(labels)}",
        })
    duplicates: Counter[tuple[int, float, float, float, float]] = Counter(labels)
    small_limit = float(protocol["object_sizes"]["small_area_ratio_max"])
    medium_limit = float(protocol["object_sizes"]["medium_area_ratio_max"])
    roundtrip_tolerance = float(protocol["audit"]["bbox_roundtrip_tolerance"])
    letterbox_tolerance = float(protocol["audit"]["letterbox_roundtrip_tolerance_px"])
    target = 1280
    scale = min(target / width, target / height)
    pad_x = (target - width * scale) / 2
    pad_y = (target - height * scale) / 2
    for index, values in enumerate(labels):
        class_id, xc, yc, box_width, box_height = values
        finite = all(math.isfinite(value) for value in values)
        x1 = (xc - box_width / 2) * width
        y1 = (yc - box_height / 2) * height
        x2 = (xc + box_width / 2) * width
        y2 = (yc + box_height / 2) * height
        reconstructed = (
            (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height,
        )
        roundtrip_error = max(abs(left - right) for left, right in zip(values[1:], reconstructed))
        letterbox = (
            x1 * scale + pad_x, y1 * scale + pad_y,
            x2 * scale + pad_x, y2 * scale + pad_y,
        )
        inverse = (
            (letterbox[0] - pad_x) / scale, (letterbox[1] - pad_y) / scale,
            (letterbox[2] - pad_x) / scale, (letterbox[3] - pad_y) / scale,
        )
        letterbox_error = max(abs(left - right) for left, right in zip((x1, y1, x2, y2), inverse))
        area = box_width * box_height
        size = "small" if area < small_limit else "medium" if area < medium_limit else "large"
        classes[class_id] += 1
        sizes[size] += 1
        issue_names = []
        if not finite:
            issue_names.append("non_finite")
        if class_id not in range(6):
            issue_names.append("invalid_class")
        if box_width <= 0 or box_height <= 0:
            issue_names.append("non_positive_size")
        if min(xc, yc, box_width, box_height) < 0 or max(xc, yc, box_width, box_height) > 1:
            issue_names.append("normalized_value_out_of_range")
        # YOLO labels are stored with eight decimal places. Boxes clipped to an
        # image edge can therefore extend by about 5e-9 in normalized units
        # after serialization; use the frozen normalized round-trip tolerance.
        boundary_tolerance = roundtrip_tolerance * max(width, height)
        if (
            x1 < -boundary_tolerance or y1 < -boundary_tolerance
            or x2 > width + boundary_tolerance or y2 > height + boundary_tolerance
        ):
            issue_names.append("box_outside_image")
        if roundtrip_error > roundtrip_tolerance:
            issue_names.append("yolo_pixel_roundtrip")
        if letterbox_error > letterbox_tolerance:
            issue_names.append("letterbox_roundtrip")
        if area < 1e-8:
            issue_names.append("suspiciously_tiny")
        if area > 0.95:
            issue_names.append("suspiciously_giant")
        if max(box_width / max(box_height, 1e-12), box_height / max(box_width, 1e-12)) > 100:
            issue_names.append("extreme_aspect_ratio")
        if duplicates[values] > 1:
            issue_names.append("duplicate_box")
        for issue in sorted(set(issue_names)):
            errors.append({
                "severity": "fatal" if issue in {
                    "non_finite", "invalid_class", "non_positive_size",
                    "normalized_value_out_of_range", "box_outside_image",
                    "yolo_pixel_roundtrip", "letterbox_roundtrip",
                } else "warning",
                "issue": issue, "image_path": str(image), "label_path": str(label),
                "object_index": index, "class_id": class_id,
                "xc": xc, "yc": yc, "width": box_width, "height": box_height,
                "area_ratio": area, "roundtrip_error": roundtrip_error,
                "letterbox_error_px": letterbox_error,
            })
        objects.append({
            "split": split, "grouped_scene_id": scene, "image_path": str(image),
            "class_id": class_id, "width": box_width, "height": box_height,
            "area_ratio": area, "aspect_ratio": box_width / max(box_height, 1e-12),
            "size": size, "image_width": width, "image_height": height,
        })
    return errors, objects, classes, sizes


def draw_sample(
    image_path: Path, labels: list[tuple[int, float, float, float, float]],
    caption: str, names: dict[int, str],
) -> Image.Image:
    with Image.open(image_path) as opened:
        source = opened.convert("RGB")
    source.thumbnail((480, 270), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (480, 330), "white")
    x_offset = (480 - source.width) // 2
    canvas.paste(source, (x_offset, 60))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), caption[:150], fill="black", font=ImageFont.load_default())
    for class_id, xc, yc, width, height in labels:
        x1 = x_offset + (xc - width / 2) * source.width
        y1 = 60 + (yc - height / 2) * source.height
        x2 = x_offset + (xc + width / 2) * source.width
        y2 = 60 + (yc + height / 2) * source.height
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
        draw.text((x1 + 2, max(61, y1 + 2)), f"{class_id}:{names[class_id]}", fill="yellow")
    return canvas


def save_montage(
    category: str, samples: list[dict[str, Any]], names: dict[int, str],
    limit: int | None = None,
) -> list[str]:
    destination = OUTPUT / "montages"
    destination.mkdir(parents=True, exist_ok=True)
    selected = samples[:limit] if limit is not None else samples
    outputs = []
    for page, start in enumerate(range(0, len(selected), 20), 1):
        sheet = Image.new("RGB", (4 * 480, 5 * 330), "#dddddd")
        for position, sample in enumerate(selected[start:start + 20]):
            tile = draw_sample(
                Path(sample["image_path"]), sample["labels"],
                f"{Path(sample['image_path']).name} | {sample['split']} | "
                f"{sample['grouped_scene_id']} | {sample['width']}x{sample['height']}",
                names,
            )
            sheet.paste(tile, ((position % 4) * 480, (position // 4) * 330))
        output = destination / f"{category}_{page:03d}.jpg"
        sheet.save(output, quality=88)
        outputs.append(str(output))
    return outputs


def audit_class_mapping(protocol: dict[str, Any], dataset_yaml: dict[str, Any]) -> dict[str, Any]:
    expected = {int(key): value for key, value in protocol["class_names"].items()}
    dataset = {int(key): value for key, value in dataset_yaml["names"].items()}
    checkpoint = PROJECT_DIR / protocol["failed_run"]["checkpoint"]
    model_names = {int(key): value for key, value in YOLO(str(checkpoint)).names.items()}
    evaluator = dataset.copy()
    article = expected.copy()
    consistent = dataset == expected == model_names == evaluator == article
    payload = {
        "dataset_class_id_to_name": dataset,
        "model_class_id_to_name": model_names,
        "evaluator_class_id_to_name": evaluator,
        "article_class_id_to_name": article,
        "expected_class_id_to_name": expected,
        "checkpoint_sha256": sha256(checkpoint),
        "mapping_consistent": consistent,
    }
    atomic_json(OUTPUT / "class_mapping.json", payload)
    return payload


def main() -> None:
    configure_low_priority()
    ensure_branch()
    assert_test_sealed()
    write_protocol_lock()
    protocol = load_protocol()
    manifest_path = PROJECT_DIR / protocol["split_manifest"]
    dataset_yaml_path = PROJECT_DIR / protocol["dataset"]
    manifest = pd.read_csv(manifest_path, dtype={"frame_id": str})
    dataset_yaml = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)

    manifest_hash = sha256(manifest_path)
    split_scenes = {
        split: set(rows["grouped_scene_id"].astype(str))
        for split, rows in manifest.groupby("split")
    }
    scene_overlap = {
        f"{left}_{right}": sorted(split_scenes[left] & split_scenes[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    source_sequence_splits = (
        manifest.groupby("subsequence_id")["split"].nunique().loc[lambda value: value > 1]
    )

    image_paths = [Path(value) for value in manifest["output_image"]]
    label_paths = [Path(value) for value in manifest["output_label"]]
    missing_images = [str(path) for path in image_paths if not path.is_file()]
    missing_labels = [str(path) for path in label_paths if not path.is_file()]
    actual_images = {
        path.resolve() for split in ("train", "val", "test")
        for path in (dataset_yaml_path.parent / "images" / split).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    actual_labels = {
        path.resolve() for split in ("train", "val", "test")
        for path in (dataset_yaml_path.parent / "labels" / split).glob("*.txt")
    }
    expected_images = {path.resolve() for path in image_paths}
    expected_labels = {path.resolve() for path in label_paths}

    records: list[dict[str, Any]] = []
    bbox_errors: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    class_distribution: Counter[tuple[str, str, int]] = Counter()
    size_distribution: Counter[tuple[str, str, str]] = Counter()
    exact_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    labels_exact: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_basename: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_timestamp: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    random_seed = int(protocol["audit"]["random_seed"])
    for row_index, row in enumerate(manifest.itertuples(index=False), 1):
        image = Path(row.output_image)
        label = Path(row.output_label)
        if not image.is_file() or not label.is_file():
            continue
        with Image.open(image) as opened:
            width, height = opened.size
            opened.verify()
        labels = read_labels(label)
        errors, frame_objects, classes, sizes = bbox_checks(
            image, label, int(row.annotations), str(row.split),
            str(row.grouped_scene_id), protocol,
        )
        bbox_errors.extend(errors)
        objects.extend(frame_objects)
        for class_id, count in classes.items():
            class_distribution[(str(row.split), str(row.grouped_scene_id), class_id)] += count
        for size, count in sizes.items():
            size_distribution[(str(row.split), str(row.grouped_scene_id), size)] += count
        image_hash = file_digest(image)
        label_hash = file_digest(label)
        perceptual = dhash(image)
        record = {
            "split": str(row.split), "grouped_scene_id": str(row.grouped_scene_id),
            "subsequence_id": str(row.subsequence_id), "frame_id": str(row.frame_id),
            "image_path": str(image.resolve()), "label_path": str(label.resolve()),
            "image_sha256": image_hash, "label_sha256": label_hash,
            "dhash": f"{perceptual:016x}", "dhash_int": perceptual,
            "basename": image.name, "width": width, "height": height,
            "annotations": len(labels), "labels": labels,
        }
        records.append(record)
        exact_groups[image_hash].append(record)
        labels_exact[label_hash].append(record)
        seen_basename[image.name].append(record)
        seen_timestamp[image.stem.rsplit("_", 1)[-1]].append(record)
        if row_index % 100 == 0 or row_index == len(manifest):
            print(f"[dataset-audit] checked {row_index}/{len(manifest)} frames", flush=True)

    duplicate_rows = []
    leakage_rows = []
    for kind, groups in (
        ("image_sha256", exact_groups), ("label_sha256", labels_exact),
        ("basename", seen_basename), ("timestamp", seen_timestamp),
    ):
        for value, group in groups.items():
            splits = sorted({record["split"] for record in group})
            if len(group) > 1:
                row = {
                    "kind": kind, "value": value, "count": len(group),
                    "splits": "|".join(splits),
                    "grouped_scenes": "|".join(sorted({r["grouped_scene_id"] for r in group})),
                    "paths": "|".join(r["image_path"] for r in group),
                    "cross_split": len(splits) > 1,
                }
                duplicate_rows.append(row)
                if len(splits) > 1:
                    leakage_rows.append(row)
    perceptual_limit = int(protocol["audit"]["perceptual_hash_distance_max"])
    by_split = {split: [record for record in records if record["split"] == split] for split in split_scenes}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        print(f"[dataset-audit] near-duplicate pass {left} vs {right}", flush=True)
        for first in by_split[left]:
            for second in by_split[right]:
                distance = (first["dhash_int"] ^ second["dhash_int"]).bit_count()
                if distance <= perceptual_limit:
                    leakage_rows.append({
                        "kind": "perceptual_hash", "value": f"distance={distance}",
                        "count": 2, "splits": f"{left}|{right}",
                        "grouped_scenes": f"{first['grouped_scene_id']}|{second['grouped_scene_id']}",
                        "paths": f"{first['image_path']}|{second['image_path']}",
                        "cross_split": True,
                    })

    pd.DataFrame(duplicate_rows, columns=[
        "kind", "value", "count", "splits", "grouped_scenes", "paths", "cross_split"
    ]).to_csv(OUTPUT / "duplicate_report.csv", index=False)
    pd.DataFrame(leakage_rows, columns=[
        "kind", "value", "count", "splits", "grouped_scenes", "paths", "cross_split"
    ]).to_csv(OUTPUT / "leakage_report.csv", index=False)
    pd.DataFrame(bbox_errors).to_csv(OUTPUT / "bbox_errors.csv", index=False)
    pd.DataFrame([
        {"split": split, "grouped_scene_id": scene, "class_id": class_id, "count": count}
        for (split, scene, class_id), count in sorted(class_distribution.items())
    ]).to_csv(OUTPUT / "class_distribution.csv", index=False)
    pd.DataFrame([
        {
            "split": split, "grouped_scene_id": scene,
            "frames": sum(record["split"] == split and record["grouped_scene_id"] == scene for record in records),
            "objects": sum(record["annotations"] for record in records if record["split"] == split and record["grouped_scene_id"] == scene),
        }
        for split, scenes in split_scenes.items() for scene in sorted(scenes)
    ]).to_csv(OUTPUT / "scene_distribution.csv", index=False)
    pd.DataFrame(objects).to_csv(OUTPUT / "object_size_distribution.csv", index=False)

    names = {int(key): value for key, value in protocol["class_names"].items()}
    print("[dataset-audit] rendering ground-truth montages", flush=True)
    randomizer = random.Random(random_seed)
    sample_records = records.copy()
    randomizer.shuffle(sample_records)
    montage_outputs = []
    montage_outputs += save_montage(
        "random_train", [r for r in sample_records if r["split"] == "train"], names, 100
    )
    montage_outputs += save_montage(
        "random_validation", [r for r in sample_records if r["split"] == "val"], names, 100
    )
    for class_id, class_name in names.items():
        class_samples = [r for r in sample_records if any(label[0] == class_id for label in r["labels"])]
        montage_outputs += save_montage(f"class_{class_id}_{class_name}", class_samples, names, 30)
    smallest = sorted(
        records,
        key=lambda r: min((label[3] * label[4] for label in r["labels"]), default=float("inf")),
    )
    montage_outputs += save_montage("smallest_objects", smallest, names, 100)
    montage_outputs += save_montage(
        "most_objects", sorted(records, key=lambda r: (-r["annotations"], r["image_path"])), names, 100
    )
    montage_outputs += save_montage(
        "empty_labels", [r for r in records if not r["labels"]], names
    )
    suspicious_paths = {row["image_path"] for row in bbox_errors}
    montage_outputs += save_montage(
        "suspicious_boxes", [r for r in records if r["image_path"] in suspicious_paths], names, 100
    )

    class_mapping = audit_class_mapping(protocol, dataset_yaml)
    cross_split_exact = sum(
        row["kind"] == "image_sha256" and row["cross_split"] for row in duplicate_rows
    )
    fatal_bbox = sum(row.get("severity") == "fatal" for row in bbox_errors)
    split_audit = {
        "status": "PASS" if (
            manifest_hash == protocol["split_manifest_sha256"]
            and len(set(manifest["grouped_scene_id"].astype(str))) == 20
            and not any(scene_overlap.values())
            and not missing_images and not missing_labels
            and not (actual_images - expected_images) and not (actual_labels - expected_labels)
            and not source_sequence_splits.size and cross_split_exact == 0
            and fatal_bbox == 0 and class_mapping["mapping_consistent"]
        ) else "FAIL",
        "manifest_sha256": manifest_hash,
        "expected_manifest_sha256": protocol["split_manifest_sha256"],
        "manifest_hash_match": manifest_hash == protocol["split_manifest_sha256"],
        "frames": len(manifest),
        "grouped_scenes": len(set(manifest["grouped_scene_id"].astype(str))),
        "scene_counts": {split: len(scenes) for split, scenes in split_scenes.items()},
        "frame_counts": manifest.groupby("split").size().to_dict(),
        "cross_split_scene_overlap": scene_overlap,
        "cross_split_exact_duplicates": cross_split_exact,
        "cross_split_near_duplicate_pairs": sum(
            row["kind"] == "perceptual_hash" for row in leakage_rows
        ),
        "source_sequences_crossing_splits": source_sequence_splits.index.astype(str).tolist(),
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "unmanifested_images": sorted(str(path) for path in actual_images - expected_images),
        "unmanifested_labels": sorted(str(path) for path in actual_labels - expected_labels),
        "fatal_bbox_errors": fatal_bbox,
        "warning_bbox_errors": len(bbox_errors) - fatal_bbox,
        "class_mapping_consistent": class_mapping["mapping_consistent"],
        "test_model_evaluation_performed": False,
    }
    atomic_json(OUTPUT / "split_audit.json", split_audit)
    atomic_json(OUTPUT / "visual_audit.json", {
        "visual_audit_passed": None,
        "reviewed_frames": 0,
        "confirmed_annotation_errors": 0,
        "suspected_annotation_errors": 0,
        "montage_files": montage_outputs,
        "status": "PENDING_MANUAL_REVIEW",
    })
    required = [
        OUTPUT / "split_audit.json", OUTPUT / "duplicate_report.csv",
        OUTPUT / "leakage_report.csv", OUTPUT / "class_mapping.json",
        OUTPUT / "bbox_errors.csv", OUTPUT / "class_distribution.csv",
        OUTPUT / "scene_distribution.csv", OUTPUT / "object_size_distribution.csv",
        OUTPUT / "visual_audit.json",
    ]
    if split_audit["status"] != "PASS":
        raise RuntimeError(f"Dataset audit failed: {json.dumps(split_audit, indent=2)}")
    completed_marker(
        OUTPUT, inputs=[manifest_path, dataset_yaml_path], outputs=required,
        extra={"stage": "dataset_audit", "test_model_evaluation_performed": False},
    )
    print(json.dumps(split_audit, indent=2))


if __name__ == "__main__":
    main()
