from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from audit_final_practice import load_manifest, split_leakage
from build_yolo_dataset import (
    CANDIDATE_CLASSES,
    STREAM_NAME,
    belongs_to_stream,
    get_group_name,
    get_object_types,
    normalize_shapes,
)
from download_osdar23_direct import RAW_DIR, RAW_EXCLUSIONS_PATH, SEQUENCES, load_frame_exclusions


PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST = PROJECT_DIR / "data/yolo_osdar23/manifest.csv"
OUTPUT = PROJECT_DIR / "outputs/final_practice/audit"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def label_file(sequence: str) -> Path:
    preferred = RAW_DIR / sequence / f"{sequence}_labels.json"
    if preferred.is_file():
        return preferred
    matches = sorted((RAW_DIR / sequence).glob("*_labels.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one OpenLABEL file for {sequence}, found {len(matches)}")
    return matches[0]


def safe_frame_path(sequence: str, uri: str) -> tuple[Path, str]:
    relative = PurePosixPath(uri.lstrip("/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe URI: {uri}")
    relative_text = PurePosixPath(sequence, *relative.parts).as_posix()
    return RAW_DIR / sequence / Path(*relative.parts), relative_text


def validate_shape(shape: dict[str, Any], width: int, height: int) -> str | None:
    values = shape.get("val")
    if not isinstance(values, list):
        return "shape_without_numeric_val"
    if "bbox" in str(shape.get("name", "")) or len(values) == 4:
        try:
            cx, cy, box_width, box_height = (float(value) for value in values)
        except (TypeError, ValueError):
            return "non_numeric_bbox"
        if box_width <= 0 or box_height <= 0:
            return "non_positive_bbox"
        tolerance = 1.0
        if (
            cx - box_width / 2 < -tolerance
            or cy - box_height / 2 < -tolerance
            or cx + box_width / 2 > width + tolerance
            or cy + box_height / 2 > height + tolerance
        ):
            return "bbox_outside_image"
    return None


def sample_digest(path: Path) -> str:
    """Cheap duplicate screen; full corruption is handled by PIL.verify()."""
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(65536))
        if size > 65536:
            handle.seek(max(0, size - 65536))
            digest.update(handle.read(65536))
    return digest.hexdigest()


def audit_scene(sequence: str, exclusions: set[str]) -> tuple[dict[str, object], list[dict[str, str]]]:
    errors: list[str] = []
    frame_rows: list[dict[str, str]] = []
    expected = found = objects_count = 0
    digests: dict[str, list[str]] = defaultdict(list)
    try:
        labels = label_file(sequence)
        payload = json.loads(labels.read_text(encoding="utf-8-sig"))
        openlabel = payload.get("openlabel")
        if not isinstance(openlabel, dict):
            raise ValueError("missing openlabel object")
        frames = openlabel.get("frames")
        if not isinstance(frames, dict):
            raise ValueError("openlabel.frames is not an object")
        object_types = get_object_types(openlabel)
        expected = len(frames)
        seen_uris: set[str] = set()
        for frame_id, frame in frames.items():
            frame_errors: list[str] = []
            try:
                stream = frame["frame_properties"]["streams"][STREAM_NAME]
                uri = stream["uri"]
                if not isinstance(uri, str) or not uri.strip():
                    raise ValueError("empty camera URI")
                image_path, relative = safe_frame_path(sequence, uri)
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"frame {frame_id}: {error}")
                continue
            if uri in seen_uris:
                frame_errors.append("duplicate_frame_uri")
            seen_uris.add(uri)
            excluded = relative in exclusions
            width = height = 0
            if not image_path.is_file():
                if not excluded:
                    frame_errors.append("missing_image")
            else:
                try:
                    with Image.open(image_path) as image:
                        width, height = image.size
                        image.verify()
                    if width <= 0 or height <= 0:
                        frame_errors.append("invalid_image_dimensions")
                    else:
                        found += 1
                    digests[sample_digest(image_path)].append(relative)
                except Exception as error:  # Pillow emits several format-specific errors.
                    if not excluded:
                        frame_errors.append(f"unreadable_image:{type(error).__name__}")
            frame_objects = frame.get("objects", {}) if isinstance(frame, dict) else {}
            if not isinstance(frame_objects, dict):
                frame_errors.append("frame_objects_not_object")
                frame_objects = {}
            if width and height:
                for object_id, frame_object in frame_objects.items():
                    if not isinstance(frame_object, dict):
                        continue
                    object_type = object_types.get(str(object_id))
                    if object_type not in CANDIDATE_CLASSES:
                        continue
                    object_data = frame_object.get("object_data", {})
                    if not isinstance(object_data, dict):
                        continue
                    has_geometry = False
                    for geometry_type in ("bbox", "poly2d"):
                        for shape in normalize_shapes(object_data.get(geometry_type)):
                            if not belongs_to_stream(shape):
                                continue
                            has_geometry = True
                            issue = validate_shape(shape, width, height)
                            if issue:
                                frame_errors.append(f"{object_id}:{issue}")
                    if has_geometry:
                        objects_count += 1
            if excluded and image_path.is_file() and not frame_errors:
                # The exclusion may document a truncated file that Pillow rejected;
                # a now-valid recovered frame is flagged so the split can be rebuilt deliberately.
                frame_errors.append("configured_exclusion_present")
            unapproved = [item for item in frame_errors if not excluded]
            errors.extend(f"frame {frame_id}: {item}" for item in unapproved)
            frame_rows.append({
                "sequence_id": get_group_name(sequence),
                "scene_name": sequence,
                "frame_id": str(frame_id),
                "image_path": str(image_path),
                "excluded": str(excluded).lower(),
                "status": "excluded" if excluded else ("pass" if not frame_errors else "fail"),
                "error_message": ";".join(frame_errors),
            })
        for paths in digests.values():
            if len(paths) > 1:
                errors.append("duplicate_image_content:" + "|".join(paths))
    except Exception as error:
        errors.append(f"openlabel:{type(error).__name__}:{error}")
        labels = RAW_DIR / sequence / f"{sequence}_labels.json"
    excluded_count = sum(1 for row in frame_rows if row["excluded"] == "true")
    status = "PASS_WITH_EXCLUSIONS" if not errors and excluded_count else ("PASS" if not errors else "FAIL")
    return ({
        "sequence_id": get_group_name(sequence),
        "scene_name": sequence,
        "split": "",
        "frames_expected": expected,
        "frames_found": found,
        "labels_expected": 1,
        "labels_found": int(labels.is_file()),
        "objects_count": objects_count,
        "validation_status": status,
        "error_message": "; ".join(errors),
    }, frame_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit every raw OSDaR23 OpenLABEL scene")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    exclusions = load_frame_exclusions(RAW_EXCLUSIONS_PATH)
    manifest = load_manifest(args.manifest)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        source_splits[row["source_sequence"]].add(row["split"])
    scenes: list[dict[str, object]] = []
    frames: list[dict[str, str]] = []
    for sequence in SEQUENCES:
        scene, scene_frames = audit_scene(sequence, exclusions)
        splits = source_splits.get(sequence, set())
        scene["split"] = next(iter(splits)) if len(splits) == 1 else "|".join(sorted(splits))
        if len(splits) != 1:
            scene["validation_status"] = "FAIL"
            scene["error_message"] = (str(scene["error_message"]) + "; invalid_scene_split").strip("; ")
        scenes.append(scene)
        frames.extend(scene_frames)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "scene_manifest.csv", list(scenes[0]), scenes)
    split_rows = [{
        "sequence_id": row["sequence_id"],
        "scene_name": row["source_sequence"],
        "image_path": row["image_path"],
        "frame_id": row["frame_id"],
        "split": row["split"],
    } for row in manifest]
    write_csv(args.output / "split_manifest.csv", list(split_rows[0]), split_rows)
    write_csv(args.output / "frame_validation.csv", list(frames[0]), frames)
    leakage = split_leakage(manifest)
    failed = [row["scene_name"] for row in scenes if row["validation_status"] == "FAIL"]
    summary = {
        "status": "PASS" if not failed and not any(leakage.values()) else "FAIL",
        "scenes_expected": len(SEQUENCES),
        "scenes_checked": len(scenes),
        "scenes_pass": sum(row["validation_status"] == "PASS" for row in scenes),
        "scenes_pass_with_exclusions": sum(row["validation_status"] == "PASS_WITH_EXCLUSIONS" for row in scenes),
        "scenes_failed": failed,
        "frames_expected": sum(int(row["frames_expected"]) for row in scenes),
        "frames_found_and_readable": sum(int(row["frames_found"]) for row in scenes),
        "configured_exclusions": len(exclusions),
        "split_sequence_counts": {
            split: len({row["sequence_id"] for row in manifest if row["split"] == split})
            for split in ("train", "val", "test")
        },
        "split_intersections": {name: sorted(values) for name, values in leakage.items()},
        "statistical_unit": "sequence_id",
    }
    (args.output / "split_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
