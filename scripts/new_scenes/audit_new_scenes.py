from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from scripts.new_scenes.common import (
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_csv,
    atomic_json,
    config,
    dhash,
    hamming,
    parse_bool,
    read_csv,
    require_columns,
    resolve,
    schema_columns,
    sha256,
)


def _input_paths() -> dict[str, Path]:
    return {
        key: PROJECT / value for key, value in config()["inputs"].items()
    }


def _write_waiting(missing: list[str]) -> dict[str, Any]:
    protocol = config()
    template = PROJECT / "protocol/new_scenes_v1/templates/NEW_SCENES_MANIFEST.csv"
    (OUTPUT / "NEW_SCENES_MANIFEST.csv").parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "NEW_SCENES_MANIFEST.csv").write_text(
        template.read_text(encoding="utf-8"), encoding="utf-8"
    )
    atomic_csv(
        OUTPUT / "HARD_NEGATIVE_COUNTS.csv",
        pd.DataFrame(columns=["scene_id", "category", "count"]),
    )
    card = """# Railway person new scenes v1 — Dataset Card

Status: `WAITING_FOR_NEW_SCENES`

No acquisition files were supplied. This document is a schema-level placeholder,
not evidence that the required railway scenes exist.

Required: 8–12 independent scenes, 1,500–3,000 manually verified frames,
at least three camera/capture points, two illumination conditions and one
causal 20-frame fragment per scene. Test remains sealed.
"""
    (OUTPUT / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    payload = {
        "protocol_id": protocol["protocol_id"],
        "status": "WAITING_FOR_NEW_SCENES",
        "missing_inputs": missing,
        "counts": {
            "scenes": 0,
            "frames": 0,
            "person_boxes": 0,
            "hard_negatives": 0,
        },
        "training_authorized": False,
        "independent_data_protocol": "BLOCKED_BY_NEW_SCENES_GATE",
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    atomic_json(OUTPUT / "NEW_SCENES_AUDIT.json", payload)
    atomic_json(OUTPUT / "STATUS.json", payload)
    return payload


def _old_identifiers() -> tuple[set[str], set[str], set[str], set[str]]:
    protocol = config()
    development = read_csv(
        PROJECT / protocol["immutable_old_data"]["development_manifest"]
    )
    sealed = read_csv(
        PROJECT / protocol["immutable_old_data"]["sealed_test_manifest"]
    )

    def values(rows: list[dict[str, str]], keys: tuple[str, ...]) -> set[str]:
        return {
            row.get(key, "")
            for row in rows
            for key in keys
            if row.get(key, "")
        }

    return (
        values(development, ("group", "grouped_scene_id", "sequence_id")),
        values(development, ("sequence", "subsequence_id")),
        values(sealed, ("group", "grouped_scene_id", "sequence_id")),
        values(sealed, ("sequence", "subsequence_id")),
    )


def _old_development_hashes() -> set[str]:
    cache = OUTPUT / "audit/OLD_DEVELOPMENT_HASHES.csv"
    if cache.is_file():
        return set(pd.read_csv(cache)["sha256"].astype(str))
    rows = read_csv(
        PROJECT / config()["immutable_old_data"]["development_manifest"]
    )
    hashes = []
    for row in rows:
        source = row.get("source_image") or row.get("output_image") or ""
        path = resolve(source) if source else Path()
        if source and path.is_file():
            hashes.append({"sha256": sha256(path)})
    atomic_csv(cache, pd.DataFrame(hashes))
    return {row["sha256"] for row in hashes}


def _annotation_digest(rows: list[dict[str, str]]) -> str:
    canonical = json.dumps(
        sorted(rows, key=lambda row: row["annotation_id"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _maximum_consecutive(frames: list[int]) -> int:
    best = current = 0
    prior: int | None = None
    for frame in sorted(set(frames)):
        current = current + 1 if prior is not None and frame == prior + 1 else 1
        best = max(best, current)
        prior = frame
    return best


def _timestamp_key(value: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    try:
        return float(text)
    except ValueError:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        return dt.datetime.fromisoformat(normalized).timestamp()


def audit() -> dict[str, Any]:
    assert_locked()
    protocol = config()
    paths = _input_paths()
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        return _write_waiting(missing)

    manifest = read_csv(paths["manifest"])
    annotations = read_csv(paths["annotations"])
    hard_negatives = read_csv(paths["hard_negatives"])
    reviews = read_csv(paths["review_log"])
    require_columns(
        paths["manifest"],
        manifest,
        schema_columns("NEW_SCENES_MANIFEST"),
    )
    require_columns(
        paths["annotations"],
        annotations,
        schema_columns("NEW_SCENES_ANNOTATIONS"),
    )
    require_columns(
        paths["hard_negatives"],
        hard_negatives,
        schema_columns("HARD_NEGATIVE_AUDIT"),
    )
    require_columns(
        paths["review_log"],
        reviews,
        schema_columns("ANNOTATION_REVIEW_LOG"),
    )

    failures: collections.Counter[str] = collections.Counter()
    image_ids = [row["image_id"] for row in manifest]
    if len(image_ids) != len(set(image_ids)):
        failures["duplicate_image_id"] += len(image_ids) - len(set(image_ids))
    manifest_by_image = {row["image_id"]: row for row in manifest}
    annotations_by_image: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in annotations:
        annotations_by_image[row["image_id"]].append(row)
    hard_by_image: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in hard_negatives:
        hard_by_image[row["image_id"]].append(row)
    reviews_by_image: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in reviews:
        reviews_by_image[row["image_id"]].append(row)

    old_dev_scenes, old_dev_sequences, sealed_scenes, sealed_sequences = (
        _old_identifiers()
    )
    old_hashes = _old_development_hashes()
    allowed_roles = set(protocol["acquisition_gate"]["required_split_roles"])
    allowed_strata = set(protocol["acquisition_gate"]["required_strata"])
    allowed_occlusion = set(protocol["annotation"]["occlusion_levels"])
    allowed_sizes = set(protocol["annotation"]["person_size_bins"])
    allowed_hard = set(protocol["annotation"]["hard_negative_categories"])

    scene_rows: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    sequence_scenes: dict[str, set[str]] = collections.defaultdict(set)
    sequence_roles: dict[str, set[str]] = collections.defaultdict(set)
    source_video_scenes: dict[str, set[str]] = collections.defaultdict(set)
    source_video_roles: dict[str, set[str]] = collections.defaultdict(set)
    scene_roles: dict[str, set[str]] = collections.defaultdict(set)
    sequence_order: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    hashes: dict[str, str] = {}
    fingerprints: list[tuple[str, str, str, int]] = []
    observed_strata: set[str] = set()
    person_boxes = 0
    hard_rows: list[dict[str, Any]] = []
    image_statistics: list[dict[str, Any]] = []

    for row in manifest:
        image_id = row["image_id"]
        scene = row["scene_id"]
        sequence = row["sequence_id"]
        role = row["split_role"]
        scene_rows[scene].append(row)
        sequence_scenes[sequence].add(scene)
        sequence_roles[sequence].add(role)
        source_video_scenes[row["source_video_id"]].add(scene)
        source_video_roles[row["source_video_id"]].add(role)
        scene_roles[scene].add(role)
        if role not in allowed_roles:
            failures["invalid_split_role"] += 1
        if scene in old_dev_scenes or sequence in old_dev_sequences:
            failures["old_development_leakage"] += 1
        if scene in sealed_scenes or sequence in sealed_sequences:
            failures["sealed_test_leakage"] += 1
        try:
            if not parse_bool(row["new_scene"]):
                failures["not_new_scene"] += 1
            temporal = parse_bool(row["temporal_eval_eligible"])
        except ValueError:
            failures["invalid_metadata"] += 1
            temporal = False
        if row["annotation_status"] != protocol["acquisition_gate"][
            "allowed_annotation_status"
        ]:
            failures["invalid_metadata"] += 1
        if not row["reviewer_role"] or not row["review_timestamp"]:
            failures["invalid_metadata"] += 1
        try:
            frame_number = int(row["frame_number"])
            timestamp = _timestamp_key(row["timestamp"])
            if float(row["nominal_fps"]) <= 0:
                raise ValueError("non-positive nominal FPS")
            sequence_order[sequence].append((frame_number, timestamp))
        except (TypeError, ValueError, OverflowError):
            failures["invalid_frame_order_metadata"] += 1
        strata = {
            item.strip() for item in row["strata"].split(";") if item.strip()
        }
        if not strata or not strata <= allowed_strata:
            failures["invalid_metadata"] += 1
        observed_strata |= strata
        image_path = resolve(row["image_path"])
        if not image_path.is_file():
            failures["missing_images"] += 1
            continue
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
        except Exception:
            failures["broken_images"] += 1
            continue
        digest = sha256(image_path)
        if digest in hashes or digest in old_hashes:
            failures["exact_duplicates"] += 1
        hashes[digest] = image_id
        fingerprints.append((image_id, scene, role, dhash(image_path)))

        person_rows = annotations_by_image.get(image_id, [])
        if "person_present" in strata and not person_rows:
            failures["missing_annotation_rows"] += 1
        if "person_present" not in strata and person_rows:
            failures["invalid_metadata"] += 1
        for annotation in person_rows:
            try:
                box = [
                    float(annotation[key]) for key in ("x1", "y1", "x2", "y2")
                ]
            except ValueError:
                failures["invalid_bbox"] += 1
                continue
            if (
                annotation["class_name"] != protocol["target"]["class_name"]
                or annotation["occlusion_level"] not in allowed_occlusion
                or annotation["person_size_bin"] not in allowed_sizes
                or annotation["review_status"] != "VERIFIED"
            ):
                failures["invalid_metadata"] += 1
            try:
                parse_bool(annotation["border_flag"])
            except ValueError:
                failures["invalid_metadata"] += 1
            if not (
                0 <= box[0] < box[2] <= width
                and 0 <= box[1] < box[3] <= height
            ):
                failures["invalid_bbox"] += 1
            person_boxes += 1
        for hard in hard_by_image.get(image_id, []):
            try:
                box = [float(hard[key]) for key in ("x1", "y1", "x2", "y2")]
            except ValueError:
                failures["invalid_bbox"] += 1
                continue
            if (
                hard["category"] not in allowed_hard
                or hard["review_status"] != "VERIFIED"
            ):
                failures["invalid_metadata"] += 1
            if not (
                0 <= box[0] < box[2] <= width
                and 0 <= box[1] < box[3] <= height
            ):
                failures["invalid_bbox"] += 1
            hard_rows.append({
                "scene_id": scene,
                "category": hard["category"],
            })
        if len(reviews_by_image.get(image_id, [])) != 1:
            failures["review_log_coverage"] += 1
        else:
            review = reviews_by_image[image_id][0]
            if (
                review["annotation_sha256"]
                != _annotation_digest(person_rows)
                or not review["reviewer_role"]
                or not review["timestamp"]
                or not review["commit"]
            ):
                failures["review_log_integrity"] += 1
        image_statistics.append({
            "image_id": image_id,
            "scene_id": scene,
            "sequence_id": sequence,
            "split_role": role,
            "person_boxes": len(person_rows),
            "hard_negatives": len(hard_by_image.get(image_id, [])),
            "temporal_eval_eligible": temporal,
            "image_sha256": digest,
        })

    unknown_annotation_images = set(annotations_by_image) - set(manifest_by_image)
    unknown_hard_images = set(hard_by_image) - set(manifest_by_image)
    unknown_review_images = set(reviews_by_image) - set(manifest_by_image)
    failures["invalid_metadata"] += (
        len(unknown_annotation_images)
        + len(unknown_hard_images)
        + len(unknown_review_images)
    )
    failures["scene_leakage"] += sum(
        len(values) != 1 for values in sequence_scenes.values()
    )
    failures["sequence_leakage"] += sum(
        len(values) != 1 for values in sequence_roles.values()
    )
    failures["source_video_fragment_leakage"] += sum(
        len(values) != 1 for values in source_video_scenes.values()
    )
    failures["source_video_split_leakage"] += sum(
        len(values) != 1 for values in source_video_roles.values()
    )
    failures["train_validation_intersections"] += sum(
        len(values) != 1 for values in scene_roles.values()
    )
    for ordered in sequence_order.values():
        frame_numbers = [item[0] for item in ordered]
        if len(frame_numbers) != len(set(frame_numbers)):
            failures["duplicate_frame_number"] += (
                len(frame_numbers) - len(set(frame_numbers))
            )
        by_frame = sorted(ordered)
        if any(
            current[1] <= prior[1]
            for prior, current in zip(by_frame, by_frame[1:])
        ):
            failures["non_monotonic_timestamp"] += 1

    maximum_hamming = int(protocol["deduplication"]["dhash_hamming_maximum"])
    for index, left in enumerate(fingerprints):
        for right in fingerprints[index + 1 :]:
            if left[2] == right[2]:
                continue
            if hamming(left[3], right[3]) <= maximum_hamming:
                failures["cross_split_near_duplicates"] += 1

    gate = protocol["acquisition_gate"]
    scenes = len(scene_rows)
    frames = len(manifest)
    if not int(gate["independent_scenes_minimum"]) <= scenes <= int(
        gate["independent_scenes_maximum"]
    ):
        failures["scene_count"] += 1
    if not int(gate["total_frames_minimum"]) <= frames <= int(
        gate["total_frames_maximum"]
    ):
        failures["total_frame_count"] += 1
    for scene, rows in scene_rows.items():
        if not int(gate["frames_per_scene_minimum"]) <= len(rows) <= int(
            gate["frames_per_scene_maximum"]
        ):
            failures["frames_per_scene"] += 1
        temporal_ok = False
        by_sequence: dict[str, list[int]] = collections.defaultdict(list)
        for row in rows:
            try:
                if parse_bool(row["temporal_eval_eligible"]):
                    by_sequence[row["sequence_id"]].append(int(row["frame_number"]))
            except ValueError:
                continue
        temporal_ok = any(
            _maximum_consecutive(sequence_frames)
            >= int(gate["continuous_fragment_minimum_frames"])
            for sequence_frames in by_sequence.values()
        )
        if gate["require_temporal_fragment_per_scene"] and not temporal_ok:
            failures["missing_temporal_fragment"] += 1
    camera_points = {
        (row["camera_id"], row["capture_point_id"]) for row in manifest
    }
    if len(camera_points) < int(
        gate["distinct_camera_or_capture_points_minimum"]
    ):
        failures["camera_capture_diversity"] += 1
    if len({row["illumination"] for row in manifest}) < int(
        gate["illumination_conditions_minimum"]
    ):
        failures["illumination_diversity"] += 1
    role_scenes = {
        role: len({row["scene_id"] for row in manifest if row["split_role"] == role})
        for role in allowed_roles
    }
    for role, minimum in gate["required_split_roles"].items():
        if role_scenes.get(role, 0) < int(minimum):
            failures["split_role_scene_support"] += 1
    if not allowed_strata <= observed_strata:
        failures["missing_required_strata"] += 1

    hard_frame = pd.DataFrame(hard_rows)
    if hard_frame.empty:
        hard_counts = pd.DataFrame(columns=["scene_id", "category", "count"])
    else:
        hard_counts = (
            hard_frame.groupby(["scene_id", "category"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
    atomic_csv(OUTPUT / "HARD_NEGATIVE_COUNTS.csv", hard_counts)
    atomic_csv(OUTPUT / "audit/IMAGE_STATISTICS.csv", pd.DataFrame(image_statistics))
    atomic_csv(OUTPUT / "NEW_SCENES_MANIFEST.csv", pd.DataFrame(manifest))
    status = "PASS" if not {key: value for key, value in failures.items() if value} else "FAIL"
    payload = {
        "protocol_id": protocol["protocol_id"],
        "status": status,
        "counts": {
            "scenes": scenes,
            "frames": frames,
            "person_boxes": person_boxes,
            "hard_negatives": len(hard_negatives),
            "camera_capture_points": len(camera_points),
            "illumination_conditions": len(
                {row["illumination"] for row in manifest}
            ),
            "role_scenes": role_scenes,
        },
        "failures": {
            key: int(value) for key, value in sorted(failures.items()) if value
        },
        "inputs": {
            key: {"path": str(path), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "training_authorized": False,
        "independent_data_protocol": (
            "ELIGIBLE_FOR_EXECUTION_LOCK"
            if status == "PASS"
            else "BLOCKED_BY_NEW_SCENES_GATE"
        ),
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    atomic_json(OUTPUT / "NEW_SCENES_AUDIT.json", payload)
    atomic_json(OUTPUT / "STATUS.json", payload)
    card = f"""# Railway person new scenes v1 — Dataset Card

Status: `{status}`

- Independent scenes: {scenes}
- Frames: {frames}
- Person boxes: {person_boxes}
- Hard-negative audit boxes: {len(hard_negatives)}
- Camera/capture points: {len(camera_points)}
- Illumination conditions: {len({row["illumination"] for row in manifest})}
- Split roles: {json.dumps(role_scenes, sort_keys=True)}
- Test: `SEALED`, access count `0`

The statistical unit is `scene_id`; `sequence_id` is indivisible. Hard-negative
categories are audit metadata and are not detector classes.
"""
    (OUTPUT / "DATASET_CARD.md").write_text(card, encoding="utf-8")
    return payload


def main() -> int:
    argparse.ArgumentParser().parse_args()
    payload = audit()
    print(json.dumps(payload, indent=2))
    return 1 if payload["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
