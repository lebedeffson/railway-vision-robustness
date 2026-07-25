from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from PIL import Image

from canonical_m4_tiling import assign_ground_truth_to_tiles
from person_v8.common import (
    OUTPUT_ROOT,
    ROOT,
    assert_test_sealed,
    atomic_json,
    dhash,
    hamming,
    load_config,
    parse_bool,
    read_csv,
    read_yolo_labels,
    require_columns,
    resolve_data_path,
    sha256_file,
    verify_declared_hashes,
    write_csv,
)


DEFAULT_MANIFEST = ROOT / "data/person_v8/acquisition_manifest.csv"
DEFAULT_CORRECTIONS = ROOT / "data/person_v8/correction_log.csv"
STATUS_PATH = OUTPUT_ROOT / "acquisition/ACQUISITION_STATUS.json"


def _required_columns(schema_path: Path) -> list[str]:
    return list(json.loads(schema_path.read_text(encoding="utf-8"))["required_columns"])


def _old_development_hashes(config: dict[str, Any]) -> set[str]:
    manifest = ROOT / config["immutable_inputs"]["prior_development_manifest"]["path"]
    rows = read_csv(manifest)
    hashes: set[str] = set()
    cache_rows: list[dict[str, str]] = []
    for row in rows:
        source = row.get("source_image") or row.get("output_image")
        if not source:
            continue
        path = resolve_data_path(source)
        if not path.exists():
            continue
        digest = sha256_file(path)
        hashes.add(digest)
        cache_rows.append({"path": str(path), "sha256": digest})
    write_csv(
        OUTPUT_ROOT / "audit/development_hash_cache.csv",
        cache_rows,
        ["path", "sha256"],
    )
    return hashes


def _sealed_identifiers(config: dict[str, Any]) -> tuple[set[str], set[str]]:
    # Metadata-only read: no sealed image or label path is opened here.
    manifest = ROOT / config["immutable_inputs"]["sealed_test_manifest"]["path"]
    rows = read_csv(manifest)
    scenes = {row.get("grouped_scene_id") or row.get("group") or "" for row in rows}
    sources = {row.get("source_video_id") or row.get("sequence") or "" for row in rows}
    return scenes - {""}, sources - {""}


def _best_tiled_size(
    labels: list[dict[str, Any]], tiling: dict[str, Any], image_size: int
) -> list[dict[str, Any]]:
    assigned = assign_ground_truth_to_tiles(labels, tiling)
    by_gt: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for tile_labels in assigned.values():
        for label in tile_labels:
            by_gt[int(label["source_gt_id"])].append(label)
    result = []
    for index, label in enumerate(labels):
        candidates = by_gt.get(index, [])
        if not candidates:
            raise RuntimeError(f"GT {index} lost after tiling")
        best = max(candidates, key=lambda item: (item["visible_fraction"], _area(item["box"])))
        x1, y1, x2, y2 = best["box"]
        scale = image_size / float(tiling["tiling"]["tile_width"])
        result.append(
            {
                "width_px": (x2 - x1) * scale,
                "height_px": (y2 - y1) * scale,
                "visible_fraction": float(best["visible_fraction"]),
            }
        )
    return result


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def audit(manifest_path: Path, corrections_path: Path) -> dict[str, Any]:
    config = load_config()
    assert_test_sealed(config)
    verify_declared_hashes(config)
    if not manifest_path.exists() or not corrections_path.exists():
        payload = {
            "protocol_id": config["protocol_id"],
            "status": "WAITING_FOR_NEW_DATA",
            "manifest_present": manifest_path.exists(),
            "correction_log_present": corrections_path.exists(),
            "test_status": "SEALED",
            "test_access_count": 0,
            "training_authorized": False,
        }
        atomic_json(STATUS_PATH, payload)
        return payload

    rows = read_csv(manifest_path)
    corrections = read_csv(corrections_path)
    require_columns(
        manifest_path,
        rows,
        _required_columns(ROOT / "protocol/v8/schemas/acquisition_manifest.schema.json"),
    )
    require_columns(
        corrections_path,
        corrections,
        _required_columns(ROOT / "protocol/v8/schemas/correction_log.schema.json"),
    )
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image_id in acquisition manifest")
    correction_by_image = collections.defaultdict(list)
    for correction in corrections:
        correction_by_image[correction["image_id"]].append(correction)

    expected_width = int(config["acquisition"]["source_frame_width"])
    expected_height = int(config["acquisition"]["source_frame_height"])
    old_hashes = _old_development_hashes(config)
    sealed_scenes, sealed_sources = _sealed_identifiers(config)
    tiling = load_config(ROOT / config["immutable_inputs"]["tiling_protocol"]["path"])
    image_size = int(config["model"]["image_size"])

    exact_seen: dict[str, str] = {}
    perceptual_seen: list[tuple[str, int]] = []
    dedup_clusters: set[str] = set()
    last_timestamp_by_subsequence: dict[str, float] = {}
    scene_stats: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    frame_stats: list[dict[str, Any]] = []
    failures: collections.Counter[str] = collections.Counter()
    gt_total = small_total = medium_total = large_total = 0
    under = collections.Counter()
    border_total = 0

    for row in rows:
        image_id = row["image_id"]
        image_path = resolve_data_path(row["image_path"])
        label_path = resolve_data_path(row["label_path"])
        if not image_path.exists():
            failures["missing_images"] += 1
            continue
        if not label_path.exists():
            failures["missing_labels"] += 1
            continue
        if row["annotation_status"] != config["acquisition"]["allowed_annotation_status"]:
            failures["unverified_annotations"] += 1
        try:
            if not parse_bool(row["new_scene"]):
                failures["not_new_scene"] += 1
        except ValueError:
            failures["not_new_scene"] += 1
        reasons = {item.strip() for item in row["selection_reasons"].split(";") if item.strip()}
        if not reasons or not reasons <= set(config["acquisition"]["selection_reasons"]):
            failures["invalid_selection_reasons"] += 1
        if not row["reviewer_role"] or not row["review_timestamp"] or not row["source_license"]:
            failures["manifest_review_incomplete"] += 1
        cluster = row["dedup_cluster_id"]
        if not cluster or cluster in dedup_clusters:
            failures["duplicate_or_missing_dedup_cluster"] += 1
        dedup_clusters.add(cluster)
        try:
            timestamp = float(row["frame_timestamp"])
            subsequence = row["subsequence_id"]
            prior_timestamp = last_timestamp_by_subsequence.get(subsequence)
            if (
                prior_timestamp is not None
                and abs(timestamp - prior_timestamp)
                < float(config["deduplication"]["minimum_time_separation_seconds"])
            ):
                failures["insufficient_time_separation"] += 1
            last_timestamp_by_subsequence[subsequence] = timestamp
        except ValueError:
            failures["invalid_frame_timestamp"] += 1
        if len(correction_by_image[image_id]) != 1:
            failures["correction_log_coverage"] += 1
        else:
            correction = correction_by_image[image_id][0]
            if correction["new_annotation_sha256"] != sha256_file(label_path):
                failures["correction_hash_mismatch"] += 1
            if not correction["reviewer_role"] or not correction["timestamp"] or not correction["commit"]:
                failures["correction_log_incomplete"] += 1

        scene = row["grouped_scene_id"]
        if scene in sealed_scenes or row["source_video_id"] in sealed_sources:
            failures["scene_leakage"] += 1
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
        if (width, height) != (expected_width, expected_height):
            failures["unexpected_image_dimensions"] += 1
            continue
        try:
            labels = read_yolo_labels(label_path, width, height)
        except ValueError as error:
            if "class" in str(error):
                failures["invalid_class_ids"] += 1
            else:
                failures["invalid_geometry"] += 1
            continue
        try:
            tiled = _best_tiled_size(labels, tiling, image_size)
        except RuntimeError:
            failures["lost_gt_after_tiling"] += 1
            continue

        exact = sha256_file(image_path)
        if exact in old_hashes or exact in exact_seen:
            failures["cross_split_duplicates"] += 1
        exact_seen[exact] = image_id
        fingerprint = dhash(image_path)
        for prior_id, prior_hash in perceptual_seen:
            if hamming(fingerprint, prior_hash) <= int(config["deduplication"]["dhash_hamming_maximum"]):
                failures["near_duplicates"] += 1
                break
        perceptual_seen.append((image_id, fingerprint))

        frame_small = frame_medium = frame_large = frame_border = 0
        for label, transformed in zip(labels, tiled, strict=True):
            area_ratio = _area(label["box"]) / float(width * height)
            if area_ratio < float(config["size_definition"]["small_area_ratio_below"]):
                frame_small += 1
            elif area_ratio < float(config["size_definition"]["medium_area_ratio_below"]):
                frame_medium += 1
            else:
                frame_large += 1
            minimum_side = min(transformed["width_px"], transformed["height_px"])
            for threshold in (2, 4, 8, 16):
                if minimum_side < threshold:
                    under[str(threshold)] += 1
            if transformed["visible_fraction"] < 1.0 - 1e-9:
                frame_border += 1
        count = len(labels)
        gt_total += count
        small_total += frame_small
        medium_total += frame_medium
        large_total += frame_large
        border_total += frame_border
        stats = scene_stats[scene]
        stats["frames"] += 1
        stats["person_gt"] += count
        stats["small_gt"] += frame_small
        stats["medium_gt"] += frame_medium
        stats["large_gt"] += frame_large
        stats["positive_frames"] += int(count > 0)
        stats["negative_frames"] += int(count == 0)
        frame_stats.append(
            {
                "image_id": image_id,
                "grouped_scene_id": scene,
                "person_gt": count,
                "small_gt": frame_small,
                "medium_gt": frame_medium,
                "large_gt": frame_large,
                "border_gt": frame_border,
                "image_sha256": exact,
                "dhash": f"{fingerprint:016x}",
            }
        )

    scene_rows = [
        {"grouped_scene_id": scene, **dict(stats)}
        for scene, stats in sorted(scene_stats.items())
    ]
    write_csv(
        OUTPUT_ROOT / "audit/scene_statistics.csv",
        scene_rows,
        [
            "grouped_scene_id",
            "frames",
            "person_gt",
            "small_gt",
            "medium_gt",
            "large_gt",
            "positive_frames",
            "negative_frames",
        ],
    )
    write_csv(
        OUTPUT_ROOT / "audit/frame_statistics.csv",
        frame_stats,
        [
            "image_id",
            "grouped_scene_id",
            "person_gt",
            "small_gt",
            "medium_gt",
            "large_gt",
            "border_gt",
            "image_sha256",
            "dhash",
        ],
    )

    minimums = config["acquisition"]
    if len(scene_stats) < int(minimums["independent_new_scene_minimum"]):
        failures["insufficient_new_scenes"] += 1
    if len(rows) < int(minimums["selected_frame_minimum"]):
        failures["insufficient_frames"] += 1
    if gt_total < int(minimums["person_box_minimum"]):
        failures["insufficient_person_boxes"] += 1
    small_fraction = small_total / gt_total if gt_total else 0.0
    if small_fraction < float(minimums["small_or_distant_fraction_minimum"]):
        failures["insufficient_small_or_distant_fraction"] += 1

    status = "PASS" if not failures else "FAIL"
    payload = {
        "protocol_id": config["protocol_id"],
        "status": status,
        "training_authorized": False,
        "test_status": "SEALED",
        "test_access_count": 0,
        "counts": {
            "new_scenes": len(scene_stats),
            "frames": len(rows),
            "person_gt": gt_total,
            "small_gt": small_total,
            "medium_gt": medium_total,
            "large_gt": large_total,
            "small_fraction": small_fraction,
            "border_gt": border_total,
            "boxes_below_tiled_pixels": dict(under),
        },
        "failures": dict(sorted(failures.items())),
        "inputs": {
            "acquisition_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "correction_log": {
                "path": str(corrections_path),
                "sha256": sha256_file(corrections_path),
            },
        },
    }
    atomic_json(OUTPUT_ROOT / "audit/CPU_GATE.json", payload)
    atomic_json(STATUS_PATH, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    args = parser.parse_args()
    result = audit(args.manifest, args.corrections)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"PASS", "WAITING_FOR_NEW_DATA"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
