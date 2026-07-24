from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.person_v5.prepare_crowdhuman import iter_odgt


PROJECT = Path(__file__).resolve().parents[2]
DOWNLOADS = PROJECT / "data/crowdhuman_downloads"
DATASET = PROJECT / "data/crowdhuman"
OUTPUT = PROJECT / "outputs/person_v5/data_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def size_bucket(width: float, height: float) -> str:
    area = width * height
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def height_bucket(height: float) -> str:
    for limit in (2, 4, 8, 16, 32):
        if height < limit:
            return f"lt_{limit}px"
    return "ge_32px"


def image_map(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if path.stem in result:
            raise RuntimeError(f"Duplicate CrowdHuman image ID: {path.stem}")
        result[path.stem] = path
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_passes(
    split_summaries: dict[str, Any],
    *,
    corrupt_images: int,
    cross_split_duplicates: int,
    id_intersection: int,
    forbidden_test_artifacts: int,
) -> bool:
    """Return the frozen data-integrity gate result.

    Source boxes that become smaller than the protocol's 1 px minimum after
    clipping are recorded and excluded by the converter. They are therefore a
    data-quality warning, not an integrity failure.
    """
    return (
        all(
            summary["record_count_matches"]
            and summary["missing_images"] == 0
            for summary in split_summaries.values()
        )
        and corrupt_images == 0
        and cross_split_duplicates == 0
        and id_intersection == 0
        and forbidden_test_artifacts == 0
    )


def audit() -> dict[str, Any]:
    if (PROJECT / "outputs/person_v3/test/TEST_OPENED.json").exists():
        raise RuntimeError("Railway test marker exists")
    forbidden = [
        path
        for root in (DOWNLOADS, DATASET)
        for path in root.rglob("*")
        if "crowdhuman_test" in path.name.lower()
    ]
    if forbidden:
        raise RuntimeError(f"CrowdHuman test artifact found: {forbidden[0]}")
    images = image_map(DATASET / "raw/Images")
    expected = {"train": 15000, "val": 4370}
    image_hash_roles: dict[str, list[tuple[str, str]]] = defaultdict(list)
    split_summaries: dict[str, Any] = {}
    size_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    corrupt_rows: list[dict[str, Any]] = []
    all_ids: dict[str, set[str]] = {}
    for split in ("train", "val"):
        annotation = DOWNLOADS / f"annotation_{split}.odgt"
        records = list(iter_odgt(annotation))
        all_ids[split] = {str(record["ID"]) for record in records}
        size_counts: Counter[str] = Counter()
        height_counts: Counter[str] = Counter()
        total_boxes = 0
        ignored = 0
        missing = []
        for record in records:
            identifier = str(record["ID"])
            image_path = images.get(identifier)
            if image_path is None:
                missing.append(identifier)
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as error:
                corrupt_rows.append(
                    {
                        "split": split,
                        "image_id": identifier,
                        "error": str(error),
                    }
                )
                continue
            image_hash_roles[sha256(image_path)].append((split, identifier))
            scale = min(640.0 / width, 640.0 / height)
            for item in record.get("gtboxes", []):
                if item.get("tag") != "person":
                    ignored += 1
                    continue
                if int(item.get("extra", {}).get("ignore", 0)) == 1:
                    ignored += 1
                    continue
                x, y, box_width, box_height = map(float, item["vbox"])
                x1 = max(0.0, min(float(width), x))
                y1 = max(0.0, min(float(height), y))
                x2 = max(0.0, min(float(width), x + box_width))
                y2 = max(0.0, min(float(height), y + box_height))
                clipped_width = x2 - x1
                clipped_height = y2 - y1
                if clipped_width < 1.0 or clipped_height < 1.0:
                    invalid_rows.append(
                        {
                            "split": split,
                            "image_id": identifier,
                            "x": x,
                            "y": y,
                            "width": box_width,
                            "height": box_height,
                            "reason": "empty_after_clipping",
                        }
                    )
                    continue
                model_width = clipped_width * scale
                model_height = clipped_height * scale
                size_counts[size_bucket(model_width, model_height)] += 1
                height_counts[height_bucket(model_height)] += 1
                total_boxes += 1
        split_summaries[split] = {
            "annotation_records": len(records),
            "expected_records": expected[split],
            "record_count_matches": len(records) == expected[split],
            "visible_person_boxes": total_boxes,
            "ignored_annotations": ignored,
            "missing_images": len(missing),
            "size_640_letterbox": dict(sorted(size_counts.items())),
            "height_640_letterbox": dict(sorted(height_counts.items())),
        }
        for bucket, count in sorted(size_counts.items()):
            size_rows.append(
                {
                    "split": split,
                    "measure": "COCO_area_at_640_letterbox",
                    "bucket": bucket,
                    "count": count,
                }
            )
        for bucket, count in sorted(height_counts.items()):
            size_rows.append(
                {
                    "split": split,
                    "measure": "height_at_640_letterbox",
                    "bucket": bucket,
                    "count": count,
                }
            )
    duplicate_rows = []
    cross_split_duplicates = 0
    for digest, roles in sorted(image_hash_roles.items()):
        if len(roles) < 2:
            continue
        splits = {split for split, _ in roles}
        if len(splits) > 1:
            cross_split_duplicates += 1
        duplicate_rows.append(
            {
                "sha256": digest,
                "count": len(roles),
                "cross_split": len(splits) > 1,
                "members": ";".join(
                    f"{split}:{identifier}" for split, identifier in roles
                ),
            }
        )
    id_intersection = sorted(all_ids["train"] & all_ids["val"])
    write_csv(OUTPUT / "box_size_distribution.csv", size_rows)
    write_csv(OUTPUT / "invalid_boxes.csv", invalid_rows)
    write_csv(OUTPUT / "corrupt_images.csv", corrupt_rows)
    write_csv(OUTPUT / "duplicate_images.csv", duplicate_rows)
    passed = audit_passes(
        split_summaries,
        corrupt_images=len(corrupt_rows),
        cross_split_duplicates=cross_split_duplicates,
        id_intersection=len(id_intersection),
        forbidden_test_artifacts=len(forbidden),
    )
    warnings = []
    if invalid_rows:
        warnings.append(
            "source_vbox_excluded_after_clipping_below_1px_minimum"
        )
    if duplicate_rows:
        warnings.append("within_split_exact_image_duplicates_recorded")
    payload = {
        "status": "PASS" if passed else "FAIL",
        "protocol_id": "canonical-v5-person-data-first-v1",
        "image_root_count": len(images),
        "splits": split_summaries,
        "invalid_boxes": len(invalid_rows),
        "excluded_unusable_boxes": len(invalid_rows),
        "excluded_box_policy": (
            "clip_to_image_bounds_then_exclude_if_width_or_height_below_1px"
        ),
        "corrupt_images": len(corrupt_rows),
        "duplicate_image_groups": len(duplicate_rows),
        "cross_split_duplicate_groups": cross_split_duplicates,
        "train_val_ID_intersection": len(id_intersection),
        "crowdhuman_test_artifacts": len(forbidden),
        "railway_test_opened": False,
        "warnings": warnings,
    }
    atomic_json(OUTPUT / "crowdhuman_data_audit.json", payload)
    if not passed:
        raise RuntimeError("CrowdHuman data audit failed")
    return payload


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
