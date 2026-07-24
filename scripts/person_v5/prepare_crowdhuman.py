from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


PROJECT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT / "configs/canonical_v5_person_data_first.yaml"
LOCK_PATH = (
    PROJECT
    / "protocols/canonical_v5_person_data_first_v1/protocol_lock.json"
)


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


def iter_odgt(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid ODGT JSON"
                ) from error


def visible_person_boxes(
    record: dict[str, Any], width: int, height: int
) -> tuple[list[tuple[float, float, float, float]], int]:
    boxes: list[tuple[float, float, float, float]] = []
    ignored = 0
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
            ignored += 1
            continue
        boxes.append(
            (
                ((x1 + x2) / 2.0) / width,
                ((y1 + y2) / 2.0) / height,
                clipped_width / width,
                clipped_height / height,
            )
        )
    return boxes, ignored


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
        handle.extractall(destination)


def find_image(images: Path, identifier: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = images / f"{identifier}{suffix}"
        if candidate.is_file():
            return candidate
    matches = list(images.rglob(f"{identifier}.*"))
    return matches[0] if len(matches) == 1 else None


def convert_split(
    annotation: Path,
    images: Path,
    output_images: Path,
    output_labels: Path,
) -> dict[str, Any]:
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    records = 0
    linked_images = 0
    missing_images: list[str] = []
    boxes_total = 0
    ignored_total = 0
    for record in iter_odgt(annotation):
        records += 1
        identifier = str(record["ID"])
        source = find_image(images, identifier)
        if source is None:
            missing_images.append(identifier)
            continue
        with Image.open(source) as image:
            width, height = image.size
        boxes, ignored = visible_person_boxes(record, width, height)
        destination = output_images / source.name
        if not destination.exists():
            destination.symlink_to(source.resolve())
        label = output_labels / f"{source.stem}.txt"
        label.write_text(
            "".join(
                "0 " + " ".join(f"{value:.10f}" for value in box) + "\n"
                for box in boxes
            ),
            encoding="utf-8",
        )
        linked_images += 1
        boxes_total += len(boxes)
        ignored_total += ignored
    return {
        "records": records,
        "linked_images": linked_images,
        "missing_images": missing_images,
        "visible_person_boxes": boxes_total,
        "ignored_annotations": ignored_total,
    }


def prepare(source: Path, destination: Path) -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise RuntimeError("Freeze and commit canonical v5 protocol first")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["protocol_id"] != "canonical-v5-person-data-first-v1":
        raise RuntimeError("Wrong protocol lock")
    required = (
        "CrowdHuman_train01.zip",
        "CrowdHuman_train02.zip",
        "CrowdHuman_train03.zip",
        "CrowdHuman_val.zip",
        "annotation_train.odgt",
        "annotation_val.odgt",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing CrowdHuman files: {missing}")
    raw_images = destination / "raw/Images"
    raw_images.mkdir(parents=True, exist_ok=True)
    for name in required[:4]:
        safe_extract(source / name, destination / "raw")
    discovered = list((destination / "raw").rglob("*.jpg"))
    for image in discovered:
        target = raw_images / image.name
        if image.parent != raw_images and not target.exists():
            target.symlink_to(image.resolve())
    converted = destination / "yolo_visible_person"
    splits = {}
    for split in ("train", "val"):
        splits[split] = convert_split(
            source / f"annotation_{split}.odgt",
            raw_images,
            converted / split / "images",
            converted / split / "labels",
        )
    payload = {
        "status": (
            "PASS"
            if all(not result["missing_images"] for result in splits.values())
            else "FAIL"
        ),
        "protocol_id": lock["protocol_id"],
        "source_file_sha256": {
            name: sha256(source / name) for name in required
        },
        "annotation_view": "visible_person_vbox",
        "test_downloaded": False,
        "splits": splits,
    }
    audit = (
        PROJECT
        / "outputs/person_v5/protocol/"
        "crowdhuman_conversion_audit.json"
    )
    atomic_json(audit, payload)
    if payload["status"] != "PASS":
        raise RuntimeError("CrowdHuman conversion audit failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT / "data/crowdhuman",
    )
    parser.add_argument(
        "--accept-noncommercial-research-terms",
        action="store_true",
        help="required acknowledgement of the CrowdHuman image terms",
    )
    args = parser.parse_args()
    if not args.accept_noncommercial_research_terms:
        raise SystemExit(
            "Refusing to prepare images without explicit CrowdHuman terms "
            "acknowledgement"
        )
    print(
        json.dumps(
            prepare(args.source.resolve(), args.destination.resolve()),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
