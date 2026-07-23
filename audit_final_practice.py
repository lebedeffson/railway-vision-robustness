from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data/yolo_osdar23/manifest.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/final_practice/00_audit"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 2026


def canonical_path(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")

    required = {"split", "frame_id", "output_image"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")

    for row in rows:
        # The builder's ``group`` is the independent railway scene. Raw
        # ``sequence`` may contain adjacent numbered parts and is retained only
        # as provenance.
        sequence_id = (
            row.get("sequence_id")
            or row.get("group")
            or row.get("sequence")
            or ""
        ).strip()
        if not sequence_id:
            raise RuntimeError("Every manifest row must have sequence_id/group")
        row["sequence_id"] = sequence_id
        row["source_sequence"] = row.get("sequence", sequence_id)
        row["image_path"] = canonical_path(row["output_image"])

    return rows


def split_leakage(rows: Iterable[dict[str, str]]) -> dict[str, set[str]]:
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_split[row["split"]].add(row["sequence_id"])

    return {
        "train_val": by_split["train"] & by_split["val"],
        "train_test": by_split["train"] & by_split["test"],
        "val_test": by_split["val"] & by_split["test"],
    }


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def parse_label_file(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    boxes: list[tuple[int, float, float, float, float]] = []
    issues: list[str] = []
    if not path.is_file():
        return boxes, ["missing_label_file"]

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            issues.append(f"line_{line_number}:expected_5_fields")
            continue
        try:
            class_value, x, y, width, height = (float(value) for value in fields)
        except ValueError:
            issues.append(f"line_{line_number}:non_numeric")
            continue
        class_id = int(class_value)
        if class_value != class_id or class_id < 0:
            issues.append(f"line_{line_number}:invalid_class_id")
        if not all(0.0 <= value <= 1.0 for value in (x, y, width, height)):
            issues.append(f"line_{line_number}:coordinate_out_of_range")
        if width <= 0.0 or height <= 0.0:
            issues.append(f"line_{line_number}:non_positive_box")
        boxes.append((class_id, x, y, width, height))
    return boxes, issues


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_example(image_path: Path, boxes: list[tuple[int, float, float, float, float]], output: Path) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for class_id, x, y, box_width, box_height in boxes:
        left = (x - box_width / 2.0) * width
        top = (y - box_height / 2.0) * height
        right = (x + box_width / 2.0) * width
        bottom = (y + box_height / 2.0) * height
        draw.rectangle((left, top, right, bottom), outline="red", width=3)
        draw.text((left + 2, top + 2), str(class_id), fill="yellow")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((1920, 1080))
    image.save(output, quality=90)


def audit(manifest_path: Path, output: Path, sample_size: int, seed: int) -> dict[str, object]:
    rows = load_manifest(manifest_path)
    leakage = split_leakage(rows)

    split_rows = [
        {
            "sequence_id": row["sequence_id"],
            "source_sequence": row["source_sequence"],
            "image_path": row["image_path"],
            "frame_id": row["frame_id"],
            "split": row["split"],
        }
        for row in rows
    ]
    write_csv(
        output / "split_audit.csv",
        ["sequence_id", "source_sequence", "image_path", "frame_id", "split"],
        split_rows,
    )

    class_counts: Counter[tuple[str, int]] = Counter()
    image_counts: Counter[tuple[str, int]] = Counter()
    issue_rows: list[dict[str, object]] = []
    parsed: list[tuple[dict[str, str], Path, list[tuple[int, float, float, float, float]]]] = []

    for row in rows:
        image = Path(row["image_path"])
        labels = label_path(image)
        boxes, issues = parse_label_file(labels)
        expected = row.get("annotations", "")
        if expected:
            try:
                if int(expected) != len(boxes):
                    issues.append(f"annotation_count_mismatch:{expected}!={len(boxes)}")
            except ValueError:
                issues.append("invalid_manifest_annotation_count")
        if not image.is_file():
            issues.append("missing_image_file")
        for class_id in {box[0] for box in boxes}:
            image_counts[(row["split"], class_id)] += 1
        for class_id, *_ in boxes:
            class_counts[(row["split"], class_id)] += 1
        for issue in issues:
            issue_rows.append({
                "sequence_id": row["sequence_id"],
                "image_path": row["image_path"],
                "label_path": str(labels),
                "issue": issue,
            })
        parsed.append((row, image, boxes))

    class_rows = []
    for split, class_id in sorted(set(class_counts) | set(image_counts)):
        class_rows.append({
            "split": split,
            "class_id": class_id,
            "annotations": class_counts[(split, class_id)],
            "images_with_class": image_counts[(split, class_id)],
        })
    write_csv(
        output / "class_statistics.csv",
        ["split", "class_id", "annotations", "images_with_class"],
        class_rows,
    )
    write_csv(
        output / "annotation_issues.csv",
        ["sequence_id", "image_path", "label_path", "issue"],
        issue_rows,
    )

    available = [item for item in parsed if item[1].is_file()]
    rng = random.Random(seed)
    chosen = rng.sample(available, min(sample_size, len(available)))
    examples = output / "annotation_examples"
    if examples.exists():
        shutil.rmtree(examples)
    for index, (row, image, boxes) in enumerate(chosen, 1):
        name = f"{index:03d}_{row['split']}_{row['sequence_id']}_{image.stem}.jpg"
        render_example(image, boxes, examples / name)

    summary = {
        "status": "PASS" if not any(leakage.values()) and not issue_rows else "FAIL",
        "manifest": str(manifest_path.resolve()),
        "rows": len(rows),
        "images": len({row["image_path"] for row in rows}),
        "sequence_ids": len({row["sequence_id"] for row in rows}),
        "split_sequence_counts": {
            split: len({row["sequence_id"] for row in rows if row["split"] == split})
            for split in ("train", "val", "test")
        },
        "leakage": {name: sorted(values) for name, values in leakage.items()},
        "annotation_issue_count": len(issue_rows),
        "rendered_examples": len(chosen),
        "statistical_unit": "sequence_id",
        "sequence_id_source": "manifest.sequence_id, else manifest.group",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit sequence split and YOLO labels")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit(args.manifest, args.output, args.sample_size, args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
