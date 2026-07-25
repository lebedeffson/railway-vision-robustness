from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/canonical_v8_person_active_data.yaml"
OUTPUT_ROOT = ROOT / "outputs/person_v8"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_columns(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    if not rows:
        raise ValueError(f"{path}: contains no data rows")
    missing = sorted(set(columns) - set(rows[0]))
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def resolve_data_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_yolo_labels(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        class_id = int(fields[0])
        if class_id != 0:
            raise ValueError(f"{path}:{line_number}: invalid person class {class_id}")
        cx, cy, box_width, box_height = map(float, fields[1:])
        values = (cx, cy, box_width, box_height)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{path}:{line_number}: coordinates outside [0, 1]")
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError(f"{path}:{line_number}: non-positive box")
        x1 = (cx - box_width / 2.0) * width
        y1 = (cy - box_height / 2.0) * height
        x2 = (cx + box_width / 2.0) * width
        y2 = (cy + box_height / 2.0) * height
        tolerance = 1e-6
        if x1 < -tolerance or y1 < -tolerance or x2 > width + tolerance or y2 > height + tolerance:
            raise ValueError(f"{path}:{line_number}: box extends outside image")
        labels.append(
            {
                "class_id": class_id,
                "box": [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)],
            }
        )
    return labels


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def assert_test_sealed(config: dict[str, Any]) -> None:
    test = config["test"]
    if not test["sealed"] or test["readable"] or test["reusable"]:
        raise RuntimeError("Canonical v8 requires a physically sealed, non-reusable test")
    marker = ROOT / test["opened_marker"]
    if marker.exists():
        raise RuntimeError(f"Railway test has already been opened: {marker}")


def verify_declared_hashes(config: dict[str, Any], include_optional_weights: bool = False) -> None:
    for name, item in config["immutable_inputs"].items():
        path = ROOT / item["path"]
        if name == "b0_initialization" and not include_optional_weights and not path.exists():
            continue
        if not path.exists():
            raise RuntimeError(f"Missing immutable input {name}: {path}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Immutable input hash mismatch for {name}: {actual}")

