from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image


PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "configs/new_scenes_v1.yaml"
OUTPUT = PROJECT / "outputs/new_scenes_v1"
LOCK = PROJECT / "protocol/new_scenes_v1/ACQUISITION_LOCK.json"


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT / candidate


def parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    raise ValueError(f"Invalid boolean: {value}")


def schema_columns(name: str) -> list[str]:
    path = PROJECT / f"protocol/new_scenes_v1/schemas/{name}.schema.json"
    return list(json.loads(path.read_text(encoding="utf-8"))["required_columns"])


def require_columns(path: Path, rows: list[dict[str, str]], expected: list[str]) -> None:
    if rows:
        present = list(rows[0])
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            present = next(csv.reader(handle), [])
    missing = [column for column in expected if column not in present]
    if missing:
        raise ValueError(f"{path} misses required columns: {missing}")


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        grey = image.convert("L").resize((9, 8))
        pixels = list(grey.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value <<= 1
            value |= int(
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def assert_test_sealed() -> None:
    protocol = config()
    for key in ("marker", "legacy_marker"):
        marker = PROJECT / protocol["test"][key]
        if marker.exists():
            raise RuntimeError(f"Railway test is not sealed: {marker}")


def verify_parent_state() -> dict[str, Any]:
    path = PROJECT / config()["frozen_parent"]["state"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("v8b", "temporal_safety_v1", "temporal_verifier_v1", "crop_verifier_v1"):
        item = payload[key]
        artifact = PROJECT / item["artifact"]
        if not artifact.is_file() or sha256(artifact) != item["sha256"]:
            raise RuntimeError(f"Frozen parent artifact changed: {key}")
    if payload["test_status"] != "SEALED" or payload["test_access_count"] != 0:
        raise RuntimeError("Frozen parent test boundary changed")
    return payload


def assert_locked() -> dict[str, Any]:
    assert_test_sealed()
    verify_parent_state()
    if not LOCK.is_file():
        raise RuntimeError("New-scenes acquisition protocol is not locked")
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    if payload["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("New-scenes config changed after lock")
    for relative, expected in payload["files"].items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Locked new-scenes file changed: {relative}")
    return payload

