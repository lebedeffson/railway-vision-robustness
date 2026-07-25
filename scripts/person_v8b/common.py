from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/canonical_v8b_person_failure_risk.yaml"
PROTOCOL_PATH = ROOT / "protocol/v8b/V8B_PROTOCOL.md"
LOCK_PATH = ROOT / "protocol/v8b/V8B_PROTOCOL_LOCK.json"
OUTPUT_ROOT = ROOT / "outputs/person_v8b"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_csv(
    path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def assert_test_sealed() -> None:
    config = load_config()
    if not config["test"]["sealed"]:
        raise RuntimeError("V8b test is not declared sealed")
    markers = [
        ROOT / config["test"]["opened_marker"],
        ROOT / "outputs/person_v3/test/TEST_OPENED.json",
        ROOT / "outputs/person_v8/test/TEST_OPENED.json",
    ]
    opened = [str(path) for path in markers if path.exists()]
    if opened:
        raise RuntimeError(f"Test access marker exists: {opened}")


def verify_immutable_inputs() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, item in load_config()["immutable_inputs"].items():
        path = ROOT / item["path"]
        if not path.is_file():
            raise RuntimeError(f"Missing immutable v8b input {name}: {path}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(
                f"Immutable v8b input changed for {name}: {actual} != {item['sha256']}"
            )
        result[name] = actual
    return result


def assert_locked() -> dict[str, Any]:
    assert_test_sealed()
    if not LOCK_PATH.is_file():
        raise RuntimeError("V8B_PROTOCOL_LOCK.json is absent")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["protocol_id"] != load_config()["protocol_id"]:
        raise RuntimeError("V8b protocol lock ID mismatch")
    for raw_path, expected in lock["files"].items():
        path = ROOT / raw_path
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"V8b locked file changed: {path}")
    verify_immutable_inputs()
    return lock

