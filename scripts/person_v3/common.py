from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_DIR / "configs/canonical_v3_person_safety.yaml"
OUTPUT_ROOT = PROJECT_DIR / "outputs/person_v3"
PROTOCOL_ROOT = OUTPUT_ROOT / "protocol"
AUDIT_ROOT = OUTPUT_ROOT / "audit"
DATASET_ROOT = OUTPUT_ROOT / "dataset"
FOLDS_PATH = PROTOCOL_ROOT / "folds.json"
LOCK_PATH = PROTOCOL_ROOT / "protocol_lock.json"
TEST_MARKER = OUTPUT_ROOT / "test/TEST_OPENED.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def assert_test_sealed() -> None:
    if TEST_MARKER.exists():
        raise RuntimeError(f"Person v3 test is already open: {TEST_MARKER}")
    if not load_protocol()["data"]["test_sealed"]:
        raise RuntimeError("Person v3 protocol does not seal test")


def assert_locked() -> dict[str, Any]:
    assert_test_sealed()
    if not LOCK_PATH.is_file():
        raise RuntimeError("Person v3 protocol lock is absent")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("protocol_sha256") != sha256(PROTOCOL_PATH):
        raise RuntimeError("Person v3 protocol changed after lock")
    if not lock.get("training_allowed"):
        raise RuntimeError("Person v3 training is blocked by CPU audit")
    return lock

