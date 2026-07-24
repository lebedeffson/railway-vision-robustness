from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "configs/canonical_v5_person_data_first.yaml"
PROTOCOL_LOCK = (
    PROJECT
    / "protocols/canonical_v5_person_data_first_v1/protocol_lock.json"
)
OUTPUT = PROJECT / "outputs/person_v5"
RUNTIME_LOCK = OUTPUT / "protocol/runtime_lock.json"
TEST_MARKER = PROJECT / "outputs/person_v3/test/TEST_OPENED.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def assert_test_sealed() -> None:
    if TEST_MARKER.exists():
        raise RuntimeError(f"Railway test marker exists: {TEST_MARKER}")


def assert_runtime_locked() -> dict[str, Any]:
    assert_test_sealed()
    if not RUNTIME_LOCK.is_file():
        raise RuntimeError("Canonical v5 runtime lock is missing")
    lock = json.loads(RUNTIME_LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "LOCKED" or not lock["training_allowed"]:
        raise RuntimeError("Canonical v5 runtime is not training-ready")
    if sha256(CONFIG) != lock["protocol_sha256"]:
        raise RuntimeError("Canonical v5 protocol changed after runtime lock")
    for relative, expected in lock["code_sha256"].items():
        if sha256(PROJECT / relative) != expected:
            raise RuntimeError(f"Canonical v5 runtime code changed: {relative}")
    return lock

