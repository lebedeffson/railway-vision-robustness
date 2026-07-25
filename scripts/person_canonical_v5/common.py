from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "configs/person_v5/protocol.yaml"
OUTPUT = PROJECT / "outputs/person_canonical_v5"
PROTOCOL_ROOT = PROJECT / "protocols/person_canonical_v5_range_aware_v1"
PROTOCOL_LOCK = PROTOCOL_ROOT / "protocol_lock.json"
DIAGNOSTIC_LOCK = PROTOCOL_ROOT / "development_diagnostic_lock.json"
TEST_MARKER = PROJECT / "outputs/person_v3/test/TEST_OPENED.json"


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
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def assert_test_sealed() -> None:
    if TEST_MARKER.exists():
        raise RuntimeError(f"Railway test marker exists: {TEST_MARKER}")
    if not load_protocol()["claim_boundary"]["test_sealed"]:
        raise RuntimeError("Person canonical v5 protocol does not seal test")


def assert_diagnostic_locked() -> dict[str, Any]:
    assert_test_sealed()
    if not DIAGNOSTIC_LOCK.is_file():
        raise RuntimeError("Development diagnostic is not locked")
    lock = json.loads(DIAGNOSTIC_LOCK.read_text(encoding="utf-8"))
    if lock["protocol_sha256"] != sha256(CONFIG):
        raise RuntimeError("Protocol changed after development diagnostic lock")
    if not lock["diagnostic_allowed"] or lock["training_allowed"]:
        raise RuntimeError("Development diagnostic lock has invalid permissions")
    return lock

