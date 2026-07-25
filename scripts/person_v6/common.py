from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "configs/canonical_v6_person_temporal_tnorm.yaml"
LOCK = PROJECT / "protocol/v6/V6_T0_PROTOCOL_LOCK.json"
OUTPUT = PROJECT / "outputs/person_v6_temporal"


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def assert_locked() -> dict[str, Any]:
    marker = PROJECT / config()["test_marker"]
    if marker.exists():
        raise RuntimeError("Railway test has been opened")
    if not LOCK.is_file():
        raise RuntimeError("Canonical v6 T0 protocol is not locked")
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    if payload["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Canonical v6 config changed after lock")
    for relative, expected in payload["implementation_sha256"].items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Canonical v6 implementation changed: {relative}")
    if payload["test_status"] != "SEALED" or payload["attacks_status"] != "BLOCKED":
        raise RuntimeError("Canonical v6 isolation contract failed")
    return payload

