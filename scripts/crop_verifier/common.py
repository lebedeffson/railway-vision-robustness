from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "configs/crop_verifier_v1.yaml"
OUTPUT = PROJECT / "outputs/crop_verifier_v1"
LOCK = OUTPUT / "protocol/protocol_lock.json"


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


def assert_test_sealed() -> None:
    protocol = config()
    for key in ("marker", "legacy_marker"):
        marker = PROJECT / protocol["test_access"][key]
        if marker.exists():
            raise RuntimeError(f"Railway test is not sealed: {marker}")


def assert_locked() -> dict[str, Any]:
    assert_test_sealed()
    if not LOCK.is_file():
        raise RuntimeError("Crop verifier protocol is not locked")
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    if payload["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Crop verifier config changed after lock")
    for relative, expected in payload["implementation_sha256"].items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Crop verifier implementation changed: {relative}")
    if payload["test_status"] != "SEALED" or payload["test_access_count"] != 0:
        raise RuntimeError("Crop verifier test isolation failed")
    return payload

