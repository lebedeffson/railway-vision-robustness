from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "configs/canonical_v7_crop_verifier_amendment.yaml"
LOCK = PROJECT / "protocol/v7/V7_CROP_PROTOCOL_LOCK.json"
OUTPUT = PROJECT / "outputs/person_v7_crop_verifier"


def sha256(path: Path) -> str:
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def assert_locked() -> dict[str, Any]:
    protocol = config()
    if (PROJECT / protocol["test_marker"]).exists():
        raise RuntimeError("Railway test has been opened")
    if not LOCK.is_file():
        raise RuntimeError("Canonical v7 crop amendment is not locked")
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    if payload["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Canonical v7 crop config changed after lock")
    for relative, expected in payload["implementation_sha256"].items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Canonical v7 crop implementation changed: {relative}")
    parent_gate = PROJECT / protocol["parent_gate"]
    gate = json.loads(parent_gate.read_text(encoding="utf-8"))
    if gate["V0_GATE"] != "FAIL":
        raise RuntimeError("Crop amendment requires the frozen V0 failure")
    checkpoint = PROJECT / protocol["encoder"]["checkpoint"]
    if sha256(checkpoint) != protocol["encoder"]["checkpoint_sha256"]:
        raise RuntimeError("CrowdHuman checkpoint hash mismatch")
    if payload["test_status"] != "SEALED" or payload["attacks_status"] != "BLOCKED":
        raise RuntimeError("Crop amendment isolation contract failed")
    return payload

