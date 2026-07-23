from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PROJECT_DIR
    / "configs/rescue_v2/canonical_v2_small_signal_rescue_v2.yaml"
)


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_DIR, text=True
    ).strip()


def assert_frozen_inputs() -> dict[str, Any]:
    protocol = load_protocol()
    parent = str(protocol["parent_commit"])
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent, "HEAD"],
        cwd=PROJECT_DIR,
    ).returncode != 0:
        raise RuntimeError(f"Frozen parent is not an ancestor of HEAD: {parent}")
    checkpoint = PROJECT_DIR / protocol["parent_checkpoint"]
    manifest = PROJECT_DIR / protocol["split_manifest"]
    if sha256(checkpoint) != protocol["parent_checkpoint_sha256"]:
        raise RuntimeError("Parent checkpoint SHA-256 mismatch")
    if sha256(manifest) != protocol["split_manifest_sha256"]:
        raise RuntimeError("Split manifest SHA-256 mismatch")
    return protocol


def assert_test_sealed() -> None:
    protocol = assert_frozen_inputs()
    marker = (
        PROJECT_DIR
        / protocol["scientific_boundaries"]["test_open_marker"]
    )
    if marker.exists():
        raise RuntimeError(f"Rescue-v2 test is already open: {marker}")
    if not protocol["scientific_boundaries"]["test_sealed"]:
        raise RuntimeError("Frozen rescue-v2 protocol does not seal test")


def assert_role_allowed(role: str, quality_gate_path: Path | None = None) -> None:
    assert_test_sealed()
    if role in {"audit", "micro"}:
        return
    if role == "full_training":
        gate = PROJECT_DIR / "outputs/rescue_v2/micro/selection_gate.json"
        if not gate.is_file():
            raise RuntimeError("Full training blocked: micro selection gate missing")
        payload = json.loads(gate.read_text(encoding="utf-8"))
        if not payload.get("micro_gate_passed"):
            raise RuntimeError("Full training blocked: micro gate did not pass")
        return
    if role in {"test", "attack", "article"}:
        if quality_gate_path is None or not quality_gate_path.is_file():
            raise RuntimeError(f"{role} blocked: quality gate missing")
        payload = json.loads(quality_gate_path.read_text(encoding="utf-8"))
        if not payload.get("quality_gate_passed"):
            raise RuntimeError(f"{role} blocked: quality gate did not pass")
        if not payload.get("checkpoint_frozen") or not payload.get("thresholds_frozen"):
            raise RuntimeError(f"{role} blocked: checkpoint/thresholds not frozen")
        return
    raise ValueError(f"Unknown pipeline role: {role}")
