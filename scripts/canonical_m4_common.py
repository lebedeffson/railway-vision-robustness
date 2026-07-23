from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_DIR / "configs/canonical_v2_m4_full_protocol.yaml"
SCHEMA_PATH = (
    PROJECT_DIR / "configs/schemas/canonical_v2_m4_full_protocol.schema.json"
)
OUTPUT_ROOT = PROJECT_DIR / "outputs/canonical_m4"
PROTOCOL_LOCK = OUTPUT_ROOT / "protocol/protocol_lock.json"
QUALITY_GATE = OUTPUT_ROOT / "validation/quality_gate.json"
TEST_MARKER = OUTPUT_ROOT / "final/TEST_OPENED.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT_DIR, text=True
    ).strip()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _check_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    if "const" in schema and value != schema["const"]:
        raise RuntimeError(
            f"Protocol schema mismatch at {path}: {value!r} != {schema['const']!r}"
        )
    expected = schema.get("type")
    if expected == "object" and not isinstance(value, dict):
        raise RuntimeError(f"Protocol schema requires object at {path}")
    if expected == "array" and not isinstance(value, list):
        raise RuntimeError(f"Protocol schema requires array at {path}")
    if isinstance(value, dict):
        missing = sorted(set(schema.get("required", [])) - set(value))
        if missing:
            raise RuntimeError(f"Protocol schema missing at {path}: {missing}")
        for name, child_schema in schema.get("properties", {}).items():
            if name in value:
                _check_schema(value[name], child_schema, f"{path}.{name}")


def validate_protocol_schema() -> dict[str, Any]:
    protocol = load_protocol()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    _check_schema(protocol, schema)
    if protocol["dataset"]["image_width"] != 4112:
        raise RuntimeError("M4 protocol is not frozen to audited image width")
    if protocol["dataset"]["image_height"] != 2504:
        raise RuntimeError("M4 protocol is not frozen to audited image height")
    if protocol["scene_cv"]["groups"] != "grouped_scene_id":
        raise RuntimeError("Scene CV must use grouped_scene_id")
    if len(protocol["full_training"]["seeds"]) != 3:
        raise RuntimeError("Canonical M4 requires exactly three full-training seeds")
    if protocol["evaluation"]["safety_threshold_rule"] != "maximum_F2":
        raise RuntimeError("Safety threshold rule must be frozen before validation")
    return protocol


def verify_frozen_inputs() -> dict[str, Any]:
    protocol = validate_protocol_schema()
    checks = {
        protocol["provenance"]["micro_selection"]:
            protocol["provenance"]["micro_selection_sha256"],
        protocol["provenance"]["m4_micro_result"]:
            protocol["provenance"]["m4_micro_result_sha256"],
        protocol["provenance"]["m4_micro_checkpoint"]:
            protocol["provenance"]["m4_micro_checkpoint_sha256"],
        protocol["dataset"]["yaml"]: protocol["dataset"]["yaml_sha256"],
        protocol["dataset"]["manifest"]: protocol["dataset"]["manifest_sha256"],
        protocol["dataset"]["split_manifest"]:
            protocol["dataset"]["split_manifest_sha256"],
        protocol["model"]["initialization"]:
            protocol["model"]["initialization_sha256"],
    }
    for relative, expected in checks.items():
        path = PROJECT_DIR / relative
        if not path.is_file():
            raise RuntimeError(f"Frozen M4 input is absent: {path}")
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Frozen M4 input hash mismatch: {relative}: {observed} != {expected}"
            )
    parent = str(protocol["parent_commit"])
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent, "HEAD"],
        cwd=PROJECT_DIR,
    ).returncode != 0:
        raise RuntimeError(f"M4 parent commit is not an ancestor: {parent}")
    return protocol


def expected_protocol_lock() -> dict[str, Any]:
    protocol = verify_frozen_inputs()
    protocol_commit = (
        json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))["protocol_commit"]
        if PROTOCOL_LOCK.is_file()
        else git("rev-parse", "HEAD")
    )
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, "HEAD"],
        cwd=PROJECT_DIR,
    ).returncode != 0:
        raise RuntimeError("Locked M4 protocol commit is not an ancestor of HEAD")
    return {
        "status": "LOCKED",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "schema_path": str(SCHEMA_PATH.resolve()),
        "schema_sha256": sha256(SCHEMA_PATH),
        "parent_commit": protocol["parent_commit"],
        "protocol_commit": protocol_commit,
        "split_manifest_sha256": protocol["dataset"]["split_manifest_sha256"],
        "dataset_manifest_sha256": protocol["dataset"]["manifest_sha256"],
        "initialization_sha256": protocol["model"]["initialization_sha256"],
        "micro_selection_sha256":
            protocol["provenance"]["micro_selection_sha256"],
        "test_sealed": True,
        "test_opened": False,
    }


def create_protocol_lock() -> dict[str, Any]:
    payload = expected_protocol_lock()
    payload["locked_at"] = now()
    if PROTOCOL_LOCK.exists():
        current = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
        comparable = {key: current.get(key) for key in payload if key != "locked_at"}
        expected = {key: value for key, value in payload.items() if key != "locked_at"}
        if comparable != expected:
            raise RuntimeError("Existing M4 protocol lock differs from frozen protocol")
        return current
    atomic_json(PROTOCOL_LOCK, payload)
    return payload


def assert_protocol_locked() -> dict[str, Any]:
    if not PROTOCOL_LOCK.is_file():
        raise RuntimeError("Canonical M4 full training blocked: protocol lock missing")
    lock = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    expected = expected_protocol_lock()
    for key, value in expected.items():
        if lock.get(key) != value:
            raise RuntimeError(f"Canonical M4 protocol lock mismatch: {key}")
    return lock


def assert_test_sealed() -> None:
    protocol = verify_frozen_inputs()
    if not protocol["test_sealed"]:
        raise RuntimeError("Canonical M4 protocol does not seal test")
    if TEST_MARKER.exists():
        raise RuntimeError(f"Canonical M4 test is already open: {TEST_MARKER}")


def assert_role_allowed(role: str) -> None:
    assert_protocol_locked()
    if role in {"tiling_audit", "scene_cv", "full_training", "validation"}:
        assert_test_sealed()
        return
    if role in {"normalization", "attack_calibration"}:
        assert_test_sealed()
        if not QUALITY_GATE.is_file():
            raise RuntimeError(f"{role} blocked: validation quality gate missing")
        gate = json.loads(QUALITY_GATE.read_text(encoding="utf-8"))
        if not gate.get("quality_gate_passed"):
            raise RuntimeError(f"{role} blocked: validation quality gate failed")
        return
    if role in {"test", "attack", "article"}:
        if not QUALITY_GATE.is_file():
            raise RuntimeError(f"{role} blocked: validation quality gate missing")
        gate = json.loads(QUALITY_GATE.read_text(encoding="utf-8"))
        if not gate.get("quality_gate_passed"):
            raise RuntimeError(f"{role} blocked: validation quality gate failed")
        if not gate.get("checkpoint_frozen") or not gate.get("thresholds_frozen"):
            raise RuntimeError(f"{role} blocked: checkpoint/thresholds not frozen")
        return
    raise ValueError(f"Unknown canonical M4 role: {role}")
