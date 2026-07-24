from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.person_v5.common import (
    CONFIG,
    OUTPUT,
    PROJECT,
    PROTOCOL_LOCK,
    RUNTIME_LOCK,
    assert_test_sealed,
    atomic_json,
    now,
    sha256,
)


CODE = (
    "configs/canonical_v5_person_data_first_runtime.yaml",
    "scripts/person_v5/common.py",
    "scripts/person_v5/train.py",
    "scripts/person_v5/hard_mining.py",
    "scripts/person_v5/evaluate.py",
    "scripts/person_v5/run_pipeline.py",
)
EVIDENCE = (
    "outputs/person_v5/protocol/data_acquisition_manifest.json",
    "outputs/person_v5/protocol/crowdhuman_conversion_audit.json",
    "outputs/person_v5/data_audit/crowdhuman_data_audit.json",
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT, text=True
    ).strip()


def lock() -> dict[str, Any]:
    assert_test_sealed()
    if git("status", "--short"):
        raise RuntimeError("Commit canonical v5 runtime before locking")
    protocol_lock = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    if sha256(CONFIG) != protocol_lock["protocol_sha256"]:
        raise RuntimeError("Prospective v5 protocol changed")
    evidence_hashes = {}
    for relative in EVIDENCE:
        path = PROJECT / relative
        if not path.is_file():
            raise RuntimeError(f"Required v5 data evidence missing: {relative}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["status"] != "PASS":
            raise RuntimeError(f"V5 data evidence failed: {relative}")
        evidence_hashes[relative] = sha256(path)
    code_hashes = {relative: sha256(PROJECT / relative) for relative in CODE}
    payload = {
        "status": "LOCKED",
        "protocol_id": "canonical-v5-person-data-first-v1",
        "created_at": now(),
        "git_commit": git("rev-parse", "HEAD"),
        "protocol_sha256": sha256(CONFIG),
        "protocol_lock_sha256": sha256(PROTOCOL_LOCK),
        "data_evidence_sha256": evidence_hashes,
        "code_sha256": code_hashes,
        "test_sealed": True,
        "training_allowed": True,
    }
    atomic_json(RUNTIME_LOCK, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(lock(), indent=2))
