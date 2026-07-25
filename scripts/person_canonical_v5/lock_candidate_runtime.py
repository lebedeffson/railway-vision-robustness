from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from scripts.person_canonical_v5.common import (
    PROJECT,
    PROTOCOL_ROOT,
    assert_test_sealed,
    atomic_json,
    now,
    sha256,
)


CONFIG = PROJECT / "configs/person_v5/candidate_runtime.yaml"
DECISION = (
    PROTOCOL_ROOT / "development_diagnostic_decision.json"
)
LOCK = PROTOCOL_ROOT / "candidate_runtime_lock.json"
CODE = (
    "configs/person_v5/candidate_runtime.yaml",
    "scripts/person_canonical_v5/train_candidates.py",
    "scripts/person_canonical_v5/evaluate_candidate.py",
    "scripts/person_canonical_v5/candidate_gate.py",
    "scripts/person_canonical_v5/run_candidate_pipeline.py",
    "scripts/person_canonical_v5/materialize_pasting.py",
    "scripts/person_canonical_v5/render_pasting_audit.py",
    "scripts/person_canonical_v5/build_instance_bank.py",
    "src/data/openlabel_person_geometry.py",
    "src/models/p2_head.py",
    "src/models/coordinate_attention.py",
    "src/augmentation/person_pasting.py",
    "systemd/tnorm-person-v5-candidates.service",
)


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT, text=True
    ).strip()


def validate() -> dict[str, Any]:
    assert_test_sealed()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    required = config["diagnostic_requirements"]
    if decision["diagnostic_status"] != required["status"]:
        raise RuntimeError("Development diagnostic did not pass")
    if decision["gradual_transfer_allowed"] is not False:
        raise RuntimeError("Unexpected gradual-transfer eligibility")
    if decision["V5_E_status"] != required["V5_E_status"]:
        raise RuntimeError("V5-E decision changed")
    if config["execution"]["V5-E"] != "excluded_by_diagnostic":
        raise RuntimeError("Candidate runtime must exclude V5-E")
    if config["test_sealed"] is not True:
        raise RuntimeError("Candidate runtime must seal test")
    gate = config["gate"]["require_all"]
    if (
        float(gate["macro_mAP50_min"]) != 0.45
        or float(gate["macro_recall_min"]) != 0.45
        or float(gate["macro_small_recall_min"]) != 0.30
        or float(gate["worst_fold_recall_min"]) != 0.30
    ):
        raise RuntimeError("Candidate development gate changed")
    return {"config": config, "decision": decision}


def lock() -> dict[str, Any]:
    if git("status", "--short"):
        raise RuntimeError("Commit candidate runtime before locking")
    validate()
    missing = [path for path in CODE if not (PROJECT / path).is_file()]
    if missing:
        raise RuntimeError(f"Candidate runtime implementation missing: {missing}")
    payload = {
        "status": "LOCKED",
        "protocol_id": validate()["config"]["protocol_id"],
        "created_at": now(),
        "git_commit": git("rev-parse", "HEAD"),
        "config_sha256": sha256(CONFIG),
        "diagnostic_decision_sha256": sha256(DECISION),
        "implementation_sha256": {
            path: sha256(PROJECT / path) for path in CODE
        },
        "test_sealed": True,
        "training_allowed": True,
        "allowed_candidates": ["V5-A", "V5-B", "V5-C", "V5-D"],
        "excluded_candidates": ["V5-E", "V5-S"],
    }
    atomic_json(LOCK, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(lock(), indent=2))
