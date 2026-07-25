from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.person_canonical_v5.common import (
    CONFIG,
    DIAGNOSTIC_LOCK,
    PROJECT,
    PROTOCOL_LOCK,
    assert_test_sealed,
    atomic_json,
    load_protocol,
    now,
    sha256,
)


IMPLEMENTATION = (
    "src/models/coordinate_attention.py",
    "src/models/p2_head.py",
    "src/data/range_assignment.py",
    "src/augmentation/person_pasting.py",
    "src/training/gradual_transfer.py",
    "configs/person_v5/models/yolo11m-p2.yaml",
    "configs/person_v5/models/yolo11m-p2-ca.yaml",
    "scripts/person_canonical_v5/development_diagnostic.py",
)


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=PROJECT, text=True
    ).strip()


def validate(protocol: dict[str, Any]) -> dict[str, str]:
    assert_test_sealed()
    if protocol["protocol_id"] != "person-canonical-v5-range-aware-v1":
        raise RuntimeError("Unexpected person canonical v5 protocol ID")
    if protocol["architecture"]["primary_comparisons"] != [
        "V5-A_minus_B0",
        "V5-C_minus_B0",
        "V5-D_minus_B0",
    ]:
        raise RuntimeError("Primary ablation comparisons changed")
    if protocol["test_protocol"]["on_FAIL"]["test"] != "sealed":
        raise RuntimeError("FAIL policy must keep railway test sealed")
    gate = protocol["two_fold_gate"]["require_all"]
    expected = {
        "macro_mAP50_min": 0.45,
        "macro_recall_min": 0.45,
        "macro_small_recall_min": 0.30,
        "worst_fold_recall_min": 0.30,
    }
    if any(float(gate[key]) != value for key, value in expected.items()):
        raise RuntimeError("Two-fold gate changed")
    hashes: dict[str, str] = {}
    for name, item in protocol["frozen_inputs"].items():
        path = PROJECT / item["path"]
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(
                f"Frozen input changed: {name}: {actual} != {item['sha256']}"
            )
        hashes[item["path"]] = actual
    for state, state_config in protocol["development_diagnostic"][
        "states"
    ].items():
        candidates = (
            [state_config]
            if "checkpoint" in state_config
            else [state_config["fold_0"], state_config["fold_1"]]
        )
        for candidate in candidates:
            relative = candidate["checkpoint"]
            actual = sha256(PROJECT / relative)
            if actual != candidate["sha256"]:
                raise RuntimeError(
                    f"Diagnostic checkpoint changed: {state}: {relative}"
                )
            hashes[relative] = actual
    return hashes


def lock() -> dict[str, Any]:
    if git("status", "--short"):
        raise RuntimeError("Commit person canonical v5 before locking")
    protocol = load_protocol()
    hashes = validate(protocol)
    code_hashes = {
        relative: sha256(PROJECT / relative) for relative in IMPLEMENTATION
    }
    payload = {
        "status": "LOCKED",
        "protocol_id": protocol["protocol_id"],
        "created_at": now(),
        "git_commit": git("rev-parse", "HEAD"),
        "protocol_path": str(CONFIG.relative_to(PROJECT)),
        "protocol_sha256": sha256(CONFIG),
        "frozen_input_sha256": hashes,
        "implementation_sha256": code_hashes,
        "test_sealed": True,
        "training_allowed": False,
        "reason": "development_diagnostic_must_run_before_candidate_training",
    }
    atomic_json(PROTOCOL_LOCK, payload)
    diagnostic = {
        **payload,
        "status": "LOCKED",
        "diagnostic_allowed": True,
        "training_allowed": False,
        "fixed_confidence_threshold": protocol["development_diagnostic"][
            "confidence_threshold"
        ],
        "fixed_iou_threshold": protocol["development_diagnostic"][
            "iou_threshold"
        ],
    }
    atomic_json(DIAGNOSTIC_LOCK, diagnostic)
    return payload


if __name__ == "__main__":
    print(json.dumps(lock(), indent=2))

