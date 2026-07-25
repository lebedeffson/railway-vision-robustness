from __future__ import annotations

import json
import subprocess

from scripts.person_v7_crop.common import (
    CONFIG,
    LOCK,
    PROJECT,
    atomic_json,
    config,
    sha256,
)


FILES = (
    "configs/canonical_v7_crop_verifier_amendment.yaml",
    "protocol/v7/V7_CROP_AMENDMENT.md",
    "src/crop_verifier/__init__.py",
    "src/crop_verifier/crop_model.py",
    "scripts/person_v7_crop/common.py",
    "scripts/person_v7_crop/lock_protocol.py",
    "scripts/person_v7_crop/run_crop.py",
    "scripts/person_v7/run_v0.py",
    "src/tracklet_verifier/verifier.py",
    "systemd/tnorm-person-v7-crop.service",
    "tests/test_person_v7_crop_verifier.py",
)


def main() -> None:
    if LOCK.exists():
        raise RuntimeError("Canonical v7 crop lock already exists")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Commit canonical v7 crop implementation before locking")
    protocol = config()
    parent_gate = PROJECT / protocol["parent_gate"]
    parent = json.loads(parent_gate.read_text(encoding="utf-8"))
    if parent["V0_GATE"] != "FAIL":
        raise RuntimeError("V0 has not released the crop amendment")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    payload = {
        "protocol_id": protocol["protocol_id"],
        "parent_result": "canonical-v7-person-tracklet-verifier-v1:V0_FAIL",
        "parent_gate_sha256": sha256(parent_gate),
        "implementation_commit": commit,
        "config_sha256": sha256(CONFIG),
        "implementation_sha256": {
            relative: sha256(PROJECT / relative) for relative in FILES
        },
        "crowdhuman_checkpoint_sha256": protocol["encoder"][
            "checkpoint_sha256"
        ],
        "encoder_frozen": True,
        "selection_uses_train_scene_oof_only": True,
        "heldout_read_after_new_freeze_only": True,
        "fold_1_runs_only_after_fold_0_pass": True,
        "test_status": "SEALED",
        "attacks_status": "BLOCKED",
        "test_used": False,
    }
    atomic_json(LOCK, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

