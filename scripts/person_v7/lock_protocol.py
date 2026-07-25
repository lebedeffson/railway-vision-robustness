from __future__ import annotations

import json
import subprocess

from scripts.person_v7.common import CONFIG, LOCK, PROJECT, atomic_json, sha256


FILES = (
    "configs/canonical_v7_person_tracklet_verifier.yaml",
    "protocol/v7/V7_PROTOCOL.md",
    "src/tracklet_verifier/__init__.py",
    "src/tracklet_verifier/verifier.py",
    "src/temporal/ratta.py",
    "scripts/person_v7/common.py",
    "scripts/person_v7/lock_protocol.py",
    "scripts/person_v7/run_v0.py",
    "scripts/person_v3/evaluate.py",
    "scripts/audit_evaluator.py",
    "systemd/tnorm-person-v7-v0.service",
    "tests/test_person_v7_tracklet_verifier.py",
)


def main() -> None:
    if LOCK.exists():
        raise RuntimeError("Canonical v7 protocol lock already exists")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("Commit canonical v7 implementation before locking")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    payload = {
        "protocol_id": "canonical-v7-person-tracklet-verifier-v1",
        "parent_result": "canonical-v6-person-temporal-tnorm-v1:T0_FAIL",
        "implementation_commit": commit,
        "config_sha256": sha256(CONFIG),
        "implementation_sha256": {
            relative: sha256(PROJECT / relative) for relative in FILES
        },
        "models": [
            "V0-A_monotone_rank_logistic",
            "V0-B_monotone_hist_gradient_boosting",
        ],
        "model_selection_uses_train_scene_oof_only": True,
        "threshold_selection_uses_train_scene_oof_only": True,
        "heldout_fold_0_read_after_freeze_only": True,
        "fold_1_runs_only_after_fold_0_pass": True,
        "ambiguous_tracklets_excluded_from_training": True,
        "crop_verifier_status": "DEFERRED_UNLESS_V0_FAIL",
        "test_status": "SEALED",
        "attacks_status": "BLOCKED",
        "test_used": False,
    }
    atomic_json(LOCK, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
