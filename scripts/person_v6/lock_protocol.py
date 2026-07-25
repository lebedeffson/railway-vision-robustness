from __future__ import annotations

import json
import subprocess

from scripts.person_v6.common import CONFIG, LOCK, PROJECT, atomic_json, sha256


FILES = (
    "configs/canonical_v6_person_temporal_tnorm.yaml",
    "protocol/v6/ACTIVE_DATA_DRAFT_STATUS.json",
    "protocol/v6/V6_T0_PROTOCOL.md",
    "src/temporal/ratta.py",
    "scripts/person_v6/common.py",
    "scripts/person_v6/run_t0.py",
    "systemd/tnorm-person-v6-t0.service",
)


def main() -> None:
    if LOCK.exists():
        raise RuntimeError("Canonical v6 T0 lock already exists")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip():
        raise RuntimeError("Commit canonical v6 implementation before locking")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    payload = {
        "protocol_id": "canonical-v6-person-temporal-tnorm-v1",
        "implementation_commit": commit,
        "config_sha256": sha256(CONFIG),
        "implementation_sha256": {
            relative: sha256(PROJECT / relative) for relative in FILES
        },
        "fold_0_parameters_frozen_before_evaluation": True,
        "fold_1_runs_only_after_fold_0_pass": True,
        "test_status": "SEALED",
        "attacks_status": "BLOCKED",
        "test_used": False,
    }
    atomic_json(LOCK, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
