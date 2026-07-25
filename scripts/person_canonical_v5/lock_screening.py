from __future__ import annotations

import json
import subprocess

from scripts.person_canonical_v5.common import PROJECT, atomic_json, now, sha256
from scripts.person_canonical_v5.screening_common import CONFIG, LOCK


IMPLEMENTATION = (
    "configs/person_v5/expedited_screening.yaml",
    "protocol/v5/V5_EXPEDITED_SCREENING_AMENDMENT.md",
    "scripts/person_canonical_v5/materialize_pasting.py",
    "scripts/person_canonical_v5/screening_common.py",
    "scripts/person_canonical_v5/screening_data.py",
    "scripts/person_canonical_v5/screening_runner.py",
    "systemd/tnorm-person-v5-screening.service",
)


def lock() -> dict[str, object]:
    if LOCK.exists():
        raise RuntimeError(f"Screening lock already exists: {LOCK}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT, text=True
    ).strip()
    if status:
        raise RuntimeError("Commit screening implementation before locking")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    payload = {
        "protocol_id": "canonical-v5-expedited-screening-v1",
        "created_at": now(),
        "implementation_commit": commit,
        "config_sha256": sha256(CONFIG),
        "implementation_sha256": {
            relative: sha256(PROJECT / relative)
            for relative in IMPLEMENTATION
        },
        "article_evidence": False,
        "candidate_screening_only": True,
        "test_status": "SEALED",
        "attacks_status": "BLOCKED",
        "candidates": ["C0", "C1", "C2"],
        "fidelities": [0, 5, 10, 20],
        "confirmation_fold": 1,
        "formal_v5_results_modified": False,
        "test_used": False,
    }
    atomic_json(LOCK, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(lock(), indent=2))
