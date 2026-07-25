from __future__ import annotations

import json
import subprocess

from person_v8b.common import (
    LOCK_PATH,
    ROOT,
    atomic_json,
    load_config,
    sha256_file,
    verify_immutable_inputs,
)


LOCKED_FILES = [
    "configs/canonical_v8b_person_failure_risk.yaml",
    "protocol/v8b/V8B_PROTOCOL.md",
    "reports/v8b/V8B_ARTICLE_SCOPE.md",
    "scripts/person_v8b/common.py",
    "scripts/person_v8b/f0_audit.py",
    "scripts/person_v8b/extract_features.py",
    "scripts/person_v8b/analyze_loso.py",
    "scripts/person_v8b/lock_protocol.py",
    "scripts/person_v8b/run_pipeline.py",
    "systemd/tnorm-person-v8b-risk.service",
    "tests/test_person_v8b_failure_risk.py",
]


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def lock() -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise RuntimeError("Commit v8b implementation before protocol lock")
    config = load_config()
    immutable = verify_immutable_inputs()
    payload: dict[str, object] = {
        "protocol_id": config["protocol_id"],
        "implementation_commit": _git("rev-parse", "HEAD"),
        "status": "LOCKED_BEFORE_F0",
        "detector_training": "FORBIDDEN",
        "test_status": "SEALED",
        "test_access_count": 0,
        "attacks_status": "OUT_OF_SCOPE",
        "primary_endpoint": config["development_gate"]["primary_endpoint"],
        "files": {
            relative: sha256_file(ROOT / relative) for relative in LOCKED_FILES
        },
        "immutable_inputs": immutable,
    }
    atomic_json(LOCK_PATH, payload)
    return payload


def main() -> int:
    print(json.dumps(lock(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
