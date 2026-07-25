from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from person_v8.common import (
    ROOT,
    atomic_json,
    load_config,
    sha256_file,
    verify_declared_hashes,
)


LOCK_PATH = ROOT / "protocol/v8/V8_ACQUISITION_LOCK.json"
LOCKED_PATHS = [
    "configs/canonical_v8_person_active_data.yaml",
    "protocol/v8/V8_ACQUISITION_PROTOCOL.md",
    "protocol/v8/schemas/acquisition_manifest.schema.json",
    "protocol/v8/schemas/correction_log.schema.json",
    "protocol/v8/templates/acquisition_manifest.csv",
    "protocol/v8/templates/correction_log.csv",
    "scripts/person_v8/common.py",
    "scripts/person_v8/audit_active_data.py",
    "scripts/person_v8/build_splits.py",
    "scripts/person_v8/lock_acquisition.py",
    "scripts/person_v8/lock_execution.py",
    "scripts/person_v8/run_screening.py",
    "systemd/tnorm-person-v8-screening.service",
    "tests/test_person_v8_active_data.py",
]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def create_lock() -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise RuntimeError("Commit v8 acquisition implementation before locking it")
    config = load_config()
    verify_declared_hashes(config, include_optional_weights=True)
    files = {
        relative: sha256_file(ROOT / relative)
        for relative in LOCKED_PATHS
    }
    immutable = {
        name: item["sha256"]
        for name, item in config["immutable_inputs"].items()
    }
    payload: dict[str, object] = {
        "protocol_id": config["protocol_id"],
        "lock_kind": "ACQUISITION_ONLY",
        "implementation_commit": _git("rev-parse", "HEAD"),
        "status": "WAITING_FOR_NEW_DATA",
        "training_authorized": False,
        "test_status": "SEALED",
        "attack_status": "BLOCKED",
        "files": files,
        "immutable_inputs": immutable,
    }
    atomic_json(LOCK_PATH, payload)
    return payload


def main() -> int:
    argparse.ArgumentParser().parse_args()
    print(json.dumps(create_lock(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
