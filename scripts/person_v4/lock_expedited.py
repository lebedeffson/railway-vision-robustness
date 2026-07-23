from __future__ import annotations

import json

import yaml

from scripts.person_v4.common import (
    EXPEDITED_LOCK_PATH,
    EXPEDITED_PROTOCOL_PATH,
    PROTOCOL_PATH,
    assert_test_sealed,
    atomic_json,
    git,
    now,
    sha256,
)


def lock() -> dict:
    assert_test_sealed()
    if git("status", "--short"):
        raise RuntimeError(
            "Commit expedited amendment before creating its lock"
        )
    protocol = yaml.safe_load(
        EXPEDITED_PROTOCOL_PATH.read_text(encoding="utf-8")
    )
    parent_hash = sha256(PROTOCOL_PATH)
    if parent_hash != protocol["parent_protocol_sha256"]:
        raise RuntimeError("Frozen parent protocol hash mismatch")
    payload = {
        "status": "LOCKED",
        "created_at": now(),
        "created_before_A1_fold0_metrics_read": True,
        "protocol_id": protocol["protocol_id"],
        "protocol": str(EXPEDITED_PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256(EXPEDITED_PROTOCOL_PATH),
        "parent_protocol": str(PROTOCOL_PATH.resolve()),
        "parent_protocol_sha256": parent_hash,
        "parent_commit": protocol["parent_commit"],
        "git_commit": git("rev-parse", "HEAD"),
        "test_sealed": True,
    }
    atomic_json(EXPEDITED_LOCK_PATH, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(lock(), indent=2))
