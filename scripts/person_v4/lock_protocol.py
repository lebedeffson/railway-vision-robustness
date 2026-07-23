from __future__ import annotations

import json
from pathlib import Path

from scripts.person_v4.common import (
    LOCK_PATH,
    PROJECT_DIR,
    PROTOCOL_PATH,
    assert_test_sealed,
    atomic_json,
    git,
    load_protocol,
    now,
    sha256,
)


BASELINE_BUNDLE = (
    PROJECT_DIR
    / "outputs/person_v3/bundles/TNormFilter_person_v3_triage_failed.zip"
)


def lock() -> dict:
    assert_test_sealed()
    if git("status", "--short"):
        raise RuntimeError("Commit canonical v4 implementation before locking")
    protocol = load_protocol()
    if sha256(BASELINE_BUNDLE) != protocol["person_v3_baseline"]["bundle_sha256"]:
        raise RuntimeError("Person v3 baseline bundle hash mismatch")
    payload = {
        "status": "LOCKED",
        "created_at": now(),
        "protocol_id": protocol["protocol_id"],
        "protocol": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "git_commit": git("rev-parse", "HEAD"),
        "baseline_bundle": str(BASELINE_BUNDLE.resolve()),
        "baseline_bundle_sha256": sha256(BASELINE_BUNDLE),
        "test_marker": str(
            (PROJECT_DIR / protocol["data"]["test_marker"]).resolve()
        ),
        "test_sealed": True,
    }
    atomic_json(LOCK_PATH, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(lock(), indent=2))
