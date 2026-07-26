from __future__ import annotations

import json
import subprocess

from scripts.new_scenes.common import (
    CONFIG,
    LOCK,
    PROJECT,
    assert_test_sealed,
    atomic_json,
    config,
    sha256,
    verify_parent_state,
)


LOCKED_FILES = (
    "configs/new_scenes_v1.yaml",
    "configs/independent_data_v1.yaml",
    "protocol/new_scenes_v1/PARENT_STATE_FREEZE.json",
    "protocol/new_scenes_v1/PROTOCOL.md",
    "protocol/new_scenes_v1/schemas/NEW_SCENES_MANIFEST.schema.json",
    "protocol/new_scenes_v1/schemas/NEW_SCENES_ANNOTATIONS.schema.json",
    "protocol/new_scenes_v1/schemas/HARD_NEGATIVE_AUDIT.schema.json",
    "protocol/new_scenes_v1/schemas/ANNOTATION_REVIEW_LOG.schema.json",
    "protocol/new_scenes_v1/templates/NEW_SCENES_MANIFEST.csv",
    "protocol/new_scenes_v1/templates/NEW_SCENES_ANNOTATIONS.csv",
    "protocol/new_scenes_v1/templates/HARD_NEGATIVE_AUDIT.csv",
    "protocol/new_scenes_v1/templates/ANNOTATION_REVIEW_LOG.csv",
    "protocol/independent_data_v1/PROTOCOL_DRAFT.md",
    "scripts/new_scenes/__init__.py",
    "scripts/new_scenes/common.py",
    "scripts/new_scenes/audit_new_scenes.py",
    "scripts/new_scenes/lock_protocol.py",
    "scripts/new_scenes/authorize_independent_data.py",
    "scripts/new_scenes/finalize.py",
    "tests/test_new_scenes_v1.py",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def main() -> None:
    assert_test_sealed()
    parent = verify_parent_state()
    if LOCK.exists():
        raise RuntimeError("New-scenes acquisition protocol is already locked")
    if _git("status", "--porcelain"):
        raise RuntimeError("Commit new-scenes implementation before locking")
    protocol = config()
    gate = protocol["acquisition_gate"]
    if (
        gate["independent_scenes_minimum"] != 8
        or gate["independent_scenes_maximum"] != 12
        or gate["total_frames_minimum"] != 1500
        or gate["total_frames_maximum"] != 3000
    ):
        raise RuntimeError("New-scenes acquisition budget changed")
    payload = {
        "protocol_id": protocol["protocol_id"],
        "lock_kind": "ACQUISITION_ONLY",
        "implementation_commit": _git("rev-parse", "HEAD"),
        "status": "WAITING_FOR_NEW_SCENES",
        "training_authorized": False,
        "config_sha256": sha256(CONFIG),
        "files": {
            relative: sha256(PROJECT / relative) for relative in LOCKED_FILES
        },
        "parent_state_sha256": sha256(
            PROJECT / protocol["frozen_parent"]["state"]
        ),
        "parent_release": parent["release_v0_11"]["tag"],
        "parent_release_recalculation_allowed": False,
        "next_protocol": protocol["next_protocol"]["id"],
        "next_protocol_status": "BLOCKED_BY_NEW_SCENES_GATE",
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    atomic_json(LOCK, payload)
    LOCK.with_suffix(".sha256").write_text(
        f"{sha256(LOCK)}  {LOCK.name}\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

