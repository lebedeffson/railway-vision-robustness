from __future__ import annotations

import json
from pathlib import Path

from scripts.new_scenes.common import (
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_json,
    config,
    sha256,
)


EXECUTION_LOCK = (
    PROJECT / config()["next_protocol"]["execution_lock"]
)


def authorize() -> dict[str, object]:
    acquisition_lock = assert_locked()
    audit_path = OUTPUT / "NEW_SCENES_AUDIT.json"
    if not audit_path.is_file():
        raise RuntimeError("INDEPENDENT_DATA_BLOCKED: new-scenes audit is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS":
        raise RuntimeError(
            "INDEPENDENT_DATA_BLOCKED: new-scenes audit must PASS"
        )
    for key, item in audit["inputs"].items():
        path = Path(item["path"])
        if not path.is_absolute():
            path = PROJECT / path
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"INDEPENDENT_DATA_BLOCKED: changed input {key}")
    if EXECUTION_LOCK.exists():
        raise RuntimeError("Independent-data execution lock already exists")
    payload: dict[str, object] = {
        "protocol_id": config()["next_protocol"]["id"],
        "parent_data_protocol": config()["protocol_id"],
        "status": "B1_DETECTOR_STAGE_AUTHORIZED",
        "training_authorized": True,
        "B1": "AUTHORIZED",
        "B2": "BLOCKED_UNTIL_B1_DETECTOR_GATE_PASS",
        "B3": "BLOCKED_UNTIL_B2_TEMPORAL_GATE_PASS",
        "new_scenes_audit_sha256": sha256(audit_path),
        "acquisition_lock_sha256": sha256(
            PROJECT / "protocol/new_scenes_v1/ACQUISITION_LOCK.json"
        ),
        "input_sha256": {
            key: item["sha256"] for key, item in audit["inputs"].items()
        },
        "implementation_commit": acquisition_lock["implementation_commit"],
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    atomic_json(EXECUTION_LOCK, payload)
    return payload


def main() -> None:
    print(json.dumps(authorize(), indent=2))


if __name__ == "__main__":
    main()

