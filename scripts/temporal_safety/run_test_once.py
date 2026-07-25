from __future__ import annotations

import json

from scripts.temporal_safety.common import OUTPUT, assert_locked, atomic_json


def main() -> None:
    assert_locked()
    gate_path = OUTPUT / "development/DEVELOPMENT_GATE.json"
    freeze_path = OUTPUT / "development/PRE_TEST_FREEZE.json"
    if not gate_path.is_file() or not freeze_path.is_file():
        raise RuntimeError("Full development PASS and pre-test freeze are required")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "DEVELOPMENT_PASS":
        raise RuntimeError("Railway test remains sealed after development FAIL")
    marker = OUTPUT / "test/TEST_OPENED.json"
    if marker.exists():
        raise RuntimeError("Railway test has already been opened")
    atomic_json(
        marker,
        {
            "status": "OPENED_ONCE",
            "test_access_count": 1,
            "pre_test_freeze": json.loads(freeze_path.read_text(encoding="utf-8")),
        },
    )
    raise RuntimeError(
        "Test marker created only after PASS; canonical test inference adapter "
        "must be supplied by a separately reviewed execution commit"
    )


if __name__ == "__main__":
    main()

