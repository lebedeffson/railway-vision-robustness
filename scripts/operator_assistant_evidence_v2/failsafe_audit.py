from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.operator_assistant_evidence_v2.common import (
    OUTPUT,
    assert_test_sealed,
    atomic_csv,
    atomic_json,
)


CHECKS = [
    (
        "no_autonomous_alarm",
        "tests/test_review_assistant_v1.py::test_no_autonomous_alarm_output",
    ),
    (
        "no_safety_actuation",
        "tests/test_review_assistant_v1.py::test_no_safety_actuation",
    ),
    (
        "operator_decision_reversible",
        "tests/test_review_assistant_v1.py::test_operator_decision_is_reversible",
    ),
    (
        "all_review_changes_audited",
        "tests/test_review_assistant_v1.py::test_operator_decision_is_reversible",
    ),
    (
        "source_video_and_detections_immutable",
        "tests/test_operator_assistant_evidence_v2.py::test_frozen_v2_inputs_match_protocol_hashes",
    ),
    (
        "thumbnail_failure_preserves_event",
        "tests/test_review_assistant_v1.py::test_thumbnail_failure_does_not_delete_saved_event",
    ),
    (
        "restart_recovers_incomplete_processing",
        "tests/test_review_assistant_v1.py::test_resume_interrupted_video",
    ),
    (
        "duplicate_track_id_preserves_observations",
        "tests/test_review_assistant_v1.py::test_track_id_switch_does_not_duplicate_event",
    ),
    (
        "verifier_failure_routes_to_general_queue",
        "tests/test_operator_assistant_evidence_v2.py::test_verifier_exception_bypasses_filter_without_hiding_candidate",
    ),
    (
        "unknown_error_routes_to_technical_queue",
        "tests/test_review_assistant_v1.py::test_unknown_processing_error_creates_technical_review_event",
    ),
]


def main() -> None:
    assert_test_sealed()
    rows = []
    for check, node in CHECKS:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", node],
            cwd=PROJECT,
            check=False,
            capture_output=True,
            text=True,
        )
        status = "PASS" if result.returncode == 0 else "FAIL"
        evidence = (
            result.stdout.strip().splitlines()[-1]
            if result.stdout.strip()
            else result.stderr.strip().splitlines()[-1]
        )
        rows.append(
            {
                "check": check,
                "status": status,
                "test_node": node,
                "evidence": evidence,
            }
        )
    frame = pd.DataFrame(rows)
    atomic_csv(OUTPUT / "FAILSAFE_CHECKS.csv", frame)
    atomic_json(
        OUTPUT / "FAILSAFE_AUDIT.json",
        {
            "checks": len(frame),
            "passed": int(frame["status"].eq("PASS").sum()),
            "failed": int(frame["status"].eq("FAIL").sum()),
            "all_pass": bool(frame["status"].eq("PASS").all()),
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    print(
        json.dumps(
            {
                "checks": len(frame),
                "passed": int(frame["status"].eq("PASS").sum()),
            },
            sort_keys=True,
        )
    )
    if not frame["status"].eq("PASS").all():
        raise SystemExit("One or more fail-safe checks failed")


if __name__ == "__main__":
    main()
