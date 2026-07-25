from __future__ import annotations

import json

from scripts.temporal_safety.common import OUTPUT, assert_locked, atomic_json


def main() -> None:
    assert_locked()
    triage_path = OUTPUT / "triage/TRIAGE_GATE.json"
    if not triage_path.is_file():
        raise RuntimeError("Two-fold triage has not completed")
    triage = json.loads(triage_path.read_text(encoding="utf-8"))
    if triage["status"] != "TRIAGE_PASS":
        atomic_json(
            OUTPUT / "development/DEVELOPMENT_GATE.json",
            {
                "status": "DEVELOPMENT_FAIL",
                "reason": "TWO_FOLD_TRIAGE_FAIL",
                "test_status": "SEALED",
                "test_access_count": 0,
            },
        )
        return
    required = [
        OUTPUT / f"oof/fold_{fold}/predictions_and_ground_truth.csv"
        for fold in (2, 3, 4)
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        atomic_json(
            OUTPUT / "development/DEVELOPMENT_GATE.json",
            {
                "status": "BLOCKED_MISSING_DETECTOR_OOF_PREDICTIONS",
                "missing": missing,
                "reason": "Temporal protocol does not authorize retraining the frozen detector",
                "test_status": "SEALED",
                "test_access_count": 0,
            },
        )
        return
    raise NotImplementedError(
        "OOF fold 2-4 inputs appeared after the implementation lock; "
        "a prospective execution amendment is required before consuming them"
    )


if __name__ == "__main__":
    main()

