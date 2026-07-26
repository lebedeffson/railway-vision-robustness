from __future__ import annotations

import json
import subprocess

from scripts.temporal_verifier.common import (
    CONFIG,
    LOCK,
    PROJECT,
    assert_test_sealed,
    atomic_json,
    config,
    sha256,
)


IMPLEMENTATION = [
    "configs/temporal_verifier_v1.yaml",
    "protocol/temporal_verifier_v1/PROTOCOL.md",
    "src/temporal_verifier/features.py",
    "src/temporal_verifier/pipeline.py",
    "scripts/temporal_verifier/common.py",
    "scripts/temporal_verifier/lock_protocol.py",
    "scripts/temporal_verifier/run_verifier.py",
    "scripts/temporal_verifier/finalize.py",
    "tests/test_temporal_verifier_v1.py",
]


def main() -> None:
    assert_test_sealed()
    protocol = config()
    if LOCK.exists():
        raise RuntimeError("Temporal verifier protocol is already locked")
    rules = (
        len(protocol["rule_verifier"]["k_detector_hits"])
        * len(protocol["rule_verifier"]["window_frames"])
        * len(protocol["rule_verifier"]["high_confidence_threshold"])
    )
    if rules != protocol["rule_verifier"]["maximum_candidates"]:
        raise RuntimeError("Frozen rule matrix is not exactly 12 candidates")
    parent = protocol["frozen_parent"]
    inputs = {
        "parent_protocol_lock": sha256(PROJECT / parent["protocol_lock"]),
        "selected_trackers": sha256(PROJECT / parent["selected_trackers"]),
        "raw_predictions": sha256(PROJECT / parent["raw_predictions"]),
        "parent_triage_gate": sha256(
            PROJECT / "outputs/temporal_safety_v1/triage/TRIAGE_GATE.json"
        ),
        "parent_per_scene": sha256(
            PROJECT / "outputs/temporal_safety_v1/triage/PER_SCENE_RESULTS.csv"
        ),
    }
    payload = {
        "protocol_id": protocol["protocol_id"],
        "parent_commit": protocol["parent_commit"],
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "config_sha256": sha256(CONFIG),
        "implementation_sha256": {
            relative: sha256(PROJECT / relative) for relative in IMPLEMENTATION
        },
        "frozen_input_sha256": inputs,
        "primary_tracker": protocol["tracker"]["primary"],
        "sensitivity_tracker": protocol["tracker"]["sensitivity"],
        "rule_candidates": rules,
        "support_excludes_folds": protocol["data"]["support_excludes_folds"],
        "screening_fold": protocol["data"]["screening_fold"],
        "confirmation_fold": protocol["data"]["confirmation_fold"],
        "test_status": "SEALED",
        "test_access_count": 0,
        "v8b_unchanged": True,
    }
    atomic_json(LOCK, payload)
    LOCK.with_suffix(".sha256").write_text(
        f"{sha256(LOCK)}  {LOCK.name}\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
