from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.temporal_safety.common import (
    CONFIG,
    LOCK,
    PROJECT,
    assert_test_sealed,
    atomic_json,
    config,
    sha256,
)


IMPLEMENTATION = [
    "configs/temporal_safety_v1.yaml",
    "protocol/temporal_safety_v1/PROTOCOL.md",
    "protocol/temporal_safety_v1/PRE_METRIC_AMENDMENT_001.json",
    "protocol/temporal_safety_v1/PRE_METRIC_AMENDMENT_002.json",
    "src/temporal_safety/tracker_base.py",
    "src/temporal_safety/bytetrack_adapter.py",
    "src/temporal_safety/ocsort_adapter.py",
    "src/temporal_safety/temporal_score.py",
    "src/temporal_safety/interpolation.py",
    "src/temporal_safety/sequence_loader.py",
    "src/temporal_safety/evaluator.py",
    "src/temporal_safety/metrics.py",
    "scripts/temporal_safety/build_sequence_index.py",
    "scripts/temporal_safety/generate_detector_predictions.py",
    "scripts/temporal_safety/run_tracker_triage.py",
    "scripts/temporal_safety/run_development_evaluation.py",
    "scripts/temporal_safety/run_test_once.py",
    "scripts/temporal_safety/finalize_article.py",
    "scripts/temporal_safety/build_failure_bundle.py",
]


def main() -> None:
    assert_test_sealed()
    protocol = config()
    if LOCK.exists():
        raise RuntimeError("Temporal safety protocol is already locked")
    for tracker, grid in protocol["tracker_grids"].items():
        if len(grid) > protocol["selection"]["maximum_configurations_per_tracker"]:
            raise RuntimeError(f"{tracker} grid exceeds frozen maximum")
    implementation_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()
    payload = {
        "protocol_id": protocol["protocol_id"],
        "parent_commit": protocol["parent_commit"],
        "implementation_commit": implementation_commit,
        "config_sha256": sha256(CONFIG),
        "implementation_sha256": {
            relative: sha256(PROJECT / relative) for relative in IMPLEMENTATION
        },
        "detector_source_protocol": protocol["detector"]["source_protocol"],
        "detector_checkpoint_sha256": {
            f"fold_{fold}": protocol["detector"][f"fold_{fold}"]["checkpoint_sha256"]
            for fold in (0, 1)
        },
        "detector_config_sha256": sha256(
            PROJECT / "configs/canonical_v3_person_safety.yaml"
        ),
        "development_manifest_sha256": sha256(
            PROJECT / protocol["data"]["development_manifest"]
        ),
        "test_manifest_sha256": sha256(
            PROJECT / protocol["data"]["sealed_test_manifest"]
        ),
        "tracker_candidates": list(protocol["tracker_grids"]),
        "parameters_selected_on": "nine_support_scenes_excluding_folds_0_and_1",
        "fit_excluded_folds": protocol["selection"]["fit_excluded_folds"],
        "screening_folds": [0, 1],
        "test_status": "SEALED",
        "test_access_count": 0,
        "v8b_immutable": True,
        "attacks": "OUT_OF_SCOPE",
    }
    atomic_json(LOCK, payload)
    LOCK.with_suffix(".sha256").write_text(
        f"{sha256(LOCK)}  {LOCK.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
