from __future__ import annotations

import json
import subprocess

from scripts.crop_verifier.common import (
    CONFIG,
    LOCK,
    PROJECT,
    assert_test_sealed,
    atomic_json,
    config,
    sha256,
)


IMPLEMENTATION = (
    "configs/crop_verifier_v1.yaml",
    "protocol/crop_verifier_v1/PROTOCOL.md",
    "src/crop_verifier_v1/__init__.py",
    "src/crop_verifier_v1/encoder.py",
    "src/crop_verifier_v1/model.py",
    "scripts/crop_verifier/__init__.py",
    "scripts/crop_verifier/common.py",
    "scripts/crop_verifier/lock_protocol.py",
    "scripts/crop_verifier/extract_embeddings.py",
    "scripts/crop_verifier/run_crop_verifier.py",
    "scripts/crop_verifier/finalize.py",
    "tests/test_crop_verifier_v1.py",
)


def main() -> None:
    assert_test_sealed()
    protocol = config()
    if LOCK.exists():
        raise RuntimeError("Crop verifier protocol is already locked")
    if protocol["models"]["candidates"] != [
        "track_only",
        "visual_only",
        "combined",
    ]:
        raise RuntimeError("Frozen model comparison changed")
    checkpoint = PROJECT / protocol["encoder"]["checkpoint"]
    if sha256(checkpoint) != protocol["encoder"]["checkpoint_sha256"]:
        raise RuntimeError("Frozen visual checkpoint hash mismatch")
    frozen = protocol["frozen_inputs"]
    inputs = {
        "detector_predictions": sha256(
            PROJECT / frozen["detector_predictions"]
        ),
        "tracker_parameters": sha256(PROJECT / frozen["tracker_parameters"]),
        "parent_protocol_lock": sha256(PROJECT / frozen["parent_protocol_lock"]),
        "parent_decision": sha256(PROJECT / frozen["parent_decision"]),
        "visual_checkpoint": sha256(checkpoint),
    }
    for role in ("support", "screening", "confirmation"):
        root = PROJECT / frozen["tracks_root"] / role
        for filename in (
            "track_observations.csv",
            "temporal_additions.csv",
            "track_features.csv",
            "track_labels.csv",
        ):
            inputs[f"{role}_{filename}"] = sha256(root / filename)
    payload = {
        "protocol_id": protocol["protocol_id"],
        "parent_protocol": protocol["parent_protocol"],
        "parent_final_commit": protocol["parent_final_commit"],
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
        "config_sha256": sha256(CONFIG),
        "implementation_sha256": {
            relative: sha256(PROJECT / relative) for relative in IMPLEMENTATION
        },
        "frozen_input_sha256": inputs,
        "primary_tracker": frozen["tracker"],
        "model_candidates": protocol["models"]["candidates"],
        "screening_fold": protocol["data"]["screening_fold"],
        "confirmation_fold": protocol["data"]["confirmation_fold"],
        "test_status": "SEALED",
        "test_access_count": 0,
        "detector_frozen": True,
        "tracker_frozen": True,
        "parent_results_frozen": True,
    }
    atomic_json(LOCK, payload)
    LOCK.with_suffix(".sha256").write_text(
        f"{sha256(LOCK)}  {LOCK.name}\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
