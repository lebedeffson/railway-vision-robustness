from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.article_evidence_v1.common import (
    CONFIG_PATH,
    OUTPUT,
    PROJECT,
    atomic_json,
    check_no_test_markers,
    config,
    ensure_development_path,
    sha256,
    tree_sha256,
)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()


def main() -> None:
    settings = config()
    check_no_test_markers()
    inputs = settings["inputs"]
    resolved: dict[str, Path] = {
        key: PROJECT / value for key, value in inputs.items()
    }
    missing = [key for key, path in resolved.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Required saved artifacts missing: {missing}")
    for path in resolved.values():
        ensure_development_path(path)

    provenance = {
        "git_commit": git_commit(),
        "detector_predictions_sha256": tree_sha256(resolved["raw_predictions"]),
        "tracker_outputs_sha256": tree_sha256(resolved["tracker_outputs"]),
        "track_verifier_outputs_sha256": tree_sha256(
            resolved["track_verifier_sweep"]
        ),
        "false_track_audit_sha256": tree_sha256(resolved["false_track_audit"]),
        "event_config_sha256": tree_sha256(resolved["event_config"]),
        "development_manifest_sha256": tree_sha256(
            resolved["development_manifest"]
        ),
        "crop_verifier_model_sha256": tree_sha256(
            resolved["crop_verifier_model"]
        ),
        "v8b_feature_manifest_sha256": tree_sha256(
            resolved["v8b_feature_manifest"]
        ),
        "config_sha256": sha256(CONFIG_PATH),
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    lock = {
        "protocol_id": settings["protocol_id"],
        "created_from_git_commit": provenance["git_commit"],
        "config_sha256": provenance["config_sha256"],
        "threshold_grid": settings["threshold_baseline"]["grid"],
        "metrics": [
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "fn",
            "fn_per_frame",
            "fp_per_frame",
            "false_alarms_per_min",
            "maximum_consecutive_miss",
            "mean_consecutive_miss",
            "detections_per_frame",
            "tracks_per_scene",
        ],
        "methods": settings["threshold_baseline"]["methods"],
        "statistical_unit": settings["threshold_baseline"]["statistical_unit"],
        "bootstrap_iterations": settings["statistics"]["bootstrap_iterations"],
        "bootstrap_seed": settings["statistics"]["bootstrap_seed"],
        "holm_family": settings["statistics"]["holm_family"],
        "comparisons": settings["threshold_baseline"]["comparisons"],
        "track_verifier_threshold_grid_points": settings["track_verifier"][
            "threshold_grid_points"
        ],
        "gate": settings["gate"],
        "gate_distance": settings["gate"]["gate_distance"],
        "runtime": settings["runtime"],
        "false_track_categories": settings["false_tracks"][
            "semantic_categories"
        ],
        "new_training": "FORBIDDEN",
        "new_data": "FORBIDDEN",
        "article_editing": "OUT_OF_SCOPE",
        "test_status": "SEALED",
        "test_access_count": 0,
        "immutable_after_first_calculation": True,
    }
    output_lock = OUTPUT / "COMPUTATION_LOCK.json"
    if output_lock.exists():
        existing = json.loads(output_lock.read_text(encoding="utf-8"))
        if existing != lock:
            raise RuntimeError("COMPUTATION_LOCK.json differs from frozen lock")
    else:
        atomic_json(output_lock, lock)
    output_provenance = OUTPUT / "INPUT_PROVENANCE.json"
    if output_provenance.exists():
        existing = json.loads(output_provenance.read_text(encoding="utf-8"))
        if existing != provenance:
            raise RuntimeError("INPUT_PROVENANCE.json differs from frozen inputs")
    else:
        atomic_json(output_provenance, provenance)
    print(json.dumps({"lock": lock, "provenance": provenance}, indent=2))


if __name__ == "__main__":
    main()
