from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    QUALITY_GATE,
    TEST_MARKER,
    atomic_json,
    now,
    sha256,
)


PYTHON = PROJECT_DIR / ".venv/bin/python"
PROTOCOL_PATH = PROJECT_DIR / "configs/canonical_v2_m4_expedited_protocol.yaml"
TRIAGE_REPORT = OUTPUT_ROOT / "expedited/triage/early_generalization_report.json"
STATUS = OUTPUT_ROOT / "expedited/pipeline_status.json"
SEED = 20260722


def run_script(script: str, *arguments: str) -> None:
    subprocess.run(
        [str(PYTHON), "-u", f"scripts/{script}", *arguments],
        cwd=PROJECT_DIR,
        check=True,
    )


def set_status(stage: str, status: str, **extra: Any) -> None:
    payload = (
        json.loads(STATUS.read_text(encoding="utf-8"))
        if STATUS.is_file()
        else {"protocol_id": "canonical-v2-m4-expedited-v1", "stages": {}}
    )
    payload["updated_at"] = now()
    payload["current_stage"] = stage
    payload["stages"][stage] = {"status": status, **extra}
    atomic_json(STATUS, payload)


def build_gate() -> dict[str, Any]:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    evaluation_root = OUTPUT_ROOT / f"validation/seed_{SEED}"
    evaluation = json.loads(
        (evaluation_root / "evaluation_result.json").read_text(encoding="utf-8")
    )
    threshold = json.loads(
        (evaluation_root / "threshold_selection.json").read_text(encoding="utf-8")
    )
    checkpoint = Path(evaluation["checkpoint"])
    required = {
        "checkpoint_hash_verified":
            sha256(checkpoint) == evaluation["checkpoint_sha256"],
        "evaluator_consistency_passed":
            bool(evaluation["evaluator_consistency_passed"]),
        "all_validation_frames_present":
            bool(evaluation["all_expected_frames_present"]),
        "no_duplicate_predictions":
            bool(evaluation["no_duplicate_predictions"]),
        "no_missing_scene": bool(evaluation["no_missing_scene"]),
        "no_nan_or_inf": bool(evaluation["no_nan_or_inf"]),
    }
    gate_config = protocol["official_validation_gate"]
    map_pass = float(evaluation["mAP50"]) >= float(gate_config["map50_min"])
    recall_pass = float(evaluation["safety_recall"]) >= float(
        gate_config["safety_recall_min"]
    )
    passed = map_pass and recall_pass and all(required.values())
    gate = {
        "protocol_id": protocol["protocol_id"],
        "created_at": now(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "validation_map50": float(evaluation["mAP50"]),
        "validation_safety_recall": float(evaluation["safety_recall"]),
        "map50_requirement": float(gate_config["map50_min"]),
        "recall_requirement": float(gate_config["safety_recall_min"]),
        "map50_passed": map_pass,
        "recall_passed": recall_pass,
        "required_checks": required,
        "quality_gate_passed": passed,
        "checkpoint_frozen": True,
        "thresholds_frozen": True,
        "standard_threshold": float(threshold["standard"]["threshold"]),
        "safety_threshold": float(threshold["safety"]["threshold"]),
        "test_opened": False,
        "test_marker_absent": not TEST_MARKER.exists(),
        "selection_rule": "single_prospectively_fixed_seed_20260722",
    }
    atomic_json(OUTPUT_ROOT / "expedited/quality_gate.json", gate)
    atomic_json(QUALITY_GATE, gate)
    selection = {
        "status": "FROZEN",
        "protocol_id": protocol["protocol_id"],
        "selected_seed": SEED,
        "selection_rule": gate["selection_rule"],
        "checkpoint": gate["checkpoint"],
        "checkpoint_sha256": gate["checkpoint_sha256"],
        "selection_split": "validation",
        "test_used": False,
    }
    atomic_json(OUTPUT_ROOT / "expedited/checkpoint_selection.json", selection)
    pd.DataFrame([{**evaluation, "selected": True}]).to_csv(
        OUTPUT_ROOT / "expedited/seed_comparison.csv", index=False
    )
    return gate


def failure_bundle(gate: dict[str, Any]) -> Path:
    import zipfile

    destination = (
        OUTPUT_ROOT / "bundles/TNormFilter_M4_expedited_validation_gate_failed.zip"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        paths = [
            PROTOCOL_PATH,
            TRIAGE_REPORT,
            OUTPUT_ROOT / "expedited/quality_gate.json",
            OUTPUT_ROOT / "expedited/checkpoint_selection.json",
            OUTPUT_ROOT / f"validation/seed_{SEED}/evaluation_result.json",
            OUTPUT_ROOT / f"validation/seed_{SEED}/per_scene_metrics.csv",
            OUTPUT_ROOT / f"validation/seed_{SEED}/per_class_metrics.csv",
            OUTPUT_ROOT / f"validation/seed_{SEED}/per_size_metrics.csv",
            STATUS,
        ]
        for path in paths:
            if path.is_file():
                archive.write(path, path.relative_to(PROJECT_DIR))
        archive.writestr(
            "run_summary.json",
            json.dumps(gate, indent=2, ensure_ascii=False) + "\n",
        )
    return destination


def main() -> None:
    if TEST_MARKER.exists():
        raise RuntimeError("Expedited validation cannot start after test opening")
    triage = json.loads(TRIAGE_REPORT.read_text(encoding="utf-8"))
    if triage.get("status") != "promising":
        raise RuntimeError("Expedited full seed blocked: fold 0 is not promising")
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if int(protocol["expedited_full_training"]["seed"]) != SEED:
        raise RuntimeError("Expedited seed differs from frozen protocol")
    set_status("full_training_seed_20260722", "running", started_at=now())
    run_script("train_canonical_m4.py", "--mode", "official", "--seed", str(SEED))
    run_script("evaluate_canonical_m4.py", "--mode", "official", "--seed", str(SEED))
    set_status("full_training_seed_20260722", "success", finished_at=now())
    gate = build_gate()
    if gate["quality_gate_passed"]:
        set_status(
            "official_validation_gate",
            "success",
            finished_at=now(),
            next_stage="reduced_post_gate_amendment",
            test_opened=False,
        )
    else:
        bundle = failure_bundle(gate)
        set_status(
            "official_validation_gate",
            "failed",
            finished_at=now(),
            test_opened=False,
            downstream="skipped",
            diagnostic_bundle=str(bundle.resolve()),
            diagnostic_bundle_sha256=sha256(bundle),
        )


if __name__ == "__main__":
    main()

