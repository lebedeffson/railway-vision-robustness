from __future__ import annotations

import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    PROTOCOL_LOCK,
    QUALITY_GATE,
    assert_role_allowed,
    atomic_json,
    load_protocol,
    now,
    sha256,
)
from train_canonical_m4 import training_root


PYTHON = PROJECT_DIR / ".venv/bin/python"
PIPELINE_STATUS = OUTPUT_ROOT / "final/pre_gate_pipeline_status.json"


def set_status(stage: str, status: str, **extra: Any) -> None:
    current = (
        json.loads(PIPELINE_STATUS.read_text(encoding="utf-8"))
        if PIPELINE_STATUS.is_file()
        else {"protocol_id": load_protocol()["protocol_id"], "stages": {}}
    )
    current["updated_at"] = now()
    current["current_stage"] = stage
    current["stages"][stage] = {
        **current["stages"].get(stage, {}),
        "status": status,
        **extra,
    }
    atomic_json(PIPELINE_STATUS, current)


def run_script(script: str, *arguments: str) -> None:
    subprocess.run(
        [str(PYTHON), "-u", f"scripts/{script}", *arguments],
        cwd=PROJECT_DIR,
        check=True,
    )


def wait_for_tiling() -> None:
    stage = "tiling_audit"
    set_status(stage, "running", started_at=now())
    audit = OUTPUT_ROOT / "tiling_audit/tiling_audit.json"
    while True:
        if audit.is_file():
            payload = json.loads(audit.read_text(encoding="utf-8"))
            if payload.get("status") == "PASS" and not payload.get("test_used"):
                set_status(
                    stage,
                    "success",
                    finished_at=now(),
                    output=str(audit.resolve()),
                    output_sha256=sha256(audit),
                )
                return
            raise RuntimeError("Canonical M4 tiling audit failed")
        time.sleep(30)


def scene_cv(protocol: dict[str, Any]) -> None:
    seed = int(protocol["scene_cv"]["seed"])
    rows = []
    for fold in range(int(protocol["scene_cv"]["folds"])):
        stage = f"scene_cv_fold_{fold}"
        set_status(stage, "running", started_at=now())
        run_script(
            "train_canonical_m4.py",
            "--mode", "cv", "--seed", str(seed), "--fold", str(fold),
        )
        run_script(
            "evaluate_canonical_m4.py",
            "--mode", "cv", "--seed", str(seed), "--fold", str(fold),
        )
        evaluation = (
            training_root("cv", seed, fold) / "evaluation/evaluation_result.json"
        )
        payload = json.loads(evaluation.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"Scene-CV fold failed: {fold}")
        rows.append(payload)
        set_status(
            stage,
            "success",
            finished_at=now(),
            evaluation=str(evaluation.resolve()),
            checkpoint_sha256=payload["checkpoint_sha256"],
        )
    destination = OUTPUT_ROOT / "scene_cv/aggregate_metrics.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    atomic_json(OUTPUT_ROOT / "scene_cv/scene_cv_summary.json", {
        "status": "PASS",
        "folds": len(rows),
        "group_field": "grouped_scene_id",
        "test_used": False,
        "mean_scene_macro_map50":
            float(pd.DataFrame(rows)["scene_macro_map50"].mean()),
        "mean_scene_macro_safety_recall":
            float(pd.DataFrame(rows)["scene_macro_safety_recall"].mean()),
        "aggregate_sha256": sha256(destination),
    })


def full_training(protocol: dict[str, Any]) -> None:
    for seed_value in protocol["full_training"]["seeds"]:
        seed = int(seed_value)
        stage = f"full_training_seed_{seed}"
        set_status(stage, "running", started_at=now())
        run_script(
            "train_canonical_m4.py",
            "--mode", "official", "--seed", str(seed),
        )
        run_script(
            "evaluate_canonical_m4.py",
            "--mode", "official", "--seed", str(seed),
        )
        evaluation = OUTPUT_ROOT / f"validation/seed_{seed}/evaluation_result.json"
        payload = json.loads(evaluation.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"Official validation failed for seed {seed}")
        set_status(
            stage,
            "success",
            finished_at=now(),
            evaluation=str(evaluation.resolve()),
            checkpoint_sha256=payload["checkpoint_sha256"],
        )


def select_checkpoint_and_gate(protocol: dict[str, Any]) -> dict[str, Any]:
    selection_root = OUTPUT_ROOT / "selection"
    selection_root.mkdir(parents=True, exist_ok=True)
    rows = []
    scene_rows = []
    for seed_value in protocol["full_training"]["seeds"]:
        seed = int(seed_value)
        evaluation = OUTPUT_ROOT / f"validation/seed_{seed}"
        payload = json.loads(
            (evaluation / "evaluation_result.json").read_text(encoding="utf-8")
        )
        rows.append(payload)
        frame = pd.read_csv(evaluation / "per_scene_metrics.csv")
        frame.insert(0, "seed", seed)
        scene_rows.append(frame)
    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        [
            "scene_macro_map50",
            "scene_macro_safety_recall",
            "safety_fn_per_frame",
            "scene_safety_recall_std",
            "mean_tiling_inference_latency_ms",
            "seed",
        ],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)
    comparison["selected"] = False
    comparison.loc[0, "selected"] = True
    comparison_path = selection_root / "seed_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    per_scene_path = selection_root / "per_scene_seed_comparison.csv"
    pd.concat(scene_rows, ignore_index=True).to_csv(per_scene_path, index=False)
    selected = comparison.iloc[0].to_dict()
    checkpoint = Path(str(selected["checkpoint"]))
    threshold_path = (
        OUTPUT_ROOT
        / f"validation/seed_{int(selected['seed'])}/threshold_selection.json"
    )
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    decision = {
        "status": "FROZEN",
        "selected_at": now(),
        "protocol_id": protocol["protocol_id"],
        "rule": protocol["checkpoint_selection"]["lexicographic_rule"],
        "selected_seed": int(selected["seed"]),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "standard_threshold": float(threshold["standard"]["threshold"]),
        "safety_threshold": float(threshold["safety"]["threshold"]),
        "threshold_source": str(threshold_path.resolve()),
        "selection_split": "validation",
        "test_used": False,
        "all_seed_results": str(comparison_path.resolve()),
    }
    atomic_json(selection_root / "checkpoint_selection.json", decision)
    required = {
        "checkpoint_hash_verified":
            selected["checkpoint_sha256"] == sha256(checkpoint),
        "evaluator_consistency_passed":
            bool(selected["evaluator_consistency_passed"]),
        "all_validation_frames_present":
            bool(selected["all_expected_frames_present"]),
        "no_duplicate_predictions": bool(selected["no_duplicate_predictions"]),
        "no_missing_scene": bool(selected["no_missing_scene"]),
        "no_nan_or_inf": bool(selected["no_nan_or_inf"]),
    }
    map_pass = float(selected["mAP50"]) >= float(
        protocol["quality_gate"]["validation_map50_min"]
    )
    recall_pass = float(selected["safety_recall"]) >= float(
        protocol["quality_gate"]["validation_safety_recall_min"]
    )
    passed = map_pass and recall_pass and all(required.values())
    gate = {
        "protocol_id": protocol["protocol_id"],
        "created_at": now(),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "split_manifest_sha256": protocol["dataset"]["split_manifest_sha256"],
        "validation_map50": float(selected["mAP50"]),
        "validation_safety_recall": float(selected["safety_recall"]),
        "map50_requirement": float(protocol["quality_gate"]["validation_map50_min"]),
        "recall_requirement":
            float(protocol["quality_gate"]["validation_safety_recall_min"]),
        "map50_passed": map_pass,
        "recall_passed": recall_pass,
        "required_checks": required,
        "quality_gate_passed": passed,
        "checkpoint_frozen": True,
        "thresholds_frozen": True,
        "standard_threshold": decision["standard_threshold"],
        "safety_threshold": decision["safety_threshold"],
        "test_opened": False,
    }
    atomic_json(QUALITY_GATE, gate)
    return gate


def failure_bundle(gate: dict[str, Any]) -> Path:
    destination = Path(load_protocol()["failure_policy"]["bundle"])
    if not destination.is_absolute():
        destination = PROJECT_DIR / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    roots = [
        PROJECT_DIR / "configs/canonical_v2_m4_full_protocol.yaml",
        PROJECT_DIR / "configs/schemas/canonical_v2_m4_full_protocol.schema.json",
        PROTOCOL_LOCK,
        OUTPUT_ROOT / "tiling_audit/tiling_audit.json",
        OUTPUT_ROOT / "scene_cv",
        OUTPUT_ROOT / "full_training",
        OUTPUT_ROOT / "validation",
        OUTPUT_ROOT / "selection",
        PIPELINE_STATUS,
    ]
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in roots:
            if root.is_file():
                archive.write(root, root.relative_to(PROJECT_DIR))
            elif root.is_dir():
                for path in sorted(root.rglob("*")):
                    if path.is_file() and path.suffix not in {".pt", ".jpg", ".png"}:
                        archive.write(path, path.relative_to(PROJECT_DIR))
        archive.writestr("GATE_FAILURE.json", json.dumps(gate, indent=2))
    temporary.replace(destination)
    (destination.with_suffix(destination.suffix + ".sha256")).write_text(
        f"{sha256(destination)}  {destination.name}\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    assert_role_allowed("scene_cv")
    protocol = load_protocol()
    try:
        wait_for_tiling()
        scene_cv(protocol)
        full_training(protocol)
        set_status("checkpoint_selection", "running", started_at=now())
        gate = select_checkpoint_and_gate(protocol)
        set_status(
            "checkpoint_selection",
            "success",
            finished_at=now(),
            quality_gate_passed=gate["quality_gate_passed"],
        )
        if not gate["quality_gate_passed"]:
            bundle = failure_bundle(gate)
            set_status(
                "validation_gate",
                "failed",
                finished_at=now(),
                test_opened=False,
                failure_bundle=str(bundle.resolve()),
                failure_bundle_sha256=sha256(bundle),
            )
            return
        set_status(
            "validation_gate",
            "success",
            finished_at=now(),
            test_opened=False,
            post_gate_ready=True,
        )
        post_gate = PROJECT_DIR / "scripts/run_canonical_m4_post_gate.py"
        if not post_gate.is_file():
            raise RuntimeError("Quality gate passed but post-gate runner is absent")
        run_script("run_canonical_m4_post_gate.py")
    except Exception as error:
        set_status(
            "pipeline",
            "failed",
            finished_at=now(),
            error=f"{type(error).__name__}: {error}",
            test_opened=False,
        )
        raise


if __name__ == "__main__":
    main()
