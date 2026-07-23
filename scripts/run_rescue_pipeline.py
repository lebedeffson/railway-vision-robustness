from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from rescue_common import (
    PROJECT_DIR,
    OUTPUT_ROOT,
    assert_test_sealed,
    atomic_json,
    ensure_branch,
    load_protocol,
    now,
)


STATUS = OUTPUT_ROOT / "pipeline_status.json"
PYTHON = PROJECT_DIR / ".venv/bin/python"
POLICY_PATH = PROJECT_DIR / "configs/rescue/candidate_execution_policy.yaml"


def read_status() -> dict[str, Any]:
    if STATUS.is_file():
        return json.loads(STATUS.read_text(encoding="utf-8"))
    return {"protocol_id": load_protocol()["protocol_id"], "status": "created", "stages": {}}


def write_status(payload: dict[str, Any]) -> None:
    payload["updated_at"] = now()
    atomic_json(STATUS, payload)


def stage(name: str, command: list[str], outputs: list[Path]) -> None:
    payload = read_status()
    record = payload.setdefault("stages", {}).get(name, {})
    if record.get("status") == "success" and all(path.is_file() for path in outputs):
        print(f"[rescue] {name}: resume skip", flush=True)
        return
    if not record and all(path.is_file() for path in outputs):
        payload["stages"][name] = {
            "status": "success", "started_at": None, "finished_at": now(),
            "command": command, "outputs": [str(path) for path in outputs],
            "error": None, "recovered_from_verified_outputs": True,
        }
        write_status(payload)
        print(f"[rescue] {name}: recovered from existing outputs", flush=True)
        return
    payload["status"] = "running"
    payload["current_stage"] = name
    payload["stages"][name] = {
        "status": "running", "started_at": now(), "command": command,
        "outputs": [str(path) for path in outputs], "error": None,
    }
    write_status(payload)
    try:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
        missing = [str(path) for path in outputs if not path.is_file()]
        if missing:
            raise RuntimeError(f"Stage {name} missing outputs: {missing}")
    except subprocess.CalledProcessError as error:
        payload = read_status()
        if error.returncode == 75:
            payload["status"] = "blocked_infrastructure"
            payload["current_stage"] = None
            payload["stages"][name].update({
                "status": "blocked", "finished_at": now(),
                "error": "cuda_device_not_visible_to_execution_environment",
            })
            write_status(payload)
            raise SystemExit(
                "Rescue pipeline is resumable but blocked: CUDA is not visible "
                "to this execution environment"
            ) from error
        payload["status"] = "failed"
        payload["current_stage"] = None
        payload["stages"][name].update({
            "status": "failed", "finished_at": now(),
            "error": f"{type(error).__name__}: {error}",
        })
        write_status(payload)
        raise
    except Exception as error:
        payload = read_status()
        payload["status"] = "failed"
        payload["current_stage"] = None
        payload["stages"][name].update({
            "status": "failed", "finished_at": now(),
            "error": f"{type(error).__name__}: {error}",
        })
        write_status(payload)
        raise
    payload = read_status()
    payload["stages"][name].update({
        "status": "success", "finished_at": now(), "error": None,
    })
    payload["current_stage"] = None
    write_status(payload)


def candidate_stage(candidate: str, seed: int, base: str | None = None) -> None:
    output = OUTPUT_ROOT / "runs" / candidate / f"seed_{seed}"
    command = [
        str(PYTHON), "scripts/train_rescue_candidate.py",
        "--candidate", candidate, "--seed", str(seed),
    ]
    if base:
        command.extend(["--base-candidate", base])
    stage(
        f"candidate_{candidate}_seed_{seed}", command,
        [output / "COMPLETED.json", output / "evaluation/candidate_result.json"],
    )


def result(candidate: str, seed: int) -> dict[str, Any]:
    path = OUTPUT_ROOT / "runs" / candidate / f"seed_{seed}/evaluation/candidate_result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def imbalance_is_material(policy: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    classes = pd.read_csv(OUTPUT_ROOT / "audit/class_distribution.csv")
    train_classes = classes[classes["split"] == "train"].groupby("class_id")["count"].sum()
    class_ratio = float(train_classes.max() / max(train_classes.min(), 1))
    scenes = pd.read_csv(OUTPUT_ROOT / "audit/scene_distribution.csv")
    train_scenes = scenes[scenes["split"] == "train"]["frames"]
    scene_ratio = float(train_scenes.max() / max(train_scenes.min(), 1))
    material = (
        class_ratio >= float(policy["r4_material_class_imbalance_ratio_min"])
        or scene_ratio >= float(policy["r4_material_scene_frame_ratio_min"])
    )
    return material, {"class_count_ratio": class_ratio, "scene_frame_ratio": scene_ratio}


def choose_initial_base(candidates: list[str], seed: int) -> str:
    rows = [result(candidate, seed) for candidate in candidates]
    rows.sort(key=lambda row: (
        -float(row["scene_macro_map50"]),
        -float(row["scene_macro_safety_recall"]),
        float(row["validation_safety_fn_per_frame"]),
        float(row["scene_map50_std"]),
    ))
    return str(rows[0]["candidate"])


def main() -> None:
    ensure_branch()
    assert_test_sealed()
    protocol = load_protocol()
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    initial_seed = int(protocol["selection"]["initial_seed"])
    stage(
        "data_audit",
        [str(PYTHON), "scripts/audit_dataset.py"],
        [OUTPUT_ROOT / "audit/COMPLETED.json", OUTPUT_ROOT / "audit/split_audit.json"],
    )
    stage(
        "visual_audit",
        [str(PYTHON), "scripts/record_visual_audit.py"],
        [OUTPUT_ROOT / "audit/visual_audit.json"],
    )
    stage(
        "rescue_dataset",
        [str(PYTHON), "scripts/prepare_rescue_dataset.py"],
        [
            PROJECT_DIR / "data/yolo_osdar23_rescue_v1/data.yaml",
            PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv",
            OUTPUT_ROOT / "audit/data_repair.json",
        ],
    )
    stage(
        "evaluator_audit",
        [str(PYTHON), "scripts/audit_evaluator.py"],
        [OUTPUT_ROOT / "evaluator/COMPLETED.json", OUTPUT_ROOT / "evaluator/evaluator_audit.json"],
    )
    stage(
        "micro_overfit",
        [str(PYTHON), "scripts/run_micro_overfit.py"],
        [OUTPUT_ROOT / "micro_overfit/COMPLETED.json", OUTPUT_ROOT / "micro_overfit/result.json"],
    )
    micro = json.loads((OUTPUT_ROOT / "micro_overfit/result.json").read_text(encoding="utf-8"))
    if micro.get("status") != "PASS":
        raise RuntimeError("Rescue candidate matrix is blocked by micro-overfit failure")
    stage(
        "current_checkpoint_diagnosis",
        [str(PYTHON), "scripts/analyze_current_checkpoint.py"],
        [
            OUTPUT_ROOT / "current_checkpoint/COMPLETED.json",
            OUTPUT_ROOT / "current_checkpoint/summary.json",
        ],
    )
    stage(
        "candidate_R0_reference",
        [str(PYTHON), "scripts/evaluate_pretrained_reference.py"],
        [OUTPUT_ROOT / f"runs/R0/seed_{initial_seed}/COMPLETED.json"],
    )
    candidate_stage("R1", initial_seed)
    candidate_stage("R2", initial_seed)
    initial_candidates = ["R1", "R2"]
    r1 = result("R1", initial_seed)
    r2 = result("R2", initial_seed)
    improvement = (
        float(r2["validation_small_object_safety_recall"])
        - float(r1["validation_small_object_safety_recall"])
    )
    r3_decision = {
        "R1_small_recall": r1["validation_small_object_safety_recall"],
        "R2_small_recall": r2["validation_small_object_safety_recall"],
        "improvement": improvement,
        "required": policy["r3_small_object_recall_improvement_min"],
        "run_R3": improvement >= float(policy["r3_small_object_recall_improvement_min"]),
    }
    atomic_json(OUTPUT_ROOT / "comparison/R3_decision.json", r3_decision)
    if r3_decision["run_R3"]:
        candidate_stage("R3", initial_seed)
        initial_candidates.append("R3")
    else:
        payload = read_status()
        payload.setdefault("stages", {})["candidate_R3"] = {
            "status": "skipped", "finished_at": now(),
            "error": "R2_small_object_recall_improvement_below_frozen_threshold",
            "outputs": [str(OUTPUT_ROOT / "comparison/R3_decision.json")],
        }
        write_status(payload)
    material, imbalance = imbalance_is_material(policy)
    imbalance["run_R4"] = material
    atomic_json(OUTPUT_ROOT / "comparison/R4_decision.json", imbalance)
    if material:
        base = choose_initial_base(initial_candidates, initial_seed)
        candidate_stage("R4", initial_seed, base)
        initial_candidates.append("R4")
    else:
        payload = read_status()
        payload.setdefault("stages", {})["candidate_R4"] = {
            "status": "skipped", "finished_at": now(),
            "error": "class_and_scene_imbalance_below_frozen_thresholds",
            "outputs": [str(OUTPUT_ROOT / "comparison/R4_decision.json")],
        }
        write_status(payload)

    initial_rows = [result(candidate, initial_seed) for candidate in initial_candidates]
    initial_rows.sort(key=lambda row: (
        -float(row["scene_macro_map50"]),
        -float(row["scene_macro_safety_recall"]),
        float(row["validation_safety_fn_per_frame"]),
        float(row["scene_map50_std"]),
    ))
    finalists = [
        str(row["candidate"])
        for row in initial_rows[:int(protocol["selection"]["maximum_finalists"])]
    ]
    atomic_json(OUTPUT_ROOT / "comparison/finalists.json", {
        "status": "FROZEN", "finalists": finalists,
        "selection_uses_test": False, "initial_seed": initial_seed,
    })
    for candidate in finalists:
        base = None
        if candidate == "R4":
            weights = pd.read_csv(
                OUTPUT_ROOT / "runs/R4" / f"seed_{initial_seed}/sampler_weights.csv"
            )
            base = choose_initial_base([name for name in initial_candidates if name != "R4"], initial_seed)
        for seed in protocol["selection"]["finalist_seeds"]:
            if int(seed) == initial_seed:
                continue
            candidate_stage(candidate, int(seed), base)
    stage(
        "finalize_rescue",
        [str(PYTHON), "scripts/finalize_rescue.py"],
        [
            OUTPUT_ROOT / "final/quality_gate.json",
            OUTPUT_ROOT / "final/rescue_report.md",
            OUTPUT_ROOT / "final/run_summary.json",
        ],
    )
    gate = json.loads((OUTPUT_ROOT / "final/quality_gate.json").read_text(encoding="utf-8"))
    payload = read_status()
    payload["status"] = (
        "rescue_quality_gate_passed" if gate["quality_gate_passed"]
        else "stopped_rescue_quality_gate_failed"
    )
    payload["quality_gate_passed"] = gate["quality_gate_passed"]
    payload["test_opened"] = False
    payload["current_stage"] = None
    write_status(payload)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
