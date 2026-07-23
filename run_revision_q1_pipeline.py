from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from revision_q1.protocol import load_protocol, output_root


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL = load_protocol()
ROOT = output_root(PROTOCOL)
STATUS = ROOT / "pipeline_status.json"
PYTHON = PROJECT_DIR / ".venv/bin/python"
NORMALIZATION_STATS = ROOT / "normalization/layer_channel_statistics.pt"
STAGE1_NORMALIZATION_ROOT = ROOT / "normalization/stage1"
STAGE1_NORMALIZATION_STATS = (
    STAGE1_NORMALIZATION_ROOT / "normalization/layer_channel_statistics.pt"
)
VAL_MATRIX = ROOT / "raw/normalization_validation_matrix.csv"
TEST_MATRIX = ROOT / "raw/stage2_best_test_matrix.csv"
SENSITIVITY_MATRIX = ROOT / "raw/stage1_best_sensitivity_matrix.csv"
STAGE_NAMES = (
    "scene_difficulty", "normalization_fit", "normalization_validation_matrix",
    "normalization_selection", "normalization_fit_stage1",
    "stage2_primary_matrix", "stage2_statistics",
    "stage1_sensitivity_matrix", "stage1_statistics", "checkpoint_sequence_metrics",
    "checkpoint_comparison", "scene_difficulty_analysis", "spatial_stress",
    "cross_checkpoint_transfer", "figures", "bundle",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_status() -> dict:
    if STATUS.is_file():
        return json.loads(STATUS.read_text(encoding="utf-8"))
    return {"protocol_id": PROTOCOL["protocol_id"], "stages": {}, "updated_at": now()}


def write_status(payload: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now()
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def command_hash(command: list[str]) -> str:
    return hashlib.sha256("\0".join(command).encode()).hexdigest()


def run_stage(name: str, command: list[str], outputs: list[Path]) -> None:
    status = load_status()
    stage = status["stages"].get(name, {})
    if outputs and all(path.is_file() for path in outputs):
        status["stages"][name] = {
            **stage, "status": "success", "finished_at": stage.get("finished_at") or now(),
            "input_hash": command_hash(command), "output_files": [str(path) for path in outputs],
            "error": None,
        }
        write_status(status)
        print(f"[Q1] {name}: already complete", flush=True)
        return
    status["stages"][name] = {
        "status": "running", "started_at": now(), "finished_at": None,
        "input_hash": command_hash(command), "output_files": [], "error": None,
    }
    write_status(status)
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{name}.log"
    print(f"[Q1] {name}: {' '.join(command)}", flush=True)
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"})
    try:
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=PROJECT_DIR, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        missing = [str(path) for path in outputs if not path.is_file()]
        if missing:
            raise RuntimeError(f"Stage {name} did not create {missing}")
    except Exception as error:
        status = load_status()
        status["stages"][name].update({
            "status": "failed", "finished_at": now(),
            "error": f"{type(error).__name__}: {error}",
        })
        write_status(status)
        raise
    status = load_status()
    status["stages"][name].update({
        "status": "success", "finished_at": now(),
        "output_files": [str(path) for path in outputs], "error": None,
    })
    write_status(status)


def python_module(module: str, *arguments: str) -> list[str]:
    return [str(PYTHON), "-m", module, *arguments]


def selected_normalization() -> str:
    payload = json.loads(
        (ROOT / "config/normalization_selection.json").read_text(encoding="utf-8")
    )
    return str(payload["selected_normalization"])


def initialize() -> None:
    for name in ("normalization", "raw", "tables", "figures", "statistics", "tests", "logs", "config"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_DIR / "config/revision_q1_protocol.yaml", ROOT / "revision_protocol.yaml")
    shutil.copy2(PROJECT_DIR / "revision_q1/hypotheses.md", ROOT / "hypotheses.md")
    for test in (PROJECT_DIR / "tests").glob("test_revision_q1*.py"):
        shutil.copy2(test, ROOT / "tests" / test.name)
    status = load_status()
    status.setdefault("status", "pending")
    for name in STAGE_NAMES:
        status["stages"].setdefault(name, {
            "status": "pending", "started_at": None, "finished_at": None,
            "input_hash": None, "output_files": [], "error": None,
        })
    write_status(status)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable TNormFilter Q1 pipeline")
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    initialize()
    if args.initialize_only:
        print(f"Initialized Q1 status: {STATUS}")
        return
    main_bundle = PROJECT_DIR / "outputs/bundles/TNormFilter_final_practice.zip"
    if not main_bundle.is_file():
        raise RuntimeError("Q1 revision waits for successful completion of the main practice bundle")
    run_stage(
        "scene_difficulty",
        python_module("revision_q1.scene_difficulty"),
        [ROOT / "raw/scene_difficulty.csv", ROOT / "config/scene_difficulty_thresholds.json"],
    )
    run_stage(
        "normalization_fit",
        python_module("revision_q1.collect_normalization"),
        [NORMALIZATION_STATS, ROOT / "figures/01_feature_normalization_by_layer.png"],
    )
    run_stage(
        "normalization_validation_matrix",
        [
            str(PYTHON), "run_final_matrix.py", "--split", "val", "--workers", "1",
            "--model", str(PROJECT_DIR / PROTOCOL["primary_checkpoint"]),
            "--checkpoint-name", "stage2_best", "--revision-stats", str(NORMALIZATION_STATS),
            "--normalizations", ",".join(PROTOCOL["normalization"]["variants"]),
            "--fgsm-eps", "1,4", "--pgd-eps", "0.25,1", "--pgd-steps", "20",
            "--no-adaptive-pgd", "--defenses", "none,tnorm,median,bilateral",
            "--output", str(VAL_MATRIX),
        ],
        [VAL_MATRIX, VAL_MATRIX.with_suffix(".json")],
    )
    run_stage(
        "normalization_selection",
        python_module("revision_q1.select_normalization", "--input", str(VAL_MATRIX)),
        [ROOT / "config/normalization_selection.json", ROOT / "tables/02_normalization_ablation.csv"],
    )
    mode = selected_normalization()
    run_stage(
        "normalization_fit_stage1",
        python_module(
            "revision_q1.collect_normalization", "--model",
            str(PROJECT_DIR / PROTOCOL["sensitivity_checkpoint"]),
            "--output", str(STAGE1_NORMALIZATION_ROOT),
        ),
        [STAGE1_NORMALIZATION_STATS],
    )
    run_stage(
        "stage2_primary_matrix",
        [
            str(PYTHON), "run_final_matrix.py", "--split", "test", "--workers", "1",
            "--model", str(PROJECT_DIR / PROTOCOL["primary_checkpoint"]),
            "--checkpoint-name", "stage2_best", "--revision-stats", str(NORMALIZATION_STATS),
            "--normalizations", mode, "--fgsm-eps", "0.5,1,2,4,8",
            "--pgd-eps", "0.1,0.25,0.5,1,2", "--pgd-steps", "20,40",
            "--adaptive-pgd-eps", "0.25,0.5,1", "--adaptive-pgd-steps", "20,40",
            "--seeds", "42,123,999", "--defenses", "none,tnorm,bilateral,gaussian,median,jpeg",
            "--output", str(TEST_MATRIX),
        ],
        [TEST_MATRIX, TEST_MATRIX.with_suffix(".json")],
    )
    run_stage(
        "stage2_statistics",
        python_module(
            "revision_q1.analyze", "--input", str(TEST_MATRIX), "--split", "test",
            "--normalization", mode,
        ),
        [ROOT / "tables/05_damage_models_D0_D4.csv", ROOT / "tables/06_recovery_models_R0_R4.csv"],
    )
    run_stage(
        "stage1_sensitivity_matrix",
        [
            str(PYTHON), "run_final_matrix.py", "--split", "test", "--workers", "1",
            "--model", str(PROJECT_DIR / PROTOCOL["sensitivity_checkpoint"]),
            "--checkpoint-name", "stage1_best", "--revision-stats", str(STAGE1_NORMALIZATION_STATS),
            "--normalizations", mode, "--fgsm-eps", "1,4",
            "--pgd-eps", "0.25,1", "--pgd-steps", "20",
            "--adaptive-pgd-eps", "1", "--adaptive-pgd-steps", "20,40",
            "--seeds", "42,123,999", "--defenses", "none,tnorm,median,bilateral",
            "--output", str(SENSITIVITY_MATRIX),
        ],
        [SENSITIVITY_MATRIX, SENSITIVITY_MATRIX.with_suffix(".json")],
    )
    sensitivity_root = ROOT / "statistics/stage1"
    run_stage(
        "stage1_statistics",
        python_module(
            "revision_q1.analyze", "--input", str(SENSITIVITY_MATRIX), "--split", "test",
            "--normalization", mode, "--output", str(sensitivity_root),
        ),
        [sensitivity_root / "tables/05_damage_models_D0_D4.csv"],
    )
    run_stage(
        "checkpoint_sequence_metrics",
        python_module("revision_q1.evaluate_checkpoints_by_sequence"),
        [ROOT / "raw/checkpoint_sequence_metrics.csv"],
    )
    run_stage(
        "checkpoint_comparison",
        python_module(
            "revision_q1.compare_checkpoints", "--stage2", str(TEST_MATRIX),
            "--stage1", str(SENSITIVITY_MATRIX),
        ),
        [ROOT / "tables/07_checkpoint_paired_comparison.csv", ROOT / "tables/08_checkpoint_sensitivity.csv"],
    )
    run_stage(
        "scene_difficulty_analysis",
        python_module("revision_q1.analyze_scene_difficulty", "--input", str(TEST_MATRIX)),
        [ROOT / "tables/09_scene_difficulty_analysis.csv"],
    )
    run_stage(
        "spatial_stress",
        python_module("revision_q1.run_spatial_stress"),
        [ROOT / "tables/10_spatial_stress_test.csv"],
    )
    run_stage(
        "cross_checkpoint_transfer",
        python_module("revision_q1.run_transfer"),
        [ROOT / "tables/11_transfer_attack.csv"],
    )
    run_stage(
        "figures",
        python_module("revision_q1.plot_revision", "--matrix", str(TEST_MATRIX)),
        [ROOT / "figures/11_adaptive_vs_nonadaptive.png"],
    )
    run_stage(
        "bundle",
        python_module("revision_q1.build_bundle"),
        [
            PROJECT_DIR / "outputs/bundles/TNormFilter_revision_q1.zip",
            PROJECT_DIR / "outputs/bundles/TNormFilter_revision_q1.zip.sha256",
        ],
    )
    status = load_status()
    status["status"] = "success"
    write_status(status)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        status = load_status()
        status["status"] = "failed"
        write_status(status)
        raise
