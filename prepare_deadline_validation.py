from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/final_practice/deadline"
BASELINE = PROJECT_DIR / "outputs/final_practice/deadline_baseline"
VAL = PROJECT_DIR / "outputs/final_practice/unified_diagnostics_val_raw.csv"
AUDIT = PROJECT_DIR / "outputs/final_practice/audit"
PYTHON = PROJECT_DIR / ".venv/bin/python"
STATUS = ROOT / "pipeline_status.json"
STAGES = (
    "baseline_snapshot", "matrix_integrity", "nms_timeout_audit",
    "legacy_recovery_recalculation", "matrix_integrity_after_nms",
    "normalization_fit", "pilot_selection",
    "pilot_matrix", "pilot_gate", "canonical_budget_selection",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status_payload() -> dict:
    if STATUS.is_file():
        return json.loads(STATUS.read_text(encoding="utf-8"))
    return {
        "mode": "deadline_minimal_v1", "status": "running",
        "full_q1_service": "masked", "stages": {
            name: {"status": "pending", "error": None} for name in STAGES
        },
    }


def write_status(payload: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now()
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def run_stage(name: str, command: list[str], outputs: list[Path]) -> None:
    payload = status_payload()
    if outputs and all(path.is_file() for path in outputs):
        payload["stages"][name] = {
            "status": "success", "finished_at": now(),
            "outputs": [str(path) for path in outputs], "error": None,
        }
        write_status(payload)
        print(f"[deadline] {name}: already complete", flush=True)
        return
    payload["stages"][name] = {
        "status": "running", "started_at": now(), "outputs": [], "error": None,
    }
    write_status(payload)
    try:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
        missing = [str(path) for path in outputs if not path.is_file()]
        if missing:
            raise RuntimeError(f"{name}: missing outputs {missing}")
    except Exception as error:
        payload = status_payload()
        payload["stages"][name].update({
            "status": "failed", "finished_at": now(),
            "error": f"{type(error).__name__}: {error}",
        })
        payload["status"] = "failed"
        write_status(payload)
        raise
    payload = status_payload()
    payload["stages"][name].update({
        "status": "success", "finished_at": now(),
        "outputs": [str(path) for path in outputs], "error": None,
    })
    write_status(payload)


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def snapshot_baseline() -> None:
    raw = BASELINE / "raw/unified_diagnostics_val_raw.csv"
    hardlink_or_copy(VAL, raw)
    hardlink_or_copy(VAL.with_suffix(".json"), BASELINE / "configs/unified_diagnostics_val_raw.json")
    checkpoint = PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/weights/best.pt"
    hardlink_or_copy(checkpoint, BASELINE / "checkpoint/stage2_best.pt")
    log_path = BASELINE / "logs/tnorm-wait-train.journal.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.is_file():
        output = subprocess.check_output([
            "journalctl", "--user", "-u", "tnorm-wait-train.service",
            "--since", "2026-07-22 00:20:00", "--all", "--no-pager",
        ])
        log_path.write_bytes(output)
    files = sorted(path for path in BASELINE.rglob("*") if path.is_file())
    import pandas as pd
    rows = pd.read_csv(VAL, low_memory=False)
    completed = rows[[
        "sequence_id", "image_path", "attack", "adaptive", "epsilon", "steps",
        "restart", "seed", "defense",
    ]].drop_duplicates()
    completed.to_csv(BASELINE / "completed_conditions.csv", index=False)
    files = sorted(path for path in BASELINE.rglob("*") if path.is_file())
    manifest = {str(path.relative_to(BASELINE)): sha256(path) for path in files}
    (BASELINE / "manifest.json").write_text(
        json.dumps({
            "created_at": now(), "status": "frozen_deadline_baseline",
            "files": manifest,
        }, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    for name in ("config", "pilot/raw", "tables", "figures", "statistics", "logs"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_DIR / "config/deadline_protocol.yaml", ROOT / "config/deadline_protocol.yaml")
    payload = status_payload()
    payload["status"] = "running"
    payload["full_q1_service"] = "masked"
    write_status(payload)
    if not (BASELINE / "manifest.json").is_file():
        snapshot_baseline()
    run_stage("baseline_snapshot", ["true"], [BASELINE / "manifest.json"])
    run_stage(
        "matrix_integrity", [str(PYTHON), "deadline_matrix_audit.py"],
        [AUDIT / "matrix_integrity.json", AUDIT / "missing_conditions.csv"],
    )
    integrity = json.loads((AUDIT / "matrix_integrity.json").read_text(encoding="utf-8"))
    if integrity.get("status") != "PASS":
        raise RuntimeError("Existing matrix integrity artifact is not PASS")
    run_stage(
        "nms_timeout_audit", [str(PYTHON), "deadline_nms_audit.py"],
        [AUDIT / "nms_timeout_audit.json", AUDIT / "nms_timeout_cases.csv"],
    )
    nms_status = json.loads((AUDIT / "nms_timeout_audit.json").read_text(encoding="utf-8"))
    if nms_status.get("status") not in {"PASS", "RERUN_FAILED_EXCLUDED"}:
        raise RuntimeError(f"Unexpected NMS audit status: {nms_status.get('status')}")
    run_stage(
        "legacy_recovery_recalculation", [
            str(PYTHON), "recalculate_legacy_recovery.py",
        ], [AUDIT / "legacy_recovery_recalculation.json"],
    )
    post_nms = AUDIT / "post_nms"
    run_stage(
        "matrix_integrity_after_nms", [
            str(PYTHON), "deadline_matrix_audit.py", "--output", str(post_nms),
        ], [post_nms / "matrix_integrity.json", post_nms / "missing_conditions.csv"],
    )
    post_integrity = json.loads(
        (post_nms / "matrix_integrity.json").read_text(encoding="utf-8")
    )
    if post_integrity.get("status") != "PASS":
        raise RuntimeError("Post-NMS matrix integrity artifact is not PASS")
    run_stage(
        "normalization_fit", [
            str(PYTHON), "-m", "revision_q1.collect_normalization",
            "--output", str(ROOT), "--workers", "0",
        ],
        [ROOT / "normalization/layer_channel_statistics.pt", ROOT / "tables/02_normalization_ablation.csv"],
    )
    run_stage(
        "pilot_selection", [str(PYTHON), "deadline_select_pilot.py"],
        [ROOT / "config/pilot_manifest.csv", ROOT / "config/pilot_data.yaml"],
    )
    pilot = ROOT / "pilot/raw/pilot_matrix.csv"
    protocol = yaml.safe_load(
        (PROJECT_DIR / "config/deadline_protocol.yaml").read_text(encoding="utf-8")
    )
    pilot_attacks = protocol["pilot"]["attacks"]
    run_stage(
        "pilot_matrix", [
            str(PYTHON), "run_final_matrix.py",
            "--data", str(ROOT / "config/pilot_data.yaml"),
            "--split", "val", "--workers", "0", "--checkpoint-name", "stage2_best",
            "--revision-stats", str(ROOT / "normalization/layer_channel_statistics.pt"),
            "--normalizations", "N1_quantile,N2_robust_sigmoid,N3_zscore_sigmoid",
            "--fgsm-eps", ",".join(map(str, pilot_attacks["fgsm_epsilon_px"])),
            "--pgd-eps", ",".join(map(str, pilot_attacks["pgd_epsilon_px"])),
            "--pgd-steps", ",".join(map(str, pilot_attacks["pgd_steps"])),
            "--adaptive-pgd-eps", ",".join(
                map(str, pilot_attacks["adaptive_pgd_epsilon_px"])
            ),
            "--adaptive-pgd-steps", ",".join(
                map(str, pilot_attacks["adaptive_pgd_steps"])
            ),
            "--seeds", "42,123,999", "--defenses", "none,tnorm,bilateral",
            "--nms-max-time-img", "10", "--output", str(pilot),
        ],
        [pilot, pilot.with_suffix(".json")],
    )
    run_stage(
        "pilot_gate", [str(PYTHON), "deadline_pilot_gate.py"],
        [ROOT / "pilot/pilot_gate.json"],
    )
    pilot_gate = json.loads((ROOT / "pilot/pilot_gate.json").read_text(encoding="utf-8"))
    if pilot_gate.get("status") != "PASS":
        raise RuntimeError("Existing pilot gate artifact is not PASS")
    budget_selection = ROOT / "config/canonical_budget_selection.json"
    run_stage(
        "canonical_budget_selection", [
            str(PYTHON), "deadline_select_budgets.py", "--input", str(pilot),
            "--output", str(budget_selection),
        ], [budget_selection, budget_selection.with_suffix(".csv")],
    )
    cache_plan = {
        "stage2_test": "single_pass_canonical_matrix_with_N1_metrics",
        "normalization_variants": "same_pilot_attacks_and_feature_tensors_no_repeated_inference",
        "validation_baseline": (
            "legacy matrix has no adversarial tensors or raw activations; six warning images "
            "are the only targeted reruns"
        ),
        "resume_unit": "complete_image_condition_checkpoint",
    }
    (ROOT / "config/cache_reuse_plan.json").write_text(
        json.dumps(cache_plan, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "status": "PASS", "created_at": now(),
        "pilot_gate": str(ROOT / "pilot/pilot_gate.json"),
        "full_q1_service": "masked",
        "cache_reuse": cache_plan,
    }
    (ROOT / "deadline_validation_preparation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "run_summary.json").write_text(json.dumps({
        "status": "VALIDATION_READY", "created_at": now(),
        "full_q1_service": "masked",
        "pilot_use": "code_smoke_only_not_article_results",
        "validation_sequences": 3,
        "deferred_extended_analysis": yaml.safe_load(
            (PROJECT_DIR / "config/deadline_protocol.yaml").read_text(encoding="utf-8")
        )["deferred_extended_analysis"],
    }, indent=2) + "\n", encoding="utf-8")
    payload = status_payload()
    payload["status"] = "validation_ready"
    write_status(payload)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
