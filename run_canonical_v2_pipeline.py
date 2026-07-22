from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/canonical_v2"
STATUS = ROOT / "pipeline_status.json"
PYTHON = PROJECT_DIR / ".venv/bin/python"
LEGACY_VAL = PROJECT_DIR / "outputs/final_practice/unified_diagnostics_val_raw.csv"
PROTOCOL = yaml.safe_load(
    (PROJECT_DIR / "config/canonical_v2_protocol.yaml").read_text(encoding="utf-8")
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_status() -> dict[str, Any]:
    if STATUS.is_file(): return json.loads(STATUS.read_text(encoding="utf-8"))
    return {"protocol": PROTOCOL["protocol_id"], "status": "waiting", "stages": {}}


def write_status(payload: dict[str, Any]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True); payload["updated_at"] = now()
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def stage(name: str, command: list[str], outputs: list[Path]) -> None:
    payload = read_status(); record = payload.setdefault("stages", {}).get(name, {})
    if record.get("status") == "success" and all(path.is_file() for path in outputs):
        print(f"[canonical-v2] {name}: resume skip", flush=True); return
    payload["status"] = "running"; payload["stages"][name] = {
        "status": "running", "started_at": now(), "command": command,
        "outputs": [str(path) for path in outputs], "error": None,
    }; write_status(payload)
    try:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
        missing = [str(path) for path in outputs if not path.is_file()]
        if missing: raise RuntimeError(f"Missing outputs: {missing}")
    except Exception as error:
        payload = read_status(); payload["status"] = "failed"
        payload["stages"][name].update({
            "status": "failed", "finished_at": now(),
            "error": f"{type(error).__name__}: {error}",
        }); write_status(payload); raise
    payload = read_status(); payload["stages"][name].update({
        "status": "success", "finished_at": now(), "error": None,
    }); write_status(payload)


def wait_for_legacy() -> None:
    blocker = ROOT / "LEGACY_TEST_BLOCKED"
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text(
        "Current 3-scene legacy test is forbidden; canonical v2 uses its own split and output.\n",
        encoding="utf-8",
    )
    payload = read_status(); payload["status"] = "waiting_for_legacy_validation_198"
    payload["legacy_test_blocked"] = True; write_status(payload)
    while not LEGACY_VAL.is_file():
        time.sleep(30)
    # The legacy child has atomically renamed its final CSV, so it is safe to stop
    # the paused obsolete controller before it can launch test inference.
    subprocess.run(["systemctl", "--user", "stop", "tnorm-wait-train.service"], check=False)


def quality_gate(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary["quality_gate"]["passed"]:
        payload = read_status(); payload["status"] = "stopped_baseline_quality_gate"
        payload["canonical_attacks_started"] = False
        payload["quality_gate"] = summary["quality_gate"]; write_status(payload)
        raise SystemExit("Canonical v2 stopped: validation quality gate was not reached")


def main() -> None:
    wait_for_legacy()
    stage("legacy_nms_g_pilot", [str(PYTHON), "prepare_deadline_validation.py"], [PROJECT_DIR / "outputs/final_practice/deadline/pilot/pilot_gate.json", PROJECT_DIR / "outputs/final_practice/audit/legacy_recovery_recalculation.json"])
    stage("legacy_statistics", [str(PYTHON), "analyze_final_practice.py", "--input", str(LEGACY_VAL), "--output", "outputs/final_practice/09_statistics_val", "--bootstrap", "2000"], [PROJECT_DIR / "outputs/final_practice/09_statistics_val/model_comparison_m0_m4.csv"])
    stage("legacy_bundle", [str(PYTHON), "build_legacy_baseline.py"], [PROJECT_DIR / "outputs/bundles/TNormFilter_legacy_baseline.zip", PROJECT_DIR / "outputs/bundles/TNormFilter_legacy_baseline.zip.sha256"])
    stage("current_baseline_rescue", [str(PYTHON), "baseline_rescue_audit.py"], [ROOT / "baseline_rescue_current/baseline_rescue_summary.json", ROOT / "baseline_rescue_current/threshold_selection.json"])
    stage("split_v2", [str(PYTHON), "create_split_v2.py"], [ROOT / "split/split_v2_manifest.csv", ROOT / "split/split_v2_summary.json", ROOT / "split/split_v2_hash.txt", PROJECT_DIR / "data/yolo_osdar23_v2/data.yaml"])
    stage("train_v2", [str(PYTHON), "train_canonical_v2.py"], [ROOT / "training/yolo11m_canonical_v2/weights/best.pt", ROOT / "training/yolo11m_canonical_v2/TRAINING_COMPLETE"])
    checkpoint = ROOT / "training/yolo11m_canonical_v2/weights/best.pt"
    v2_audit = ROOT / "baseline_rescue_v2"
    stage("v2_clean_and_threshold", [str(PYTHON), "baseline_rescue_audit.py", "--model", str(checkpoint), "--data", "data/yolo_osdar23_v2/data.yaml", "--output", str(v2_audit)], [v2_audit / "baseline_rescue_summary.json", v2_audit / "threshold_selection.json"])
    quality_gate(v2_audit / "baseline_rescue_summary.json")
    normalization = ROOT / "normalization_v2"
    stage("normalization_v2", [str(PYTHON), "-m", "revision_q1.collect_normalization", "--model", str(checkpoint), "--data", "data/yolo_osdar23_v2/data.yaml", "--output", str(normalization), "--workers", "0"], [normalization / "normalization/layer_channel_statistics.pt", normalization / "normalization/normalization_manifest.json"])
    confidence = json.loads((v2_audit / "threshold_selection.json").read_text(encoding="utf-8"))["safety"]["confidence"]
    val_matrix = ROOT / "raw/canonical_validation.csv"
    common = ["--model", str(checkpoint), "--checkpoint-name", "canonical_v2", "--data", "data/yolo_osdar23_v2/data.yaml", "--workers", "1", "--revision-stats", str(normalization / "normalization/layer_channel_statistics.pt"), "--normalizations", "N1_quantile", "--confidence", str(confidence), "--seeds", "42,123,999", "--defenses", "none,tnorm,bilateral,median", "--nms-max-time-img", "10"]
    attacks = PROTOCOL["validation_attacks"]
    stage("canonical_validation", [str(PYTHON), "run_final_matrix.py", "--split", "val", "--output", str(val_matrix), *common, "--fgsm-eps", ",".join(map(str, attacks["fgsm_epsilon_px"])), "--pgd-eps", ",".join(map(str, attacks["pgd20_epsilon_px"])), "--pgd-steps", "20", "--adaptive-pgd-eps", ",".join(map(str, attacks["adaptive_pgd20_epsilon_px"])), "--adaptive-pgd-steps", "20"], [val_matrix, val_matrix.with_suffix(".json")])
    budgets = ROOT / "config/canonical_budget_selection.json"
    stage("freeze_budgets", [str(PYTHON), "deadline_select_budgets.py", "--strict-absolute", "--input", str(val_matrix), "--output", str(budgets)], [budgets, budgets.with_suffix(".csv")])
    frozen = json.loads(budgets.read_text(encoding="utf-8"))["selected"]
    test_matrix = ROOT / "raw/canonical_test.csv"
    stage("canonical_test", [str(PYTHON), "run_final_matrix.py", "--split", "test", "--output", str(test_matrix), *common, "--fgsm-eps", ",".join(map(str, frozen["fgsm_epsilon_px"])), "--pgd-eps", ",".join(map(str, frozen["pgd_epsilon_px"])), "--pgd-steps", "20", "--adaptive-pgd-eps", ",".join(map(str, frozen["adaptive_pgd_epsilon_px"])), "--adaptive-pgd-steps", "20"], [test_matrix, test_matrix.with_suffix(".json")])
    analysis = ROOT / "analysis_test"
    stage("canonical_statistics", [str(PYTHON), "-m", "revision_q1.analyze", "--input", str(test_matrix), "--output", str(analysis), "--normalization", "N1_quantile", "--split", "test", "--bootstrap", "5000"], [analysis / "tables/12_multiple_comparison_corrections.csv", analysis / "tables/13_scene_macro_loso.csv"])
    stage("canonical_bundle", [str(PYTHON), "finalize_canonical_v2.py"], [PROJECT_DIR / "outputs/bundles/TNormFilter_canonical_final.zip", PROJECT_DIR / "outputs/bundles/TNormFilter_canonical_final.zip.sha256"])
    payload = read_status(); payload["status"] = "success"; write_status(payload)


if __name__ == "__main__":
    main()
