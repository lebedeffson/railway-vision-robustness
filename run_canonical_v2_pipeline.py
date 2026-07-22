from __future__ import annotations

from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/canonical_v2"
STATUS = ROOT / "pipeline_status.json"
PYTHON = PROJECT_DIR / ".venv/bin/python"
LEGACY_VAL = PROJECT_DIR / "outputs/final_practice/unified_diagnostics_val_raw.csv"
LEGACY_CONFIG = LEGACY_VAL.with_suffix(".json")
LEGACY_COMPLETE = PROJECT_DIR / "outputs/final_practice/legacy_validation.complete.json"
LOCK = PROJECT_DIR / "outputs/locks/canonical_v2.lock"
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_legacy_completion_marker() -> dict[str, Any]:
    """Gate the handoff on a closed, complete and internally balanced CSV."""
    expected_frames = 198
    expected_rows_per_frame = 450
    condition_columns = (
        "attack", "adaptive", "epsilon_px", "steps", "restart", "seed",
        "defense", "layer",
    )
    frame_counts: Counter[str] = Counter()
    signatures: defaultdict[str, set[tuple[str, ...]]] = defaultdict(set)
    duplicate_rows = 0
    seen: set[tuple[str, ...]] = set()
    total_rows = 0
    with LEGACY_VAL.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", *condition_columns}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError("Legacy validation CSV does not contain its condition key")
        for row in reader:
            total_rows += 1
            image = row["image_path"]
            signature = tuple(row[column] for column in condition_columns)
            key = (image, *signature)
            duplicate_rows += int(key in seen)
            seen.add(key)
            frame_counts[image] += 1
            signatures[image].add(signature)
    completed = sum(count == expected_rows_per_frame for count in frame_counts.values())
    partial = sum(count != expected_rows_per_frame for count in frame_counts.values())
    reference = max(signatures.values(), key=len, default=set())
    missing_conditions = sum(len(reference - value) for value in signatures.values())
    unexpected_conditions = sum(len(value - reference) for value in signatures.values())
    payload = {
        "status": "PASS",
        "created_at": now(),
        "csv_path": str(LEGACY_VAL.resolve()),
        "config_path": str(LEGACY_CONFIG.resolve()),
        "csv_sha256": sha256(LEGACY_VAL),
        "config_sha256": sha256(LEGACY_CONFIG),
        "total_rows": total_rows,
        "completed_frames": completed,
        "partial_frames": partial,
        "missing_conditions": missing_conditions,
        "unexpected_conditions": unexpected_conditions,
        "duplicate_rows": duplicate_rows,
        "csv_successfully_closed": True,
        "expected_frames": expected_frames,
        "expected_rows_per_frame": expected_rows_per_frame,
    }
    valid = (
        completed == expected_frames
        and len(frame_counts) == expected_frames
        and total_rows == expected_frames * expected_rows_per_frame
        and partial == 0
        and missing_conditions == 0
        and unexpected_conditions == 0
        and duplicate_rows == 0
    )
    if not valid:
        payload["status"] = "FAIL"
        raise RuntimeError(f"Legacy validation completion gate failed: {payload}")
    temporary = LEGACY_COMPLETE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(LEGACY_COMPLETE)
    checksum = LEGACY_COMPLETE.with_suffix(".sha256")
    checksum.write_text(f"{payload['csv_sha256']}  {LEGACY_VAL.name}\n", encoding="utf-8")
    return payload


def wait_for_legacy() -> None:
    blocker = ROOT / "LEGACY_TEST_BLOCKED"
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text(
        "Current 3-scene legacy test is forbidden; canonical v2 uses its own split and output.\n",
        encoding="utf-8",
    )
    payload = read_status(); payload["status"] = "waiting_for_legacy_validation_198"
    payload["legacy_test_blocked"] = True; write_status(payload)
    while not (LEGACY_VAL.is_file() and LEGACY_CONFIG.is_file()):
        time.sleep(30)
    marker = create_legacy_completion_marker()
    payload = read_status()
    payload["legacy_validation_marker"] = marker
    payload["status"] = "legacy_validation_complete"
    write_status(payload)
    # Stop the paused obsolete controller only after the atomic completion marker.
    subprocess.run(["systemctl", "--user", "stop", "tnorm-wait-train.service"], check=False)


def build_quality_gate_failure_bundle(summary_path: Path) -> Path:
    summary_path = summary_path.resolve()
    destination = PROJECT_DIR / "outputs/bundles/TNormFilter_baseline_gate_failed.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    roots = [
        summary_path.parent,
        ROOT / "split",
        # Include both the exported checkpoint selection/provenance and the
        # Ultralytics validation artifacts. Checkpoint weights remain excluded
        # by the extension allow-list below.
        ROOT / "training",
        ROOT / "pipeline_status.json",
        PROJECT_DIR / "config/canonical_v2_protocol.yaml",
    ]
    allowed = {".csv", ".json", ".png", ".yaml", ".yml", ".txt"}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in roots:
            candidates = [root] if root.is_file() else sorted(root.rglob("*"))
            for path in candidates:
                if not path.is_file() or path.suffix.lower() not in allowed:
                    continue
                archive.write(path, path.relative_to(PROJECT_DIR))
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{sha256(destination)}  {destination.name}\n", encoding="utf-8"
    )
    return destination


def quality_gate(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output = summary_path.parent
    clean = output / "clean_metrics_train_val_test.csv"
    scenes = output / "clean_metrics_per_scene.csv"
    thresholds = output / "threshold_selection.json"
    required = [clean, scenes, thresholds, output / "ap_by_class.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Quality gate evidence is incomplete: {missing}")
    clean_rows = list(csv.DictReader(clean.open(encoding="utf-8")))
    scene_rows = list(csv.DictReader(scenes.open(encoding="utf-8")))
    class_rows = list(csv.DictReader((output / "ap_by_class.csv").open(encoding="utf-8")))
    threshold_payload = json.loads(thresholds.read_text(encoding="utf-8"))
    validation_scenes = {
        row["sequence_id"] for row in scene_rows
        if row.get("split") == "val" and row.get("operating_point") == "standard"
    }
    finite = all(
        math.isfinite(float(row[name]))
        for row in clean_rows if row.get("split") in {"train", "val"}
        for name in ("precision", "recall", "f1", "f2", "fn_per_frame", "mAP50", "mAP50-95")
    )
    technical_checks = {
        "five_validation_scenes_evaluated": len(validation_scenes) == 5,
        "metrics_are_finite": finite,
        "threshold_fit_on_validation_only": (
            threshold_payload.get("selection_split") == "val"
            and not threshold_payload.get("test_used_for_selection", True)
        ),
        "checkpoint_hash_matches_summary": (
            threshold_payload.get("checkpoint_sha256") == summary.get("checkpoint_sha256")
        ),
        "class_mapping_is_canonical": {
            row.get("class_name") for row in class_rows if row.get("split") == "val"
        } == {"person", "signal", "road_vehicle", "train", "animal", "bicycle"},
        "metrics_csv_complete": {
            (row.get("split"), row.get("operating_point")) for row in clean_rows
        } == {
            ("train", "standard"), ("train", "safety"),
            ("val", "standard"), ("val", "safety"),
        },
        "test_not_evaluated": "test" not in summary.get("evaluated_splits", []),
    }
    gate_record = {
        **summary["quality_gate"],
        "technical_checks": technical_checks,
        "passed": bool(summary["quality_gate"]["passed"] and all(technical_checks.values())),
    }
    (output / "quality_gate.json").write_text(
        json.dumps(gate_record, indent=2) + "\n", encoding="utf-8"
    )
    if not gate_record["passed"]:
        bundle = PROJECT_DIR / "outputs/bundles/TNormFilter_baseline_gate_failed.zip"
        payload = read_status(); payload["status"] = "stopped_baseline_quality_gate"
        payload["canonical_attacks_started"] = False
        payload["quality_gate_passed"] = False
        payload["quality_gate"] = gate_record
        payload["quality_gate_failure_bundle"] = str(bundle)
        gate_stage = payload.setdefault("stages", {}).setdefault("baseline_quality_gate", {})
        gate_stage.update({
            "status": "failed",
            "finished_at": now(),
            "outputs": [str(output / "quality_gate.json"), str(bundle)],
            "error": "validation_quality_gate_not_reached",
        })
        stage_order = PROTOCOL.get("canonical_stage_order", [])
        gate_index = stage_order.index("baseline_quality_gate")
        for name in stage_order[gate_index + 1:]:
            payload["stages"].setdefault(name, {
                "status": "skipped",
                "finished_at": now(),
                "outputs": [],
                "error": "blocked_by_validation_quality_gate",
            })
        write_status(payload)
        build_quality_gate_failure_bundle(summary_path)
        raise SystemExit("Canonical v2 stopped: validation quality gate was not reached")


def main() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("Another canonical v2 pipeline owns outputs/locks/canonical_v2.lock") from error
    lock_handle.seek(0); lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()}\nstarted_at={now()}\n")
    lock_handle.flush()
    wait_for_legacy()
    stage("legacy_nms_g_pilot", [str(PYTHON), "finalize_legacy_smoke.py"], [PROJECT_DIR / "outputs/final_practice/deadline/pilot/legacy_smoke_disposition.json", PROJECT_DIR / "outputs/final_practice/audit/legacy_recovery_recalculation.json"])
    stage("legacy_nms_contract", [str(PYTHON), "finalize_legacy_nms_audit.py"], [PROJECT_DIR / "outputs/final_practice/audit/nms_timeout_recheck.csv", PROJECT_DIR / "outputs/final_practice/audit/nms_audit_summary.json"])
    stage("legacy_statistics", [str(PYTHON), "analyze_final_practice.py", "--input", str(LEGACY_VAL), "--output", "outputs/final_practice/09_statistics_val", "--bootstrap", "2000"], [PROJECT_DIR / "outputs/final_practice/09_statistics_val/model_comparison_m0_m4.csv"])
    stage("legacy_bundle", [str(PYTHON), "build_legacy_baseline.py"], [PROJECT_DIR / "outputs/bundles/TNormFilter_legacy_baseline.zip", PROJECT_DIR / "outputs/bundles/TNormFilter_legacy_baseline.zip.sha256"])
    legacy_threshold = ROOT / "legacy_threshold_calibration"
    stage("legacy_threshold_calibration", [str(PYTHON), "baseline_rescue_audit.py", "--output", str(legacy_threshold), "--splits", "val"], [legacy_threshold / "baseline_rescue_summary.json", legacy_threshold / "threshold_selection.json"])
    stage("split_v2", [str(PYTHON), "create_split_v2.py"], [ROOT / "split/split_v2_manifest.csv", ROOT / "split/split_v2_summary.json", ROOT / "split/split_v2_hash.txt", PROJECT_DIR / "data/yolo_osdar23_v2/data.yaml"])
    stage("canonical_v2_pilot_selection_unbiased", [str(PYTHON), "canonical_v2_select_pilot.py"], [ROOT / "pilot/pilot_frames.csv", ROOT / "pilot/pilot_selection.json", ROOT / "pilot/pilot_data.yaml"])
    stage("train_v2", [str(PYTHON), "train_canonical_v2.py"], [ROOT / "training/yolo11m_canonical_v2/weights/best.pt", ROOT / "training/yolo11m_canonical_v2/TRAINING_COMPLETE"])
    stage("finalize_training_provenance", [str(PYTHON), "finalize_canonical_training.py"], [ROOT / "training/checkpoint_selection.csv", ROOT / "training/checkpoint_provenance.json"])
    checkpoint = ROOT / "training/yolo11m_canonical_v2/weights/best.pt"
    v2_audit = ROOT / "baseline_rescue_v2"
    stage("canonical_v2_threshold_calibration", [str(PYTHON), "baseline_rescue_audit.py", "--model", str(checkpoint), "--data", "data/yolo_osdar23_v2/data.yaml", "--output", str(v2_audit), "--splits", "train,val"], [v2_audit / "baseline_rescue_summary.json", v2_audit / "threshold_selection.json", v2_audit / "threshold_sweep_per_scene.csv", v2_audit / "checkpoint_hash_verification.json"])
    quality_gate(v2_audit / "baseline_rescue_summary.json")
    normalization = ROOT
    stage("normalization_v2", [str(PYTHON), "-m", "revision_q1.collect_normalization", "--model", str(checkpoint), "--data", "data/yolo_osdar23_v2/data.yaml", "--output", str(normalization), "--workers", "0"], [normalization / "normalization/layer_channel_statistics.pt", normalization / "normalization/normalization_manifest.json"])
    stage("freeze_normalization", [str(PYTHON), "select_canonical_normalization.py"], [normalization / "normalization/normalization_selection.json", normalization / "normalization/normalization_manifest.json"])
    normalization_mode = json.loads(
        (normalization / "normalization/normalization_selection.json").read_text(encoding="utf-8")
    )["selected_normalization"]
    confidence = json.loads((v2_audit / "threshold_selection.json").read_text(encoding="utf-8"))["safety"]["confidence"]
    pilot_matrix = ROOT / "pilot/pilot_metrics.csv"
    stage("canonical_v2_pilot_matrix", [str(PYTHON), "run_final_matrix.py", "--model", str(checkpoint), "--checkpoint-name", "canonical_v2", "--data", str(ROOT / "pilot/pilot_data.yaml"), "--manifest", "data/yolo_osdar23_v2/manifest.csv", "--split", "val", "--workers", "0", "--revision-stats", str(normalization / "normalization/layer_channel_statistics.pt"), "--normalizations", normalization_mode, "--confidence", str(confidence), "--seeds", "42,123,999", "--defenses", "none,tnorm,bilateral,median", "--nms-max-time-img", "10", "--fgsm-eps", "1", "--pgd-eps", "0.25", "--pgd-steps", "20", "--adaptive-pgd-eps", "0.25", "--adaptive-pgd-steps", "20", "--output", str(pilot_matrix)], [pilot_matrix, pilot_matrix.with_suffix(".json")])
    stage("canonical_v2_pilot_gate", [str(PYTHON), "canonical_v2_pilot_gate.py"], [ROOT / "pilot/pilot_gate.json", ROOT / "pilot/pilot_report.md", ROOT / "pilot/attack_budget_audit.csv", ROOT / "pilot/adaptive_gradient_audit.json"])
    val_matrix = ROOT / "raw/canonical_validation.csv"
    common = ["--model", str(checkpoint), "--checkpoint-name", "canonical_v2", "--data", "data/yolo_osdar23_v2/data.yaml", "--manifest", "data/yolo_osdar23_v2/manifest.csv", "--workers", "1", "--revision-stats", str(normalization / "normalization/layer_channel_statistics.pt"), "--normalizations", normalization_mode, "--confidence", str(confidence), "--seeds", "42,123,999", "--defenses", "none,tnorm,bilateral,median", "--nms-max-time-img", "10"]
    attacks = PROTOCOL["validation_attacks"]
    stage("canonical_validation", [str(PYTHON), "run_final_matrix.py", "--split", "val", "--output", str(val_matrix), *common, "--fgsm-eps", ",".join(map(str, attacks["fgsm_epsilon_px"])), "--pgd-eps", ",".join(map(str, attacks["pgd20_epsilon_px"])), "--pgd-steps", "20", "--adaptive-pgd-eps", ",".join(map(str, attacks["adaptive_pgd20_epsilon_px"])), "--adaptive-pgd-steps", "20"], [val_matrix, val_matrix.with_suffix(".json")])
    provenance_validation = ROOT / "config/provenance_after_validation.json"
    stage("verify_validation_provenance", [str(PYTHON), "verify_canonical_provenance.py", "--checkpoint", str(checkpoint), "--thresholds", str(v2_audit / "threshold_selection.json"), "--matrix-config", str(val_matrix.with_suffix(".json")), "--output", str(provenance_validation)], [provenance_validation])
    budgets = ROOT / "config/canonical_budget_selection.json"
    stage("freeze_budgets", [str(PYTHON), "deadline_select_budgets.py", "--strict-absolute", "--input", str(val_matrix), "--output", str(budgets)], [budgets, budgets.with_suffix(".csv"), ROOT / "config/frozen_attack_budgets.yaml"])
    frozen = json.loads(budgets.read_text(encoding="utf-8"))["selected"]
    clean_test = v2_audit / "test_evaluation"
    stage("canonical_v2_clean_test", [str(PYTHON), "baseline_rescue_audit.py", "--model", str(checkpoint), "--data", "data/yolo_osdar23_v2/data.yaml", "--output", str(clean_test), "--splits", "test", "--frozen-thresholds", str(v2_audit / "threshold_selection.json")], [clean_test / "baseline_rescue_summary.json", clean_test / "clean_metrics_train_val_test.csv"])
    stage("export_canonical_v2_clean", [str(PYTHON), "export_canonical_v2_clean.py"], [ROOT / "calibration/threshold_selection.json", ROOT / "tables/clean_test_metrics.csv", ROOT / "tables/clean_test_per_class.csv", ROOT / "tables/clean_test_per_scene.csv", ROOT / "tables/clean_test_object_sizes.csv"])
    provenance_before = ROOT / "config/provenance_before_test_attacks.json"
    stage("verify_provenance_before_test_attacks", [str(PYTHON), "verify_canonical_provenance.py", "--checkpoint", str(checkpoint), "--thresholds", str(v2_audit / "threshold_selection.json"), "--clean-test-summary", str(clean_test / "baseline_rescue_summary.json"), "--matrix-config", str(val_matrix.with_suffix(".json")), "--output", str(provenance_before)], [provenance_before])
    test_matrix = ROOT / "raw/canonical_test.csv"
    stage("canonical_test", [str(PYTHON), "run_final_matrix.py", "--split", "test", "--output", str(test_matrix), *common, "--fgsm-eps", ",".join(map(str, frozen["fgsm_epsilon_px"])), "--pgd-eps", ",".join(map(str, frozen["pgd_epsilon_px"])), "--pgd-steps", "20", "--adaptive-pgd-eps", ",".join(map(str, frozen["adaptive_pgd_epsilon_px"])), "--adaptive-pgd-steps", "20"], [test_matrix, test_matrix.with_suffix(".json")])
    provenance_final = ROOT / "config/provenance_final.json"
    stage("verify_final_provenance", [str(PYTHON), "verify_canonical_provenance.py", "--checkpoint", str(checkpoint), "--thresholds", str(v2_audit / "threshold_selection.json"), "--clean-test-summary", str(clean_test / "baseline_rescue_summary.json"), "--matrix-config", str(val_matrix.with_suffix(".json")), "--matrix-config", str(test_matrix.with_suffix(".json")), "--output", str(provenance_final)], [provenance_final])
    analysis = ROOT / "analysis_test"
    stage("canonical_matrix_audit", [str(PYTHON), "audit_canonical_matrix.py"], [ROOT / "audit/canonical_matrix_audit.json", ROOT / "audit/canonical_nms_audit.csv"])
    stage("canonical_statistics", [str(PYTHON), "-m", "revision_q1.analyze", "--protocol", "config/canonical_v2_analysis.yaml", "--input", str(test_matrix), "--output", str(analysis), "--normalization", normalization_mode, "--split", "test", "--bootstrap", "5000"], [analysis / "tables/12_multiple_comparison_corrections.csv", analysis / "tables/13_scene_macro_loso.csv"])
    latency = ROOT / "latency"
    stage("canonical_latency", [str(PYTHON), "benchmark_latency.py", "--model", str(checkpoint), "--data", "data/yolo_osdar23_v2/data.yaml", "--revision-stats", str(normalization / "normalization/layer_channel_statistics.pt"), "--output", str(latency), "--warmup", "30", "--repetitions", "100", "--batch", "1"], [latency / "latency_summary.csv", latency / "latency_raw.csv", latency / "environment.json"])
    stage("canonical_outputs", [str(PYTHON), "build_canonical_outputs.py"], [ROOT / "report/output_build.json", ROOT / "tables/15_nms_audit.csv", ROOT / "figures/10_latency_tradeoff.png"])
    article = PROJECT_DIR / "outputs/article"
    stage("canonical_article_fill", [str(PYTHON), "article/fill_article_results.py"], [article / "TNorm_RZD_article_final.docx", article / "TNorm_RZD_article_final.pdf", article / "TNorm_RZD_supplementary.pdf", article / "result_mapping_resolved.json"])
    stage("canonical_article_validation", [str(PYTHON), "article/validate_final_article.py"], [article / "article_validation.json"])
    stage("canonical_acceptance_tests", [str(PYTHON), "run_canonical_acceptance_tests.py"], [ROOT / "tests/unittest.txt", ROOT / "tests/test_summary.json"])
    stage("canonical_bundle", [str(PYTHON), "finalize_canonical_v2.py"], [PROJECT_DIR / "outputs/bundles/TNormFilter_canonical_final.zip", PROJECT_DIR / "outputs/bundles/TNormFilter_canonical_final.zip.sha256"])
    payload = read_status(); payload["status"] = "success"; write_status(payload)


if __name__ == "__main__":
    main()
