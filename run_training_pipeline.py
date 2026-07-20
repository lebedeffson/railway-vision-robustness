from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from download_osdar23_direct import SEQUENCES, sequence_is_complete


PROJECT_DIR = Path(__file__).resolve().parent
STATE_PATH = PROJECT_DIR / "outputs/final_practice/training_pipeline_state.json"
MARKERS = PROJECT_DIR / "outputs/final_practice/pipeline_markers"


def write_state(stage: str, status: str, **extra: object) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"stage": stage, "status": status, **extra}, indent=2) + "\n",
        encoding="utf-8",
    )


def run(
    stage: str,
    script: str,
    *arguments: str,
    expected: tuple[str, ...] = (),
) -> None:
    marker = MARKERS / f"{stage}.json"
    expected_paths = [PROJECT_DIR / value for value in expected]
    if marker.is_file() and all(path.exists() for path in expected_paths):
        write_state(stage, "SKIP_PASS", marker=str(marker))
        return
    write_state(stage, "RUNNING", command=[sys.executable, script, *arguments])
    subprocess.run([sys.executable, script, *arguments], cwd=PROJECT_DIR, check=True)
    missing = [str(path) for path in expected_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"{stage}: expected outputs missing: {missing}")
    MARKERS.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({
        "stage": stage,
        "status": "PASS",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, script, *arguments],
        "outputs": [str(path) for path in expected_paths],
    }, indent=2) + "\n", encoding="utf-8")
    write_state(stage, "PASS", marker=str(marker))


def main() -> None:
    missing = [sequence for sequence in SEQUENCES if not sequence_is_complete(sequence)]
    if missing:
        write_state("download", "BLOCKED", missing_sequences=missing)
        raise SystemExit(
            f"Dataset download is incomplete: {len(missing)} sequences missing"
        )
    run(
        "build_dataset", "build_yolo_dataset.py",
        expected=("data/yolo_osdar23/data.yaml", "data/yolo_osdar23/manifest.csv"),
    )
    run(
        "validate_labels", "validate_yolo_dataset.py",
        expected=("outputs/label_preview",),
    )
    run(
        "sequence_audit", "audit_final_practice.py",
        expected=("outputs/final_practice/00_audit/audit_summary.json",),
    )
    run(
        "train", "train_yolo_baseline.py",
        expected=(
            "outputs/training/yolo11m_baseline_stage2/weights/best.pt",
            "outputs/training/yolo11m_baseline_stage2/TRAINING_COMPLETE",
        ),
    )
    run(
        "preflight", "final_practice_preflight.py",
        expected=("outputs/final_practice/preflight.json",),
    )
    run(
        "clean_test", "evaluate_baseline.py",
        expected=("outputs/evaluation/clean_test_baseline/clean_test_metrics.json",),
    )
    run(
        "clean_utility", "evaluate_clean_utility.py",
        expected=("outputs/final_practice/03_clean_utility/clean_utility_summary.csv",),
    )
    run(
        "feature_normalization", "extract_feature_consistency.py",
        "--eval-split", "val", "--quick", "--rebuild-stats",
        "--batch", "1", "--workers", "0",
        expected=(
            "outputs/diagnostics/feature_consistency/feature_normalization_val.pt",
            "outputs/diagnostics/feature_consistency/feature_consistency_val.csv",
        ),
    )
    run(
        "final_matrix_val", "run_final_matrix.py",
        "--split", "val", "--workers", "1",
        "--output", "outputs/final_practice/unified_diagnostics_val_raw.csv",
        expected=("outputs/final_practice/unified_diagnostics_val_raw.csv",),
    )
    run(
        "statistics_val", "analyze_final_practice.py",
        "--input", "outputs/final_practice/unified_diagnostics_val_raw.csv",
        "--output", "outputs/final_practice/09_statistics_val", "--bootstrap", "2000",
        expected=("outputs/final_practice/09_statistics_val/model_comparison_m0_m4.csv",),
    )
    run(
        "freeze_validation", "freeze_validation_protocol.py",
        expected=("outputs/final_practice/validation_freeze.json",),
    )
    run(
        "final_matrix_test", "run_final_matrix.py",
        "--split", "test", "--workers", "1",
        "--output", "outputs/final_practice/unified_diagnostics_raw.csv",
        expected=("outputs/final_practice/unified_diagnostics_raw.csv",),
    )
    run(
        "export_consistency", "export_consistency_tables.py",
        expected=(
            "outputs/final_practice/attack_consistency_raw.csv",
            "outputs/final_practice/defense_consistency_raw.csv",
        ),
    )
    run(
        "statistics_test", "analyze_final_practice.py", "--bootstrap", "2000",
        expected=("outputs/final_practice/09_statistics/model_comparison_m0_m4.csv",),
    )
    run(
        "latency", "benchmark_latency.py", "--warmup", "30", "--repetitions", "100", "--batch", "1",
        expected=("outputs/final_practice/08_latency/latency.csv",),
    )
    run(
        "figures", "plot_final_practice.py",
        expected=("outputs/final_practice/figures/08_layerwise_recovery.png",),
    )
    run(
        "delivery_zip", "build_final_delivery.py",
        expected=(
            "outputs/bundles/TNormFilter_final_practice.zip",
            "outputs/bundles/TNormFilter_final_practice.zip.sha256",
        ),
    )
    write_state("complete", "PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_state("failed", "FAIL", error=f"{type(error).__name__}: {error}")
        raise
