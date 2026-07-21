from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parent
STATUS_PATH = PROJECT_DIR / "outputs/pipeline_status.json"
STAGES = (
    "dataset_recovery",
    "dataset_validation",
    "stage1_training",
    "stage2_training",
    "clean_evaluation",
    "attack_generation",
    "feature_extraction",
    "attack_consistency",
    "defense_consistency",
    "statistics",
    "figures",
    "report",
    "bundle",
)
VALID_STATUSES = {"pending", "running", "success", "failed", "skipped"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_stage() -> dict[str, object]:
    return {
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "input_hash": None,
        "output_files": [],
        "error": None,
    }


def load(path: Path = STATUS_PATH) -> dict[str, object]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {}
    stages = payload.setdefault("stages", {})
    for stage in STAGES:
        current = empty_stage()
        if isinstance(stages.get(stage), dict):
            current.update(stages[stage])
        stages[stage] = current
    payload["schema_version"] = 1
    payload["updated_at"] = now()
    return payload


def save(payload: dict[str, object], path: Path = STATUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now()
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def digest_inputs(values: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(Path(item)) for item in values):
        path = Path(value)
        digest.update(value.encode("utf-8"))
        if path.is_file():
            digest.update(str(path.stat().st_size).encode("ascii"))
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        elif path.exists():
            digest.update(str(path.stat().st_mtime_ns).encode("ascii"))
        else:
            digest.update(b"MISSING")
    return digest.hexdigest()


def update(
    stage: str,
    status: str,
    *,
    inputs: Iterable[str | Path] = (),
    outputs: Iterable[str | Path] = (),
    error: str | None = None,
    path: Path = STATUS_PATH,
) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid pipeline status: {status}")
    payload = load(path)
    entry = payload["stages"][stage]
    if status == "running" and not entry.get("started_at"):
        entry["started_at"] = now()
    if status in {"success", "failed", "skipped"}:
        entry["started_at"] = entry.get("started_at") or now()
        entry["finished_at"] = now()
    elif status == "pending":
        entry["started_at"] = None
        entry["finished_at"] = None
    entry["status"] = status
    input_values = list(inputs)
    if input_values:
        entry["input_hash"] = digest_inputs(input_values)
    output_values = [str(Path(item)) for item in outputs]
    if output_values:
        entry["output_files"] = output_values
    entry["error"] = error
    save(payload, path)


def sync(path: Path = STATUS_PATH) -> dict[str, object]:
    """Reconcile inexpensive on-disk evidence without disturbing active jobs."""
    from download_osdar23_direct import SEQUENCES, sequence_is_usable

    payload = load(path)
    evidence: dict[str, tuple[str, list[Path], list[Path]]] = {}
    if all(sequence_is_usable(name) for name in SEQUENCES):
        evidence["dataset_recovery"] = (
            "success",
            [PROJECT_DIR / "config/raw_frame_exclusions.json"],
            [PROJECT_DIR / "download_osdar23_direct.py", PROJECT_DIR / "config/raw_frame_exclusions.json"],
        )
    audit = PROJECT_DIR / "outputs/final_practice/00_audit/audit_summary.json"
    if audit.is_file() and json.loads(audit.read_text(encoding="utf-8")).get("status") == "PASS":
        evidence["dataset_validation"] = (
            "success", [audit],
            [PROJECT_DIR / "data/yolo_osdar23/manifest.csv", PROJECT_DIR / "audit_final_practice.py"],
        )
    stage1 = PROJECT_DIR / "outputs/training/yolo11m_baseline_stage1"
    stage2 = PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2"
    if (stage1 / "TRAINING_COMPLETE").is_file():
        evidence["stage1_training"] = (
            "success", [stage1 / "weights/best.pt"],
            [PROJECT_DIR / "data/yolo_osdar23/data.yaml", PROJECT_DIR / "train_yolo_baseline.py"],
        )
    elif (stage1 / "weights/last.pt").is_file():
        evidence["stage1_training"] = (
            "running", [stage1 / "weights/last.pt"],
            [PROJECT_DIR / "data/yolo_osdar23/data.yaml", PROJECT_DIR / "train_yolo_baseline.py"],
        )
    if (stage2 / "TRAINING_COMPLETE").is_file() and (stage2 / "weights/best.pt").is_file():
        evidence["stage2_training"] = (
            "success", [stage2 / "weights/best.pt"],
            [stage1 / "weights/best.pt", PROJECT_DIR / "train_yolo_baseline.py"],
        )
    elif (stage2 / "weights/last.pt").is_file():
        evidence["stage2_training"] = (
            "running", [stage2 / "weights/last.pt"],
            [stage1 / "weights/best.pt", PROJECT_DIR / "train_yolo_baseline.py"],
        )

    output_map = {
        "clean_evaluation": [PROJECT_DIR / "outputs/evaluation/clean_test_baseline/clean_test_metrics.json"],
        "attack_generation": [PROJECT_DIR / "outputs/final_practice/unified_diagnostics_raw.csv"],
        "feature_extraction": [PROJECT_DIR / "outputs/diagnostics/feature_consistency/feature_consistency_val.csv"],
        "attack_consistency": [PROJECT_DIR / "outputs/final_practice/raw/attack_consistency.csv"],
        "defense_consistency": [PROJECT_DIR / "outputs/final_practice/raw/defense_consistency.csv"],
        "statistics": [PROJECT_DIR / "outputs/final_practice/09_statistics/model_comparison_m0_m4.csv"],
        "figures": [PROJECT_DIR / "outputs/final_practice/figures/09_clean_utility_vs_latency.png"],
        "report": [PROJECT_DIR / "outputs/final_practice/TNormFilter_final_practice_report.pdf"],
        "bundle": [PROJECT_DIR / "outputs/bundles/TNormFilter_final_practice.zip"],
    }
    for stage, outputs in output_map.items():
        if all(item.is_file() for item in outputs):
            evidence[stage] = (
                "success", outputs,
                [PROJECT_DIR / {
                    "clean_evaluation": "evaluate_baseline.py",
                    "attack_generation": "run_final_matrix.py",
                    "feature_extraction": "extract_feature_consistency.py",
                    "attack_consistency": "export_consistency_tables.py",
                    "defense_consistency": "export_consistency_tables.py",
                    "statistics": "analyze_final_practice.py",
                    "figures": "plot_final_practice.py",
                    "report": "build_final_report.py",
                    "bundle": "build_final_delivery.py",
                }[stage]],
            )
    for stage, (status, outputs, inputs) in evidence.items():
        entry = payload["stages"][stage]
        entry["status"] = status
        entry["started_at"] = entry.get("started_at") or now()
        entry["finished_at"] = now() if status == "success" else None
        entry["output_files"] = [str(item) for item in outputs]
        entry["input_hash"] = digest_inputs(inputs)
        entry["error"] = None
    save(payload, path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Maintain resumable final-practice status")
    parser.add_argument("command", choices=("sync", "show"), nargs="?", default="sync")
    parser.add_argument("--path", type=Path, default=STATUS_PATH)
    args = parser.parse_args()
    payload = sync(args.path) if args.command == "sync" else load(args.path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
