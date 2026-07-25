from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.temporal_safety.common import (
    CONFIG,
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_csv,
    atomic_json,
    sha256,
)
from src.temporal_safety.metrics import paired_scene_bootstrap


def loso_rows(
    merged: pd.DataFrame, tracker: str
) -> list[dict[str, float | str]]:
    rows = []
    for scene in sorted(merged["grouped_scene_id"].astype(str).unique()):
        reduced = merged[merged["grouped_scene_id"].astype(str).ne(scene)]
        rows.append(
            {
                "tracker": tracker,
                "excluded_scene": scene,
                "delta_recall": float(
                    (
                        reduced["recall_candidate"]
                        - reduced["recall_baseline"]
                    ).mean()
                ),
                "delta_FN_per_frame": float(
                    (
                        reduced["FN_per_frame_candidate"]
                        - reduced["FN_per_frame_baseline"]
                    ).mean()
                ),
                "delta_false_alarms_per_minute": float(
                    (
                        reduced["false_alarms_per_minute_candidate"]
                        - reduced["false_alarms_per_minute_baseline"]
                    ).mean()
                ),
            }
        )
    return rows


def build_bundle(files: list[Path]) -> Path:
    bundle = OUTPUT / "bundles/railway_person_temporal_safety_development_fail.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        manifest = []
        for path in files:
            if not path.is_file():
                raise RuntimeError(f"Missing final artifact: {path}")
            relative = path.relative_to(PROJECT).as_posix()
            archive.write(path, relative)
            manifest.append(f"{sha256(path)}  {relative}")
        archive.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    with zipfile.ZipFile(bundle) as archive:
        prohibited = (
            "data/raw/",
            "/images/",
            ".pt",
            "sealed_test_manifest.csv",
            "raw_predictions.parquet",
        )
        for name in archive.namelist():
            if any(token in name for token in prohibited):
                raise RuntimeError(f"Prohibited payload in public bundle: {name}")
    (bundle.parent / "MANIFEST.sha256").write_text(
        f"{sha256(bundle)}  {bundle.name}\n", encoding="utf-8"
    )
    return bundle


def main() -> None:
    lock = assert_locked()
    gate_path = OUTPUT / "triage/TRIAGE_GATE.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["status"] != "DEVELOPMENT_FAIL":
        raise RuntimeError("Negative finalizer requires frozen development FAIL")
    per_scene = pd.read_csv(OUTPUT / "triage/PER_SCENE_RESULTS.csv")
    baseline = per_scene[per_scene["system"].eq("frame_detector")].copy()
    bootstrap_rows = []
    loso = []
    effects = []
    iterations = 10000
    seed = 20260726
    for order, tracker in enumerate(("bytetrack", "ocsort")):
        candidate = per_scene[per_scene["system"].eq(tracker)].copy()
        merged = baseline.merge(
            candidate,
            on=["fold", "grouped_scene_id"],
            suffixes=("_baseline", "_candidate"),
        )
        for metric in ("recall", "FN_per_frame", "false_alarms_per_minute", "f1"):
            interval = paired_scene_bootstrap(
                dict(
                    zip(
                        merged["grouped_scene_id"],
                        merged[f"{metric}_baseline"],
                    )
                ),
                dict(
                    zip(
                        merged["grouped_scene_id"],
                        merged[f"{metric}_candidate"],
                    )
                ),
                iterations,
                seed + order,
            )
            bootstrap_rows.append({"tracker": tracker, "metric": metric, **interval})
        loso.extend(loso_rows(merged, tracker))
        effects.extend(
            {
                "tracker": tracker,
                "fold": int(row.fold),
                "grouped_scene_id": str(row.grouped_scene_id),
                "delta_recall": float(row.recall_candidate - row.recall_baseline),
                "delta_FN_per_frame": float(
                    row.FN_per_frame_candidate - row.FN_per_frame_baseline
                ),
                "delta_false_alarms_per_minute": float(
                    row.false_alarms_per_minute_candidate
                    - row.false_alarms_per_minute_baseline
                ),
                "delta_f1": float(row.f1_candidate - row.f1_baseline),
            }
            for row in merged.itertuples(index=False)
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    loso_frame = pd.DataFrame(loso)
    effects_frame = pd.DataFrame(effects)
    atomic_csv(OUTPUT / "statistics/PAIRED_SCENE_BOOTSTRAP.csv", bootstrap)
    atomic_csv(OUTPUT / "statistics/LOSO_SENSITIVITY.csv", loso_frame)
    atomic_csv(OUTPUT / "statistics/PER_SCENE_EFFECTS.csv", effects_frame)

    latency = []
    for fold in (0, 1):
        detector = json.loads(
            (
                PROJECT
                / f"outputs/person_v3/scene_cv/fold_{fold}/seed_20260723/"
                "evaluation/evaluation_result.json"
            ).read_text(encoding="utf-8")
        )
        for tracker in ("bytetrack", "ocsort"):
            item = json.loads(
                (
                    OUTPUT / f"triage/fold_{fold}/{tracker}/metrics.json"
                ).read_text(encoding="utf-8")
            )
            latency.append(
                {
                    "fold": fold,
                    "system": tracker,
                    "detector_mean_ms": detector["mean_runtime_ms"],
                    "tracker_mean_ms": item["tracker_latency_ms_per_frame"],
                    "full_pipeline_mean_ms": detector["mean_runtime_ms"]
                    + item["tracker_latency_ms_per_frame"],
                    "full_pipeline_p95_ms": np.nan,
                    "p95_status": "NOT_MEASURED_FROM_CACHED_PREDICTIONS_AFTER_FAIL",
                    "peak_vram": "NOT_REMEASURED_AFTER_FAIL",
                    "cpu_usage": "NOT_REMEASURED_AFTER_FAIL",
                }
            )
    atomic_csv(OUTPUT / "latency/FULL_PIPELINE_LATENCY.csv", pd.DataFrame(latency))

    summary = {
        "protocol_id": gate["protocol_id"],
        "status": "DEVELOPMENT_FAIL",
        "implementation_commit": lock["implementation_commit"],
        "recall_signal": {
            row["tracker"]: row["deltas"]["recall"] for row in gate["decisions"]
        },
        "relative_FN_reduction": {
            row["tracker"]: row["deltas"]["relative_FN_reduction"]
            for row in gate["decisions"]
        },
        "relative_false_alarm_increase": {
            row["tracker"]: row["deltas"]["relative_false_alarm_increase"]
            for row in gate["decisions"]
        },
        "delta_F1": {
            row["tracker"]: row["deltas"]["F1"] for row in gate["decisions"]
        },
        "bootstrap_recall_CI": {
            row.tracker: [float(row.ci_low), float(row.ci_high)]
            for row in bootstrap[
                bootstrap["metric"].eq("recall")
            ].itertuples(index=False)
        },
        "full_development": "BLOCKED_BY_TWO_FOLD_TRIAGE_FAIL",
        "test_status": "SEALED",
        "test_access_count": 0,
        "v8b_modified": False,
        "positive_article": "BLOCKED",
    }
    atomic_json(OUTPUT / "FINAL_DEVELOPMENT_SUMMARY.json", summary)
    report = [
        "# Railway Person Temporal Safety v1 — Development Result",
        "",
        "Status: `DEVELOPMENT_FAIL`",
        "",
        "Both causal trackers recovered additional true person detections, but neither",
        "satisfied the prospectively frozen false-alarm and F1 constraints.",
        "",
        "## Frozen two-fold gate",
        "",
        "| Tracker | Fold-macro ΔRecall | FN/frame reduction | False-alarm increase | ΔF1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for decision in gate["decisions"]:
        delta = decision["deltas"]
        report.append(
            f"| {decision['tracker']} | {delta['recall']:+.4f} | "
            f"{delta['relative_FN_reduction']:+.2%} | "
            f"{delta['relative_false_alarm_increase']:+.2%} | "
            f"{delta['F1']:+.4f} |"
        )
    report.extend(
        [
            "",
            "Recall and FN/frame met their triage conditions. False alarms/min and F1",
            "failed for both trackers, so no winner was selected.",
            "",
            "## Paired scene bootstrap",
            "",
            "| Tracker | Scene-macro ΔRecall | 95% CI | Scene-macro ΔFN/frame | 95% CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for tracker in ("bytetrack", "ocsort"):
        recall_row = bootstrap[
            bootstrap["tracker"].eq(tracker) & bootstrap["metric"].eq("recall")
        ].iloc[0]
        fn_row = bootstrap[
            bootstrap["tracker"].eq(tracker)
            & bootstrap["metric"].eq("FN_per_frame")
        ].iloc[0]
        report.append(
            f"| {tracker} | {recall_row['estimate']:+.4f} | "
            f"[{recall_row['ci_low']:+.4f}, {recall_row['ci_high']:+.4f}] | "
            f"{fn_row['estimate']:+.4f} | "
            f"[{fn_row['ci_low']:+.4f}, {fn_row['ci_high']:+.4f}] |"
        )
    report.extend(
        [
            "",
            "The positive Recall signal is real on these six development scenes, but it",
            "is operationally unusable under the frozen false-alarm constraint.",
            "",
            "## Runtime",
            "",
            "Cached detector mean latency plus measured tracker overhead was 115.9–156.7",
            "ms/frame across the two folds. Full-pipeline p95, VRAM, and CPU were not",
            "remeasured after the gate FAIL; no value is imputed.",
            "",
            "## Scope and integrity",
            "",
            "- Tracker parameters used nine support scenes excluding folds 0 and 1.",
            "- ByteTrack and OC-SORT consumed identical frozen proposals per fold.",
            "- Test access count remained zero.",
            "- V8b was not modified and T-norms were out of scope.",
            "- Full development, a positive article, and test evaluation are blocked.",
        ]
    )
    final_report = OUTPUT / "reports/FINAL_DEVELOPMENT_REPORT.md"
    final_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    files = [
        CONFIG,
        PROJECT / "protocol/temporal_safety_v1/PROTOCOL.md",
        PROJECT / "protocol/v9/V9_CLOSURE.json",
        OUTPUT / "protocol/protocol_lock.json",
        OUTPUT / "data/SEQUENCE_INDEX_AUDIT.json",
        OUTPUT / "baseline/RAW_PREDICTIONS_AUDIT.json",
        OUTPUT / "selection/TRACKER_GRID_RESULTS.csv",
        OUTPUT / "selection/SELECTED_TRACKER_CONFIGS.json",
        gate_path,
        OUTPUT / "triage/PER_SCENE_RESULTS.csv",
        OUTPUT / "statistics/PAIRED_SCENE_BOOTSTRAP.csv",
        OUTPUT / "statistics/LOSO_SENSITIVITY.csv",
        OUTPUT / "statistics/PER_SCENE_EFFECTS.csv",
        OUTPUT / "latency/FULL_PIPELINE_LATENCY.csv",
        OUTPUT / "decision_trace.json",
        OUTPUT / "development/DEVELOPMENT_GATE.json",
        OUTPUT / "reports/TEMPORAL_SAFETY_DEVELOPMENT_REPORT.md",
        final_report,
        OUTPUT / "FINAL_DEVELOPMENT_SUMMARY.json",
        PROJECT / "src/temporal_safety/tracker_base.py",
        PROJECT / "src/temporal_safety/bytetrack_adapter.py",
        PROJECT / "src/temporal_safety/ocsort_adapter.py",
        PROJECT / "src/temporal_safety/evaluator.py",
        PROJECT / "src/temporal_safety/metrics.py",
        PROJECT / "scripts/temporal_safety/run_tracker_triage.py",
        PROJECT / "tests/test_temporal_safety_v1.py",
        PROJECT / "tests/test_temporal_safety_finalization.py",
    ]
    build_bundle(files)


if __name__ == "__main__":
    main()
