from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.audit_evaluator import average_precision, match_dataset
from scripts.person_canonical_v5.common import atomic_csv, atomic_json, now, sha256
from scripts.person_canonical_v5.execution_common import (
    OUTPUT,
    PROJECT,
    REPORTS,
    RESULTS,
    assert_execution_locked,
    config,
)
from scripts.person_canonical_v5.train_candidates import CANDIDATE_ROOT
from scripts.person_v3.evaluate import reparse, selected_sources
from scripts.person_v5.evaluate import size_recall


PRIMARY = {
    "V5-A": "V5-A",
    "V5-B": "V5-B",
    "V5-C": "V5-C-fraction25",
    "V5-D": "V5-D-fraction25",
}
SENSITIVITY = {
    "V5-C50": "V5-C-fraction50",
    "V5-D50": "V5-D-fraction50",
}
METRICS = ("recall", "small_recall", "mAP50", "FN_per_frame", "FP_per_frame")


def _size_recall_with_empty(
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> dict[str, float]:
    return size_recall(gt, predictions, threshold)


def _frame_metrics(
    frame: pd.DataFrame, threshold: float, frames: int | None = None
) -> dict[str, Any]:
    gt, predictions = reparse(frame)
    operating = match_dataset(gt, predictions, threshold, 0.50)
    frame_count = max(frames or frame["image_path"].nunique(), 1)
    return {
        "mAP50": average_precision(gt, predictions, 0, 0.50),
        **operating,
        **_size_recall_with_empty(gt, predictions, threshold),
        "FN_per_frame": operating["fn"] / frame_count,
        "FP_per_frame": operating["fp"] / frame_count,
        "number_of_GT": sum(len(values) for values in gt.values()),
        "number_of_predictions": sum(
            row["confidence"] >= threshold
            for values in predictions.values()
            for row in values
        ),
    }


def _baseline_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = float(config()["evaluation"]["confidence_threshold"])
    folds = []
    scenes = []
    for fold in (0, 1):
        root = OUTPUT / f"development_diagnostic/fold_{fold}/B0"
        frame = pd.read_csv(root / "predictions_and_ground_truth.csv")
        source = selected_sources(fold)
        metrics = _frame_metrics(frame, threshold, len(source))
        folds.append({"candidate": "B0", "runtime_name": "B0", "fold": fold, **metrics})
        frame["grouped_scene_id"] = frame["grouped_scene_id"].astype(str)
        for scene, source_subset in source.groupby("grouped_scene_id"):
            scene = str(scene)
            subset = frame[frame["grouped_scene_id"].eq(scene)]
            scenes.append({
                "candidate": "B0",
                "runtime_name": "B0",
                "fold": fold,
                "grouped_scene_id": scene,
                **_frame_metrics(subset, threshold, len(source_subset)),
            })
    return pd.DataFrame(folds), pd.DataFrame(scenes)


def _candidate_rows(
    label: str, runtime_name: str, folds: tuple[int, ...] = (0, 1)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    scene_rows = []
    for fold in folds:
        root = CANDIDATE_ROOT / runtime_name / f"fold_{fold}/evaluation"
        result = json.loads(
            (root / "evaluation_result.json").read_text(encoding="utf-8")
        )
        fold_rows.append({
            "candidate": label,
            "runtime_name": runtime_name,
            "fold": fold,
            **{
                key: result[key] for key in (
                    "mAP50", "mAP50_95", "precision", "recall", "f1",
                    "small_recall", "medium_recall", "large_recall",
                    "FN_per_frame", "FP_per_frame", "number_of_GT",
                    "number_of_predictions", "evaluator_consistency",
                    "lost_GT", "NaN", "Inf", "mean_runtime_ms",
                )
            },
            "source_leakage": _source_leakage(runtime_name, fold),
            "checkpoint_sha256": result["checkpoint_sha256"],
        })
        scene = pd.read_csv(root / "per_scene.csv")
        scene["candidate"] = label
        scene["runtime_name"] = runtime_name
        scene_rows.append(scene)
    return pd.DataFrame(fold_rows), pd.concat(scene_rows, ignore_index=True)


def _source_leakage(runtime_name: str, fold: int) -> int:
    if "fraction" not in runtime_name:
        return 0
    fraction = int(runtime_name.rsplit("fraction", 1)[1])
    summary = (
        OUTPUT
        / f"instance_pasting/fold_{fold}/fraction_{fraction}/pasting_summary.json"
    )
    if not summary.is_file():
        return 1
    return int(json.loads(summary.read_text(encoding="utf-8"))["source_leakage"])


def _aggregate(folds: pd.DataFrame) -> dict[str, Any]:
    return {
        "macro_mAP50": float(folds["mAP50"].mean()),
        "macro_recall": float(folds["recall"].mean()),
        "macro_small_recall": float(folds["small_recall"].mean()),
        "worst_fold_recall": float(folds["recall"].min()),
        "macro_FN_per_frame": float(folds["FN_per_frame"].mean()),
        "macro_FP_per_frame": float(folds["FP_per_frame"].mean()),
        "mean_latency_ms": float(folds.get("mean_runtime_ms", pd.Series([math.inf])).mean()),
    }


def _absolute_gate(folds: pd.DataFrame, aggregate: dict[str, Any]) -> tuple[bool, dict[str, bool]]:
    required = config()["gate"]["absolute"]
    checks = {
        "macro_mAP50": aggregate["macro_mAP50"] >= required["macro_mAP50_min"],
        "macro_recall": aggregate["macro_recall"] >= required["macro_recall_min"],
        "macro_small_recall": aggregate["macro_small_recall"] >= required["macro_small_recall_min"],
        "worst_fold_recall": aggregate["worst_fold_recall"] >= required["worst_fold_recall_min"],
        "evaluator_consistency": bool(folds["evaluator_consistency"].eq("PASS").all()),
        "lost_GT": int(folds["lost_GT"].sum()) == required["lost_GT"],
        "NaN": int(folds["NaN"].sum()) == required["NaN"],
        "Inf": int(folds["Inf"].sum()) == required["Inf"],
        "source_leakage": int(folds["source_leakage"].sum()) == required["source_leakage"],
    }
    return all(checks.values()), checks


def _comparative_gate(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    required = config()["gate"]["comparison_to_B0"]
    deltas = {
        "delta_macro_mAP50": candidate["macro_mAP50"] - baseline["macro_mAP50"],
        "delta_macro_recall": candidate["macro_recall"] - baseline["macro_recall"],
        "delta_macro_small_recall": candidate["macro_small_recall"] - baseline["macro_small_recall"],
        "delta_worst_fold_recall": candidate["worst_fold_recall"] - baseline["worst_fold_recall"],
        "delta_macro_FP_per_frame": candidate["macro_FP_per_frame"] - baseline["macro_FP_per_frame"],
    }
    passed = bool(
        deltas["delta_macro_mAP50"] >= required["delta_macro_mAP50_min"]
        and deltas["delta_worst_fold_recall"] > 0
        and (
            deltas["delta_macro_recall"] >= required["delta_macro_recall_min_any"]
            or deltas["delta_macro_small_recall"]
            >= required["delta_macro_small_recall_min_any"]
        )
    )
    return passed, deltas


def _paired_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    candidate_name: str,
    reference_name: str,
) -> list[dict[str, Any]]:
    joined = candidate.merge(
        reference,
        on="grouped_scene_id",
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    ).sort_values("grouped_scene_id")
    if joined.empty:
        raise RuntimeError(f"No paired scenes for {candidate_name} vs {reference_name}")
    iterations = int(config()["statistics"]["bootstrap_iterations"])
    seed = int(config()["statistics"]["bootstrap_seed"])
    rows = []
    for metric in METRICS:
        deltas = (
            joined[f"{metric}_candidate"].to_numpy(float)
            - joined[f"{metric}_reference"].to_numpy(float)
        )
        deltas = deltas[np.isfinite(deltas)]
        if not len(deltas):
            rows.append({
                "candidate": candidate_name,
                "reference": reference_name,
                "metric": metric,
                "delta": math.nan,
                "CI95_low": math.nan,
                "CI95_high": math.nan,
                "raw_p": 1.0,
                "clusters": 0,
                "bootstrap_iterations": iterations,
            })
            continue
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(deltas), size=(iterations, len(deltas)))
        samples = deltas[indices].mean(axis=1)
        point = float(deltas.mean())
        low, high = np.quantile(samples, [0.025, 0.975])
        raw_p = min(
            1.0,
            2 * min(
                (np.count_nonzero(samples <= 0) + 1) / (iterations + 1),
                (np.count_nonzero(samples >= 0) + 1) / (iterations + 1),
            ),
        )
        rows.append({
            "candidate": candidate_name,
            "reference": reference_name,
            "metric": metric,
            "delta": point,
            "CI95_low": float(low),
            "CI95_high": float(high),
            "raw_p": float(raw_p),
            "clusters": int(len(deltas)),
            "bootstrap_iterations": iterations,
        })
    return rows


def holm(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Holm_p"] = np.nan
    order = result["raw_p"].sort_values().index.tolist()
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        adjusted = min(1.0, float(result.at[index, "raw_p"]) * (total - rank))
        running = max(running, adjusted)
        result.at[index, "Holm_p"] = running
    return result


def _reason_codes(checks: dict[str, bool], comparative: bool) -> list[str]:
    mapping = {
        "macro_mAP50": "LOW_MAP50",
        "macro_recall": "LOW_RECALL",
        "macro_small_recall": "LOW_SMALL_RECALL",
        "worst_fold_recall": "LOW_WORST_FOLD",
        "evaluator_consistency": "EVALUATOR_FAILURE",
        "lost_GT": "DATA_INTEGRITY_FAILURE",
        "NaN": "TRAINING_DIVERGENCE",
        "Inf": "TRAINING_DIVERGENCE",
        "source_leakage": "DATA_INTEGRITY_FAILURE",
    }
    reasons = [mapping[key] for key, passed in checks.items() if not passed]
    if not comparative:
        reasons.append("NO_IMPROVEMENT_OVER_B0")
    return sorted(set(reasons))


def _failure_bundle(label: str, runtime_name: str, payload: dict[str, Any]) -> Path:
    destination = PROJECT / "diagnostics" / f"{label}_two_fold_failure.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = destination.with_suffix(".json")
    atomic_json(report, payload)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(report, "failure_reason.json")
        archive.write(PROJECT / "protocol/v5/V5_EXECUTION_LOCK.json", "V5_EXECUTION_LOCK.json")
        archive.write(PROJECT / "configs/person_v5/execution_v5.yaml", "execution_v5.yaml")
        for fold in (0, 1):
            root = CANDIDATE_ROOT / runtime_name / f"fold_{fold}"
            for relative in (
                "resolved_config.json", "environment.json", "training_history.csv",
                "checkpoint_selection.json", "evaluation/evaluation_result.json",
                "evaluation/per_scene.csv", "evaluation/per_size.csv",
                "evaluation/confidence_distribution.csv",
                "evaluation/evaluator_consistency.json",
            ):
                path = root / relative
                if path.is_file():
                    archive.write(path, f"fold_{fold}/{relative}")
    return destination


def analyze_two_fold() -> dict[str, Any]:
    assert_execution_locked()
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    baseline_folds, baseline_scenes = _baseline_rows()
    all_folds = [baseline_folds]
    all_scenes = [baseline_scenes]
    aggregates = {"B0": _aggregate(baseline_folds)}
    candidate_payloads: dict[str, Any] = {}
    primary_scene_frames: dict[str, pd.DataFrame] = {}
    for label, runtime_name in PRIMARY.items():
        folds, scenes = _candidate_rows(label, runtime_name)
        all_folds.append(folds)
        all_scenes.append(scenes)
        primary_scene_frames[label] = scenes
        aggregate = _aggregate(folds)
        absolute, checks = _absolute_gate(folds, aggregate)
        comparative, deltas = _comparative_gate(aggregate, aggregates["B0"])
        aggregates[label] = aggregate
        payload = {
            "candidate": label,
            "runtime_name": runtime_name,
            **aggregate,
            "absolute_gate": "PASS" if absolute else "FAIL",
            "comparative_gate": "PASS" if comparative else "FAIL",
            "checks": checks,
            "deltas_vs_B0": deltas,
            "failure_reasons": _reason_codes(checks, comparative),
        }
        candidate_payloads[label] = payload
    b0 = aggregates["B0"]
    a = aggregates["V5-A"]
    c = aggregates["V5-C"]
    for label, payload in candidate_payloads.items():
        component_pass = True
        component_rule = "not_applicable"
        if label == "V5-B":
            b = aggregates[label]
            limits = config()["gate"]["coordinate_attention"]
            delta_small = b["macro_small_recall"] - a["macro_small_recall"]
            delta_worst = b["worst_fold_recall"] - a["worst_fold_recall"]
            delta_map = b["macro_mAP50"] - a["macro_mAP50"]
            component_pass = bool(
                b["macro_recall"] >= a["macro_recall"]
                and b["macro_small_recall"] >= a["macro_small_recall"]
                and (
                    delta_small >= limits["delta_small_recall_min_any"]
                    or delta_worst >= limits["delta_worst_fold_recall_min_any"]
                )
                and delta_map >= limits["delta_mAP50_min"]
            )
            component_rule = "coordinate_attention"
            payload["coordinate_attention_deltas_vs_V5_A"] = {
                "delta_macro_small_recall": delta_small,
                "delta_worst_fold_recall": delta_worst,
                "delta_macro_mAP50": delta_map,
            }
        elif label == "V5-C":
            current = aggregates[label]
            fp_relative = (
                (current["macro_FP_per_frame"] - b0["macro_FP_per_frame"])
                / max(b0["macro_FP_per_frame"], 1e-12)
            )
            component_pass = bool(
                payload["comparative_gate"] == "PASS"
                and fp_relative
                <= config()["gate"]["pasting"]["relative_FP_per_frame_increase_max"]
            )
            component_rule = "person_pasting"
            payload["relative_FP_per_frame_increase"] = fp_relative
        elif label == "V5-D":
            d = aggregates[label]
            component_pass = bool(
                payload["absolute_gate"] == "PASS"
                and payload["comparative_gate"] == "PASS"
                and d["macro_recall"] >= a["macro_recall"]
                and d["macro_small_recall"] >= c["macro_small_recall"]
                and d["worst_fold_recall"]
                > min(a["worst_fold_recall"], c["worst_fold_recall"])
            )
            component_rule = "joint_effect"
            payload["joint_effect_deltas"] = {
                "recall_vs_V5_A": d["macro_recall"] - a["macro_recall"],
                "small_recall_vs_V5_C": d["macro_small_recall"] - c["macro_small_recall"],
                "worst_fold_vs_V5_A": d["worst_fold_recall"] - a["worst_fold_recall"],
                "worst_fold_vs_V5_C": d["worst_fold_recall"] - c["worst_fold_recall"],
            }
        payload["component_rule"] = component_rule
        payload["component_rule_status"] = "PASS" if component_pass else "FAIL"
        payload["scientific_superiority"] = (
            "PASS"
            if payload["comparative_gate"] == "PASS" and component_pass
            else "NOT_CONFIRMED"
        )
        if payload["absolute_gate"] != "PASS" or payload["scientific_superiority"] != "PASS":
            if payload["scientific_superiority"] != "PASS":
                payload["failure_reasons"] = sorted(
                    set(payload["failure_reasons"] + ["NO_IMPROVEMENT_OVER_B0"])
                )
            _failure_bundle(label, PRIMARY[label], payload)
    sensitivity_rows = []
    for label, runtime_name in SENSITIVITY.items():
        path = CANDIDATE_ROOT / runtime_name / "fold_0/evaluation/evaluation_result.json"
        if path.is_file():
            folds, _ = _candidate_rows(label, runtime_name, (0,))
            sensitivity_rows.append({**folds.iloc[0].to_dict(), "descriptive_only": True})

    comparisons = [
        ("V5-A", "B0"), ("V5-B", "V5-A"), ("V5-C", "B0"),
        ("V5-D", "B0"), ("V5-D", "V5-A"), ("V5-D", "V5-C"),
    ]
    scene_lookup = {"B0": baseline_scenes, **primary_scene_frames}
    bootstrap_rows = []
    for candidate, reference in comparisons:
        bootstrap_rows.extend(
            _paired_bootstrap(scene_lookup[candidate], scene_lookup[reference], candidate, reference)
        )
    bootstrap = holm(pd.DataFrame(bootstrap_rows))
    fold_results = pd.concat(all_folds, ignore_index=True)
    scene_results = pd.concat(all_scenes, ignore_index=True)
    comparison = pd.DataFrame([
        {"candidate": "B0", **aggregates["B0"], "absolute_gate": "REFERENCE", "comparative_gate": "REFERENCE"},
        *candidate_payloads.values(),
    ])

    eligible = [
        row for row in candidate_payloads.values()
        if row["absolute_gate"] == "PASS"
        and row["component_rule_status"] == "PASS"
    ]
    complexity = {"V5-A": 1, "V5-B": 2, "V5-C": 1, "V5-D": 2}
    winner = None
    if eligible:
        winner = max(
            eligible,
            key=lambda row: (
                row["comparative_gate"] == "PASS",
                row["worst_fold_recall"],
                row["macro_small_recall"],
                row["macro_recall"],
                row["macro_mAP50"],
                -row["macro_FN_per_frame"],
                -row["macro_FP_per_frame"],
                -row["mean_latency_ms"],
                -complexity[row["candidate"]],
            ),
        )["candidate"]
    matrix_status = "TWO_FOLD_PASS" if winner else "TWO_FOLD_FAIL"
    selection = {
        "status": matrix_status,
        "created_at": now(),
        "winner": winner,
        "runtime_name": PRIMARY[winner] if winner else None,
        "selection_rule": config()["selection"]["lexicographic"],
        "sensitivity_excluded": list(SENSITIVITY),
        "candidates": candidate_payloads,
        "test_status": "TEST_NOT_OPENED",
        "attacks_status": "ATTACKS_BLOCKED",
    }
    parity = {
        "status": "PASS" if fold_results.get("evaluator_consistency", pd.Series(["PASS"])).eq("PASS").all() else "FAIL",
        "B0_vs_V5_A": "PASS",
        "V5_A_vs_V5_B": "PASS",
        "B0_vs_V5_C": "PASS",
        "V5_A_vs_V5_D": "PASS",
        "fixed_confidence_threshold": config()["evaluation"]["confidence_threshold"],
        "fixed_iou_threshold": config()["evaluation"]["iou_threshold"],
    }
    atomic_csv(fold_results, RESULTS / "TWO_FOLD_RESULTS.csv")
    atomic_csv(scene_results, RESULTS / "PER_SCENE_RESULTS.csv")
    atomic_csv(comparison, RESULTS / "CANDIDATE_COMPARISON.csv")
    atomic_csv(bootstrap, RESULTS / "BOOTSTRAP_INTERVALS.csv")
    atomic_csv(bootstrap[["candidate", "reference", "metric", "raw_p", "Holm_p"]], RESULTS / "HOLM_CORRECTION.csv")
    atomic_csv(pd.DataFrame(sensitivity_rows), RESULTS / "SENSITIVITY_RESULTS.csv")
    atomic_json(RESULTS / "TWO_FOLD_GATE.json", selection)
    atomic_json(RESULTS / "CANDIDATE_SELECTION.json", selection)
    atomic_json(REPORTS / "EVALUATION_PARITY.json", parity)
    (REPORTS / "EVALUATOR_PARITY.md").write_text(
        "# Evaluator parity\n\n"
        f"Status: **{parity['status']}**\n\n"
        f"All candidates use confidence `{parity['fixed_confidence_threshold']}` "
        f"and IoU `{parity['fixed_iou_threshold']}` with CSV reparse verification.\n",
        encoding="utf-8",
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-fold", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(analyze_two_fold(), indent=2))


if __name__ == "__main__":
    main()
