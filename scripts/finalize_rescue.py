from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rescue_common import (
    PROJECT_DIR,
    OUTPUT_ROOT,
    atomic_json,
    environment_snapshot,
    load_protocol,
    now,
    run_text,
    sha256,
)


COMPARISON = OUTPUT_ROOT / "comparison"
FINAL = OUTPUT_ROOT / "final"
POLICY = PROJECT_DIR / "configs/rescue/candidate_execution_policy.yaml"


def result_paths() -> list[Path]:
    return sorted(OUTPUT_ROOT.glob("runs/R[1-4]/seed_*/evaluation/candidate_result.json"))


def collect_results() -> pd.DataFrame:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths()]
    if not rows:
        raise RuntimeError("No trained rescue candidate results are available")
    return pd.DataFrame(rows)


def finalist_ranking(frame: pd.DataFrame) -> list[str]:
    initial_seed = int(load_protocol()["selection"]["initial_seed"])
    initial = frame[frame["seed"] == initial_seed].copy()
    initial = initial.sort_values(
        [
            "scene_macro_map50", "scene_macro_safety_recall",
            "validation_safety_fn_per_frame", "scene_map50_std",
        ],
        ascending=[False, False, True, True],
    )
    return initial["candidate"].drop_duplicates().head(2).tolist()


def summarize_seeds(frame: pd.DataFrame, finalists: list[str]) -> pd.DataFrame:
    metrics = [
        "validation_mAP50", "validation_safety_recall",
        "scene_macro_map50", "scene_macro_safety_recall",
        "validation_safety_fn_per_frame", "minimum_scene_safety_recall",
    ]
    rows = []
    for candidate in finalists:
        scope = frame[frame["candidate"] == candidate]
        row: dict[str, Any] = {
            "candidate": candidate, "seed_count": len(scope),
            "seeds": "|".join(map(str, sorted(scope["seed"].astype(int)))),
        }
        for metric in metrics:
            values = scope[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_worst"] = (
                float(values.max()) if metric == "validation_safety_fn_per_frame"
                else float(values.min())
            )
        rows.append(row)
    return pd.DataFrame(rows)


def select_winner(frame: pd.DataFrame, summary: pd.DataFrame) -> tuple[str, pd.Series, pd.Series]:
    protocol = load_protocol()
    required_seeds = len(protocol["selection"]["finalist_seeds"])
    eligible = summary[summary["seed_count"] == required_seeds].copy()
    if eligible.empty:
        raise RuntimeError("Finalists do not have all prospectively frozen seeds")
    eligible = eligible.sort_values(
        [
            "scene_macro_map50_median",
            "scene_macro_safety_recall_median",
            "validation_safety_fn_per_frame_median",
            "scene_macro_map50_std",
            "scene_macro_safety_recall_std",
        ],
        ascending=[False, False, True, True, True],
    )
    aggregate = eligible.iloc[0]
    candidate = str(aggregate["candidate"])
    scope = frame[frame["candidate"] == candidate].copy()
    median = float(aggregate["scene_macro_map50_median"])
    scope["distance_to_median"] = (scope["scene_macro_map50"] - median).abs()
    selected = scope.sort_values(
        ["distance_to_median", "scene_macro_safety_recall", "validation_safety_fn_per_frame"],
        ascending=[True, False, True],
    ).iloc[0]
    return candidate, selected, aggregate


def copy_thresholds(selected: pd.Series) -> Path:
    source = (
        OUTPUT_ROOT / "runs" / str(selected["candidate"])
        / f"seed_{int(selected['seed'])}" / "evaluation/threshold_selection.json"
    )
    destination = FINAL / "threshold_selection.json"
    shutil.copy2(source, destination)
    return destination


def build_report(
    quality_gate: dict[str, Any], frame: pd.DataFrame, seed_summary: pd.DataFrame
) -> str:
    audit = json.loads((OUTPUT_ROOT / "audit/split_audit.json").read_text(encoding="utf-8"))
    visual = json.loads((OUTPUT_ROOT / "audit/visual_audit.json").read_text(encoding="utf-8"))
    micro = json.loads((OUTPUT_ROOT / "micro_overfit/result.json").read_text(encoding="utf-8"))
    evaluator = json.loads((OUTPUT_ROOT / "evaluator/evaluator_audit.json").read_text(encoding="utf-8"))
    lines = [
        "# Canonical v2 Rescue v1 report", "",
        "## 1. Исходное состояние", "",
        "Canonical v2 был остановлен на validation quality gate; исходный failed-run сохранён.",
        "",
        "## 2. Зафиксированный rescue-протокол", "",
        f"- protocol: `{quality_gate['protocol_id']}`",
        f"- split SHA-256: `{quality_gate['split_manifest_sha256']}`",
        "- test usage before gate: forbidden",
        "",
        "## 3. Проверка split", "",
        f"- status: {audit['status']}",
        f"- grouped scenes: {audit['scene_counts']}",
        f"- exact/near cross-split duplicates: {audit['cross_split_exact_duplicates']}/"
        f"{audit['cross_split_near_duplicate_pairs']}",
        "",
        "## 4. Проверка class mapping", "",
        f"- consistent: {audit['class_mapping_consistent']}",
        "",
        "## 5. Проверка разметки", "",
        f"- fatal bbox errors: {audit['fatal_bbox_errors']}",
        f"- confirmed repaired train duplicate: {visual['confirmed_annotation_errors']}",
        "",
        "## 6. Micro-overfit", "",
        f"- status: {micro['status']}",
        f"- mAP50/Recall: {micro['mAP50']:.6f}/{micro['recall']:.6f}",
        "",
        "## 7. Проверка evaluator", "",
        f"- status: {evaluator['status']}",
        "",
        "## 8. Диагностика исходного checkpoint", "",
        "См. `current_checkpoint/summary.json` и `error_taxonomy.csv`.",
        "",
        "## 9. Описание кандидатов", "",
        "R0 pretrained reference; R1 conservative 640; R2 staged 960; "
        "R3 conditional 1280; R4 conditional balanced sampling.",
        "",
        "## 10. Результаты по seeds", "",
        seed_summary.to_markdown(index=False),
        "",
        "## 11. Результаты по классам", "",
        "См. candidate evaluation `ap_by_class.csv`.",
        "",
        "## 12. Результаты по сценам", "",
        "См. `comparison/per_scene_comparison.csv`.",
        "",
        "## 13. Результаты по размерам объектов", "",
        "См. candidate evaluation `clean_metrics_by_class_and_size.csv`.",
        "",
        "## 14. Threshold calibration", "",
        "Standard=max F1; safety=max F2; validation only.",
        "",
        "## 15. Выбор checkpoint", "",
        f"- candidate: {quality_gate.get('candidate')}",
        f"- seed: {quality_gate.get('seed')}",
        f"- checkpoint SHA-256: `{quality_gate.get('checkpoint_sha256')}`",
        "",
        "## 16. Quality gate", "",
        f"- validation mAP50: {quality_gate['validation_map50']:.6f}",
        f"- validation safety Recall: {quality_gate['validation_safety_recall']:.6f}",
        f"- passed: {quality_gate['quality_gate_passed']}",
        "",
        "## 17. Разрешённые и skipped стадии", "",
        (
            "Canonical test/attacks разрешены только через физический rescue_gate."
            if quality_gate["quality_gate_passed"]
            else "Test, attacks and article finalization remain skipped."
        ),
        "",
        "## 18. Ограничения", "",
        "Пять validation grouped scenes; редкие классы представлены малым числом сцен.",
        "",
        "## 19. SHA-256 и provenance", "",
        f"- current commit: `{run_text(['git', 'rev-parse', 'HEAD'])}`",
        "",
        "## 20. Команды воспроизведения", "",
        "```bash",
        "PYTHONPATH=scripts:. .venv/bin/python scripts/run_rescue_pipeline.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def build_bundle(passed: bool, failure_stage: str | None = None) -> Path:
    name = (
        "TNormFilter_baseline_rescue_v1_passed.zip"
        if passed else "TNormFilter_baseline_rescue_v1_gate_failed.zip"
    )
    destination = OUTPUT_ROOT / "bundles" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    allowed = {
        ".json", ".csv", ".yaml", ".yml", ".txt", ".md", ".png", ".jpg",
    }
    sources = [
        OUTPUT_ROOT,
        PROJECT_DIR / "configs/rescue",
        PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv",
        PROJECT_DIR / "data/yolo_osdar23_rescue_v1/data.yaml",
    ]
    candidates = []
    for root in sources:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            if OUTPUT_ROOT / "bundles" in path.parents:
                continue
            if OUTPUT_ROOT / "micro_overfit/dataset" in path.parents:
                continue
            candidates.append(path)
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        checksums = []
        for source in sorted(set(candidates)):
            relative = source.relative_to(PROJECT_DIR)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            checksums.append(f"{sha256(target)}  {relative.as_posix()}")
        (staging / "checksums.sha256").write_text(
            "\n".join(checksums) + "\n", encoding="utf-8"
        )
        atomic_json(staging / "manifest.json", {
            "protocol_id": load_protocol()["protocol_id"],
            "created_at": now(),
            "git_commit": run_text(["git", "rev-parse", "HEAD"]),
            "git_branch": run_text(["git", "branch", "--show-current"]),
            "quality_gate_passed": passed,
            "quality_gate_evaluated": failure_stage != "micro_overfit",
            "failure_stage": failure_stage,
            "files": len(checksums),
            "weights_included": False,
            "environment": environment_snapshot(),
        })
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging))
    checksum = sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{checksum}  {destination.name}\n", encoding="utf-8"
    )
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Rescue bundle failed CRC verification")
    return destination


def finalize_micro_failure() -> None:
    protocol = load_protocol()
    FINAL.mkdir(parents=True, exist_ok=True)
    COMPARISON.mkdir(parents=True, exist_ok=True)
    micro = json.loads(
        (OUTPUT_ROOT / "micro_overfit/result.json").read_text(encoding="utf-8")
    )
    quality_gate = {
        "protocol_id": protocol["protocol_id"],
        "split_manifest_sha256": protocol["split_manifest_sha256"],
        "quality_gate_passed": False,
        "quality_gate_evaluated": False,
        "blocked_by": "micro_overfit",
        "micro_overfit_status": micro["status"],
        "micro_overfit_mAP50": micro["mAP50"],
        "micro_overfit_recall": micro["recall"],
        "micro_overfit_mAP50_requirement": protocol["micro_overfit"]["map50_min"],
        "micro_overfit_recall_requirement": protocol["micro_overfit"]["recall_min"],
        "test_opened": False,
    }
    decision = {
        "protocol_id": protocol["protocol_id"],
        "winner": None,
        "checkpoint_frozen": False,
        "thresholds_frozen": False,
        "quality_gate_passed": False,
        "quality_gate_evaluated": False,
        "candidate_matrix_status": "skipped_micro_overfit_failure",
        "test_used": False,
    }
    atomic_json(FINAL / "quality_gate.json", quality_gate)
    atomic_json(COMPARISON / "selection_decision.json", decision)
    report = "\n".join([
        "# Canonical v2 Rescue v1 diagnostic report",
        "",
        "## Outcome",
        "",
        "The prospectively frozen micro-overfit gate did not pass. The R0-R4 "
        "candidate matrix, test, attacks, and article finalization were not run.",
        "",
        "## Micro-overfit",
        "",
        f"- frames: {micro['frames']}",
        f"- mAP50: {micro['mAP50']:.6f} (required {protocol['micro_overfit']['map50_min']:.2f})",
        f"- Recall: {micro['recall']:.6f} (required {protocol['micro_overfit']['recall_min']:.2f})",
        f"- F1: {micro['f1']:.6f}",
        f"- initial/final train loss sum: {micro['initial_train_loss_sum']:.6f}/"
        f"{micro['final_train_loss_sum']:.6f}",
        f"- gradient NaN/Inf: {micro['gradient_nan_or_inf']}",
        f"- checkpoint SHA-256: `{micro['checkpoint_sha256']}`",
        "",
        "## Scientific boundary",
        "",
        "- quality gate evaluated: false",
        "- test opened: false",
        "- attacks: skipped",
        "- hypotheses H1-H4: not evaluated",
        "",
        "Detailed class, scene, threshold, gradient, prediction, audit, and "
        "training artifacts are included in this diagnostic archive.",
        "",
    ])
    (FINAL / "rescue_report.md").write_text(report, encoding="utf-8")
    expected_bundle = (
        OUTPUT_ROOT / "bundles" / "TNormFilter_baseline_rescue_v1_gate_failed.zip"
    )
    atomic_json(FINAL / "run_summary.json", {
        "failure_stage": "micro_overfit",
        "micro_overfit": micro,
        "quality_gate": quality_gate,
        "selection": decision,
        "bundle": str(expected_bundle.resolve()),
        "test_opened": False,
        "attacks": "skipped",
        "article_finalization": "skipped",
        "hypotheses_H1_H4": "not_evaluated",
    })
    bundle = build_bundle(False, failure_stage="micro_overfit")
    print(json.dumps({
        "failure_stage": "micro_overfit",
        "bundle": str(bundle),
        "bundle_sha256": sha256(bundle),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-failure", action="store_true")
    args = parser.parse_args()
    if args.micro_failure:
        finalize_micro_failure()
        return
    protocol = load_protocol()
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    COMPARISON.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    frame = collect_results()
    frame.to_csv(COMPARISON / "candidate_summary.csv", index=False)
    finalists = finalist_ranking(frame)
    seed_summary = summarize_seeds(frame, finalists)
    seed_summary.to_csv(COMPARISON / "seed_summary.csv", index=False)
    scene_frames = []
    for path in result_paths():
        result = json.loads(path.read_text(encoding="utf-8"))
        scene = pd.read_csv(path.parent / "metrics_per_scene_with_ap.csv")
        scene["candidate"] = result["candidate"]
        scene["seed"] = result["seed"]
        scene_frames.append(scene)
    pd.concat(scene_frames, ignore_index=True).to_csv(
        COMPARISON / "per_scene_comparison.csv", index=False
    )
    candidate, selected, aggregate = select_winner(frame, seed_summary)
    checkpoint = Path(selected["checkpoint_path"])
    gate = protocol["quality_gate"]
    selected_pass = bool(selected["quality_gate_passed"])
    median_map_pass = (
        float(aggregate["validation_mAP50_median"]) >= float(gate["validation_map50_min"])
    )
    median_recall_pass = (
        float(aggregate["validation_safety_recall_median"])
        >= float(gate["validation_safety_recall_min"])
    )
    quality_gate = {
        "protocol_id": protocol["protocol_id"],
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "split_manifest_sha256": protocol["split_manifest_sha256"],
        "candidate": candidate, "seed": int(selected["seed"]),
        "validation_map50": float(selected["validation_mAP50"]),
        "validation_standard_recall": float(selected["validation_standard_recall"]),
        "validation_safety_recall": float(selected["validation_safety_recall"]),
        "validation_minimum_scene_safety_recall": float(
            selected["minimum_scene_safety_recall"]
        ),
        "median_seed_validation_map50": float(aggregate["validation_mAP50_median"]),
        "median_seed_validation_safety_recall": float(
            aggregate["validation_safety_recall_median"]
        ),
        "map50_requirement": float(gate["validation_map50_min"]),
        "recall_requirement": float(gate["validation_safety_recall_min"]),
        "scene_recall_requirement": float(gate["validation_scene_recall_min"]),
        "selected_checkpoint_passed": selected_pass,
        "median_map50_passed": median_map_pass,
        "median_recall_passed": median_recall_pass,
        "quality_gate_passed": bool(selected_pass and median_map_pass and median_recall_pass),
        "test_opened": False,
    }
    atomic_json(FINAL / "quality_gate.json", quality_gate)
    atomic_json(FINAL / "frozen_checkpoint.json", {
        "status": "FROZEN" if quality_gate["quality_gate_passed"] else "NOT_FROZEN_GATE_FAILED",
        "candidate": candidate, "seed": int(selected["seed"]),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "selection_uses_test": False,
    })
    copy_thresholds(selected)
    decision = {
        "protocol_id": protocol["protocol_id"],
        "finalists": finalists,
        "selection_order": protocol["selection"],
        "winner": candidate, "selected_seed": int(selected["seed"]),
        "quality_gate_passed": quality_gate["quality_gate_passed"],
        "test_used": False,
    }
    atomic_json(COMPARISON / "selection_decision.json", decision)
    report = build_report(quality_gate, frame, seed_summary)
    (FINAL / "rescue_report.md").write_text(report, encoding="utf-8")
    expected_bundle = OUTPUT_ROOT / "bundles" / (
        "TNormFilter_baseline_rescue_v1_passed.zip"
        if quality_gate["quality_gate_passed"]
        else "TNormFilter_baseline_rescue_v1_gate_failed.zip"
    )
    atomic_json(FINAL / "run_summary.json", {
        "quality_gate": quality_gate,
        "selection": decision,
        "bundle": str(expected_bundle.resolve()),
        "test_opened": False,
        "canonical_downstream": (
            "allowed_by_rescue_gate" if quality_gate["quality_gate_passed"]
            else "skipped_baseline_quality_failure"
        ),
    })
    bundle = build_bundle(quality_gate["quality_gate_passed"])
    print(json.dumps({
        "quality_gate": quality_gate,
        "bundle": str(bundle), "bundle_sha256": sha256(bundle),
    }, indent=2))


if __name__ == "__main__":
    main()
