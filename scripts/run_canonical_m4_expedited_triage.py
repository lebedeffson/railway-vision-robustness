from __future__ import annotations

import json
import math
import os
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from canonical_m4_common import PROJECT_DIR, atomic_json, now, sha256


PYTHON = PROJECT_DIR / ".venv/bin/python"
FULL_ROOT = PROJECT_DIR / "outputs/canonical_m4"
FOLD_ROOT = FULL_ROOT / "scene_cv/fold_0/seed_20260722"
TRAINING_MARKER = FOLD_ROOT / "TRAINING_COMPLETE.json"
EVALUATION_ROOT = FOLD_ROOT / "evaluation"
TRIAGE_ROOT = FULL_ROOT / "expedited/triage"
PROTOCOL_PATH = PROJECT_DIR / "configs/canonical_v2_m4_expedited_protocol.yaml"
TEST_MARKER = FULL_ROOT / "final/TEST_OPENED.json"
FULL_STATUS = FULL_ROOT / "final/pre_gate_pipeline_status.json"
EXPEDITED_STATUS = FULL_ROOT / "expedited/pipeline_status.json"
INTERNAL_ARTICLE = (
    FULL_ROOT / "expedited/TNorm_RZD_article_internal_triage.md"
)


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def classify(
    evaluation: dict[str, Any],
    per_scene: pd.DataFrame,
    lost_gt: int,
    protocol: dict[str, Any],
) -> tuple[str, list[str]]:
    triage = protocol["early_triage"]
    reasons: list[str] = []
    finite_values = [
        evaluation.get("mAP50"),
        evaluation.get("mAP50_95"),
        evaluation.get("safety_precision"),
        evaluation.get("safety_recall"),
        evaluation.get("safety_f1"),
    ]
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in finite_values
    ):
        reasons.append("non_finite_metric")
    if float(evaluation.get("mAP50", -math.inf)) < float(triage["map50_min"]):
        reasons.append("map50_below_triage_threshold")
    if float(evaluation.get("safety_recall", -math.inf)) < float(
        triage["recall_min"]
    ):
        reasons.append("recall_below_triage_threshold")
    if not bool(evaluation.get("evaluator_consistency_passed")):
        reasons.append("evaluator_consistency_failed")
    if not bool(evaluation.get("no_missing_scene")):
        reasons.append("missing_held_out_scene")
    if int(lost_gt) != 0:
        reasons.append("lost_ground_truth")
    if not bool(evaluation.get("no_nan_or_inf")):
        reasons.append("nan_or_inf_detected")
    scene_gt = per_scene["tp"].astype(float) + per_scene["fn"].astype(float)
    catastrophic = (
        per_scene.empty
        or (
            scene_gt.gt(0)
            & per_scene["recall"].astype(float).le(
                float(triage["catastrophic_scene_failure"]["threshold"])
            )
        ).any()
    )
    if catastrophic:
        reasons.append("catastrophic_scene_failure")
    return ("hard_fail" if reasons else "promising"), reasons


def stop_full_service() -> None:
    subprocess.run(
        ["systemctl", "--user", "stop", "tnorm-canonical-m4.service"],
        check=True,
    )
    subprocess.run(
        ["systemctl", "--user", "disable", "tnorm-canonical-m4.service"],
        check=False,
    )


def mark_full_protocol_partial(status: str) -> None:
    payload = (
        json.loads(FULL_STATUS.read_text(encoding="utf-8"))
        if FULL_STATUS.is_file()
        else {"protocol_id": "canonical-v2-m4-full-v1", "stages": {}}
    )
    payload["updated_at"] = now()
    payload["current_stage"] = "expedited_triage"
    payload["expedited_triage_status"] = status
    payload["stages"]["scene_cv_fold_0"] = {
        "status": "completed_for_expedited_triage",
        "evaluation": str(
            (EVALUATION_ROOT / "evaluation_result.json").resolve()
        ),
        "canonical_claim_role": "none",
    }
    for fold in range(1, 5):
        payload["stages"][f"scene_cv_fold_{fold}"] = {
            "status": "skipped_by_expedited_triage"
        }
    for seed in (20260722, 20260723, 20260724):
        payload["stages"][f"full_training_seed_{seed}"] = {
            "status": (
                "moved_to_expedited_protocol"
                if seed == 20260722 and status == "promising"
                else "skipped_by_expedited_triage"
            )
        }
    atomic_json(FULL_STATUS, payload)


def diagnostic_bundle(report: dict[str, Any]) -> Path:
    destination = (
        FULL_ROOT / "bundles/TNormFilter_M4_expedited_triage_hard_fail.zip"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    paths = [
        PROTOCOL_PATH,
        TRIAGE_ROOT / "early_generalization_report.json",
        TRIAGE_ROOT / "early_generalization_report.md",
        EXPEDITED_STATUS,
        INTERNAL_ARTICLE,
        EVALUATION_ROOT / "evaluation_result.json",
        EVALUATION_ROOT / "evaluator_consistency.json",
        EVALUATION_ROOT / "per_scene_metrics.csv",
        EVALUATION_ROOT / "per_class_metrics.csv",
        EVALUATION_ROOT / "per_size_metrics.csv",
        FOLD_ROOT / "TRAINING_COMPLETE.json",
        FULL_STATUS,
        PROJECT_DIR / "outputs/canonical_m4/tiling_audit/tiling_audit.json",
    ]
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file():
                archive.write(path, path.relative_to(PROJECT_DIR))
        archive.writestr(
            "run_summary.json",
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        )
    os.replace(temporary, destination)
    return destination


def write_hard_fail_outputs(report: dict[str, Any]) -> None:
    atomic_json(
        EXPEDITED_STATUS,
        {
            "protocol_id": report["protocol_id"],
            "status": "stopped_after_fold_0_hard_fail",
            "updated_at": now(),
            "test_opened": False,
            "stages": {
                "fold_0_training": {"status": "success"},
                "fold_0_independent_evaluation": {"status": "success"},
                "early_generalization_triage": {"status": "hard_fail"},
                "remaining_scene_cv_folds": {
                    "status": "skipped_by_expedited_triage"
                },
                "full_training": {"status": "skipped_by_expedited_triage"},
                "official_validation": {"status": "skipped_by_expedited_triage"},
                "clean_test": {"status": "skipped_by_expedited_triage"},
                "FGSM": {"status": "skipped_by_expedited_triage"},
                "PGD": {"status": "skipped_by_expedited_triage"},
                "adaptive_PGD": {"status": "skipped_by_expedited_triage"},
                "D2_D3": {"status": "skipped_by_expedited_triage"},
                "R2_R3": {"status": "skipped_by_expedited_triage"},
                "final_article": {"status": "blocked_negative_internal_only"},
            },
        },
    )
    INTERNAL_ARTICLE.parent.mkdir(parents=True, exist_ok=True)
    INTERNAL_ARTICLE.write_text(
        "\n".join([
            "# TNormFilter M4: внутренняя отрицательная редакция",
            "",
            "M4 overlapping tiling прошёл диагностический micro-overfit, "
            "что подтвердило техническую обучаемость малых объектов на "
            "фиксированном micro-наборе. Это не являлось оценкой обобщения.",
            "",
            "На независимых train-сценах fold 0 глобальный fused evaluator "
            f"получил mAP50={report['mAP50']:.6f} и "
            f"Recall={report['recall']:.6f}. Оба значения ниже заранее "
            "зафиксированных triage-порогов 0.25 и 0.35.",
            "",
            "Следовательно, M4 не подтвердил достаточную переносимость на "
            "held-out scenes в ускоренном go/no-go. Полный CV остановлен, "
            "официальный validation и test не открывались, атаки и гипотезы "
            "H1-H4 не проверялись.",
            "",
            "Этот документ является внутренним диагностическим результатом "
            "и не является финальной журнальной статьёй.",
            "",
        ]),
        encoding="utf-8",
    )


def write_report() -> dict[str, Any]:
    protocol = load_protocol()
    evaluation = json.loads(
        (EVALUATION_ROOT / "evaluation_result.json").read_text(encoding="utf-8")
    )
    consistency = json.loads(
        (EVALUATION_ROOT / "evaluator_consistency.json").read_text(encoding="utf-8")
    )
    per_scene = pd.read_csv(EVALUATION_ROOT / "per_scene_metrics.csv")
    per_class = pd.read_csv(EVALUATION_ROOT / "per_class_metrics.csv")
    per_size = pd.read_csv(EVALUATION_ROOT / "per_size_metrics.csv")
    tiling = json.loads(
        (FULL_ROOT / "tiling_audit/tiling_audit.json").read_text(encoding="utf-8")
    )
    lost_gt = int(tiling["lost_source_gt_instances"])
    status, reasons = classify(evaluation, per_scene, lost_gt, protocol)
    class_names = {
        0: "person",
        1: "signal",
        2: "road vehicle",
        3: "train",
        4: "animal",
        5: "bicycle",
    }
    class_rows = []
    for row in per_class.to_dict(orient="records"):
        row["class_name"] = class_names.get(int(row["class_id"]), "unknown")
        class_rows.append({
            key: (
                None
                if isinstance(value, float) and not math.isfinite(value)
                else value
            )
            for key, value in row.items()
        })
    report = {
        "status": status,
        "role": "early_computational_go_no_go_not_canonical_validation",
        "created_at": now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "parent_protocol": protocol["parent_protocol"],
        "fold": 0,
        "seed": 20260722,
        "checkpoint": evaluation["checkpoint"],
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "frames": int(evaluation["frames"]),
        "scenes": int(evaluation["scenes"]),
        "mAP50": float(evaluation["mAP50"]),
        "mAP50_95": float(evaluation["mAP50_95"]),
        "precision": float(evaluation["safety_precision"]),
        "recall": float(evaluation["safety_recall"]),
        "f1": float(evaluation["safety_f1"]),
        "fn": int(evaluation["safety_fn"]),
        "fn_per_frame": float(evaluation["safety_fn_per_frame"]),
        "small_recall": float(evaluation["small_recall"]),
        "medium_recall": float(evaluation["medium_recall"]),
        "large_recall": float(evaluation["large_recall"]),
        "per_class": class_rows,
        "per_scene": per_scene.to_dict(orient="records"),
        "lost_gt": lost_gt,
        "duplicate_predictions": not bool(
            evaluation["no_duplicate_predictions"]
        ),
        "missing_scene_count": (
            0 if bool(evaluation["no_missing_scene"]) else max(0, 2 - len(per_scene))
        ),
        "evaluator_consistency": consistency,
        "hard_fail_reasons": reasons,
        "test_opened": TEST_MARKER.exists(),
        "recommended_action": (
            "stop_and_build_diagnostic_bundle"
            if status == "hard_fail"
            else "run_one_fixed_full_seed_then_official_validation"
        ),
    }
    if report["test_opened"]:
        raise RuntimeError("Expedited triage found an illegal test-open marker")
    atomic_json(TRIAGE_ROOT / "early_generalization_report.json", report)
    lines = [
        "# M4 early generalization triage",
        "",
        "This is a computational go/no-go result, not canonical validation.",
        "",
        f"- Status: `{status}`",
        f"- mAP50: `{report['mAP50']:.6f}`",
        f"- Recall: `{report['recall']:.6f}`",
        f"- Small Recall: `{report['small_recall']:.6f}`",
        f"- Held-out scenes: `{report['scenes']}`",
        f"- Hard-fail reasons: `{', '.join(reasons) if reasons else 'none'}`",
        "- Test opened: `false`",
        "",
    ]
    (TRIAGE_ROOT / "early_generalization_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


def main() -> None:
    TRIAGE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        TRIAGE_ROOT / "watcher_status.json",
        {
            "status": "waiting_for_fold_0_training",
            "started_at": now(),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "test_opened": TEST_MARKER.exists(),
        },
    )
    while not TRAINING_MARKER.is_file():
        if TEST_MARKER.exists():
            raise RuntimeError("Test opened while expedited triage was waiting")
        time.sleep(1)
    stop_full_service()
    atomic_json(
        TRIAGE_ROOT / "watcher_status.json",
        {
            "status": "evaluating_fold_0",
            "training_marker": str(TRAINING_MARKER.resolve()),
            "training_marker_sha256": sha256(TRAINING_MARKER),
            "test_opened": False,
        },
    )
    subprocess.run(
        [
            str(PYTHON),
            "-u",
            "scripts/evaluate_canonical_m4.py",
            "--mode",
            "cv",
            "--seed",
            "20260722",
            "--fold",
            "0",
        ],
        cwd=PROJECT_DIR,
        check=True,
    )
    report = write_report()
    mark_full_protocol_partial(report["status"])
    if report["status"] == "hard_fail":
        write_hard_fail_outputs(report)
        bundle = diagnostic_bundle(report)
        report["diagnostic_bundle"] = str(bundle.resolve())
        report["diagnostic_bundle_sha256"] = sha256(bundle)
        atomic_json(TRIAGE_ROOT / "early_generalization_report.json", report)
    else:
        subprocess.run(
            ["systemctl", "--user", "start", "tnorm-canonical-m4-expedited.service"],
            check=True,
        )
    atomic_json(
        TRIAGE_ROOT / "watcher_status.json",
        {
            "status": "complete",
            "finished_at": now(),
            "triage_result": report["status"],
            "test_opened": False,
        },
    )


if __name__ == "__main__":
    main()
