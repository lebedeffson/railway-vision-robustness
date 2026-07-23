from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scripts.audit_evaluator import match_dataset
from scripts.person_v3.run_pipeline import size_recall
from scripts.person_v4.common import (
    EXPEDITED_LOCK_PATH,
    EXPEDITED_OUTPUT_ROOT,
    EXPEDITED_PROTOCOL_PATH,
    OUTPUT_ROOT,
    PROJECT_DIR,
    assert_locked,
    atomic_json,
    now,
    sha256,
)
from scripts.person_v4.run_pipeline import (
    combined,
    oof_gate,
    run,
    threshold_sweep,
)
from scripts.person_v4.train import training_root


STATUS = EXPEDITED_OUTPUT_ROOT / "pipeline_status.json"
TRACE = EXPEDITED_OUTPUT_ROOT / "decision_trace.json"
SKIPPED = EXPEDITED_OUTPUT_ROOT / "skipped_stages.json"
GATE_TOLERANCE = 1.0e-12


def protocol() -> dict[str, Any]:
    return yaml.safe_load(
        EXPEDITED_PROTOCOL_PATH.read_text(encoding="utf-8")
    )


def write_status(stage: str, value: str, **details: Any) -> None:
    payload = (
        json.loads(STATUS.read_text(encoding="utf-8"))
        if STATUS.is_file()
        else {
            "protocol_id": protocol()["protocol_id"],
            "stages": {},
            "test_opened": False,
        }
    )
    payload["updated_at"] = now()
    payload["current_stage"] = stage
    payload["stages"][stage] = {"status": value, **details}
    atomic_json(STATUS, payload)


def trace(event: str, **details: Any) -> None:
    payload = (
        json.loads(TRACE.read_text(encoding="utf-8"))
        if TRACE.is_file()
        else {
            "protocol_id": protocol()["protocol_id"],
            "created_at": now(),
            "events": [],
            "test_opened": False,
        }
    )
    payload["events"].append(
        {"sequence": len(payload["events"]) + 1, "at": now(),
         "event": event, **details}
    )
    atomic_json(TRACE, payload)


def record_skip(stage: str, reason: str) -> None:
    payload = (
        json.loads(SKIPPED.read_text(encoding="utf-8"))
        if SKIPPED.is_file()
        else {"protocol_id": protocol()["protocol_id"], "stages": {}}
    )
    payload["stages"][stage] = {"status": reason, "at": now()}
    atomic_json(SKIPPED, payload)
    write_status(stage, reason)
    trace("stage_skipped", stage=stage, reason=reason)


def mark_parent_protocol_stopped() -> None:
    path = OUTPUT_ROOT / "pipeline_status.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {
            "protocol_id": "canonical-v4-person-dg-nwd-v1",
            "stages": {},
        }
    )
    payload["updated_at"] = now()
    payload["current_stage"] = "expedited_amendment"
    payload["stages"]["full_protocol"] = {
        "status": "stopped_by_expedited_amendment",
        "amendment": protocol()["protocol_id"],
        "test_opened": False,
    }
    atomic_json(path, payload)


def run_fold(variant: str, fold: int) -> dict[str, Any]:
    stage = f"{variant}_fold_{fold}"
    result_path = (
        training_root(variant, fold)
        / "evaluation/evaluation_result.json"
    )
    if not result_path.is_file():
        write_status(stage, "running", started_at=now())
        trace("fold_started", variant=variant, fold=fold)
        run("train.py", "--variant", variant, "--fold", str(fold))
        run("evaluate.py", "--variant", variant, "--fold", str(fold))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["status"] != "PASS":
        raise RuntimeError(f"{variant} fold {fold} technical FAIL")
    write_status(
        stage,
        "success",
        checkpoint_sha256=result["checkpoint_sha256"],
        mAP50=result["mAP50"],
        finished_at=now(),
    )
    trace(
        "fold_completed",
        variant=variant,
        fold=fold,
        checkpoint_sha256=result["checkpoint_sha256"],
    )
    return result


def summarize(variant: str, folds: list[int]) -> dict[str, Any]:
    gt, predictions, by_fold, _, _ = combined(variant, folds)
    sweep = threshold_sweep(gt, predictions)
    selected = sweep.sort_values(
        ["f1", "recall", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]
    threshold = float(selected["threshold"])
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        fold_gt, fold_predictions = by_fold[fold]
        evaluation_path = (
            PROJECT_DIR
            / f"outputs/person_v3/scene_cv/fold_{fold}/"
            "seed_20260723/evaluation/evaluation_result.json"
            if variant == "A0"
            else training_root(variant, fold)
            / "evaluation/evaluation_result.json"
        )
        evaluation = json.loads(
            evaluation_path.read_text(encoding="utf-8")
        )
        fold_rows.append(
            {
                "variant": variant,
                "fold": fold,
                "threshold": threshold,
                "mAP50": float(evaluation["mAP50"]),
                "mAP50_95": float(evaluation["mAP50_95"]),
                **match_dataset(
                    fold_gt, fold_predictions, threshold, 0.50
                ),
                **size_recall(
                    fold_gt, fold_predictions, threshold
                ),
                "evaluator_consistency": evaluation[
                    "evaluator_consistency"
                ],
                "lost_GT": int(evaluation["lost_GT"]),
                "missing_scenes": int(evaluation["missing_scenes"]),
            }
        )
    frame = pd.DataFrame(fold_rows)
    finite = np.isfinite(
        frame[
            ["mAP50", "recall", "f1", "small_recall"]
        ].to_numpy()
    ).all()
    technical = bool(
        finite
        and frame["evaluator_consistency"].eq("PASS").all()
        and frame["lost_GT"].eq(0).all()
        and frame["missing_scenes"].eq(0).all()
    )
    destination = (
        EXPEDITED_OUTPUT_ROOT
        / "metrics"
        / variant
        / ("fold_" + "_".join(map(str, folds)))
    )
    destination.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(destination / "threshold_sweep.csv", index=False)
    frame.to_csv(destination / "fold_metrics.csv", index=False)
    result = {
        "variant": variant,
        "folds": folds,
        "threshold": threshold,
        "macro_mAP50": float(frame["mAP50"].mean()),
        "macro_recall": float(frame["recall"].mean()),
        "macro_F1": float(frame["f1"].mean()),
        "macro_small_recall": float(
            frame["small_recall"].mean()
        ),
        "worst_fold_recall": float(frame["recall"].min()),
        "tp": int(selected["tp"]),
        "fp": int(selected["fp"]),
        "fn": int(selected["fn"]),
        "technical_checks_passed": technical,
        "NaN_Inf": 0 if finite else 1,
        "lost_GT": int(frame["lost_GT"].sum()),
        "missing_scenes": int(frame["missing_scenes"].sum()),
        "evaluator_consistency": (
            "PASS"
            if frame["evaluator_consistency"].eq("PASS").all()
            else "FAIL"
        ),
        "test_opened": False,
    }
    atomic_json(destination / "summary.json", result)
    return result


def fold0_decision(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    gate = protocol()["execution_tree"]["fold0"]["require_all"]
    result = {
        "variant": candidate["variant"],
        "fold": 0,
        "delta_mAP50": (
            candidate["macro_mAP50"] - baseline["macro_mAP50"]
        ),
        "delta_recall": (
            candidate["macro_recall"] - baseline["macro_recall"]
        ),
        "delta_small_recall": (
            candidate["macro_small_recall"]
            - baseline["macro_small_recall"]
        ),
        "technical_checks_passed": candidate[
            "technical_checks_passed"
        ],
    }
    result["passed"] = bool(
        result["delta_mAP50"]
        >= float(gate["delta_mAP50_min"]) - GATE_TOLERANCE
        and result["delta_recall"]
        >= float(gate["delta_recall_min"]) - GATE_TOLERANCE
        and result["delta_small_recall"]
        >= float(gate["delta_small_recall_min"]) - GATE_TOLERANCE
        and result["technical_checks_passed"]
    )
    return result


def two_fold_decision(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    gate = protocol()["execution_tree"]["two_fold"]["require_all"]
    result = {
        "variant": candidate["variant"],
        "folds": [0, 1],
        "macro_mAP50": candidate["macro_mAP50"],
        "macro_recall": candidate["macro_recall"],
        "delta_macro_small_recall": (
            candidate["macro_small_recall"]
            - baseline["macro_small_recall"]
        ),
        "worst_fold_recall": candidate["worst_fold_recall"],
        "technical_checks_passed": candidate[
            "technical_checks_passed"
        ],
    }
    result["passed"] = bool(
        result["macro_mAP50"]
        >= float(gate["macro_mAP50_min"]) - GATE_TOLERANCE
        and result["macro_recall"]
        >= float(gate["macro_recall_min"]) - GATE_TOLERANCE
        and result["delta_macro_small_recall"]
        >= (
            float(gate["delta_macro_small_recall_min"])
            - GATE_TOLERANCE
        )
        and result["worst_fold_recall"]
        >= float(gate["worst_fold_recall_min"]) - GATE_TOLERANCE
        and result["technical_checks_passed"]
    )
    return result


def save_comparison(
    name: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    row = {
        **{
            f"baseline_{key}": value
            for key, value in baseline.items()
            if not isinstance(value, (dict, list))
        },
        **{
            f"candidate_{key}": value
            for key, value in candidate.items()
            if not isinstance(value, (dict, list))
        },
        **{
            f"decision_{key}": value
            for key, value in decision.items()
            if not isinstance(value, (dict, list))
        },
    }
    path = EXPEDITED_OUTPUT_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path,
        mode="a" if path.is_file() else "w",
        header=not path.is_file(),
        index=False,
    )


def build_bundle() -> Path:
    destination = (
        EXPEDITED_OUTPUT_ROOT
        / "bundles/TNormFilter_person_v4_expedited.zip"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    roots = [
        EXPEDITED_PROTOCOL_PATH,
        EXPEDITED_LOCK_PATH,
        STATUS,
        TRACE,
        SKIPPED,
        EXPEDITED_OUTPUT_ROOT / "metrics",
        EXPEDITED_OUTPUT_ROOT / "fold0_comparisons.csv",
        EXPEDITED_OUTPUT_ROOT / "two_fold_comparisons.csv",
        EXPEDITED_OUTPUT_ROOT / "expedited_selection.json",
    ]
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for root in roots:
            if root.is_file():
                archive.write(root, root.relative_to(PROJECT_DIR))
            elif root.is_dir():
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        archive.write(
                            path, path.relative_to(PROJECT_DIR)
                        )
    return destination


def main() -> None:
    assert_locked()
    mark_parent_protocol_stopped()
    trace(
        "expedited_protocol_started",
        protocol_sha256=sha256(EXPEDITED_PROTOCOL_PATH),
        test_opened=False,
    )
    record_skip("A2", "skipped_by_expedited_amendment")
    baseline_fold0 = summarize("A0", [0])
    baseline_two = summarize("A0", [0, 1])
    winner: str | None = None
    tried: list[str] = []
    for candidate_name in ("A1", "A3"):
        tried.append(candidate_name)
        run_fold(candidate_name, 0)
        candidate_fold0 = summarize(candidate_name, [0])
        first = fold0_decision(baseline_fold0, candidate_fold0)
        save_comparison(
            "fold0_comparisons.csv",
            baseline_fold0,
            candidate_fold0,
            first,
        )
        trace("fold0_gate", **first)
        if not first["passed"]:
            record_skip(
                f"{candidate_name}_fold_1",
                "skipped_by_expedited_elimination",
            )
            continue
        run_fold(candidate_name, 1)
        candidate_two = summarize(candidate_name, [0, 1])
        second = two_fold_decision(baseline_two, candidate_two)
        save_comparison(
            "two_fold_comparisons.csv",
            baseline_two,
            candidate_two,
            second,
        )
        trace("two_fold_gate", **second)
        if second["passed"]:
            winner = candidate_name
            break
    if winner is None:
        selection = {
            "status": "NO_EXPEDITED_WINNER",
            "winner": None,
            "tried": tried,
            "remaining_folds_allowed": False,
            "test_opened": False,
            "attacks_allowed": False,
        }
        write_status(
            "selection", "NO_EXPEDITED_WINNER", **selection
        )
    else:
        for candidate_name in ("A1", "A3"):
            if candidate_name not in tried:
                record_skip(
                    candidate_name,
                    "skipped_by_early_accept",
                )
        selection = {
            "status": "EXPEDITED_WINNER",
            "winner": winner,
            "tried": tried,
            "remaining_folds_allowed": True,
            "test_opened": False,
            "attacks_allowed": False,
        }
        write_status("selection", "EXPEDITED_WINNER", **selection)
        trace("winner_selected", winner=winner)
        for fold in (2, 3, 4):
            run_fold(winner, fold)
        gate = oof_gate(winner)
        write_status("OOF_gate", gate["status"], **gate)
        trace("OOF_gate", **gate)
    atomic_json(
        EXPEDITED_OUTPUT_ROOT / "expedited_selection.json",
        selection,
    )
    output = build_bundle()
    write_status(
        "bundle",
        "success",
        path=str(output.resolve()),
        sha256=sha256(output),
        test_opened=False,
    )


if __name__ == "__main__":
    main()
