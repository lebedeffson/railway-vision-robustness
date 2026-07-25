from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from scripts.person_canonical_v5.common import atomic_json, now, sha256
from scripts.person_canonical_v5.execution_common import (
    COMPLETED,
    OUTPUT,
    PROJECT,
    RESULTS,
    assert_execution_locked,
    exclusive_execution,
    mark_complete,
    update_state,
)


PYTHON = PROJECT / ".venv/bin/python"
PRIMARY = (
    ("V5-A", "V5-A"),
    ("V5-B", "V5-B"),
    ("V5-C", "V5-C-fraction25"),
    ("V5-D", "V5-D-fraction25"),
)
SENSITIVITY = (
    ("V5-C50", "V5-C-fraction50"),
    ("V5-D50", "V5-D-fraction50"),
)


def run(module: str, *arguments: str) -> None:
    subprocess.run(
        [str(PYTHON), "-u", "-m", module, *arguments],
        cwd=PROJECT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(PROJECT), str(PROJECT / "scripts")]
            ),
        },
        check=True,
    )


def prepare_pasting(fold: int, fraction: int) -> None:
    bank = OUTPUT / f"instance_pasting/fold_{fold}/instance_bank_summary.json"
    if not bank.is_file():
        update_state("pasting", fold, "instance_bank", "ACTIVE")
        run(
            "scripts.person_canonical_v5.build_instance_bank",
            "--fold", str(fold), "--workers", "4",
        )
    bank_payload = json.loads(bank.read_text(encoding="utf-8"))
    if bank_payload["accepted"] < 200:
        raise RuntimeError(f"Fold {fold} instance bank has fewer than 200 masks")
    root = OUTPUT / f"instance_pasting/fold_{fold}/fraction_{fraction}"
    summary = root / "pasting_summary.json"
    if not summary.is_file():
        update_state("pasting", fold, f"materialize_{fraction}", "ACTIVE")
        run(
            "scripts.person_canonical_v5.materialize_pasting",
            "--fold", str(fold), "--fraction", str(fraction / 100),
        )
        run(
            "scripts.person_canonical_v5.render_pasting_audit",
            "--fold", str(fold), "--fraction", str(fraction / 100),
        )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if payload["source_leakage"] != 0 or payload["lost_GT"] != 0:
        raise RuntimeError("Pasting data-integrity audit failed")
    if payload["accepted_frames"] != payload["target_changed_frames"]:
        raise RuntimeError("Pasting fraction was not achieved")
    if payload["maximum_insertions_per_frame"] > 2:
        raise RuntimeError("Pasting exceeded maximum insertions")


def execute_candidate(label: str, runtime_name: str, fold: int) -> None:
    marker = COMPLETED / f"{runtime_name}_fold{fold}.done"
    if marker.is_file():
        return
    update_state(label, fold, "training", "ACTIVE", checkpoint="last.pt")
    run(
        "scripts.person_canonical_v5.execution_train",
        "--candidate", runtime_name, "--fold", str(fold),
    )
    update_state(label, fold, "independent_evaluator", "ACTIVE")
    run(
        "scripts.person_canonical_v5.execution_evaluate",
        "--candidate", runtime_name, "--fold", str(fold),
    )
    evaluation = (
        OUTPUT / f"candidates/{runtime_name}/fold_{fold}/evaluation/evaluation_result.json"
    )
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    mark_complete(runtime_name, fold, {
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "evaluator_consistency": payload["evaluator_consistency"],
        "lost_GT": payload["lost_GT"],
        "mAP50": payload["mAP50"],
        "recall": payload["recall"],
        "small_recall": payload["small_recall"],
    })
    update_state(label, fold, "completed", "RUNTIME_PASS")


def analyze_oof(label: str, runtime_name: str) -> dict[str, Any]:
    from scripts.person_canonical_v5.execution_analysis import (
        _aggregate,
        _candidate_rows,
    )
    from scripts.person_canonical_v5.execution_common import config

    folds, _ = _candidate_rows(label, runtime_name, (0, 1, 2, 3, 4))
    aggregate = _aggregate(folds)
    required = config()["gate"]["oof"]
    checks = {
        "macro_mAP50": aggregate["macro_mAP50"] >= required["macro_mAP50_min"],
        "macro_recall": aggregate["macro_recall"] >= required["macro_recall_min"],
        "macro_small_recall": aggregate["macro_small_recall"] >= required["macro_small_recall_min"],
        "worst_fold_recall": aggregate["worst_fold_recall"] >= required["worst_fold_recall_min"],
        "folds_with_recall_ge_0_40": int(folds["recall"].ge(0.40).sum())
        >= required["folds_with_recall_ge_0_40_min"],
        "evaluator_consistency": bool(folds["evaluator_consistency"].eq("PASS").all()),
        "lost_GT": int(folds["lost_GT"].sum()) == 0,
        "source_leakage": int(folds["source_leakage"].sum()) == 0,
        "NaN_Inf": int(folds["NaN"].sum() + folds["Inf"].sum()) == 0,
    }
    passed = all(checks.values())
    payload = {
        "status": "OOF_PASS" if passed else "OOF_FAIL",
        "candidate": label,
        "runtime_name": runtime_name,
        "created_at": now(),
        **aggregate,
        "checks": checks,
        "test_status": "TEST_NOT_OPENED",
        "attacks_status": "ATTACKS_BLOCKED",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    folds.to_csv(RESULTS / "OOF_RESULTS.csv", index=False)
    atomic_json(RESULTS / "OOF_GATE.json", payload)
    return payload


def main() -> None:
    with exclusive_execution():
        assert_execution_locked()
        update_state("V5-A", 0, "resume", "ACTIVE")
        for label, runtime_name in PRIMARY:
            if label in {"V5-C", "V5-D"}:
                for fold in (0, 1):
                    prepare_pasting(fold, 25)
                    execute_candidate(label, runtime_name, fold)
            else:
                for fold in (0, 1):
                    execute_candidate(label, runtime_name, fold)
        for label, runtime_name in SENSITIVITY:
            prepare_pasting(0, 50)
            execute_candidate(label, runtime_name, 0)
        update_state("matrix", None, "two_fold_analysis", "ACTIVE")
        run(
            "scripts.person_canonical_v5.execution_analysis",
            "--two-fold",
        )
        selection = json.loads(
            (RESULTS / "CANDIDATE_SELECTION.json").read_text(encoding="utf-8")
        )
        if selection["status"] != "TWO_FOLD_PASS":
            update_state(
                "matrix", None, "finished", "SCIENTIFIC_FAIL",
                test_status="TEST_NOT_OPENED", attacks_status="ATTACKS_BLOCKED",
            )
            return
        label = str(selection["winner"])
        runtime_name = str(selection["runtime_name"])
        for fold in (2, 3, 4):
            if label in {"V5-C", "V5-D"}:
                prepare_pasting(fold, 25)
            execute_candidate(label, runtime_name, fold)
        oof = analyze_oof(label, runtime_name)
        if oof["status"] != "OOF_PASS":
            update_state(
                label, None, "finished", "SCIENTIFIC_FAIL",
                oof_status="OOF_FAIL", test_status="TEST_NOT_OPENED",
                attacks_status="ATTACKS_BLOCKED",
            )
            return
        update_state(
            label, None, "pre_test_final_training_required", "OOF_PASS",
            test_status="TEST_NOT_OPENED", attacks_status="ATTACKS_BLOCKED",
        )


if __name__ == "__main__":
    main()
