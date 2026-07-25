from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from scripts.person_canonical_v5.common import (
    OUTPUT,
    PROJECT,
    PROTOCOL_ROOT,
    assert_test_sealed,
    atomic_json,
    now,
)
from scripts.person_canonical_v5.train_candidates import assert_runtime_locked


PYTHON = PROJECT / ".venv/bin/python"
STATUS = OUTPUT / "candidate_pipeline_status.json"
RUNTIME = PROJECT / "configs/person_v5/candidate_runtime.yaml"


def update(stage: str, state: str, **values: Any) -> None:
    protocol_id = yaml.safe_load(
        RUNTIME.read_text(encoding="utf-8")
    )["protocol_id"]
    payload = (
        json.loads(STATUS.read_text(encoding="utf-8"))
        if STATUS.is_file()
        else {
            "protocol_id": protocol_id,
            "test_opened": False,
            "stages": {},
        }
    )
    previous = payload.get("protocol_id")
    if previous and previous != protocol_id:
        payload.setdefault("superseded_protocol_ids", [])
        if previous not in payload["superseded_protocol_ids"]:
            payload["superseded_protocol_ids"].append(previous)
    payload["protocol_id"] = protocol_id
    payload["updated_at"] = now()
    payload["current_stage"] = stage
    payload["stages"][stage] = {"status": state, **values}
    atomic_json(STATUS, payload)


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


def candidate(candidate: str, fold: int) -> None:
    root = OUTPUT / f"candidates/{candidate}/fold_{fold}"
    evaluation = root / "evaluation/evaluation_result.json"
    stage = f"{candidate}_fold{fold}"
    if not evaluation.is_file():
        update(stage, "running", started_at=now())
        run(
            "scripts.person_canonical_v5.train_candidates",
            "--candidate",
            candidate,
            "--fold",
            str(fold),
        )
        run(
            "scripts.person_canonical_v5.evaluate_candidate",
            "--candidate",
            candidate,
            "--fold",
            str(fold),
        )
    result = json.loads(evaluation.read_text(encoding="utf-8"))
    update(
        stage,
        "success",
        finished_at=now(),
        checkpoint_sha256=result["checkpoint_sha256"],
        mAP50=result["mAP50"],
        evaluator_consistency=result["evaluator_consistency"],
    )


def prepare_pasting(fold: int, fraction: int) -> None:
    bank = (
        OUTPUT
        / f"instance_pasting/fold_{fold}/instance_bank_summary.json"
    )
    if not bank.is_file():
        update(f"instance_bank_fold{fold}", "running", started_at=now())
        run(
            "scripts.person_canonical_v5.build_instance_bank",
            "--fold",
            str(fold),
            "--workers",
            "4",
        )
    bank_summary = json.loads(bank.read_text(encoding="utf-8"))
    if bank_summary["accepted"] < 200:
        raise RuntimeError(f"Fold {fold} instance bank has fewer than 200 masks")
    root = OUTPUT / f"instance_pasting/fold_{fold}/fraction_{fraction}"
    summary = root / "pasting_summary.json"
    if not summary.is_file():
        update(
            f"pasting_fold{fold}_fraction{fraction}",
            "running",
            started_at=now(),
        )
        run(
            "scripts.person_canonical_v5.materialize_pasting",
            "--fold",
            str(fold),
            "--fraction",
            str(fraction / 100),
        )
        run(
            "scripts.person_canonical_v5.render_pasting_audit",
            "--fold",
            str(fold),
            "--fraction",
            str(fraction / 100),
        )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if payload["maximum_insertions_per_frame"] > 2:
        raise RuntimeError("Pasting contract exceeded two instances per frame")
    if (
        payload["accepted_frames"]
        != payload["target_changed_frames"]
    ):
        raise RuntimeError("Requested changed-frame fraction was not achieved")
    update(
        f"pasting_fold{fold}_fraction{fraction}",
        "success",
        finished_at=now(),
        accepted_frames=payload["accepted_frames"],
        inserted_GT=payload["inserted_GT"],
        test_used=False,
    )


def gate(candidate_name: str) -> dict[str, Any]:
    destination = (
        OUTPUT
        / f"candidates/{candidate_name}/two_fold/TWO_FOLD_GATE.json"
    )
    if not destination.is_file():
        run(
            "scripts.person_canonical_v5.candidate_gate",
            "--candidate",
            candidate_name,
        )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    update(
        f"{candidate_name}_two_fold_gate",
        "success" if payload["status"] == "PASS" else "failed",
        result=payload["status"],
        checks=payload["checks"],
        test_opened=False,
    )
    return payload


def main() -> None:
    assert_test_sealed()
    assert_runtime_locked()
    for architecture in ("V5-A", "V5-B"):
        for fold in (0, 1):
            candidate(architecture, fold)
        gate(architecture)
    for fraction in (25, 50):
        prepare_pasting(0, fraction)
        candidate(f"V5-C-fraction{fraction}", 0)
    fraction_lock = PROTOCOL_ROOT / "pasting_fraction_lock.json"
    if not fraction_lock.is_file():
        run(
            "scripts.person_canonical_v5.candidate_gate",
            "--select-pasting-fraction",
        )
    selected = json.loads(fraction_lock.read_text(encoding="utf-8"))
    fraction = int(selected["selected_fraction_percent"])
    prepare_pasting(1, fraction)
    candidate(f"V5-C-fraction{fraction}", 1)
    gate(f"V5-C-fraction{fraction}")
    for fold in (0, 1):
        if fold == 0:
            prepare_pasting(0, fraction)
        candidate("V5-D", fold)
    gate("V5-D")
    update(
        "two_fold_candidate_matrix",
        "success",
        finished_at=now(),
        candidates=["V5-A", "V5-B", f"V5-C-fraction{fraction}", "V5-D"],
        test_opened=False,
        next_action="full_OOF_only_for_lexicographic_gate_winner",
    )


if __name__ == "__main__":
    main()
