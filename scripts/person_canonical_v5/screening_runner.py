from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from scripts.person_canonical_v5.common import (
    PROJECT,
    atomic_csv,
    atomic_json,
    now,
    sha256,
)
from scripts.person_canonical_v5.screening_common import (
    LEADERBOARD,
    ROOT,
    append_decision,
    assert_locked,
    config,
    exclusive,
    write_status,
)
from scripts.person_canonical_v5.screening_data import audit_data, data_for


CANDIDATES = ("C0", "C1", "C2")


class ScreeningTrainer(DetectionTrainer):
    """Keep optimizer state at intermediate halving rungs for true resume."""

    def final_eval(self) -> None:
        if int(self.epoch) + 1 < int(self.epochs):
            return
        super().final_eval()


def _output_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [
            tensor
            for child in value.values()
            for tensor in _output_tensors(child)
        ]
    if isinstance(value, (list, tuple)):
        return [
            tensor
            for child in value
            for tensor in _output_tensors(child)
        ]
    return []


class StopAtEpoch:
    def __init__(self, cumulative_epoch: int) -> None:
        self.cumulative_epoch = cumulative_epoch

    def __call__(self, trainer: Any) -> None:
        if int(trainer.epoch) + 1 >= self.cumulative_epoch:
            trainer.stop = True


def _technical_smoke(candidate: str, fold: int) -> dict[str, Any]:
    audit = audit_data(candidate, fold)
    model = YOLO(str((PROJECT / config()["training"]["initialization"]).resolve()))
    module = model.model.to("cuda")
    module.train()
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    batches = int(config()["evaluation"]["technical_batches"])
    gradient_norms = []
    for _ in range(batches):
        value = torch.rand(
            1, 3,
            int(config()["training"]["imgsz"]),
            int(config()["training"]["imgsz"]),
            device="cuda",
            requires_grad=True,
        )
        output = module(value)
        tensors = _output_tensors(output)
        if not tensors:
            raise RuntimeError("Model forward emitted no differentiable tensors")
        loss = sum(
            tensor.float().square().mean()
            for tensor in tensors
            if tensor.is_floating_point()
        )
        module.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [
            parameter.grad.detach().norm()
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        gradient_norms.append(float(torch.stack(gradients).sum().cpu()))
    finite = bool(np.isfinite(gradient_norms).all())
    nonzero = bool(min(gradient_norms, default=0.0) > 0)
    if not finite or not nonzero:
        raise RuntimeError("Screening gradient-flow smoke failed")
    return {
        "status": "PASS",
        "candidate": candidate,
        "fold": fold,
        "technical_batches": batches,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in module.parameters()
                if parameter.requires_grad)
        ),
        "gradient_norm_min": min(gradient_norms),
        "gradient_norm_max": max(gradient_norms),
        "NaN_Inf": 0,
        "data_audit": audit,
        "zero_shot": {
            "status": "SHARED_INITIALIZATION_VERIFIED",
            "checkpoint_sha256": sha256(PROJECT / config()["training"]["initialization"]),
            "fixed_slice_frames": int(config()["evaluation"]["zero_shot_frames"]),
            "note": "metric inference is shared across C0/C1/C2 and is not a selection signal",
        },
        "test_used": False,
    }


def _training_root(candidate: str, fold: int) -> Path:
    return ROOT / f"candidates/{candidate}/fold_{fold}/training"


def _train_to(candidate: str, fold: int, cumulative_epoch: int) -> None:
    data = data_for(candidate, fold)
    run = _training_root(candidate, fold)
    last = run / "weights/last.pt"
    if last.is_file():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        completed_epochs = int(checkpoint.get("epoch", -1)) + 1
        if completed_epochs >= cumulative_epoch:
            return
    model = (
        YOLO(str(last))
        if last.is_file()
        else YOLO(str((PROJECT / config()["training"]["initialization"]).resolve()))
    )
    model.add_callback("on_train_epoch_end", StopAtEpoch(cumulative_epoch))
    if last.is_file():
        model.train(resume=True, trainer=ScreeningTrainer)
        return
    training = config()["training"]
    model.train(
        trainer=ScreeningTrainer,
        data=str(data.resolve()),
        task="detect",
        imgsz=int(training["imgsz"]),
        epochs=int(training["total_epochs"]),
        patience=int(training["total_epochs"]),
        batch=int(training["batch"]),
        nbs=int(training["nominal_batch_size"]),
        device=0,
        workers=int(training["workers"]),
        optimizer=str(training["optimizer"]),
        lr0=float(training["lr0"]),
        lrf=float(training["lrf"]),
        weight_decay=float(training["weight_decay"]),
        pretrained=True,
        amp=bool(training["amp"]),
        cos_lr=True,
        seed=int(training["seed"]),
        deterministic=bool(training["deterministic"]),
        cache=False,
        val=True,
        plots=True,
        save=True,
        save_period=int(training["save_period"]),
        close_mosaic=0,
        project=str(run.parent.resolve()),
        name=run.name,
        exist_ok=False,
        verbose=True,
        **training["augmentations"],
    )


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["recall"]),
        float(row["small_recall"]),
        float(row["mAP50"]),
        -float(row["FN_per_frame"]),
        -float(row["source_order"]),
    )


def _evaluate(candidate: str, fold: int, fidelity: int) -> dict[str, Any]:
    # Keep the evaluator import lazy: its legacy CLI compatibility imports
    # expect the repository's scripts directory on PYTHONPATH at runtime.
    from scripts.person_canonical_v5.execution_evaluate import _evaluate_checkpoint

    run = _training_root(candidate, fold)
    destination = ROOT / f"candidates/{candidate}/fold_{fold}/fidelity_{fidelity}"
    frozen_result = destination / "screening_evaluation.json"
    if frozen_result.is_file():
        return json.loads(frozen_result.read_text(encoding="utf-8"))
    rows = []
    cache: dict[str, dict[str, Any]] = {}
    for source_order, label in enumerate(("best", "last")):
        checkpoint = run / f"weights/{label}.pt"
        digest = sha256(checkpoint)
        if digest in cache:
            row = {**cache[digest], "label": label, "source_order": source_order}
        else:
            evaluation, _, per_scene, per_size, confidence = _evaluate_checkpoint(
                candidate,
                fold,
                checkpoint,
                destination / label,
            )
            row = {
                **evaluation,
                "label": label,
                "source_order": source_order,
            }
            cache[digest] = row
            atomic_csv(per_scene, destination / label / "per_scene.csv")
            atomic_csv(per_size, destination / label / "per_size.csv")
            atomic_csv(confidence, destination / label / "confidence_distribution.csv")
        rows.append(row)
    selected = max(rows, key=_selection_key)
    selected_checkpoint = destination / "selected.pt"
    if (
        not selected_checkpoint.is_file()
        or sha256(selected_checkpoint) != selected["checkpoint_sha256"]
    ):
        shutil.copy2(selected["checkpoint"], selected_checkpoint)
    reference = config()["reference"]
    result = {
        **selected,
        "candidate": candidate,
        "fold": fold,
        "fidelity": fidelity,
        "selected_label": selected["label"],
        "selected_checkpoint": str(selected_checkpoint.resolve()),
        "delta_mAP50": float(selected["mAP50"]) - float(reference["mAP50"]),
        "delta_recall": float(selected["recall"]) - float(reference["recall"]),
        "delta_small_recall": (
            float(selected["small_recall"]) - float(reference["small_recall"])
        ),
        "relative_FP_per_frame": (
            float(selected["FP_per_frame"]) / float(reference["FP_per_frame"])
        ),
        "checkpoint_candidates": rows,
        "article_evidence": False,
        "candidate_screening_only": True,
        "test_used": False,
    }
    atomic_json(frozen_result, result)
    return result


def _history_trend(candidate: str, fold: int, window: int = 3) -> bool:
    path = _training_root(candidate, fold) / "results.csv"
    if not path.is_file():
        return True
    frame = pd.read_csv(path)
    columns = [
        column for column in frame.columns
        if "metrics/mAP50(B)" in column or "metrics/recall(B)" in column
    ]
    if len(frame) < window or not columns:
        return True
    recent = frame[columns].tail(window)
    return any(
        float(recent[column].iloc[-1]) >= float(recent[column].iloc[0])
        for column in columns
    )


def _technical_pass(metrics: dict[str, Any]) -> bool:
    return (
        metrics["evaluator_consistency"] == "PASS"
        and int(metrics["lost_GT"]) == 0
        and int(metrics["NaN"]) == 0
        and int(metrics["Inf"]) == 0
    )


def _level1_eligible(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    rules = config()["successive_halving"]["level_1"]
    reject = rules["reject"]
    reasons = []
    simultaneous = (
        metrics["delta_mAP50"] <= reject["simultaneous_delta_mAP50_max"]
        and metrics["delta_recall"] <= reject["simultaneous_delta_recall_max"]
    )
    if simultaneous:
        reasons.append("SIMULTANEOUS_MAP50_RECALL_DEGRADATION")
    if metrics["delta_small_recall"] <= reject["delta_small_recall_max"]:
        reasons.append("SMALL_RECALL_DEGRADATION")
    if not _history_trend(metrics["candidate"], int(metrics["fold"])):
        reasons.append("NO_POSITIVE_VALIDATION_TREND")
    if not _technical_pass(metrics):
        reasons.append("TECHNICAL_FAIL")
    promote = rules["promote"]
    promotion_checks = (
        metrics["delta_mAP50"] >= promote["delta_mAP50_min"],
        metrics["delta_recall"] >= promote["delta_recall_min"],
        max(
            metrics["delta_mAP50"],
            metrics["delta_recall"],
            metrics["delta_small_recall"],
        ) >= promote["any_primary_delta_min"],
        metrics["delta_small_recall"] >= promote["delta_small_recall_min"],
    )
    if not all(promotion_checks):
        reasons.append("LEVEL_2_PROMOTION_GATE_NOT_MET")
    return not reasons, reasons


def _level2_eligible(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    rules = config()["successive_halving"]["level_2"]["promote"]
    checks = {
        "DELTA_MAP50": metrics["delta_mAP50"] >= rules["delta_mAP50_min"],
        "DELTA_RECALL": metrics["delta_recall"] >= rules["delta_recall_min"],
        "DELTA_SMALL_RECALL": (
            metrics["delta_small_recall"] >= rules["delta_small_recall_min"]
        ),
        "FP_GROWTH": (
            metrics["relative_FP_per_frame"]
            <= rules["relative_FP_per_frame_max"]
        ),
        "TECHNICAL": _technical_pass(metrics),
    }
    return all(checks.values()), [key for key, passed in checks.items() if not passed]


def _level3_pass(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    rules = config()["successive_halving"]["level_3"]["fold_0_gate"]
    improved = sum(
        metrics[key] > 0
        for key in ("delta_mAP50", "delta_recall", "delta_small_recall")
    )
    checks = {
        "MAP50": metrics["mAP50"] >= rules["mAP50_min"],
        "RECALL": metrics["recall"] >= rules["recall_min"],
        "SMALL_RECALL": metrics["small_recall"] >= rules["small_recall_min"],
        "PRIMARY_IMPROVEMENTS": improved >= rules["improved_primary_metrics_min"],
        "TECHNICAL": _technical_pass(metrics),
    }
    return all(checks.values()), [key for key, passed in checks.items() if not passed]


def _ranking(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["delta_recall"]),
        float(metrics["delta_small_recall"]),
        float(metrics["delta_mAP50"]),
        -float(metrics["FP_per_frame"]),
    )


def _write_leaderboard(rows: list[dict[str, Any]]) -> None:
    columns = [
        "candidate", "fold", "fidelity", "selected_label", "mAP50",
        "recall", "small_recall", "FP_per_frame", "delta_mAP50",
        "delta_recall", "delta_small_recall", "relative_FP_per_frame",
        "evaluator_consistency", "lost_GT", "NaN", "Inf",
    ]
    atomic_csv(pd.DataFrame(rows)[columns], LEADERBOARD)


def _two_fold_gate(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    rules = config()["successive_halving"]["level_4"]["two_fold_gate"]
    values = {
        "macro_mAP50": float(np.mean([left["mAP50"], right["mAP50"]])),
        "macro_recall": float(np.mean([left["recall"], right["recall"]])),
        "macro_small_recall": float(
            np.mean([left["small_recall"], right["small_recall"]])
        ),
        "worst_fold_recall": float(min(left["recall"], right["recall"])),
    }
    checks = {
        "macro_mAP50": values["macro_mAP50"] >= rules["macro_mAP50_min"],
        "macro_recall": values["macro_recall"] >= rules["macro_recall_min"],
        "macro_small_recall": (
            values["macro_small_recall"] >= rules["macro_small_recall_min"]
        ),
        "worst_fold_recall": (
            values["worst_fold_recall"] >= rules["worst_fold_recall_min"]
        ),
        "technical": _technical_pass(left) and _technical_pass(right),
    }
    return {
        "status": "SCREENING_PASS" if all(checks.values()) else "SCREENING_FAIL",
        **values,
        "checks": checks,
        "article_evidence": False,
        "candidate_screening_only": True,
        "test_status": "SEALED",
        "attacks_status": "BLOCKED",
    }


def main() -> None:
    with exclusive():
        assert_locked()
        write_status(stage="level_0", status="ACTIVE")
        for candidate in CANDIDATES:
            path = ROOT / f"candidates/{candidate}/fold_0/technical_smoke.json"
            if not path.is_file():
                atomic_json(path, _technical_smoke(candidate, 0))
            append_decision(candidate, 0, "PASS", ["TECHNICAL_SMOKE_PASS"])

        rows: list[dict[str, Any]] = []
        level1: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            write_status(stage="level_1", status="ACTIVE", candidate=candidate, fold=0)
            _train_to(candidate, 0, 5)
            metrics = _evaluate(candidate, 0, 5)
            rows.append(metrics)
            eligible, reasons = _level1_eligible(metrics)
            if eligible:
                level1.append(metrics)
            else:
                append_decision(candidate, 5, "REJECT", reasons, metrics)
        level1 = sorted(level1, key=_ranking, reverse=True)[:2]
        for metrics in level1:
            append_decision(
                metrics["candidate"], 5, "PROMOTE",
                ["TOP_TWO_LEVEL_1"], metrics,
            )
        promoted1 = {row["candidate"] for row in level1}
        for candidate in CANDIDATES:
            if candidate not in promoted1:
                append_decision(
                    candidate, 5, "REJECT",
                    ["NOT_SELECTED_FOR_LEVEL_2"], next(
                        row for row in rows if row["candidate"] == candidate
                    ),
                )

        level2: list[dict[str, Any]] = []
        for item in level1:
            candidate = item["candidate"]
            write_status(stage="level_2", status="ACTIVE", candidate=candidate, fold=0)
            _train_to(candidate, 0, 10)
            metrics = _evaluate(candidate, 0, 10)
            rows.append(metrics)
            eligible, reasons = _level2_eligible(metrics)
            if eligible:
                level2.append(metrics)
            else:
                append_decision(candidate, 10, "REJECT", reasons, metrics)
        if not level2:
            _write_leaderboard(rows)
            write_status(stage="finished", status="SCREENING_FAIL")
            return
        winner = max(level2, key=_ranking)
        append_decision(winner["candidate"], 10, "PROMOTE", ["LEVEL_2_WINNER"], winner)
        for item in level2:
            if item["candidate"] != winner["candidate"]:
                append_decision(
                    item["candidate"], 10, "REJECT",
                    ["NOT_SELECTED_FOR_LEVEL_3"], item,
                )

        candidate = winner["candidate"]
        write_status(stage="level_3", status="ACTIVE", candidate=candidate, fold=0)
        _train_to(candidate, 0, 20)
        fold0 = _evaluate(candidate, 0, 20)
        rows.append(fold0)
        passed, reasons = _level3_pass(fold0)
        if not passed:
            append_decision(candidate, 20, "REJECT", reasons, fold0)
            _write_leaderboard(rows)
            write_status(stage="finished", status="SCREENING_FAIL")
            return
        append_decision(candidate, 20, "PROMOTE", ["FOLD_0_GATE_PASS"], fold0)

        write_status(stage="level_4", status="ACTIVE", candidate=candidate, fold=1)
        atomic_json(
            ROOT / f"candidates/{candidate}/fold_1/technical_smoke.json",
            _technical_smoke(candidate, 1),
        )
        _train_to(candidate, 1, 20)
        fold1 = _evaluate(candidate, 1, 20)
        rows.append(fold1)
        gate = _two_fold_gate(fold0, fold1)
        atomic_json(ROOT / "TWO_FOLD_SCREENING_GATE.json", gate)
        append_decision(
            candidate,
            21,
            "PROMOTE" if gate["status"] == "SCREENING_PASS" else "REJECT",
            [gate["status"]],
            gate,
        )
        _write_leaderboard(rows)
        write_status(
            stage="finished",
            status=gate["status"],
            winner=candidate if gate["status"] == "SCREENING_PASS" else None,
            canonical_oof_required=gate["status"] == "SCREENING_PASS",
        )


if __name__ == "__main__":
    main()
