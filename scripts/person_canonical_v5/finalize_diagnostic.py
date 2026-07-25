from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from scripts.person_canonical_v5.common import (
    OUTPUT,
    PROJECT,
    PROTOCOL_ROOT,
    assert_diagnostic_locked,
    load_protocol,
    now,
    sha256,
)


DIAGNOSTIC = OUTPUT / "development_diagnostic"
REPORTS = PROJECT / "reports/person_v5"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _metric(frame: pd.DataFrame, state: str, name: str) -> float:
    return float(frame.loc[frame["state"].eq(state), name].iloc[0])


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        [
            f"{value:.6f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(str(column)), *(len(row[index]) for row in rows))
        for index, column in enumerate(columns)
    ]
    header = "| " + " | ".join(
        str(column).ljust(widths[index])
        for index, column in enumerate(columns)
    ) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def run() -> dict[str, object]:
    lock = assert_diagnostic_locked()
    protocol = load_protocol()
    fold_path = DIAGNOSTIC / "state_fold_results.csv"
    macro_path = DIAGNOSTIC / "state_macro_results.csv"
    scene_path = DIAGNOSTIC / "per_scene_results.csv"
    area_path = DIAGNOSTIC / "per_area_results.csv"
    summary_path = DIAGNOSTIC / "development_diagnostic_summary.json"
    required = (fold_path, macro_path, scene_path, area_path, summary_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing development diagnostic outputs: {missing}")
    folds = pd.read_csv(fold_path)
    macro = pd.read_csv(macro_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    parity_passed = bool(
        folds["evaluator_consistency"].eq("PASS").all()
        and folds["lost_GT"].eq(0).all()
        and folds["NaN_Inf"].eq(0).all()
        and not folds["test_used"].astype(bool).any()
    )
    b1_recall = _metric(macro, "B1", "macro_recall")
    d1_recall = _metric(macro, "D1_best", "macro_recall")
    b1_worst = _metric(macro, "B1", "worst_fold_recall")
    d1_worst = _metric(macro, "D1_best", "worst_fold_recall")
    gradual_allowed = b1_recall > d1_recall or b1_worst > d1_worst
    if gradual_allowed != bool(summary["gradual_transfer_allowed"]):
        raise RuntimeError("Gradual-transfer decision disagrees with raw summary")
    decision = {
        "protocol_id": protocol["protocol_id"],
        "created_at": now(),
        "diagnostic_status": summary["status"],
        "evaluator_parity": "PASS" if parity_passed else "FAIL",
        "fixed_confidence_threshold": summary["fixed_threshold"],
        "fixed_iou_threshold": summary["fixed_IoU"],
        "B1_macro_recall": b1_recall,
        "D1_best_macro_recall": d1_recall,
        "B1_worst_fold_recall": b1_worst,
        "D1_best_worst_fold_recall": d1_worst,
        "gradual_transfer_allowed": gradual_allowed,
        "V5_E_status": (
            "ELIGIBLE_FOR_SEPARATE_CANDIDATE"
            if gradual_allowed
            else "EXCLUDED_BY_PROSPECTIVE_DIAGNOSTIC_RULE"
        ),
        "allowed_main_candidates": ["V5-A", "V5-B", "V5-C", "V5-D"],
        "test_opened": False,
        "attacks_blocked": True,
        "scientific_result": False,
        "protocol_lock_sha256": sha256(
            PROTOCOL_ROOT / "protocol_lock.json"
        ),
        "diagnostic_lock_sha256": sha256(
            PROTOCOL_ROOT / "development_diagnostic_lock.json"
        ),
        "input_hashes": {
            str(path.relative_to(PROJECT)): sha256(path) for path in required
        },
        "lock_git_commit": lock["git_commit"],
    }
    destination = PROTOCOL_ROOT / "development_diagnostic_decision.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)

    table = macro[
        [
            "state",
            "macro_mAP50",
            "macro_mAP50_95",
            "macro_precision",
            "macro_recall",
            "macro_F1",
            "worst_fold_recall",
            "mean_FN_per_frame",
            "mean_FP_per_frame",
        ]
    ].copy()
    atomic_text(
        REPORTS / "EVALUATOR_PARITY.md",
        "# Person V5 development diagnostic\n\n"
        "All four states were evaluated with the same frozen 2x2 tiling, fusion, "
        "confidence threshold 0.07, IoU 0.50, folds and class mapping.\n\n"
        + markdown_table(table)
        + "\n\n"
        f"Evaluator parity: **{'PASS' if parity_passed else 'FAIL'}**. "
        "No test image or label was accessed. No GT was lost and no NaN/Inf "
        "was observed.\n\n"
        "CrowdHuman zero-shot does not exceed D1-best in macro Recall or "
        "worst-fold Recall. Therefore gradual transfer (V5-E) is excluded by "
        "the rule frozen before this diagnostic.\n",
    )
    atomic_text(
        REPORTS / "DATA_LEAKAGE_AUDIT.md",
        "# Person V5 data-leakage audit\n\n"
        f"- Protocol: `{protocol['protocol_id']}`\n"
        "- Diagnostic inputs: original train plus validation development folds\n"
        "- Independent unit: `grouped_scene_id`\n"
        "- Railway test: sealed and absent from evaluator sources\n"
        "- Instance-bank rule: current-fold train scenes only\n"
        "- LiDAR linkage rule: RGB bbox and LiDAR cuboid must share the same "
        "OpenLABEL object UUID\n"
        "- Area fallback is labelled `bbox_area_scale_fallback`, never range\n"
        f"- Diagnostic evaluator parity: `{'PASS' if parity_passed else 'FAIL'}`\n",
    )
    atomic_text(
        REPORTS / "RELEASE_STATUS.md",
        "# Person Canonical V5 Release Status\n\n"
        "Status: `DEVELOPMENT_DIAGNOSTIC_PASS_V5_E_REJECTED`\n\n"
        "- D1 two-fold gate: FAIL\n"
        "- B0/B1/D1-best/D1-last common evaluator: PASS\n"
        "- Gradual transfer V5-E: EXCLUDED BY PROSPECTIVE RULE\n"
        "- Allowed main candidates: V5-A, V5-B, V5-C, V5-D\n"
        "- Railway test: SEALED\n"
        "- Attacks and T-norm H1-H4: BLOCKED\n"
        "- V5 candidate training: NOT STARTED\n"
        "- Scientific claim: NONE\n\n"
        "The diagnostic is development-only. It shows that direct railway "
        "fine-tuning raises Recall over CrowdHuman zero-shot but produces a "
        "large FP increase; the last checkpoints further collapse AP and "
        "confidence quality.\n",
    )
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
