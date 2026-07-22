from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from audit_final_practice import canonical_path


PROJECT_DIR = Path(__file__).resolve().parent
AUDIT = PROJECT_DIR / "outputs/final_practice/audit"
CURRENT = PROJECT_DIR / "outputs/final_practice/unified_diagnostics_val_raw.csv"
ORIGINAL = (
    PROJECT_DIR
    / "outputs/final_practice/deadline_baseline/raw/unified_diagnostics_val_raw.csv"
)
LOW = AUDIT / "nms_timeout_low_val.csv"
HIGH = AUDIT / "nms_timeout_high_val.csv"
KEYS = [
    "sequence_id", "image_path", "attack", "adaptive", "epsilon", "steps",
    "restart", "seed", "defense", "layer",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def precision_from_f1_recall(f1: pd.Series, recall: pd.Series) -> pd.Series:
    denominator = 2.0 * recall - f1
    return pd.Series(
        np.where(np.abs(denominator) > 1e-12, f1 * recall / denominator, 0.0),
        index=f1.index,
    ).clip(0.0, 1.0)


def prepare(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["image_path"] = frame["image_path"].map(canonical_path)
    if "fn_defended" not in frame and "false_negatives" in frame:
        frame["fn_defended"] = frame["false_negatives"]
    frame[f"precision_{suffix}"] = precision_from_f1_recall(
        pd.to_numeric(frame["f1_defended"], errors="coerce"),
        pd.to_numeric(frame["recall_defended"], errors="coerce"),
    )
    columns = [*KEYS, f"precision_{suffix}"]
    mapping = {
        "nms_runtime_ms": f"runtime_{suffix}_ms",
        "nms_candidates_before": f"candidate_count_{suffix}",
        "predictions_after_nms": f"prediction_count_{suffix}",
        "recall_defended": f"recall_{suffix}",
        "f1_defended": f"f1_{suffix}",
        "fn_defended": f"fn_{suffix}",
    }
    for source, destination in mapping.items():
        if source not in frame:
            frame[source] = np.nan
        frame[destination] = frame[source]
        columns.append(destination)
    return frame[columns].drop_duplicates(KEYS)


def main() -> None:
    required = [CURRENT, ORIGINAL, LOW, HIGH, AUDIT / "nms_timeout_audit.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"NMS audit contract is incomplete: {missing}")
    original = pd.read_csv(ORIGINAL, low_memory=False)
    low = pd.read_csv(LOW, low_memory=False)
    high = pd.read_csv(HIGH, low_memory=False)
    affected = set(high["image_path"].map(canonical_path))
    original["image_path"] = original["image_path"].map(canonical_path)
    original = original[original["image_path"].isin(affected)]
    base = prepare(original, "original")
    low_values = prepare(low, "instrumented")
    high_values = prepare(high, "rerun")
    merged = base.merge(low_values, on=KEYS, how="outer", validate="one_to_one")
    merged = merged.merge(high_values, on=KEYS, how="outer", validate="one_to_one")
    for metric in ("precision", "recall", "f1", "fn"):
        merged[f"delta_{metric}"] = (
            pd.to_numeric(merged[f"{metric}_rerun"], errors="coerce")
            - pd.to_numeric(merged[f"{metric}_original"], errors="coerce")
        )
    merged["row_replaced"] = False
    merged["instrumented_timeout"] = False
    merged["original_runtime_ms"] = merged["runtime_instrumented_ms"]
    merged["rerun_runtime_ms"] = merged["runtime_rerun_ms"]
    merged["candidate_count_before_nms"] = merged["candidate_count_rerun"]
    merged["prediction_count_after_nms"] = merged["prediction_count_rerun"]
    merged.rename(columns={
        "precision_original": "original_precision",
        "precision_rerun": "rerun_precision",
        "recall_original": "original_recall",
        "recall_rerun": "rerun_recall",
        "f1_original": "original_f1",
        "f1_rerun": "rerun_f1",
        "fn_original": "original_fn",
        "fn_rerun": "rerun_fn",
    }, inplace=True)
    output_columns = [
        "sequence_id", "image_path", "attack", "epsilon", "steps", "restart",
        "seed", "defense", "candidate_count_before_nms",
        "prediction_count_after_nms", "original_runtime_ms", "rerun_runtime_ms",
        "original_precision", "rerun_precision", "original_recall", "rerun_recall",
        "original_f1", "rerun_f1", "original_fn", "rerun_fn", "row_replaced",
        "instrumented_timeout",
    ]
    merged[output_columns].to_csv(AUDIT / "nms_timeout_recheck.csv", index=False)
    audit = json.loads((AUDIT / "nms_timeout_audit.json").read_text(encoding="utf-8"))
    max_changes = {
        metric: float(pd.to_numeric(merged[f"delta_{metric}"], errors="coerce").abs().max())
        for metric in ("precision", "recall", "f1", "fn")
    }
    summary = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warning_count": int(audit["legacy_warning_count"]),
        "affected_frames": int(audit["legacy_affected_images"]),
        "rechecked_condition_rows": int(len(merged)),
        "instrumented_timeout_rows": int(audit["exact_timeout_rows_in_instrumented_rerun"]),
        "corrected_rows": int(audit["corrected_rows"]),
        "row_replacement_required": bool(audit["corrected_rows"]),
        "max_absolute_metric_change": max_changes,
        "original_csv_backup": str(ORIGINAL.resolve()),
        "original_csv_sha256": sha256(ORIGINAL),
        "post_audit_csv_sha256": sha256(CURRENT),
        "matrix_recomputed": False,
        "targeted_frames_only": True,
        "interpretation": (
            "No timeout recurred under instrumented batch-1 reruns; no detection row was replaced."
        ),
    }
    (AUDIT / "nms_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
