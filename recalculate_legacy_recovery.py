from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "outputs/final_practice/unified_diagnostics_val_raw.csv"
DEFAULT_AUDIT = PROJECT_DIR / "outputs/final_practice/audit/legacy_recovery_recalculation.json"
TAU = 1e-8


def recalculate(chunk: pd.DataFrame) -> pd.DataFrame:
    result = chunk.copy()
    result["legacy_metric_family"] = "legacy_compatibility"
    result["canonical_metric_family"] = "not_computed"
    operators = {
        "product": (
            "a_attacked_similarity", "r_restored_similarity",
            "p_clean_preservation", "g_recovery", "c_def",
        ),
        "godel": ("a_godel", "r_godel", "p_godel", "g_godel", "c_def_godel"),
        "lukasiewicz": (
            "a_lukasiewicz", "r_lukasiewicz", "p_lukasiewicz",
            "g_lukasiewicz", "c_def_lukasiewicz",
        ),
    }
    for operator, (a_col, r_col, p_col, g_col, c_col) in operators.items():
        required = {a_col, r_col, p_col, g_col, c_col}
        missing = required - set(result)
        if missing:
            raise RuntimeError(f"Legacy recovery input lacks {sorted(missing)}")
        result[f"{g_col}_legacy_stored_clipped"] = result[g_col]
        raw = (result[r_col] - result[a_col]) / (1.0 - result[a_col] + TAU)
        clipped = raw.clip(0.0, 1.0)
        result[f"{g_col}_raw"] = raw
        result[f"{g_col}_clipped"] = clipped
        result[g_col] = raw
        if operator == "product":
            result[c_col] = result[p_col] * clipped
        elif operator == "godel":
            result[c_col] = np.minimum(result[p_col], clipped)
        else:
            result[c_col] = np.maximum(0.0, result[p_col] + clipped - 1.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore raw legacy G without inference")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()
    temporary = args.input.with_suffix(args.input.suffix + ".recovery.tmp")
    rows = 0
    first = True
    minima: list[float] = []
    for chunk in pd.read_csv(args.input, chunksize=50_000, low_memory=False):
        fixed = recalculate(chunk)
        rows += len(fixed)
        minima.append(float(fixed["g_recovery_raw"].min()))
        fixed.to_csv(temporary, mode="w" if first else "a", header=first, index=False)
        first = False
    if rows == 0:
        raise RuntimeError("Legacy recovery input is empty")
    temporary.replace(args.input)
    payload = {
        "status": "PASS", "rows": rows, "inference_repeated": False,
        "g_raw_min": min(minima), "g_raw_used_for_statistics": True,
        "g_clipped_use": "C_def_and_visualization_only",
        "metric_family": "legacy_compatibility",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
