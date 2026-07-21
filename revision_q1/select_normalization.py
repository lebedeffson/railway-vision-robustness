from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from revision_q1.analyze import (
    ENDPOINTS,
    add_endpoints,
    aggregate_task,
    apply_normalization_aliases,
    boolean_series,
    model_specifications,
    out_of_fold_predictions,
)
from revision_q1.normalization import MODES
from revision_q1.protocol import assert_split_action, load_protocol, output_root


def validation_scores(data: pd.DataFrame, protocol: dict, mode: str) -> dict[str, float]:
    normalized = add_endpoints(apply_normalization_aliases(data, mode))
    values: dict[str, float] = {}
    for task, model_name in (("damage", "D3"), ("recovery", "R3")):
        specifications = model_specifications(protocol, task)
        selected = {model_name: specifications[model_name]}
        endpoint = ENDPOINTS[task][0]
        prepared = aggregate_task(normalized, task, endpoint, selected)
        models, _ = out_of_fold_predictions(
            prepared,
            endpoint,
            selected,
            "ridge",
            int(protocol["statistics"]["group_folds"]),
        )
        row = models.iloc[0]
        values[f"validation_{task}_cv_mae"] = float(row["mae"])
        values[f"validation_{task}_cv_r2"] = float(row["r2"])
        values[f"validation_{task}_cv_spearman"] = float(row["spearman"])
    values["validation_cv_mae"] = (
        values["validation_damage_cv_mae"] + values["validation_recovery_cv_mae"]
    ) / 2
    values["validation_cv_r2"] = (
        values["validation_damage_cv_r2"] + values["validation_recovery_cv_r2"]
    ) / 2
    return values


def main() -> None:
    protocol = load_protocol()
    root = output_root(protocol)
    parser = argparse.ArgumentParser(description="Select Q1 normalization on validation only")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--diagnostics", type=Path,
        default=root / "tables/02_normalization_ablation.csv",
    )
    parser.add_argument("--output", type=Path, default=root)
    args = parser.parse_args()
    assert_split_action("select_normalization", "val")
    raw = pd.read_csv(args.input)
    if raw["split"].dropna().ne("val").any():
        raise RuntimeError("Normalization selection accepts validation rows only")
    diagnostics = pd.read_csv(args.diagnostics)
    rows: list[dict[str, object]] = []
    for mode in MODES:
        scope = diagnostics[diagnostics["normalization"] == mode]
        if len(scope) != 3 or set(scope["layer"]) != {"P3", "P4", "P5"}:
            raise RuntimeError(f"{mode}: incomplete P3/P4/P5 normalization diagnostics")
        rows.append({
            "normalization": mode,
            "saturation_warning": bool(boolean_series(scope["saturation_warning"]).any()),
            "fraction_below_0.01_max": float(scope["fraction_below_0.01"].max()),
            "fraction_above_0.99_max": float(scope["fraction_above_0.99"].max()),
            **validation_scores(raw, protocol, mode),
        })
    mode_summary = pd.DataFrame(rows)
    eligible = mode_summary[~mode_summary["saturation_warning"]]
    if eligible.empty:
        eligible = mode_summary
        selection_warning = "all_normalizations_exceeded_saturation_threshold"
    else:
        selection_warning = None
    ordered = eligible.sort_values(
        ["validation_cv_r2", "validation_cv_mae", "normalization"],
        ascending=[False, True, True],
    )
    selected = str(ordered.iloc[0]["normalization"])
    diagnostics = diagnostics.drop(
        columns=[
            column for column in ("validation_cv_mae", "validation_cv_r2", "selected")
            if column in diagnostics
        ]
    )
    summary = diagnostics.merge(mode_summary, on="normalization", suffixes=("", "_mode"))
    summary["selected"] = summary["normalization"].eq(selected)
    args.output.joinpath("tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "tables/02_normalization_ablation.csv", index=False)
    payload = {
        "selected_normalization": selected,
        "selection_split": "val",
        "test_used": False,
        "rule": protocol["normalization"]["primary_selection_rule"],
        "warning": selection_warning,
    }
    args.output.joinpath("config").mkdir(parents=True, exist_ok=True)
    (args.output / "config/normalization_selection.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
