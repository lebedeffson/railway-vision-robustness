from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from audit_final_practice import canonical_path
from revision_q1.analyze import (
    ENDPOINTS,
    add_endpoints,
    aggregate_task,
    apply_normalization_aliases,
    model_specifications,
    out_of_fold_predictions,
    paired_model_bootstrap,
)
from revision_q1.protocol import load_protocol, output_root


def main() -> None:
    protocol = load_protocol()
    root = output_root(protocol)
    parser = argparse.ArgumentParser(description="Frozen validation-defined scene strata analysis")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--difficulty", type=Path, default=root / "raw/scene_difficulty.csv")
    parser.add_argument("--selection", type=Path, default=root / "config/normalization_selection.json")
    parser.add_argument("--output", type=Path, default=root)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    mode = selection["selected_normalization"]
    matrix = add_endpoints(apply_normalization_aliases(pd.read_csv(args.input), mode))
    difficulty = pd.read_csv(args.difficulty)
    matrix["canonical_image"] = matrix["image_path"].map(canonical_path)
    difficulty["canonical_image"] = difficulty["image_path"].map(canonical_path)
    merged = matrix.merge(
        difficulty[[
            "canonical_image", "object_count", "small_object_fraction",
            "object_count_stratum", "small_object_stratum",
        ]], on="canonical_image", how="left", validate="many_to_one",
    )
    if merged[["object_count_stratum", "small_object_stratum"]].isna().any().any():
        raise RuntimeError("Scene-difficulty join left unmatched experiment images")
    rows: list[dict[str, object]] = []
    for stratum_type in ("object_count_stratum", "small_object_stratum"):
        for stratum, scope in merged.groupby(stratum_type):
            for task, pair in (("damage", ("D2", "D3")), ("recovery", ("R2", "R3"))):
                specifications = model_specifications(protocol, task)
                selected_specs = {name: specifications[name] for name in pair}
                endpoint = ENDPOINTS[task][0]
                status = "confirmatory"
                try:
                    prepared = aggregate_task(scope, task, endpoint, selected_specs)
                    models, predictions = out_of_fold_predictions(
                        prepared, endpoint, selected_specs, "ridge",
                        int(protocol["statistics"]["group_folds"]),
                    )
                    gains = paired_model_bootstrap(
                        predictions, endpoint, (pair,),
                        int(protocol["bootstrap_iterations"]), int(protocol["random_seed"]),
                    )
                    gain = gains.set_index("metric")
                    sequence_count = int(prepared["sequence_id"].nunique())
                    if sequence_count < int(protocol["scene_difficulty"]["exploratory_min_sequences"]):
                        status = "exploratory"
                    row = {
                        "stratum_type": stratum_type,
                        "stratum": stratum,
                        "task": task,
                        "frames": int(scope["image_path"].nunique()),
                        "sequences": sequence_count,
                        "baseline_mae": float(models.set_index("model").loc[pair[0], "mae"]),
                        "extended_mae": float(models.set_index("model").loc[pair[1], "mae"]),
                        "delta_mae": float(gain.loc["delta_mae", "estimate"]),
                        "delta_mae_ci_low": float(gain.loc["delta_mae", "ci_low"]),
                        "delta_mae_ci_high": float(gain.loc["delta_mae", "ci_high"]),
                        "delta_r2": float(gain.loc["delta_r2", "estimate"]),
                        "delta_r2_ci_low": float(gain.loc["delta_r2", "ci_low"]),
                        "delta_r2_ci_high": float(gain.loc["delta_r2", "ci_high"]),
                        "status": status,
                    }
                except (RuntimeError, ValueError) as error:
                    row = {
                        "stratum_type": stratum_type, "stratum": stratum, "task": task,
                        "frames": int(scope["image_path"].nunique()),
                        "sequences": int(scope["sequence_id"].nunique()),
                        "status": "exploratory_unavailable", "error": str(error),
                    }
                rows.append(row)
    args.output.joinpath("tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        args.output / "tables/09_scene_difficulty_analysis.csv", index=False
    )
    print(json.dumps({"normalization": mode, "strata_results": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
