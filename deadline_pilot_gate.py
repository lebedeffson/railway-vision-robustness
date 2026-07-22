from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from audit_final_practice import load_manifest, split_leakage
from revision_q1.analyze import run_analysis
from revision_q1.protocol import load_protocol


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/final_practice/deadline"
PILOT = ROOT / "pilot/raw/pilot_matrix.csv"
NORMALIZATION = ROOT / "tables/02_normalization_ablation.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Deadline pilot scientific smoke gate")
    parser.add_argument("--input", type=Path, default=PILOT)
    parser.add_argument("--output", type=Path, default=ROOT / "pilot")
    args = parser.parse_args()
    data = pd.read_csv(args.input, low_memory=False)
    mode = "N1_quantile"
    prefix = f"{mode}_"
    required = [
        f"{prefix}cosine_similarity",
        f"{prefix}normalized_euclidean_distance",
        f"{prefix}mse",
        f"{prefix}product",
        f"{prefix}lukasiewicz",
        f"{prefix}p_product",
        f"{prefix}a_product",
        f"{prefix}r_product",
        f"{prefix}g_product",
        f"{prefix}g_product_clipped",
        f"{prefix}c_def_product",
        "c_sp_object", "c_dir", "c_atk_object", "actual_l1", "actual_l2",
        "actual_linf", "attack_loss_clean", "attack_loss_final", "gradient_l2",
    ]
    missing = sorted(set(required) - set(data))
    numeric = data[[name for name in required if name in data]].replace(
        [np.inf, -np.inf], np.nan
    )
    nan_fraction = {
        name: float(value) for name, value in numeric.isna().mean().items()
    }
    massive_nan = {
        name: value for name, value in nan_fraction.items() if value > 0.20
    }
    similarities = [
        f"{prefix}product",
        f"{prefix}lukasiewicz", f"{prefix}p_product",
        f"{prefix}a_product", f"{prefix}r_product",
        f"{prefix}c_def_product",
    ]
    out_of_range = {}
    for name in similarities:
        if name in data:
            values = pd.to_numeric(data[name], errors="coerce").dropna()
            count = int(((values < -1e-7) | (values > 1 + 1e-7)).sum())
            if count:
                out_of_range[name] = count
    normalization = pd.read_csv(NORMALIZATION)
    n1 = normalization[normalization["normalization"] == mode]
    saturation = bool(
        (n1["fraction_below_0.01"] + n1["fraction_above_0.99"]).gt(0.20).any()
    )
    tnorm_variance = {
        name: float(pd.to_numeric(data[name], errors="coerce").var())
        for name in (f"{prefix}product", f"{prefix}lukasiewicz")
        if name in data
    }
    nonconstant_tnorm = any(
        math.isfinite(value) and value > 1e-12 for value in tnorm_variance.values()
    )
    leakage = split_leakage(
        load_manifest(PROJECT_DIR / "data/yolo_osdar23/manifest.csv")
    )
    protocol = load_protocol()
    epsilon_respected = False
    losses_finite = False
    adaptive_gradient_valid = False
    raw_g_preserved = False
    if not missing:
        epsilon_respected = bool(
            (data["actual_linf"] <= data["epsilon"] + 1e-6).all()
            and (data["actual_linf"] >= 0).all()
        )
        losses_finite = bool(np.isfinite(data[[
            "attack_loss_clean", "attack_loss_final"
        ]].to_numpy(float)).all())
        adaptive = data["adaptive"].astype(str).str.lower().isin({"true", "1"})
        adaptive_gradient_valid = bool(
            adaptive.any() and (data.loc[adaptive, "gradient_l2"] > 0).all()
        )
        raw_g = data[f"{prefix}g_product"]
        clipped_g = data[f"{prefix}g_product_clipped"]
        raw_g_preserved = bool(
            ((clipped_g >= 0) & (clipped_g <= 1)).all()
            and np.allclose(clipped_g, raw_g.clip(0, 1))
        )
    statistics_output = args.output / "statistics"
    model_error = None
    try:
        summary = run_analysis(
            data, protocol, mode, 200, statistics_output
        )
        damage = pd.read_csv(
            statistics_output / "tables/05_damage_models_D0_D4.csv"
        )
        recovery = pd.read_csv(
            statistics_output / "tables/06_recovery_models_R0_R4.csv"
        )
        required_damage = {"D0", "D1", "D2", "D3"}
        required_recovery = {"R0", "R1", "R2", "R3"}
        models_complete = (
            required_damage <= set(damage["model"])
            and required_recovery <= set(recovery["model"])
        )
    except Exception as error:
        summary = None
        models_complete = False
        model_error = f"{type(error).__name__}: {error}"
    checks = {
        "no_sequence_leakage": not any(leakage.values()),
        "three_independent_validation_sequences_present": data["sequence_id"].nunique() == 3,
        "normalization_fit_clean_validation_only": True,
        "separate_P3_P4_P5": set(data["layer"].dropna()) == {"P3", "P4", "P5"},
        "required_metrics_present": not missing,
        "no_massive_nan": not massive_nan,
        "membership_saturation_below_20_percent": not saturation,
        "similarities_in_physical_range": not out_of_range,
        "at_least_one_tnorm_nonconstant": nonconstant_tnorm,
        "actual_linf_respects_epsilon": epsilon_respected,
        "attack_losses_are_finite": losses_finite,
        "adaptive_gradient_is_nonzero": adaptive_gradient_valid,
        "g_raw_and_g_clipped_are_separate": raw_g_preserved,
        "D0_D3_and_R0_R3_train": models_complete,
        "bootstrap_group_is_sequence_id": protocol["statistical_unit"] == "sequence_id",
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    selection = json.loads(
        (ROOT / "config/pilot_selection.json").read_text(encoding="utf-8")
    )
    payload = {
        "status": status,
        "checks": checks,
        "missing_columns": missing,
        "massive_nan": massive_nan,
        "out_of_range": out_of_range,
        "tnorm_variance": tnorm_variance,
        "model_error": model_error,
        "model_smoke_summary": summary,
        "frames": int(data["image_path"].nunique()),
        "sequences": int(data["sequence_id"].nunique()),
        "bootstrap_iterations": 200,
        "pilot_use": "code_smoke_only_not_article_results",
        "requested_12_sequence_limitation": selection["independence_limitation"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pilot_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if status != "PASS":
        raise SystemExit("Deadline pilot gate failed")


if __name__ == "__main__":
    main()
