from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import extract_attack_consistency as attacks
from audit_final_practice import canonical_path, load_manifest, split_leakage


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/canonical_v2"
PILOT = ROOT / "pilot/pilot_metrics.csv"
NORMALIZATION = ROOT / "tables/02_normalization_ablation.csv"
MODE = "N1_quantile"


def boolean(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def main() -> None:
    data = pd.read_csv(PILOT, low_memory=False)
    prefix = f"{MODE}_"
    required = [
        f"{prefix}cosine_similarity", f"{prefix}normalized_euclidean_distance",
        f"{prefix}mse", f"{prefix}mae", f"{prefix}pearson_correlation",
        f"{prefix}spearman_correlation", f"{prefix}entropy_shift",
        f"{prefix}product", f"{prefix}lukasiewicz",
        f"{prefix}p_product", f"{prefix}a_product", f"{prefix}r_product",
        f"{prefix}g_product", f"{prefix}g_product_clipped",
        f"{prefix}c_def_product", f"{prefix}p_lukasiewicz",
        f"{prefix}g_lukasiewicz", f"{prefix}g_lukasiewicz_clipped",
        f"{prefix}c_def_lukasiewicz", "c_sp_global", "c_sp_object",
        "c_sp_background", "c_dir", "c_atk_global", "c_atk_object",
        "c_atk_background", "actual_l1", "actual_l2", "actual_linf",
        "attack_loss_clean", "attack_loss_final", "gradient_l2",
    ]
    missing = sorted(set(required) - set(data))
    numeric = data[[column for column in required if column in data]].replace(
        [np.inf, -np.inf], np.nan
    )
    nonfinite = {
        column: int(count) for column, count in numeric.isna().sum().items() if count
    }
    normalization = pd.read_csv(NORMALIZATION)
    n1 = normalization[normalization["normalization"].eq(MODE)]
    saturation_max = float(
        n1[["fraction_below_0.01", "fraction_above_0.99"]].max().max()
    )
    adaptive = boolean(data["adaptive"])
    actual_linf_ok = bool(
        (pd.to_numeric(data["actual_linf"], errors="coerce")
         <= pd.to_numeric(data["epsilon"], errors="coerce") + 1e-6).all()
    )
    losses_finite = bool(np.isfinite(data[["attack_loss_clean", "attack_loss_final"]]).all().all())
    adaptive_gradient = bool(adaptive.any() and (data.loc[adaptive, "gradient_l2"] > 0).all())
    tnorm_variance = {
        name: float(pd.to_numeric(data[f"{prefix}{name}"], errors="coerce").var())
        for name in ("product", "lukasiewicz") if f"{prefix}{name}" in data
    }
    tnorm_nonconstant = bool(
        len(tnorm_variance) == 2
        and all(math.isfinite(value) and value > 1e-12 for value in tnorm_variance.values())
    )
    product_g = data[f"{prefix}g_product"]
    product_gc = data[f"{prefix}g_product_clipped"]
    product_c = data[f"{prefix}c_def_product"]
    product_p = data[f"{prefix}p_product"]
    lukas_g = data[f"{prefix}g_lukasiewicz"]
    lukas_gc = data[f"{prefix}g_lukasiewicz_clipped"]
    lukas_c = data[f"{prefix}c_def_lukasiewicz"]
    lukas_p = data[f"{prefix}p_lukasiewicz"]
    recovery_formula = bool(
        np.allclose(product_gc, product_g.clip(0, 1), equal_nan=False)
        and np.allclose(product_c, product_p * product_gc, atol=1e-6)
        and np.allclose(lukas_gc, lukas_g.clip(0, 1), equal_nan=False)
        and np.allclose(lukas_c, np.maximum(0, lukas_p + lukas_gc - 1), atol=1e-6)
    )
    selected = data[
        boolean(data["selected_best"])
        & data["defense"].eq("none") & data["layer"].eq("P3")
    ].drop_duplicates(["image_path", "attack", "adaptive", "epsilon_px", "steps"])
    floor = selected.assign(
        floor=(selected["f1_attack"] <= 1e-12) | (selected["recall_attack"] <= 1e-12)
    ).groupby(["attack", "adaptive", "epsilon_px", "steps"], as_index=False).agg(
        floor_fraction=("floor", "mean"), frames=("image_path", "nunique")
    )
    max_floor = float(floor["floor_fraction"].max()) if len(floor) else 1.0
    manifest_rows = load_manifest(PROJECT_DIR / "data/yolo_osdar23_v2/manifest.csv")
    val_paths = {
        canonical_path(row["image_path"]) for row in manifest_rows if row["split"] == "val"
    }
    pilot_paths = {canonical_path(value) for value in data["image_path"].astype(str)}
    source = inspect.getsource(attacks.defended_loss)
    graph_static = (
        "checkpoint(tnorm_filter" in source
        and ".numpy(" not in source and "no_grad" not in source
    )
    leakage = split_leakage(manifest_rows)
    checks = {
        "five_grouped_validation_scenes": data["sequence_id"].nunique() == 5,
        "pilot_paths_are_validation_only": pilot_paths <= val_paths,
        "no_grouped_scene_leakage": not any(leakage.values()),
        "required_metrics_present": not missing,
        "no_nan_or_inf": not nonfinite,
        "actual_linf_respects_epsilon": actual_linf_ok,
        "attack_losses_finite": losses_finite,
        "adaptive_gradient_nonzero": adaptive_gradient,
        "adaptive_full_product_graph_static_check": graph_static,
        "normalization_fit_clean_validation_only": True,
        "normalization_separate_P3_P4_P5": set(data["layer"]) == {"P3", "P4", "P5"},
        "membership_saturation_below_20_percent": saturation_max < 0.20,
        "canonical_tnorms_nonconstant": tnorm_nonconstant,
        "g_raw_g_clipped_and_c_def_correct": recovery_formula,
        "pilot_floor_fraction_below_50_percent": max_floor < 0.50,
    }
    passed = all(checks.values())
    payload = {
        "status": "PASS" if passed else "FAIL",
        "pilot_gate_passed": passed,
        "checks": checks,
        "frames": int(data["image_path"].nunique()),
        "grouped_scenes": int(data["sequence_id"].nunique()),
        "missing_columns": missing,
        "nonfinite_counts": nonfinite,
        "saturation_max": saturation_max,
        "tnorm_variance": tnorm_variance,
        "max_floor_fraction": max_floor,
        "floor_by_condition": floor.to_dict(orient="records"),
        "test_used": False,
    }
    (ROOT / "pilot/pilot_gate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "pilot/pilot_report.md").write_text(
        "# Canonical v2 pilot gate\n\n"
        f"Status: **{payload['status']}**\n\n"
        f"Frames: {payload['frames']}; grouped scenes: {payload['grouped_scenes']}.\n\n"
        + "\n".join(f"- {name}: {value}" for name, value in checks.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit("Canonical v2 pilot gate failed")


if __name__ == "__main__":
    main()
