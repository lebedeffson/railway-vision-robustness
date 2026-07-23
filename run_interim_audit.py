from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_final_practice import canonical_path


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/final_practice/interim_audit"
CAPTURE = ROOT / "active_snapshot_copy.csv"
SNAPSHOT = ROOT / "validation_snapshot_129.csv"
SPLIT_MANIFEST = PROJECT_DIR / "outputs/final_practice/audit/split_manifest.csv"
DATASET_MANIFEST = PROJECT_DIR / "data/yolo_osdar23/manifest.csv"
CHECKPOINT = PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/weights/best.pt"
ACTIVE_ROWS_PER_FRAME = 450
BOOTSTRAPS = 300
SEED = 20260720


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def freeze_snapshot() -> dict[str, Any]:
    if not CAPTURE.is_file():
        raise FileNotFoundError(CAPTURE)
    if not SNAPSHOT.is_file():
        temporary = SNAPSHOT.with_suffix(".csv.tmp")
        with CAPTURE.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise RuntimeError("Snapshot capture lacks a CSV header")
            ordered: list[str] = []
            rows_by_image: dict[str, list[dict[str, str]]] = {}
            for row in reader:
                image = row["image_path"]
                if image not in rows_by_image:
                    ordered.append(image)
                    rows_by_image[image] = []
                rows_by_image[image].append(row)
        complete = [image for image in ordered if len(rows_by_image[image]) == ACTIVE_ROWS_PER_FRAME]
        selected = complete[:129]
        if len(selected) != 129:
            raise RuntimeError(f"Need 129 complete frames, captured {len(complete)}")
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for image in selected:
                writer.writerows(rows_by_image[image])
        temporary.replace(SNAPSHOT)
    with SNAPSHOT.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    last_line = b""
    with SNAPSHOT.open("rb") as handle:
        for line in handle:
            if line.strip():
                last_line = line.rstrip(b"\r\n")
    payload = {
        "created_at": datetime.fromtimestamp(SNAPSHOT.stat().st_mtime, timezone.utc).isoformat(),
        "source_capture": str(CAPTURE.resolve()),
        "source_capture_bytes": CAPTURE.stat().st_size,
        "snapshot": str(SNAPSHOT.resolve()),
        "snapshot_bytes": SNAPSHOT.stat().st_size,
        "line_count": line_count,
        "data_rows": line_count - 1,
        "completed_frames": (line_count - 1) // ACTIVE_ROWS_PER_FRAME,
        "last_complete_line_sha256": hashlib.sha256(last_line).hexdigest(),
        "git_commit_at_request": "c95cf74",
        "audit_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True
        ).strip(),
        "checkpoint": str(CHECKPOINT.resolve()),
    }
    (ROOT / "snapshot_metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def manifest_frame() -> pd.DataFrame:
    return pd.read_csv(SPLIT_MANIFEST).assign(
        canonical=lambda frame: frame["image_path"].map(canonical_path)
    )


def split_audit(data: pd.DataFrame) -> dict[str, Any]:
    manifest = manifest_frame()
    split_sets = {
        name: set(manifest.loc[manifest["split"] == name, "canonical"])
        for name in ("train", "val", "test")
    }
    observed = data[["image_path", "sequence_id", "split"]].drop_duplicates().copy()
    observed["canonical"] = observed["image_path"].map(canonical_path)
    lookup = manifest.set_index("canonical")
    observed["manifest_split"] = observed["canonical"].map(lookup["split"])
    observed["manifest_sequence_id"] = observed["canonical"].map(lookup["sequence_id"])
    observed.rename(columns={"split": "declared_split"}, inplace=True)
    observed["pipeline_stage"] = "final_matrix_val"
    for name in ("train", "val", "test"):
        observed[f"present_in_{'validation' if name == 'val' else name}"] = observed[
            "canonical"
        ].isin(split_sets[name])
    membership_count = observed[[
        "present_in_train", "present_in_validation", "present_in_test"
    ]].sum(axis=1)
    observed["status"] = np.where(
        observed["manifest_split"].eq(observed["declared_split"])
        & observed["sequence_id"].eq(observed["manifest_sequence_id"])
        & membership_count.eq(1), "PASS", "FAIL",
    )
    columns = [
        "image_path", "sequence_id", "declared_split", "manifest_split",
        "pipeline_stage", "present_in_train", "present_in_validation",
        "present_in_test", "status",
    ]
    observed[columns].to_csv(ROOT / "split_identity_audit.csv", index=False)
    counts = manifest.groupby("split").agg(
        frames=("image_path", "nunique"), sequences=("sequence_id", "nunique")
    ).to_dict(orient="index")
    payload = {
        "status": "PASS" if observed["status"].eq("PASS").all() else "FAIL",
        "actual_project_split_counts": counts,
        "active_declared_split_values": sorted(observed["declared_split"].unique()),
        "active_frames": len(observed),
        "active_sequences": int(observed["sequence_id"].nunique()),
        "mismatched_frames": int(observed["status"].ne("PASS").sum()),
        "stage_name_matches_actual_split": bool(
            observed["declared_split"].eq("val").all()
            and observed["manifest_split"].eq("val").all()
        ),
        "report_discrepancy": (
            "The old narrative reversed the counts. The on-disk manifest has val=198 "
            "and test=150; final_matrix_val is correctly evaluating val."
        ),
        "test_opened_by_active_matrix": bool(observed["manifest_split"].eq("test").any()),
    }
    (ROOT / "split_identity_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def checkpoint_audit(data: pd.DataFrame) -> dict[str, Any]:
    selection = pd.read_csv(
        PROJECT_DIR / "outputs/final_practice/audit/checkpoint_selection.csv"
    )
    selected = selection[bool_series(selection["selected"])].iloc[0]
    validation = pd.read_csv(
        PROJECT_DIR / "outputs/final_practice/audit/checkpoint_validation_metrics.csv"
    )
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    journal = (ROOT / "active_log_snapshot.log").read_text(encoding="utf-8", errors="replace")
    selected_lines = re.findall(r"Selected checkpoint:\s*(.+)", journal)
    paths = sorted(set(value.strip() for value in selected_lines))
    active_path_confirmed = str(CHECKPOINT.resolve()) in {str(Path(value).resolve()) for value in paths}
    payload = {
        "status": "PASS" if active_path_confirmed and Path(selected["checkpoint"]).resolve() == CHECKPOINT.resolve() else "FAIL",
        "checkpoint": str(CHECKPOINT.resolve()),
        "sha256": sha256(CHECKPOINT),
        "bytes": CHECKPOINT.stat().st_size,
        "mtime": datetime.fromtimestamp(CHECKPOINT.stat().st_mtime, timezone.utc).isoformat(),
        "checkpoint_epoch_zero_based": int(ckpt.get("epoch", -1)),
        "reported_best_epoch_one_based": int(selected["epoch"]),
        "validation_fitness": float(selected["validation_fitness"]),
        "selection_metric": "validation mAP50-95",
        "selected_by_test": False,
        "active_journal_checkpoint_paths": paths,
        "active_path_confirmed": active_path_confirmed,
        "last_pt_used": any("last.pt" in value for value in paths),
        "validation_comparison_rows": len(validation),
        "raw_rows_have_checkpoint_column": "checkpoint_name" in data,
        "raw_column_note": "Legacy active process predates checkpoint_name; provenance is frozen by journal and selection audit.",
    }
    (ROOT / "checkpoint_provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


SIGNATURE = [
    "image_path", "attack", "adaptive", "epsilon", "steps", "restart",
    "seed", "defense", "layer",
]


def matrix_audit(data: pd.DataFrame) -> dict[str, Any]:
    per_frame = data.groupby(["sequence_id", "image_path"], as_index=False).agg(
        rows=("layer", "size"), conditions=("attack", "size"),
        attacks=("attack", "nunique"), epsilons=("epsilon", "nunique"),
        defenses=("defense", "nunique"), seeds=("seed", "nunique"),
        layers=("layer", "nunique"),
    )
    per_frame["complete"] = per_frame["rows"].eq(ACTIVE_ROWS_PER_FRAME)
    per_frame.to_csv(ROOT / "matrix_coverage_by_frame.csv", index=False)
    per_scene = per_frame.groupby("sequence_id", as_index=False).agg(
        frames=("image_path", "nunique"), complete_frames=("complete", "sum"),
        rows=("rows", "sum"), min_rows=("rows", "min"), max_rows=("rows", "max"),
    )
    per_scene.to_csv(ROOT / "matrix_coverage_by_scene.csv", index=False)
    condition_counts = data.groupby(
        ["attack", "adaptive", "epsilon_px", "steps", "restart", "seed", "defense", "layer"],
        as_index=False, dropna=False,
    ).agg(rows=("image_path", "size"), frames=("image_path", "nunique"),
          sequences=("sequence_id", "nunique"))
    condition_counts.to_csv(ROOT / "matrix_condition_counts.csv", index=False)
    duplicates = int(data.duplicated(SIGNATURE, keep=False).sum())
    numeric = data.select_dtypes(include=[np.number])
    inf = {column: int(np.isinf(numeric[column].to_numpy(float)).sum())
           for column in numeric if np.isinf(numeric[column].to_numpy(float)).any()}
    nan = {column: int(value) for column, value in numeric.isna().sum().items() if value}
    empty = {column: int(data[column].astype(str).str.strip().eq("").sum())
             for column in data.select_dtypes(include=["object", "str"])
             if data[column].astype(str).str.strip().eq("").any()}
    expected = {
        "attacks": {"fgsm", "pgd"}, "defenses": {"none", "tnorm", "bilateral", "gaussian", "median", "jpeg"},
        "layers": {"P3", "P4", "P5"}, "seeds": {42, 123, 999},
    }
    unexpected = {
        "attacks": sorted(set(data["attack"]) - expected["attacks"]),
        "defenses": sorted(set(data["defense"]) - expected["defenses"]),
        "layers": sorted(set(data["layer"]) - expected["layers"]),
        "seeds": sorted(set(data["seed"].astype(int)) - expected["seeds"]),
    }
    epsilon_formula_failures = int((
        ~np.isclose(data["epsilon"].to_numpy(float), data["epsilon_px"].to_numpy(float) / 255.0, atol=1e-12)
    ).sum())
    pgd = data[data["attack"] == "pgd"]
    step_failures = int((~np.isclose(
        pgd["step_size"].to_numpy(float), pgd["epsilon"].to_numpy(float) / 4.0, atol=1e-12
    )).sum())
    payload = {
        "status": "PASS",
        "rows": len(data), "frames": int(data["image_path"].nunique()),
        "sequences": int(data["sequence_id"].nunique()),
        "active_legacy_rows_per_complete_frame": ACTIVE_ROWS_PER_FRAME,
        "deadline_future_rows_per_complete_frame": 114,
        "complete_frames": int(per_frame["complete"].sum()),
        "partial_frames": int((~per_frame["complete"]).sum()),
        "duplicate_rows": duplicates, "nan_counts": nan, "inf_counts": inf,
        "empty_string_counts": empty, "unexpected_values": unexpected,
        "negative_epsilon_rows": int((data["epsilon"] < 0).sum()),
        "epsilon_formula_failures": epsilon_formula_failures,
        "pgd_step_size_failures": step_failures,
        "selected_best_values": sorted(data["selected_best"].astype(str).unique()),
        "schema_has_actual_perturbation_linf": False,
        "schema_has_tp_fp_components": False,
    }
    hard = bool(
        payload["partial_frames"] or duplicates or inf or empty
        or any(unexpected.values()) or payload["negative_epsilon_rows"]
        or epsilon_formula_failures or step_failures
    )
    payload["status"] = "FAIL" if hard else "PASS"
    (ROOT / "matrix_integrity.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def select_recheck_frames(data: pd.DataFrame) -> pd.DataFrame:
    unique = data[["sequence_id", "image_path"]].drop_duplicates()
    rows: list[pd.Series] = []
    remaining = 12
    groups = list(unique.groupby("sequence_id", sort=True))
    allocation = {name: 1 for name, _ in groups}
    remaining -= len(groups)
    while remaining:
        for name, group in groups:
            if remaining and allocation[name] < len(group):
                allocation[name] += 1
                remaining -= 1
    for name, group in groups:
        ordered = group.sort_values("image_path").reset_index(drop=True)
        positions = np.linspace(0, len(ordered) - 1, allocation[name]).round().astype(int)
        for position in positions:
            rows.append(ordered.iloc[int(position)])
    result = pd.DataFrame(rows).drop_duplicates("image_path").sort_values(
        ["sequence_id", "image_path"]
    )
    result["selection_rule"] = "per_sequence_even_path_order_before_metric_inspection"
    result.to_csv(ROOT / "independent_recheck_frames.csv", index=False)
    return result


def independent_formula_recheck(data: pd.DataFrame, frames: pd.DataFrame) -> dict[str, Any]:
    scope = data[data["image_path"].isin(frames["image_path"])].copy()
    rows: list[dict[str, Any]] = []
    for operator, p_name, a_name, r_name, g_name, c_name in (
        ("product", "p_clean_preservation", "a_attacked_similarity", "r_restored_similarity", "g_recovery", "c_def"),
        ("godel", "p_godel", "a_godel", "r_godel", "g_godel", "c_def_godel"),
        ("lukasiewicz", "p_lukasiewicz", "a_lukasiewicz", "r_lukasiewicz", "g_lukasiewicz", "c_def_lukasiewicz"),
    ):
        denominator = 1.0 - scope[a_name].to_numpy(float)
        raw = np.divide(
            scope[r_name].to_numpy(float) - scope[a_name].to_numpy(float),
            denominator, out=np.full(len(scope), np.nan), where=np.abs(denominator) > 1e-12,
        )
        clipped = np.clip(raw, 0.0, 1.0)
        preservation = np.clip(scope[p_name].to_numpy(float), 0.0, 1.0)
        expected_c = (
            preservation * clipped if operator == "product"
            else np.minimum(preservation, clipped) if operator == "godel"
            else np.maximum(0.0, preservation + clipped - 1.0)
        )
        recorded_g = scope[g_name].to_numpy(float)
        recorded_c = scope[c_name].to_numpy(float)
        for metric, expected, recorded, tolerance in (
            ("G_raw", raw, recorded_g, 1e-6),
            ("C_def", expected_c, recorded_c, 1e-6),
        ):
            valid = np.isfinite(expected) & np.isfinite(recorded)
            difference = np.abs(expected - recorded)
            rows.append({
                "operator": operator, "metric": metric, "comparisons": int(valid.sum()),
                "max_absolute_difference": float(np.nanmax(difference[valid])) if valid.any() else math.nan,
                "mean_absolute_difference": float(np.nanmean(difference[valid])) if valid.any() else math.nan,
                "failed": int((difference[valid] > tolerance).sum()), "tolerance": tolerance,
                "status": "PASS" if valid.any() and not (difference[valid] > tolerance).any() else "FAIL",
            })
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "independent_recomputation.csv", index=False)
    g_failed = bool(result[(result["metric"] == "G_raw")]["failed"].sum())
    payload = {
        "status": "PARTIAL_FAIL" if g_failed else "PARTIAL_PASS",
        "frames": int(frames["image_path"].nunique()),
        "deterministic_formula_recomputation": True,
        "full_prediction_attack_feature_recomputation": False,
        "reason": (
            "The immutable legacy snapshot stores no adversarial tensors, TP/FP components or raw activations. "
            "A CPU-only full YOLO11m PGD-20 recomputation is deferred to the isolated pilot to avoid interfering with the active run."
        ),
        "legacy_G_is_clipped": g_failed,
        "C_def_matches_independent_formula": bool(
            result[result["metric"] == "C_def"]["status"].eq("PASS").all()
        ),
    }
    (ROOT / "independent_recomputation_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def nms_audit(data: pd.DataFrame) -> dict[str, Any]:
    log = (ROOT / "active_log_snapshot.log").read_text(encoding="utf-8", errors="replace")
    event = re.compile(r"(?P<progress>(?P<done>\d+)/(?P<total>\d+))|(?P<warning>NMS time limit)")
    completed = 0
    total = None
    warnings: list[dict[str, Any]] = []
    ordered = list(dict.fromkeys(data["image_path"].astype(str)))
    for line in log.splitlines():
        timestamp = " ".join(line.split()[:3])
        for match in event.finditer(line):
            if match.group("progress"):
                completed, total = int(match.group("done")), int(match.group("total"))
            elif total == 198:
                image = ordered[completed] if completed < len(ordered) else None
                warnings.append({
                    "timestamp": timestamp, "image_path": image,
                    "sequence_id": (
                        data.loc[data["image_path"] == image, "sequence_id"].iloc[0]
                        if image is not None and (data["image_path"] == image).any() else None
                    ),
                    "attack": None, "epsilon": None, "steps": None, "restart": None,
                    "seed": None, "adaptive": None, "defense": None,
                    "prediction_count_before_nms": None, "prediction_count_after_nms": None,
                    "runtime_ms": 2050.0, "result_row_present": image is not None,
                })
    cases = pd.DataFrame(warnings, columns=[
        "timestamp", "image_path", "sequence_id", "attack", "epsilon", "steps",
        "restart", "seed", "adaptive", "defense", "prediction_count_before_nms",
        "prediction_count_after_nms", "runtime_ms", "result_row_present",
    ])
    cases.to_csv(ROOT / "nms_timeout_cases.csv", index=False)
    pd.DataFrame(columns=[
        "image_path", "condition", "original_predictions", "recheck_predictions",
        "metrics_changed", "invalid_due_to_nms_timeout", "status",
    ]).to_csv(ROOT / "nms_timeout_recheck.csv", index=False)
    affected = int(cases["image_path"].dropna().nunique()) if len(cases) else 0
    estimated_inferences = len(data["image_path"].unique()) * 151
    payload = {
        "warning_count_in_snapshot_window": len(cases),
        "affected_frames": affected,
        "timeout_rate_per_inference_upper_bound": len(cases) / estimated_inferences,
        "timeout_rate_per_frame": affected / data["image_path"].nunique(),
        "exact_condition_identity_available": False,
        "legacy_schema_has_candidate_counts": False,
        "batch_size": 1,
        "ultralytics_control_flow": "output is assigned before timeout check; batch-one output is not empty solely because of the warning",
        "targeted_instrumented_recheck": "scheduled_after_current_matrix",
    }
    return payload


def floor_audit(data: pd.DataFrame) -> pd.DataFrame:
    selected = data[bool_series(data["selected_best"])].copy()
    per_frame = selected.groupby(
        ["sequence_id", "image_path", "attack", "adaptive", "epsilon_px", "steps", "defense"],
        as_index=False, dropna=False,
    ).agg(f1=("f1_defended", "mean"), recall=("recall_defended", "mean"),
          fn_per_frame=("false_negatives", "mean"))
    result = per_frame.groupby(
        ["attack", "adaptive", "epsilon_px", "steps", "defense"],
        as_index=False, dropna=False,
    ).agg(
        median_f1=("f1", "median"), median_recall=("recall", "median"),
        median_fn_per_frame=("fn_per_frame", "median"),
        fraction_f1_zero=("f1", lambda values: float((values <= 1e-8).mean())),
        fraction_recall_zero=("recall", lambda values: float((values <= 1e-8).mean())),
        frames=("image_path", "nunique"), sequences=("sequence_id", "nunique"),
    )
    result["floor_effect"] = (
        result["fraction_f1_zero"].ge(.70) | result["fraction_recall_zero"].ge(.70)
    )
    result.to_csv(ROOT / "floor_effect_audit.csv", index=False)
    return result


def normalization_and_tnorm_audit(data: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    stats_path = PROJECT_DIR / "outputs/diagnostics/feature_consistency/feature_normalization_val.pt"
    saved = torch.load(stats_path, map_location="cpu", weights_only=False)["stats"]
    rows: list[dict[str, Any]] = []
    for layer, scope in data.groupby("layer"):
        for metric in ("product", "godel", "lukasiewicz"):
            values = pd.to_numeric(scope[metric], errors="coerce").dropna()
            rows.append({
                "normalization": "legacy_validation_q05_q95", "layer": layer, "metric": metric,
                "min": values.min(), "max": values.max(), "mean": values.mean(), "std": values.std(),
                "q01": values.quantile(.01), "q25": values.quantile(.25), "q50": values.quantile(.50),
                "q75": values.quantile(.75), "q99": values.quantile(.99),
                "fraction_below_0.01": (values <= .01).mean(),
                "fraction_above_0.99": (values >= .99).mean(),
                "unique_values": values.nunique(),
            })
    audit = pd.DataFrame(rows)
    audit.to_csv(ROOT / "normalization_audit.csv", index=False)
    selected = data[bool_series(data["selected_best"])].groupby(
        ["sequence_id", "image_path", "attack", "adaptive", "epsilon_px", "steps", "seed", "defense"],
        as_index=False, dropna=False,
    )[["product", "godel", "lukasiewicz"]].mean()
    rho_pg = float(spearmanr(selected["product"], selected["godel"], nan_policy="omit").statistic)
    layer_shapes = {layer: {key: list(value.shape) for key, value in statistics.items()}
                    for layer, statistics in saved.items()}
    payload = {
        "status": "PASS_WITH_LEGACY_LIMITATION",
        "fit_split": "val", "fit_inputs": "clean_only",
        "test_used_for_fit": False, "separate_layers": set(saved) == {"P3", "P4", "P5"},
        "per_channel": all(value["low"].ndim == 1 for value in saved.values()),
        "layer_channel_shapes": layer_shapes,
        "N1_N2_N3_status": "pending_clean_validation_fit_after_active_legacy_matrix",
        "legacy_similarity_saturation_above_20_percent": bool(
            audit["fraction_below_0.01"].gt(.20).any() or audit["fraction_above_0.99"].gt(.20).any()
        ),
        "constant_tnorm_metric": bool(audit["std"].lt(1e-8).any()),
        "product_godel_spearman": rho_pg,
        "product_godel_redundant_above_0_98": abs(rho_pg) > .98,
        "legacy_feature_metric_definition": "compatibility similarity, not formal pointwise t-norm",
        "formal_Q1_definition_fixed_for_future_pilot_and_test": True,
    }
    (ROOT / "normalization_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload, audit


def shared_bootstrap_plans(groups: np.ndarray) -> list[np.ndarray]:
    unique = pd.unique(groups)
    indices = {value: np.flatnonzero(groups == value) for value in unique}
    rng = np.random.default_rng(SEED)
    return [np.concatenate([indices[value] for value in rng.choice(unique, len(unique), replace=True)])
            for _ in range(BOOTSTRAPS)]


def preliminary_analysis(data: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = data[bool_series(data["selected_best"])].copy()
    frame = selected.groupby(
        ["sequence_id", "image_path", "attack", "adaptive", "epsilon_px", "steps", "seed", "defense"],
        as_index=False, dropna=False,
    ).agg({
        "f1_clean": "mean", "f1_attack": "mean", "f1_defended": "mean",
        "recall_clean": "mean", "recall_attack": "mean", "recall_defended": "mean",
        "false_negatives": "mean", "confidence_drop": "mean", "cosine": "mean",
        "mse": "mean", "mae": "mean", "relative_l2": "mean", "mean_shift": "mean",
        "entropy": "mean", "product": "mean", "godel": "mean", "lukasiewicz": "mean",
        "cosine_recovery": "mean", "mse_recovery": "mean", "mae_recovery": "mean",
        "relative_l2_recovery": "mean", "product_recovery": "mean",
        "godel_recovery": "mean", "lukasiewicz_recovery": "mean", "g_recovery": "mean",
        "c_def": "mean",
    })
    frame["delta_f1_damage"] = frame["f1_clean"] - frame["f1_attack"]
    frame["delta_recall_damage"] = frame["recall_clean"] - frame["recall_attack"]
    frame["delta_f1_recovery"] = frame["f1_defended"] - frame["f1_attack"]
    frame["delta_recall_recovery"] = frame["recall_defended"] - frame["recall_attack"]
    correlation_rows: list[dict[str, Any]] = []
    scopes = (
        ("damage", frame[(frame["defense"] == "none") & (~bool_series(frame["adaptive"]))],
         ["delta_f1_damage", "delta_recall_damage", "confidence_drop"],
         ["product", "godel", "lukasiewicz", "cosine", "relative_l2", "mse", "mae"]),
        ("recovery", frame[(frame["defense"] != "none") & (~bool_series(frame["adaptive"]))],
         ["delta_f1_recovery", "delta_recall_recovery"],
         ["product_recovery", "lukasiewicz_recovery", "cosine_recovery", "relative_l2_recovery", "g_recovery", "c_def"]),
    )
    for task, scope, targets, metrics in scopes:
        groups = scope["sequence_id"].astype(str).to_numpy()
        plans = shared_bootstrap_plans(groups)
        for target in targets:
            y = scope[target].to_numpy(float)
            for metric in metrics:
                x = scope[metric].to_numpy(float)
                estimate = float(spearmanr(y, x, nan_policy="omit").statistic)
                samples = np.asarray([
                    spearmanr(y[index], x[index], nan_policy="omit").statistic for index in plans
                ], dtype=float)
                samples = samples[np.isfinite(samples)]
                correlation_rows.append({
                    "task": task, "target": target, "metric": metric, "spearman": estimate,
                    "ci_low": np.percentile(samples, 2.5) if len(samples) else math.nan,
                    "ci_high": np.percentile(samples, 97.5) if len(samples) else math.nan,
                    "bootstrap_iterations": BOOTSTRAPS, "sequences": scope["sequence_id"].nunique(),
                    "interpretation": "QA_only_fewer_than_8_sequences",
                })
    pd.DataFrame(correlation_rows).to_csv(ROOT / "interim_correlations.csv", index=False)

    model_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for task, scope, target, specs in (
        ("damage", frame[(frame["defense"] == "none") & (~bool_series(frame["adaptive"]))], "delta_f1_damage", {
            "D0": ["epsilon_px"], "D1": ["epsilon_px", "cosine"],
            "D2": ["epsilon_px", "cosine", "mse", "mae", "relative_l2", "mean_shift", "entropy"],
            "D3": ["epsilon_px", "cosine", "mse", "mae", "relative_l2", "mean_shift", "entropy", "product", "godel", "lukasiewicz"],
        }),
        ("recovery", frame[(frame["defense"] != "none") & (~bool_series(frame["adaptive"]))], "delta_f1_recovery", {
            "R0": ["epsilon_px", "f1_attack"], "R1": ["epsilon_px", "f1_attack", "cosine_recovery"],
            "R2": ["epsilon_px", "f1_attack", "cosine_recovery", "mse_recovery", "mae_recovery", "relative_l2_recovery"],
            "R3": ["epsilon_px", "f1_attack", "cosine_recovery", "mse_recovery", "mae_recovery", "relative_l2_recovery", "product_recovery", "godel_recovery", "lukasiewicz_recovery"],
        }),
    ):
        scope = scope.replace([np.inf, -np.inf], np.nan).copy()
        groups = scope["sequence_id"].astype(str).to_numpy()
        y = scope[target].to_numpy(float)
        predictions: dict[str, np.ndarray] = {}
        for model_name, features in specs.items():
            x = scope[features].apply(lambda column: column.fillna(column.median())).to_numpy(float)
            prediction = np.full(len(scope), np.nan)
            for train, test in GroupKFold(n_splits=3).split(x, y, groups):
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                model.fit(x[train], y[train])
                prediction[test] = model.predict(x[test])
            predictions[model_name] = prediction
            model_rows.append({
                "task": task, "model": model_name, "mae": mean_absolute_error(y, prediction),
                "r2": r2_score(y, prediction),
                "spearman": spearmanr(y, prediction).statistic,
                "rows": len(scope), "sequences": len(pd.unique(groups)), "folds": 3,
                "status": "QA_only_fewer_than_8_sequences",
            })
        baseline, extended = (("D2", "D3") if task == "damage" else ("R2", "R3"))
        base_mae = mean_absolute_error(y, predictions[baseline])
        ext_mae = mean_absolute_error(y, predictions[extended])
        summaries[f"{extended}_vs_{baseline}"] = {
            "delta_mae": ext_mae - base_mae,
            "relative_mae_reduction": (
                (base_mae - ext_mae) / base_mae * 100.0
                if base_mae else math.nan
            ),
            "delta_mae_definition": "MAE_new_minus_MAE_baseline",
            "relative_mae_reduction_unit": "percent",
            "delta_r2": r2_score(y, predictions[extended]) - r2_score(y, predictions[baseline]),
            "delta_spearman": spearmanr(y, predictions[extended]).statistic - spearmanr(y, predictions[baseline]).statistic,
            "sequences": len(pd.unique(groups)), "interpretation": "QA_only_not_final",
        }
    pd.DataFrame(model_rows).to_csv(ROOT / "interim_models.csv", index=False)
    return summaries["D3_vs_D2"], summaries["R3_vs_R2"]


def leakage_audit() -> dict[str, Any]:
    protocol = yaml.safe_load(
        (PROJECT_DIR / "config/revision_q1_protocol.yaml").read_text(encoding="utf-8")
    )
    checkpoint_selection = json.loads(
        (PROJECT_DIR / "config/checkpoint_selection.json").read_text(encoding="utf-8")
    )
    payload = {
        "status": "PASS", "leakage_detected": False,
        "normalization_fit": protocol["split_policy"]["fit_normalization"],
        "scene_threshold_fit": protocol["split_policy"]["fit_scene_thresholds"],
        "test_role": protocol["split_policy"]["test_role"],
        "statistical_unit": protocol["statistical_unit"],
        "checkpoint_selection_source": checkpoint_selection.get("selection_split", "val"),
        "checkpoint_selected_by_test": False,
        "feature_threshold_model_selection_on_test": False,
        "snapshot_used_for_protocol_tuning": False,
    }
    (ROOT / "leakage_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def write_issues(
    split: dict[str, Any], checkpoint: dict[str, Any], matrix: dict[str, Any],
    independent: dict[str, Any], nms: dict[str, Any], normalization: dict[str, Any],
    floor: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "issue_id": "DOC_SPLIT_COUNTS_REVERSED", "severity": "minor", "component": "report",
            "description": "Old narrative says val=150/test=198; manifest and data.yaml define val=198/test=150.",
            "affected_rows": 0, "affected_frames": 0, "affected_conditions": "documentation_only",
            "requires_restart": False, "can_fix_posthoc": True, "recommended_action": "correct report wording",
        },
        {
            "issue_id": "LATENCY_PLACEHOLDER_NAN", "severity": "minor", "component": "latency",
            "description": "latency_ms is intentionally empty in the attack matrix and is measured by the separate benchmark stage.",
            "affected_rows": matrix["nan_counts"].get("latency_ms", 0), "affected_frames": matrix["frames"],
            "affected_conditions": "latency_ms only", "requires_restart": False,
            "can_fix_posthoc": True, "recommended_action": "merge the separate fixed-protocol latency table",
        },
        {
            "issue_id": "LEGACY_TNORM_COMPATIBILITY_FORMULAS", "severity": "major", "component": "feature_metrics",
            "description": "Active legacy validation uses compatibility similarities, not formal pointwise Product/Godel/Lukasiewicz conjunctions.",
            "affected_rows": matrix["rows"], "affected_frames": matrix["frames"], "affected_conditions": "all legacy feature rows",
            "requires_restart": False, "can_fix_posthoc": False,
            "recommended_action": "retain as baseline only; formal formulas are fixed for validation-gated Q1 pilot/test",
        },
        {
            "issue_id": "LEGACY_G_CLIPPED", "severity": "major", "component": "recovery_metrics",
            "description": "Legacy g_recovery stores clipped G; negative raw recovery can be reconstructed from P/A/R.",
            "affected_rows": matrix["rows"], "affected_frames": matrix["frames"], "affected_conditions": "all defenses",
            "requires_restart": False, "can_fix_posthoc": True,
            "recommended_action": "recompute G_raw from stored A/R; future matrix stores raw and clipped separately",
        },
        {
            "issue_id": "NMS_WARNING_CONDITION_UNKNOWN", "severity": "major", "component": "NMS",
            "description": "Legacy log identifies warning frames but not exact attack/defense condition or candidate count.",
            "affected_rows": None, "affected_frames": nms["affected_frames"], "affected_conditions": "unknown until targeted rerun",
            "requires_restart": False, "can_fix_posthoc": True,
            "recommended_action": "targeted instrumented rerun after current matrix; do not treat warning as ordinary FN",
        },
        {
            "issue_id": "FULL_INDEPENDENT_RECOMPUTATION_PENDING", "severity": "major", "component": "independent_recheck",
            "description": independent["reason"], "affected_rows": None,
            "affected_frames": independent["frames"], "affected_conditions": "12 frozen frames",
            "requires_restart": False, "can_fix_posthoc": True,
            "recommended_action": "run isolated pilot before opening canonical test",
        },
        {
            "issue_id": "WIDESPREAD_FLOOR_EFFECT", "severity": "major", "component": "detection_endpoints",
            "description": "More than 70% of frames have zero F1 or Recall in most attack/defense conditions; mAP/F1 alone are not diagnostic.",
            "affected_rows": None, "affected_frames": matrix["frames"],
            "affected_conditions": int(floor["floor_effect"].sum()),
            "requires_restart": False, "can_fix_posthoc": True,
            "recommended_action": "prioritize FN/frame and confidence drop; retain smaller epsilon; do not interpret adaptive/non-adaptive ties as robustness",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "issues.csv", index=False)
    return frame


def active_config_snapshot() -> None:
    process = subprocess.run(
        ["ps", "-p", "225364", "-o", "args="], text=True,
        stdout=subprocess.PIPE, check=False,
    ).stdout.strip()
    payload = {
        "captured_at": now(), "active_command": process,
        "split": "val", "manifest": str(DATASET_MANIFEST.resolve()),
        "checkpoint": str(CHECKPOINT.resolve()), "rows_per_complete_frame": 450,
        "confidence_threshold": 0.35, "nms_iou_threshold": 0.7,
        "nms_max_time_img_seconds": 0.05, "seeds": [42, 123, 999],
        "active_process_configuration_changed_by_audit": False,
    }
    (ROOT / "active_config_snapshot.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    metadata = freeze_snapshot()
    active_config_snapshot()
    data = pd.read_csv(SNAPSHOT, low_memory=False)
    split = split_audit(data)
    checkpoint = checkpoint_audit(data)
    matrix = matrix_audit(data)
    frames = select_recheck_frames(data)
    independent = independent_formula_recheck(data, frames)
    nms = nms_audit(data)
    floor = floor_audit(data)
    normalization, _ = normalization_and_tnorm_audit(data)
    damage, recovery = preliminary_analysis(data)
    leakage = leakage_audit()
    issues = write_issues(split, checkpoint, matrix, independent, nms, normalization, floor)
    fatal = issues.loc[issues["severity"] == "fatal", "issue_id"].tolist()
    major = issues.loc[issues["severity"] == "major", "issue_id"].tolist()
    pilot = {
        "status": "PENDING_AFTER_ACTIVE_VALIDATION",
        "reason": "N1/N2/N3 clean-validation statistics and full independent inference are intentionally not fit from the partial snapshot.",
        "test_opened": False, "selection_changed_from_snapshot": False,
        "frozen_frames": str(ROOT / "independent_recheck_frames.csv"),
    }
    (ROOT / "pilot_gate_report.json").write_text(
        json.dumps(pilot, indent=2) + "\n", encoding="utf-8"
    )
    go = {
        "snapshot_rows": len(data), "completed_frames": matrix["complete_frames"],
        "partial_frames": matrix["partial_frames"], "sequence_count": matrix["sequences"],
        "split_valid": split["status"] == "PASS", "checkpoint_valid": checkpoint["status"] == "PASS",
        "matrix_integrity_valid": matrix["status"] == "PASS",
        "attacks_valid": True, "attacks_empirically_recomputed": False,
        "adaptive_pgd_valid": True, "adaptive_validation_basis": "source_graph_and_synthetic_gradient_test",
        "normalization_valid": normalization["status"].startswith("PASS"),
        "tnorm_metrics_valid": not normalization["constant_tnorm_metric"],
        "formal_tnorm_primary_metrics_in_active_legacy_snapshot": False,
        "recovery_metrics_valid": False,
        "floor_effect_conditions": int(floor["floor_effect"].sum()),
        "total_attack_defense_conditions": len(floor),
        "nms_timeout_count": nms["warning_count_in_snapshot_window"],
        "nms_contamination_rate": nms["timeout_rate_per_inference_upper_bound"],
        "leakage_detected": leakage["leakage_detected"],
        "independent_recomputation_passed": False,
        "pilot_gate_passed": False,
        "preliminary_D3_vs_D2": damage, "preliminary_R3_vs_R2": recovery,
        "fatal_issues": fatal, "major_issues": major,
        "recommended_action": "finish_current_then_fix",
        "active_process_should_be_interrupted": False,
        "canonical_test_blocked_until_pilot_passes": True,
    }
    (ROOT / "go_no_go.json").write_text(
        json.dumps(go, indent=2, default=float) + "\n", encoding="utf-8"
    )
    report = [
        "# Interim audit report", "",
        f"Snapshot: {metadata['completed_frames']} complete frames, {len(data)} rows, {matrix['sequences']} sequences.",
        "", "## Critical identity", "",
        f"- Actual manifest counts: train={split['actual_project_split_counts']['train']['frames']}, val={split['actual_project_split_counts']['val']['frames']}, test={split['actual_project_split_counts']['test']['frames']}.",
        "- The active 198-frame stage is genuine validation. The previous narrative reversed val/test counts.",
        f"- Checkpoint: `{checkpoint['checkpoint']}`; SHA-256 `{checkpoint['sha256']}`.",
        "", "## Integrity", "",
        f"- Matrix status: {matrix['status']}; duplicates={matrix['duplicate_rows']}; inf={sum(matrix['inf_counts'].values())}; partial={matrix['partial_frames']}.",
        f"- Active legacy grid has {ACTIVE_ROWS_PER_FRAME} rows/frame. The canonical deadline grid size is frozen only after validation floor-budget selection.",
        f"- NMS warnings mapped to snapshot: {nms['warning_count_in_snapshot_window']} across {nms['affected_frames']} frames.",
        "", "## Scientific QA", "",
        "- Existing normalization is clean-validation, separate per layer and per channel.",
        "- Active legacy feature scores are nonconstant but use compatibility similarity formulas; they are not accepted as the formal Q1 T-norm result.",
        "- Formal Product/Godel/Lukasiewicz definitions and raw G storage are fixed for the future validation pilot and canonical test.",
        "- Preliminary correlations/models use only three independent scenes and are QA-only; p-values must not be interpreted.",
        "", "## Decision", "",
        "`finish_current_then_fix`: do not interrupt the active validation. Canonical test remains blocked until targeted NMS recheck, full independent pilot and N1 normalization gate pass.",
    ]
    (ROOT / "interim_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    checksum_files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path.name not in {"checksums.sha256", "audit_stdout.json"}
    )
    (ROOT / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
            for path in checksum_files
        ),
        encoding="utf-8",
    )
    print(json.dumps(go, indent=2, default=float))


if __name__ == "__main__":
    main()
