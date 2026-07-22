from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from audit_final_practice import load_manifest
from revision_q1.analyze import boolean_series
from revision_q1.statistics import cluster_mean_interval


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/final_practice/deadline"
FINAL_ROOT = PROJECT_DIR / "outputs/final_practice"
MAIN_MATRIX = FINAL_ROOT / "unified_diagnostics_raw.csv"
MAIN_CONFIG = MAIN_MATRIX.with_suffix(".json")
PROTOCOL = PROJECT_DIR / "config/deadline_protocol.yaml"
REVISION_PROTOCOL = PROJECT_DIR / "config/revision_q1_protocol.yaml"
MANIFEST = PROJECT_DIR / "data/yolo_osdar23/manifest.csv"
DATA = PROJECT_DIR / "data/yolo_osdar23/data.yaml"
STAGE1 = PROJECT_DIR / "outputs/training/yolo11m_baseline_stage1/weights/best.pt"
STAGE2 = PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/weights/best.pt"
PYTHON = PROJECT_DIR / ".venv/bin/python"
STATUS = ROOT / "pipeline_status.json"
ZIP_PATH = PROJECT_DIR / "outputs/bundles/TNormFilter_deadline_final.zip"
ITERATIONS = 5000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=PROJECT_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()


def read_status() -> dict[str, Any]:
    if STATUS.is_file():
        return json.loads(STATUS.read_text(encoding="utf-8"))
    return {"mode": "deadline_minimal_v1", "status": "running", "stages": {}}


def write_status(payload: dict[str, Any]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now()
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def run_stage(name: str, command: list[str], outputs: list[Path]) -> None:
    payload = read_status()
    payload.setdefault("stages", {})
    if outputs and all(path.is_file() for path in outputs):
        payload["stages"][name] = {
            "status": "success", "finished_at": now(),
            "outputs": [str(path) for path in outputs], "error": None,
        }
        write_status(payload)
        print(f"[deadline-final] {name}: already complete", flush=True)
        return
    payload["stages"][name] = {
        "status": "running", "started_at": now(), "outputs": [], "error": None,
    }
    payload["status"] = "running"
    write_status(payload)
    try:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
        missing = [str(path) for path in outputs if not path.is_file()]
        if missing:
            raise RuntimeError(f"{name}: missing outputs {missing}")
    except Exception as error:
        payload = read_status()
        payload["stages"][name].update({
            "status": "failed", "finished_at": now(),
            "error": f"{type(error).__name__}: {error}",
        })
        payload["status"] = "failed"
        write_status(payload)
        raise
    payload = read_status()
    payload["stages"][name].update({
        "status": "success", "finished_at": now(),
        "outputs": [str(path) for path in outputs], "error": None,
    })
    write_status(payload)


def validate_main_matrix() -> None:
    for path in (MAIN_MATRIX, MAIN_CONFIG, STAGE1, STAGE2):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(MAIN_CONFIG.read_text(encoding="utf-8"))
    budget_path = ROOT / "config/canonical_budget_selection.json"
    if not budget_path.is_file():
        raise FileNotFoundError(budget_path)
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    if budget.get("status") != "PASS":
        raise RuntimeError("Canonical budget selection did not pass")
    expected = {
        "split": "test",
        "checkpoint_name": "stage2_best",
        "normalizations": ["N1_quantile"],
        "defenses": ["none", "tnorm", "bilateral", "median"],
        "seeds": [42, 123, 999],
        "adaptive_pgd_eps": budget["selected"]["adaptive_pgd_epsilon_px"],
        "adaptive_pgd_steps": [20],
    }
    conditions = config.get("conditions", [])
    actual_fgsm = sorted({float(row[1]) for row in conditions if row[0] == "fgsm"})
    actual_pgd = sorted({
        float(row[1]) for row in conditions if row[0] == "pgd" and not row[3]
    })
    expected_fgsm = sorted(map(float, budget["selected"]["fgsm_epsilon_px"]))
    expected_pgd = sorted(map(float, budget["selected"]["pgd_epsilon_px"]))
    if actual_fgsm != expected_fgsm:
        mismatches = {"fgsm_eps": {"expected": expected_fgsm, "actual": actual_fgsm}}
    else:
        mismatches = {}
    if actual_pgd != expected_pgd:
        mismatches["pgd_eps"] = {"expected": expected_pgd, "actual": actual_pgd}
    mismatches.update({
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items() if config.get(key) != value
    })
    if Path(config["model"]).resolve() != STAGE2.resolve():
        mismatches["model"] = {
            "expected": str(STAGE2.resolve()), "actual": config.get("model")
        }
    if mismatches:
        raise RuntimeError(f"Canonical deadline matrix configuration mismatch: {mismatches}")
    data = pd.read_csv(MAIN_MATRIX, nrows=10)
    required = {
        "sequence_id", "checkpoint_name", "N1_quantile_product",
        "N1_quantile_cosine_similarity", "c_atk_object", "fn_clean",
        "fn_attack", "fn_defended", "nms_timeout",
    }
    missing = required - set(data)
    if missing:
        raise RuntimeError(f"Canonical deadline matrix lacks columns: {sorted(missing)}")


def select_sensitivity_frames() -> None:
    destination = ROOT / "config"
    destination.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    test_rows = pd.DataFrame([
        row for row in load_manifest(MANIFEST) if row["split"] == "test"
    ])
    grouped = {
        sequence_id: scope.sort_values("image_path").reset_index(drop=True)
        for sequence_id, scope in test_rows.groupby("sequence_id", sort=True)
    }
    allocations = {name: min(15, len(scope)) for name, scope in grouped.items()}
    while sum(allocations.values()) < 45:
        progressed = False
        for name, scope in grouped.items():
            if allocations[name] < len(scope):
                allocations[name] += 1
                progressed = True
                if sum(allocations.values()) == 45:
                    break
        if not progressed:
            raise RuntimeError("Test manifest has fewer than 45 unique frames")
    for sequence_id, scope in grouped.items():
        count = allocations[sequence_id]
        indices = np.linspace(0, len(scope) - 1, count).round().astype(int)
        if len(np.unique(indices)) != count:
            raise RuntimeError(f"Could not freeze {count} distinct frames for {sequence_id}")
        for index in indices:
            row = scope.iloc[int(index)]
            selected.append({
                "sequence_id": sequence_id,
                "image_path": row["image_path"],
                "selection_source": "deterministic_even_spacing_before_results",
            })
    frame = pd.DataFrame(selected)
    if len(frame) != 45 or frame["sequence_id"].nunique() != 3:
        raise RuntimeError("Stage 1 sensitivity must freeze 45 frames across 3 scenes")
    frame.to_csv(destination / "stage1_sensitivity_manifest.csv", index=False)
    image_list = destination / "stage1_sensitivity_images.txt"
    image_list.write_text("\n".join(frame["image_path"]) + "\n", encoding="utf-8")
    base = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    base["val"] = str(image_list)
    base["test"] = str(image_list)
    (destination / "stage1_sensitivity_data.yaml").write_text(
        yaml.safe_dump(base, sort_keys=False), encoding="utf-8"
    )
    payload = {
        "status": "FROZEN_BEFORE_STAGE1_RESULTS",
        "selection_uses_model_results": False,
        "selection_source": "45 unique evenly spaced paths, balanced as scene sizes permit",
        "frames": len(frame), "sequences": frame["sequence_id"].nunique(),
        "frames_per_sequence": frame.groupby("sequence_id").size().to_dict(),
        "interpretation": "45-frame exploratory checkpoint sensitivity; unequal 10/18/17 scene allocation",
        "manifest_sha256": sha256(destination / "stage1_sensitivity_manifest.csv"),
    }
    (destination / "stage1_sensitivity_selection.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def make_stage2_sensitivity_subset() -> None:
    selected = pd.read_csv(ROOT / "config/stage1_sensitivity_manifest.csv")
    images = set(selected["image_path"].astype(str))
    budget = json.loads(
        (ROOT / "config/canonical_budget_selection.json").read_text(encoding="utf-8")
    )["selected"]
    fgsm_epsilon = max(map(float, budget["fgsm_epsilon_px"]))
    pgd_epsilon = max(map(float, budget["pgd_epsilon_px"]))
    adaptive_epsilon = max(map(float, budget["adaptive_pgd_epsilon_px"]))
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(MAIN_MATRIX, chunksize=50_000, low_memory=False):
        scope = chunk[chunk["image_path"].astype(str).isin(images)].copy()
        adaptive = boolean_series(scope["adaptive"])
        adaptive = boolean_series(scope["adaptive"])
        conditions = (
            ((scope["attack"] == "fgsm") & np.isclose(scope["epsilon_px"], fgsm_epsilon))
            | ((scope["attack"] == "pgd") & (~adaptive)
               & np.isclose(scope["epsilon_px"], pgd_epsilon) & (scope["steps"] == 20))
            | ((scope["attack"] == "pgd") & adaptive
               & np.isclose(scope["epsilon_px"], adaptive_epsilon) & (scope["steps"] == 20))
        )
        scope = scope[
            conditions & scope["defense"].isin(["none", "tnorm"])
            & (~adaptive | (scope["attack"] == "pgd"))
        ]
        chunks.append(scope)
    if not chunks:
        raise RuntimeError("Stage 2 sensitivity subset is empty")
    output = pd.concat(chunks, ignore_index=True)
    expected_frames = len(selected)
    if output["sequence_id"].nunique() != 3 or output["image_path"].nunique() != expected_frames:
        raise RuntimeError("Stage 2 sensitivity subset lost frozen frames or sequences")
    destination = ROOT / "raw/stage2_sensitivity.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)


def selected_best_rows(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    if "selected_best" in result:
        result = result[boolean_series(result["selected_best"])]
    if "exclude_from_primary_statistics" in result:
        result = result[~boolean_series(result["exclude_from_primary_statistics"])]
    return result


def copy_csv(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def model_gain_table(analysis: Path) -> pd.DataFrame:
    return pd.read_csv(analysis / "tables/12_multiple_comparison_corrections.csv")


def primary_gain(gains: pd.DataFrame, task: str) -> pd.DataFrame:
    endpoint = "delta_f1_damage" if task == "damage" else "delta_f1_recovery"
    comparison = "D3_vs_D2" if task == "damage" else "R3_vs_R2"
    return gains[
        (gains["task"] == task) & (gains["endpoint"] == endpoint)
        & (gains["algorithm"] == "ridge") & (gains["comparison"] == comparison)
    ].copy()


def adaptive_interval(data: pd.DataFrame) -> dict[str, float | int]:
    selected = selected_best_rows(data)
    scope = selected[
        (selected["attack"] == "pgd") & np.isclose(selected["epsilon_px"], 1.0)
        & (selected["steps"] == 20) & (selected["defense"] == "tnorm")
    ].copy()
    scope["adaptive_bool"] = boolean_series(scope["adaptive"])
    per_image = scope.groupby(
        ["sequence_id", "image_path", "adaptive_bool"], as_index=False
    )["f1_defended"].mean()
    pivot = per_image.pivot(
        index=["sequence_id", "image_path"], columns="adaptive_bool", values="f1_defended"
    ).dropna()
    if False not in pivot or True not in pivot:
        return {"estimate": math.nan, "ci_low": math.nan, "ci_high": math.nan,
                "sequences": int(scope["sequence_id"].nunique()),
                "bootstrap_iterations": ITERATIONS}
    frame = pivot.reset_index()
    frame["difference"] = frame[True] - frame[False]
    return cluster_mean_interval(
        frame, "difference", iterations=ITERATIONS, seed=20260720
    )


def build_sensitivity_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for checkpoint, analysis in (
        ("stage2_best", ROOT / "analysis_stage2_sensitivity"),
        ("stage1_best", ROOT / "analysis_stage1_sensitivity"),
    ):
        correlations = pd.read_csv(analysis / "tables/03_baseline_metric_correlations.csv")
        correlations = correlations[
            (correlations["correlation"] == "spearman")
            & (correlations["layer"] == "mean")
            & correlations["endpoint"].isin(["delta_f1_damage", "delta_f1_recovery"])
            & correlations["metric"].isin([
                "cosine", "normalized_l2", "product", "lukasiewicz",
                "cosine_recovery", "normalized_l2_recovery",
                "product_recovery", "lukasiewicz_recovery",
            ])
        ]
        for row in correlations.itertuples(index=False):
            rows.append({
                "checkpoint": checkpoint, "check_type": "metric_correlation",
                "task": row.task, "metric": row.metric, "estimate": row.estimate,
                "ci_low": row.ci_low, "ci_high": row.ci_high,
                "corrected_p": row.holm_corrected_p, "sequences": row.sequences,
            })
        gains = model_gain_table(analysis)
        for task in ("damage", "recovery"):
            for row in primary_gain(gains, task).itertuples(index=False):
                rows.append({
                    "checkpoint": checkpoint, "check_type": "model_gain",
                    "task": task, "metric": row.metric, "estimate": row.estimate,
                    "ci_low": row.ci_low, "ci_high": row.ci_high,
                    "corrected_p": row.holm_corrected_p, "sequences": row.sequences,
                })
        raw = pd.read_csv(ROOT / f"raw/{'stage2' if checkpoint == 'stage2_best' else 'stage1'}_sensitivity.csv", low_memory=False)
        adaptive = adaptive_interval(raw)
        rows.append({
            "checkpoint": checkpoint, "check_type": "adaptive_pgd",
            "task": "recovery", "metric": "adaptive_minus_nonadaptive_f1",
            "estimate": adaptive["estimate"], "ci_low": adaptive["ci_low"],
            "ci_high": adaptive["ci_high"], "corrected_p": math.nan,
            "sequences": adaptive["sequences"],
        })
        chosen = selected_best_rows(raw)
        object_frame = chosen.groupby(
            ["sequence_id", "image_path"], as_index=False
        ).agg(c_atk_object=("c_atk_object", "mean"),
              c_atk_background=("c_atk_background", "mean"))
        object_frame["difference"] = (
            object_frame["c_atk_object"] - object_frame["c_atk_background"]
        )
        interval = cluster_mean_interval(
            object_frame, "difference", iterations=ITERATIONS, seed=20260720
        )
        rows.append({
            "checkpoint": checkpoint, "check_type": "object_background",
            "task": "damage", "metric": "c_atk_object_minus_background",
            "estimate": interval["estimate"], "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"], "corrected_p": math.nan,
            "sequences": interval["sequences"],
        })
    result = pd.DataFrame(rows)
    pivot = result[result["check_type"] == "metric_correlation"].pivot_table(
        index=["task", "metric"], columns="checkpoint", values="estimate", aggfunc="first"
    )
    direction = {
        (task, metric): bool(np.sign(row.get("stage1_best", np.nan)) == np.sign(row.get("stage2_best", np.nan)))
        for (task, metric), row in pivot.iterrows()
    }
    result["same_direction_across_checkpoints"] = result.apply(
        lambda row: direction.get((row["task"], row["metric"]), np.nan), axis=1
    )
    return result


def build_tables() -> None:
    tables = ROOT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    counts = manifest.groupby("split", as_index=False).agg(
        frames=("output_image", "nunique"), sequences=("sequence_id", "nunique")
    )
    counts["record_type"] = "split"
    checkpoint_rows = pd.DataFrame([
        {"record_type": "checkpoint", "split": "primary", "path": str(STAGE2),
         "sha256": sha256(STAGE2), "frames": np.nan, "sequences": np.nan},
        {"record_type": "checkpoint", "split": "sensitivity", "path": str(STAGE1),
         "sha256": sha256(STAGE1), "frames": np.nan, "sequences": np.nan},
    ])
    pd.concat([counts, checkpoint_rows], ignore_index=True, sort=False).to_csv(
        tables / "01_dataset_and_checkpoint.csv", index=False
    )
    copy_csv(FINAL_ROOT / "tables/clean_model_metrics.csv", tables / "02_clean_test_metrics.csv")
    raw = pd.read_csv(MAIN_MATRIX, low_memory=False)
    parameter_columns = [
        "attack", "adaptive", "epsilon", "epsilon_px", "steps", "step_size",
        "random_start", "restarts", "restart", "seed", "defense",
        "perturbation_norm", "attack_objective", "lambda_box", "lambda_cls", "lambda_dfl",
    ]
    raw[parameter_columns].drop_duplicates().sort_values(
        ["attack", "adaptive", "epsilon_px", "steps", "restart", "defense"]
    ).to_csv(tables / "03_attack_parameters.csv", index=False)
    chosen = selected_best_rows(raw)
    per_condition = chosen.drop_duplicates([
        "sequence_id", "image_path", "attack", "adaptive", "epsilon_px", "steps",
        "seed", "defense",
    ])
    metrics = [
        "f1_clean", "f1_attack", "f1_defended", "recall_clean", "recall_attack",
        "recall_defended", "fn_clean", "fn_attack", "fn_defended", "confidence_drop",
    ]
    robustness = per_condition.groupby(
        ["attack", "adaptive", "epsilon_px", "steps", "defense"], as_index=False
    ).agg(**{name: (name, "mean") for name in metrics},
          frames=("image_path", "nunique"), sequences=("sequence_id", "nunique"))
    robustness.to_csv(tables / "04_minimal_robustness.csv", index=False)
    analysis = ROOT / "analysis_main"
    copy_csv(analysis / "tables/03_baseline_metric_correlations.csv", tables / "05_metric_correlations.csv")
    copy_csv(analysis / "tables/04_tnorm_vs_baseline_bootstrap.csv", tables / "06_tnorm_vs_baselines.csv")
    damage = pd.read_csv(analysis / "tables/05_damage_models_D0_D4.csv")
    damage[damage["model"].isin(["D0", "D1", "D2", "D3"])].to_csv(
        tables / "07_damage_D0_D3.csv", index=False
    )
    recovery = pd.read_csv(analysis / "tables/06_recovery_models_R0_R4.csv")
    recovery[recovery["model"].isin(["R0", "R1", "R2", "R3"])].to_csv(
        tables / "08_recovery_R0_R3.csv", index=False
    )
    copy_csv(analysis / "tables/13_scene_macro_loso.csv", tables / "11_scene_macro_loso.csv")
    build_sensitivity_table().to_csv(tables / "09_stage1_sensitivity.csv", index=False)
    nms = json.loads((FINAL_ROOT / "audit/nms_timeout_audit.json").read_text(encoding="utf-8"))
    cases_path = FINAL_ROOT / "audit/nms_timeout_cases.csv"
    nms["case_rows"] = len(pd.read_csv(cases_path)) if cases_path.is_file() else 0
    pd.DataFrame([nms]).to_csv(tables / "10_nms_timeout_audit.csv", index=False)


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def errorbar(frame: pd.DataFrame, labels: pd.Series, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, max(4, len(frame) * 0.28)))
    positions = np.arange(len(frame))
    estimates = frame["estimate"].to_numpy(float)
    low = np.maximum(0, estimates - frame["ci_low"].to_numpy(float))
    high = np.maximum(0, frame["ci_high"].to_numpy(float) - estimates)
    axis.errorbar(estimates, positions, xerr=np.vstack([low, high]), fmt="o", capsize=3)
    axis.axvline(0, color="black", linewidth=.8)
    axis.set_yticks(positions, labels)
    axis.set_title(title)
    save_figure(figure, path)


def build_figures() -> None:
    figures = ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    robustness = pd.read_csv(ROOT / "tables/04_minimal_robustness.csv")
    scope = robustness[~boolean_series(robustness["adaptive"])].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for (attack, defense), group in scope.groupby(["attack", "defense"]):
        ordered = group.sort_values("epsilon_px")
        label = f"{attack}/{defense}"
        axes[0].plot(ordered["epsilon_px"], ordered["f1_defended"], "o-", label=label)
        axes[1].plot(ordered["epsilon_px"], ordered["recall_defended"], "o-", label=label)
    axes[0].set(xlabel="epsilon / 255", ylabel="F1", title="F1 versus epsilon")
    axes[1].set(xlabel="epsilon / 255", ylabel="Recall", title="Recall versus epsilon")
    axes[1].legend(fontsize=6)
    save_figure(figure, figures / "01_f1_recall_vs_epsilon.png")
    correlations = pd.read_csv(ROOT / "tables/05_metric_correlations.csv")
    correlations = correlations[
        (correlations["correlation"] == "spearman") & (correlations["layer"] == "mean")
        & correlations["endpoint"].isin(["delta_f1_damage", "delta_f1_recovery"])
    ].copy()
    labels = correlations.apply(
        lambda row: f"{row['task']} {row['metric']} (n={int(row['sequences'])}, Holm={row['holm_corrected_p']:.3g})",
        axis=1,
    )
    errorbar(correlations, labels, "Metric correlations; 95% sequence-cluster CI",
             figures / "02_metric_correlations_with_ci.png")
    deltas = pd.read_csv(ROOT / "tables/06_tnorm_vs_baselines.csv")
    deltas = deltas[
        (deltas["layer"] == "mean")
        & deltas["endpoint"].isin(["delta_f1_damage", "delta_f1_recovery"])
        & deltas["baseline"].isin(["cosine", "normalized_l2", "cosine_recovery", "normalized_l2_recovery"])
    ].copy()
    deltas["estimate"] = deltas["delta_rho"]
    labels = deltas.apply(
        lambda row: f"{row['task']} {row['tnorm']} - {row['baseline']} (n={int(row['sequences'])})", axis=1
    )
    errorbar(deltas, labels, "T-norm minus baseline absolute Spearman",
             figures / "03_tnorm_vs_baseline_delta.png")
    gains = model_gain_table(ROOT / "analysis_main")
    for task, path, title in (
        ("damage", figures / "04_damage_model_gain.png", "D3 versus D2"),
        ("recovery", figures / "05_recovery_model_gain.png", "R3 versus R2"),
    ):
        frame = primary_gain(gains, task)
        labels = frame.apply(
            lambda row: f"{row['metric']} (n={int(row['sequences'])}, Holm={row['holm_corrected_p']:.3g})", axis=1
        )
        errorbar(frame, labels, f"{title}; 95% paired cluster CI", path)
    raw = selected_best_rows(pd.read_csv(MAIN_MATRIX, low_memory=False))
    adaptive = raw[
        (raw["attack"] == "pgd") & np.isclose(raw["epsilon_px"], 1.0)
        & (raw["steps"] == 20) & (raw["defense"] == "tnorm")
    ].copy()
    adaptive["adaptive_bool"] = boolean_series(adaptive["adaptive"])
    adaptive = adaptive.groupby("adaptive_bool", as_index=False).agg(
        f1=("f1_defended", "mean"), recall=("recall_defended", "mean")
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    x = np.arange(len(adaptive))
    axis.bar(x - .16, adaptive["f1"], width=.32, label="F1")
    axis.bar(x + .16, adaptive["recall"], width=.32, label="Recall")
    axis.set_xticks(x, ["adaptive" if value else "non-adaptive" for value in adaptive["adaptive_bool"]])
    axis.set_title("Product filter: adaptive versus non-adaptive PGD")
    axis.legend()
    save_figure(figure, figures / "06_adaptive_vs_nonadaptive.png")
    sensitivity = pd.read_csv(ROOT / "tables/09_stage1_sensitivity.csv")
    sensitivity = sensitivity[
        (sensitivity["check_type"] == "metric_correlation")
        & sensitivity["metric"].isin(["product", "lukasiewicz", "product_recovery", "lukasiewicz_recovery"])
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    for checkpoint, group in sensitivity.groupby("checkpoint"):
        axis.plot(group["metric"], group["estimate"], "o-", label=checkpoint)
    axis.axhline(0, color="black", linewidth=.8)
    axis.tick_params(axis="x", rotation=30)
    matched = pd.read_csv(ROOT / "config/stage1_sensitivity_manifest.csv")["image_path"].nunique()
    axis.set_title(f"Stage 1 / Stage 2 sensitivity ({matched} matched frames, 3 scenes)")
    axis.legend()
    save_figure(figure, figures / "07_stage1_stage2_sensitivity.png")


def gain_summary(task: str) -> dict[str, Any]:
    scope = primary_gain(model_gain_table(ROOT / "analysis_main"), task)
    values = {
        row.metric: {
            "estimate": float(row.estimate), "ci_low": float(row.ci_low),
            "ci_high": float(row.ci_high), "corrected_p": float(row.holm_corrected_p),
        }
        for row in scope.itertuples(index=False)
    }
    within_observed_scenes_significant = any(
        item["corrected_p"] < .05
        and (
            item["ci_high"] < 0 if metric == "delta_mae"
            else item["ci_low"] > 0
        )
        for metric, item in values.items()
    )
    sequence_count = int(scope["sequences"].dropna().min())
    significant = within_observed_scenes_significant and sequence_count >= 5
    relative = float(scope["relative_mae_reduction"].dropna().iloc[0])
    practical = (
        relative >= 5.0 or values.get("delta_r2", {}).get("estimate", -math.inf) >= .05
        or abs(values.get("delta_spearman", {}).get("estimate", 0.0)) >= .05
    )
    return {
        "comparison": "D3_vs_D2" if task == "damage" else "R3_vs_R2",
        "statistically_confirmed": significant,
        "within_observed_scenes_corrected_signal": within_observed_scenes_significant,
        "independent_scenes": sequence_count,
        "practically_meaningful": practical,
        "relative_mae_reduction": relative, "metrics": values,
        "allowed_claim": (
            "exploratory direction within three observed scenes; no strong generalization claim"
            if sequence_count < 5 else
            "statistically confirmed and practically noticeable incremental signal"
            if significant and practical else
            "statistically reproducible but small incremental diagnostic signal"
            if significant else
            "no sequence-level confirmed advantage over standard metrics"
        ),
    }


def build_summary() -> None:
    sensitivity = pd.read_csv(ROOT / "tables/09_stage1_sensitivity.csv")
    direction = sensitivity["same_direction_across_checkpoints"].dropna()
    adaptive = adaptive_interval(pd.read_csv(MAIN_MATRIX, low_memory=False))
    nms = json.loads((FINAL_ROOT / "audit/nms_timeout_audit.json").read_text(encoding="utf-8"))
    data = pd.read_csv(MANIFEST)
    sensitivity_frames = pd.read_csv(
        ROOT / "config/stage1_sensitivity_manifest.csv"
    )["image_path"].nunique()
    stage_counts = read_status().get("stages", {})
    statuses = pd.Series([entry.get("status") for entry in stage_counts.values()]).value_counts().to_dict()
    payload = {
        "status": "PASS",
        "created_at": now(), "commit": git("rev-parse", "HEAD"),
        "checkpoints": {"primary": str(STAGE2), "sensitivity": str(STAGE1)},
        "independent_scenes": data.groupby("split")["sequence_id"].nunique().to_dict(),
        "normalization": {
            "selected": "N1_quantile", "selection_split": "validation",
            "test_used_for_selection": False,
        },
        "metric_families": {
            "legacy": "legacy_compatibility_baseline_not_formal_tnorm_evidence",
            "canonical": "normalized_pointwise_tnorm_primary_analysis",
            "primary_tnorms": ["product", "lukasiewicz"],
            "supplementary_redundant_tnorm": "godel",
        },
        "H1_damage": gain_summary("damage"),
        "H2_recovery": gain_summary("recovery"),
        "H3_checkpoint_sensitivity": {
            "status": "exploratory_due_to_three_test_scenes",
            "same_direction_fraction": float(direction.mean()) if len(direction) else math.nan,
            "matched_frames": int(sensitivity_frames), "independent_scenes": 3,
        },
        "H4_scene_difficulty": {
            "status": "deferred_extended_analysis",
            "reason": "only three independent test scenes; confirmatory strata are impossible",
        },
        "adaptive_pgd": adaptive,
        "nms_timeout_audit": nms,
        "experiment_status_counts": statuses,
        "deferred_extended_analysis": yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))[
            "deferred_extended_analysis"
        ],
        "known_limitations": [
            "The clean detector has low absolute recall and false negatives remain high.",
            "Validation and test each contain only three independent sequences; confidence intervals are correspondingly weak or wide.",
            "Bootstrap repetitions do not increase the number of independent scenes; p-values are descriptive for the observed scenes.",
            "Stage 1 sensitivity uses two checkpoints of one YOLO11m architecture and is not cross-architecture robustness evidence.",
            "Median is not included in adaptive robustness ranking because BPDA is not implemented.",
            "The failed simple threshold policy is retained as a negative result.",
        ],
    }
    (ROOT / "run_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = ROOT / "report"
    report.mkdir(parents=True, exist_ok=True)
    frozen_budgets = json.loads(
        (ROOT / "config/canonical_budget_selection.json").read_text(encoding="utf-8")
    )["selected"]
    (report / "article_methods_draft.md").write_text(
        "# Deadline practice: frozen methods text\n\n"
        "The study uses OSDaR23 with train/validation/test separation by railway sequence. "
        "The official model is YOLO11m Stage 2 best, selected only by validation mAP50-95; "
        "Stage 1 best is used for an exploratory checkpoint-sensitivity analysis.\n\n"
        "P3, P4 and P5 activations are normalized per layer and channel using validation-clean "
        "q01/q99 quantile memberships. Damage and recovery models use GroupKFold by sequence_id. "
        "All confidence intervals and paired model comparisons use 5,000 sequence-cluster bootstrap "
        "replicates and Holm-Bonferroni correction; test data are not used for feature or threshold selection.\n\n"
        f"FGSM budgets {frozen_budgets['fgsm_epsilon_px']}/255, PGD budgets "
        f"{frozen_budgets['pgd_epsilon_px']}/255 and adaptive PGD budgets "
        f"{frozen_budgets['adaptive_pgd_epsilon_px']}/255 were frozen using validation floor checks. "
        "PGD uses 20 steps, random starts, three restarts and seeds 42, 123 and 999. "
        "Adaptive PGD differentiates through Product preprocessing. Product is evaluated as "
        "preprocessing, not claimed as a "
        "universal defense.\n\n"
        "The primary claim is limited to incremental diagnostic value of Product and "
        "Lukasiewicz features beyond attack parameters, cosine and standard distances; Goedel is "
        "reported only as a supplementary redundancy ablation. Negative "
        "results and the low clean-detector recall are retained explicitly.\n",
        encoding="utf-8",
    )


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def build_bundle() -> None:
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tnorm_deadline_") as directory:
        staging = Path(directory) / "TNormFilter_deadline_final"
        staging.mkdir(parents=True)
        for name in ("tables", "figures", "statistics", "report"):
            source = ROOT / name
            copy_tree(source, staging / name)
        copy_tree(ROOT / "analysis_main", staging / "statistics/main")
        copy_tree(ROOT / "analysis_stage1_sensitivity", staging / "statistics/stage1_sensitivity")
        copy_tree(ROOT / "analysis_stage2_sensitivity", staging / "statistics/stage2_sensitivity")
        for source, relative in (
            (PROTOCOL, "configs/deadline_protocol.yaml"),
            (REVISION_PROTOCOL, "configs/revision_q1_protocol.yaml"),
            (ROOT / "config/pilot_manifest.csv", "configs/pilot_manifest.csv"),
            (ROOT / "config/stage1_sensitivity_manifest.csv", "configs/stage1_sensitivity_manifest.csv"),
            (ROOT / "pilot/pilot_gate.json", "audit/pilot_gate.json"),
            (ROOT / "config/canonical_budget_selection.json", "configs/canonical_budget_selection.json"),
            (FINAL_ROOT / "audit/legacy_recovery_recalculation.json", "audit/legacy_recovery_recalculation.json"),
            (FINAL_ROOT / "audit/matrix_integrity.json", "audit/matrix_integrity.json"),
            (FINAL_ROOT / "audit/nms_timeout_audit.json", "audit/nms_timeout_audit.json"),
            (FINAL_ROOT / "audit/nms_timeout_cases.csv", "audit/nms_timeout_cases.csv"),
            (FINAL_ROOT / "audit/missing_conditions.csv", "audit/missing_conditions.csv"),
            (FINAL_ROOT / "deadline_baseline/manifest.json", "audit/deadline_baseline_manifest.json"),
            (ROOT / "normalization/normalization_manifest.json", "audit/normalization_manifest.json"),
            (ROOT / "raw/stage1_sensitivity.csv", "raw/stage1_sensitivity.csv"),
            (ROOT / "raw/stage2_sensitivity.csv", "raw/stage2_sensitivity.csv"),
            (MAIN_MATRIX, "raw/unified_diagnostics_deadline_test.csv"),
            (MAIN_CONFIG, "configs/unified_diagnostics_deadline_test.json"),
            (PROJECT_DIR / "outputs/final_practice/deadline_baseline/logs/tnorm-wait-train.journal.log", "logs/main_service.log"),
        ):
            if source.is_file():
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        projected_status = read_status()
        projected_status["status"] = "success"
        projected_status["full_q1_service"] = "masked_and_deferred"
        projected_status.setdefault("stages", {}).setdefault("deadline_bundle", {}).update({
            "status": "success", "finished_at": now(), "error": None,
            "outputs": [str(ZIP_PATH), str(ZIP_PATH.with_suffix(".zip.sha256"))],
        })
        (staging / "pipeline_status.json").write_text(
            json.dumps(projected_status, indent=2) + "\n", encoding="utf-8"
        )
        projected_summary = json.loads(
            (ROOT / "run_summary.json").read_text(encoding="utf-8")
        )
        projected_summary["experiment_status_counts"] = pd.Series([
            entry.get("status") for entry in projected_status["stages"].values()
        ]).value_counts().to_dict()
        (staging / "run_summary.json").write_text(
            json.dumps(projected_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            "# TNormFilter deadline final\n\n"
            "Minimal validation-frozen sequence-level practice bundle. See run_summary.json. "
            "The raw dataset and model weights are excluded; their paths and hashes are recorded.\n",
            encoding="utf-8",
        )
        (staging / "git_info.txt").write_text(
            f"commit={git('rev-parse', 'HEAD')}\nstatus_begin\n{git('status', '--short')}\nstatus_end\n",
            encoding="utf-8",
        )
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        checksums = {path.relative_to(staging).as_posix(): sha256(path) for path in files}
        (staging / "checksums.sha256").write_text(
            "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
            encoding="utf-8",
        )
        manifest = {
            "protocol_id": "tnormfilter_deadline_v1", "created_at": now(),
            "commit": git("rev-parse", "HEAD"), "python": platform.python_version(),
            "pytorch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "dataset_included": False, "weights_included": False,
            "dataset_manifest_sha256": sha256(MANIFEST),
            "checkpoints": {"stage2_best": sha256(STAGE2), "stage1_best": sha256(STAGE1)},
            "files": checksums,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary = ZIP_PATH.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging.parent).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            names = set(archive.namelist())
        if bad is not None:
            raise RuntimeError(f"Corrupt deadline ZIP member: {bad}")
        required = {
            "TNormFilter_deadline_final/README.md",
            "TNormFilter_deadline_final/run_summary.json",
            "TNormFilter_deadline_final/manifest.json",
            "TNormFilter_deadline_final/checksums.sha256",
            "TNormFilter_deadline_final/tables/10_nms_timeout_audit.csv",
            "TNormFilter_deadline_final/tables/11_scene_macro_loso.csv",
        }
        if required - names:
            raise RuntimeError(f"Deadline ZIP missing {sorted(required - names)}")
        temporary.replace(ZIP_PATH)
    digest = sha256(ZIP_PATH)
    ZIP_PATH.with_suffix(".zip.sha256").write_text(
        f"{digest}  {ZIP_PATH.name}\n", encoding="utf-8"
    )


def main() -> None:
    validate_main_matrix()
    for directory in ("config", "raw", "tables", "figures", "statistics", "report", "logs"):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    run_stage(
        "stage1_selection", [str(PYTHON), __file__, "--internal-select"],
        [ROOT / "config/stage1_sensitivity_manifest.csv", ROOT / "config/stage1_sensitivity_data.yaml"],
    )
    stage1_normalization = ROOT / "stage1_normalization"
    run_stage(
        "stage1_normalization", [
            str(PYTHON), "-m", "revision_q1.collect_normalization",
            "--model", str(STAGE1), "--output", str(stage1_normalization), "--workers", "0",
        ], [stage1_normalization / "normalization/layer_channel_statistics.pt"],
    )
    stage1_matrix = ROOT / "raw/stage1_sensitivity.csv"
    selected_budgets = json.loads(
        (ROOT / "config/canonical_budget_selection.json").read_text(encoding="utf-8")
    )["selected"]
    sensitivity_fgsm = str(max(map(float, selected_budgets["fgsm_epsilon_px"])))
    sensitivity_pgd = str(max(map(float, selected_budgets["pgd_epsilon_px"])))
    sensitivity_adaptive = str(
        max(map(float, selected_budgets["adaptive_pgd_epsilon_px"]))
    )
    run_stage(
        "stage1_sensitivity_matrix", [
            str(PYTHON), "run_final_matrix.py", "--model", str(STAGE1),
            "--data", str(ROOT / "config/stage1_sensitivity_data.yaml"),
            "--split", "test", "--workers", "0", "--checkpoint-name", "stage1_best",
            "--revision-stats", str(stage1_normalization / "normalization/layer_channel_statistics.pt"),
            "--normalizations", "N1_quantile", "--fgsm-eps", sensitivity_fgsm,
            "--pgd-eps", sensitivity_pgd, "--pgd-steps", "20",
            "--adaptive-pgd-eps", sensitivity_adaptive, "--adaptive-pgd-steps", "20",
            "--seeds", "42,123,999", "--defenses", "none,tnorm",
            "--nms-max-time-img", "10", "--output", str(stage1_matrix),
        ], [stage1_matrix, stage1_matrix.with_suffix(".json")],
    )
    run_stage(
        "stage2_sensitivity_subset", [str(PYTHON), __file__, "--internal-subset"],
        [ROOT / "raw/stage2_sensitivity.csv"],
    )
    for name, matrix in (
        ("main", MAIN_MATRIX),
        ("stage1_sensitivity", stage1_matrix),
        ("stage2_sensitivity", ROOT / "raw/stage2_sensitivity.csv"),
    ):
        output = ROOT / f"analysis_{name}"
        run_stage(
            f"analysis_{name}", [
                str(PYTHON), "-m", "revision_q1.analyze", "--input", str(matrix),
                "--output", str(output), "--normalization", "N1_quantile",
                "--split", "test", "--bootstrap", str(ITERATIONS),
            ], [output / "tables/12_multiple_comparison_corrections.csv"],
        )
    run_stage(
        "deadline_tables", [str(PYTHON), __file__, "--internal-tables"],
        [ROOT / "tables/10_nms_timeout_audit.csv", ROOT / "tables/11_scene_macro_loso.csv"],
    )
    run_stage(
        "deadline_figures", [str(PYTHON), __file__, "--internal-figures"],
        [ROOT / "figures/07_stage1_stage2_sensitivity.png"],
    )
    run_stage(
        "deadline_summary", [str(PYTHON), __file__, "--internal-summary"],
        [ROOT / "run_summary.json", ROOT / "report/article_methods_draft.md"],
    )
    run_stage(
        "deadline_bundle", [str(PYTHON), __file__, "--internal-bundle"],
        [ZIP_PATH, ZIP_PATH.with_suffix(".zip.sha256")],
    )
    payload = read_status()
    payload["status"] = "success"
    payload["full_q1_service"] = "masked_and_deferred"
    payload["deadline_zip"] = str(ZIP_PATH)
    payload["deadline_zip_sha256"] = sha256(ZIP_PATH)
    write_status(payload)
    summary = json.loads((ROOT / "run_summary.json").read_text(encoding="utf-8"))
    summary["experiment_status_counts"] = pd.Series([
        entry.get("status") for entry in payload["stages"].values()
    ]).value_counts().to_dict()
    (ROOT / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS", "zip": str(ZIP_PATH), "sha256": sha256(ZIP_PATH)
    }, indent=2))


if __name__ == "__main__":
    internal = sys.argv[1] if len(sys.argv) > 1 else None
    if internal == "--internal-select":
        select_sensitivity_frames()
    elif internal == "--internal-subset":
        make_stage2_sensitivity_subset()
    elif internal == "--internal-tables":
        build_tables()
    elif internal == "--internal-figures":
        build_figures()
    elif internal == "--internal-summary":
        build_summary()
    elif internal == "--internal-bundle":
        build_bundle()
    elif internal is not None:
        raise SystemExit(f"Unknown argument: {internal}")
    else:
        main()
