from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from revision_q1.normalization import MODES
from revision_q1.protocol import assert_split_action, load_protocol, output_root
from revision_q1.statistics import (
    benjamini_hochberg,
    cluster_sample_plan,
    correlation,
    holm_bonferroni,
    materialize_cluster_sample,
    paired_cluster_delta_correlation,
    unique_cluster_samples,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
CATEGORICAL = {"attack", "defense"}
MODEL_COMPARISONS = {
    "damage": (("D2", "D3"), ("D2", "D4")),
    "recovery": (("R2", "R3"), ("R2", "R4")),
}
ENDPOINTS = {
    "damage": (
        "delta_f1_damage", "delta_recall_damage", "delta_confidence",
        "false_negatives_increase",
    ),
    "recovery": (
        "delta_f1_recovery", "delta_recall_recovery",
        "false_negatives_reduction", "normalized_quality_recovery",
    ),
}


def encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def estimator_pipeline(features: list[str], algorithm: str) -> Pipeline:
    categorical = [name for name in features if name in CATEGORICAL]
    numeric = [name for name in features if name not in CATEGORICAL]
    transformer = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder()),
        ]), categorical),
    ])
    estimator = (
        Ridge(alpha=1.0)
        if algorithm == "ridge"
        else ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20_000, random_state=20260720)
    )
    return Pipeline([("preprocess", transformer), ("estimator", estimator)])


def metric_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(target) & np.isfinite(prediction)
    if valid.sum() < 3:
        return {"mae": math.nan, "r2": math.nan, "spearman": math.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rank = float(spearmanr(target[valid], prediction[valid]).statistic)
    return {
        "mae": float(mean_absolute_error(target[valid], prediction[valid])),
        "r2": float(r2_score(target[valid], prediction[valid])),
        "spearman": rank,
    }


def apply_normalization_aliases(data: pd.DataFrame, mode: str) -> pd.DataFrame:
    result = data.copy()
    aliases = {
        "cosine": "cosine_similarity",
        "l1_distance": "l1_distance",
        "normalized_l2": "normalized_euclidean_distance",
        "mse": "mse",
        "pearson": "pearson_correlation",
        "entropy_shift": "entropy_shift",
        "product": "product",
        "godel": "godel",
        "lukasiewicz": "lukasiewicz",
        "cosine_recovery": "cosine_similarity_recovery",
        "l1_recovery": "l1_distance_recovery",
        "normalized_l2_recovery": "normalized_euclidean_distance_recovery",
        "mse_recovery": "mse_recovery",
        "pearson_recovery": "pearson_correlation_recovery",
        "entropy_recovery": "entropy_shift_recovery",
        "product_recovery": "product_recovery",
        "godel_recovery": "godel_recovery",
        "lukasiewicz_recovery": "lukasiewicz_recovery",
        "p_clean_preservation": "p_product",
        "a_attacked_similarity": "a_product",
        "r_restored_similarity": "r_product",
        "g_recovery": "g_product",
        "c_def": "c_def_product",
    }
    missing: list[str] = []
    for destination, suffix in aliases.items():
        source = f"{mode}_{suffix}"
        if source not in result:
            missing.append(source)
        else:
            result[destination] = result[source]
    if missing:
        raise RuntimeError(f"{mode}: raw matrix lacks Q1 columns {missing}")
    return result


def add_endpoints(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    definitions = {
        "delta_f1_damage": result["f1_clean"] - result["f1_attack"],
        "delta_recall_damage": result["recall_clean"] - result["recall_attack"],
        "delta_f1_recovery": result["f1_defended"] - result["f1_attack"],
        "delta_recall_recovery": result["recall_defended"] - result["recall_attack"],
        "delta_confidence": result["confidence_drop"],
        "false_negatives_increase": result["fn_attack"] - result["fn_clean"],
        "false_negatives_reduction": result["fn_attack"] - result["fn_defended"],
    }
    for name, values in definitions.items():
        result[name] = values
    if "normalized_quality_recovery" not in result:
        result["normalized_quality_recovery"] = (
            result["delta_f1_recovery"]
            / (result["delta_f1_damage"] + 1e-8)
        )
    return result


def selected_rows(data: pd.DataFrame) -> pd.DataFrame:
    result = data[data["attack"].ne("clean")].copy()
    if "selected_best" in result:
        selected = result["selected_best"].astype(str).str.lower().isin({"true", "1"})
        result = result[selected]
    return result.replace([np.inf, -np.inf], np.nan)


def boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def model_specifications(protocol: dict[str, Any], task: str) -> dict[str, list[str]]:
    key = "damage_models" if task == "damage" else "recovery_models"
    return {name: list(features) for name, features in protocol[key].items()}


def aggregate_task(
    data: pd.DataFrame, task: str, endpoint: str, specifications: dict[str, list[str]]
) -> pd.DataFrame:
    scope = selected_rows(data)
    if "adaptive" in scope:
        scope = scope[~boolean_series(scope["adaptive"])]
    if task == "damage":
        scope = scope[(scope["defense"] == "none") & (~boolean_series(scope["adaptive"]))]
        keys = ["sequence_id", "attack", "epsilon", "steps", "adaptive"]
    else:
        scope = scope[(scope["defense"] != "none") & (~boolean_series(scope["adaptive"]))]
        keys = ["sequence_id", "attack", "epsilon", "steps", "adaptive", "defense"]
    needed = {endpoint} | {feature for values in specifications.values() for feature in values}
    missing = needed - set(scope)
    if missing:
        raise RuntimeError(f"{task}: missing model columns {sorted(missing)}")
    aggregations = {
        column: ("first" if column in CATEGORICAL else "mean")
        for column in needed
        if column not in keys
    }
    return scope.groupby(keys, as_index=False, dropna=False).agg(aggregations)


def out_of_fold_predictions(
    frame: pd.DataFrame,
    target_name: str,
    specifications: dict[str, list[str]],
    algorithm: str,
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sequence_count = frame["sequence_id"].nunique()
    if sequence_count < 3:
        raise RuntimeError("At least three independent sequence_id groups are required")
    split_count = min(folds, sequence_count)
    groups = frame["sequence_id"].astype(str).to_numpy()
    target = frame[target_name].to_numpy(float)
    cv = list(GroupKFold(n_splits=split_count).split(frame, target, groups))
    prediction_table = frame[["sequence_id", target_name]].copy()
    rows: list[dict[str, object]] = []
    for model_name, features in specifications.items():
        prediction = np.full(len(frame), np.nan)
        for train, validation in cv:
            model = estimator_pipeline(features, algorithm)
            model.fit(frame.iloc[train][features], target[train])
            prediction[validation] = model.predict(frame.iloc[validation][features])
        prediction_table[f"prediction_{model_name}"] = prediction
        rows.append({
            "model": model_name,
            "algorithm": algorithm,
            **metric_values(target, prediction),
            "rows": len(frame),
            "sequences": sequence_count,
            "folds": split_count,
            "features": ",".join(features),
        })
    return pd.DataFrame(rows), prediction_table


def paired_model_bootstrap(
    predictions: pd.DataFrame,
    target_name: str,
    comparisons: tuple[tuple[str, str], ...],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    target = predictions[target_name].to_numpy(float)
    sequence_ids = predictions["sequence_id"].astype(str).to_numpy()
    plan = unique_cluster_samples(sequence_ids, iterations, seed)
    rows: list[dict[str, object]] = []
    for baseline, extended in comparisons:
        baseline_prediction = predictions[f"prediction_{baseline}"].to_numpy(float)
        extended_prediction = predictions[f"prediction_{extended}"].to_numpy(float)
        observed_baseline = metric_values(target, baseline_prediction)
        observed_extended = metric_values(target, extended_prediction)
        samples = {name: [] for name in ("delta_mae", "delta_r2", "delta_spearman")}
        for indices, multiplicity in plan:
            left = metric_values(target[indices], baseline_prediction[indices])
            right = metric_values(target[indices], extended_prediction[indices])
            samples["delta_mae"].extend([left["mae"] - right["mae"]] * multiplicity)
            samples["delta_r2"].extend([right["r2"] - left["r2"]] * multiplicity)
            samples["delta_spearman"].extend([
                right["spearman"] - left["spearman"]
            ] * multiplicity)
        observed = {
            "delta_mae": observed_baseline["mae"] - observed_extended["mae"],
            "delta_r2": observed_extended["r2"] - observed_baseline["r2"],
            "delta_spearman": (
                observed_extended["spearman"] - observed_baseline["spearman"]
            ),
        }
        for metric_name, values_list in samples.items():
            values = np.asarray(values_list, dtype=float)
            values = values[np.isfinite(values)]
            lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
            upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
            rows.append({
                "comparison": f"{extended}_vs_{baseline}",
                "metric": metric_name,
                "estimate": observed[metric_name],
                "ci_low": float(np.percentile(values, 2.5)) if len(values) else math.nan,
                "ci_high": float(np.percentile(values, 97.5)) if len(values) else math.nan,
                "p_value": min(1.0, 2 * min(lower, upper)) if len(values) else 1.0,
                "relative_mae_reduction": (
                    observed["delta_mae"] / observed_baseline["mae"]
                    if observed_baseline["mae"] else math.nan
                ),
                "bootstrap_iterations": iterations,
                "sequences": int(pd.Series(sequence_ids).nunique()),
            })
    return pd.DataFrame(rows)


def correlation_tables(
    data: pd.DataFrame, task: str, endpoint: str, iterations: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope = selected_rows(data)
    if "adaptive" in scope:
        scope = scope[~boolean_series(scope["adaptive"])]
    scope = scope[
        (scope["defense"] == "none") if task == "damage" else (scope["defense"] != "none")
    ]
    baseline = [
        "cosine", "l1_distance", "normalized_l2", "mse", "pearson", "entropy_shift",
    ]
    tnorms = ["product", "godel", "lukasiewicz"]
    if task == "recovery":
        baseline = [
            "cosine_recovery", "l1_recovery", "normalized_l2_recovery",
            "mse_recovery", "pearson_recovery", "entropy_recovery",
        ]
        tnorms = [f"{name}_recovery" for name in tnorms]
    correlation_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    layers = ["mean", *sorted(scope["layer"].dropna().unique())]
    for layer in layers:
        layer_scope = scope if layer == "mean" else scope[scope["layer"] == layer]
        keys = ["sequence_id", "image_path", "attack", "epsilon", "defense", "seed"]
        keys = [key for key in keys if key in layer_scope]
        metrics = baseline + tnorms
        aggregation = {endpoint: "mean", **{name: "mean" for name in metrics}}
        frame = layer_scope.groupby(keys, as_index=False, dropna=False).agg(aggregation)
        for metric_name in metrics:
            pair = frame[["sequence_id", endpoint, metric_name]].dropna()
            if len(pair) < 3 or pair[metric_name].nunique() < 2:
                continue
            observed_pearson = correlation(
                "pearson", pair[endpoint].to_numpy(float), pair[metric_name].to_numpy(float)
            )
            observed_spearman = correlation(
                "spearman", pair[endpoint].to_numpy(float), pair[metric_name].to_numpy(float)
            )
            sequence_ids = pair["sequence_id"].astype(str).to_numpy()
            samples = {"pearson": [], "spearman": []}
            for indices, multiplicity in unique_cluster_samples(
                sequence_ids, iterations, seed
            ):
                for kind in samples:
                    samples[kind].extend([correlation(
                        kind, pair[endpoint].to_numpy(float)[indices],
                        pair[metric_name].to_numpy(float)[indices],
                    )] * multiplicity)
            for kind, observed in (("pearson", observed_pearson), ("spearman", observed_spearman)):
                values = np.asarray(samples[kind], dtype=float)
                values = values[np.isfinite(values)]
                lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
                upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
                correlation_rows.append({
                    "task": task, "endpoint": endpoint, "layer": layer,
                    "metric": metric_name, "correlation": kind, "estimate": observed,
                    "ci_low": float(np.percentile(values, 2.5)) if len(values) else math.nan,
                    "ci_high": float(np.percentile(values, 97.5)) if len(values) else math.nan,
                    "p_value": min(1.0, 2 * min(lower, upper)) if len(values) else 1.0,
                    "frames": int(pair["image_path"].nunique()) if "image_path" in pair else len(pair),
                    "sequences": int(pair["sequence_id"].nunique()),
                    "bootstrap_iterations": iterations,
                })
        for tnorm in tnorms:
            for comparison in baseline:
                pair = frame[["sequence_id", endpoint, tnorm, comparison]].dropna()
                if len(pair) < 3:
                    continue
                result = paired_cluster_delta_correlation(
                    pair, endpoint, tnorm, comparison,
                    iterations=iterations, seed=seed,
                )
                delta_rows.append({
                    "task": task, "endpoint": endpoint, "layer": layer,
                    "tnorm": tnorm, "baseline": comparison, **result,
                })
    return pd.DataFrame(correlation_rows), pd.DataFrame(delta_rows)


def add_corrections(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    if {"task", "endpoint"} <= set(result):
        result["hypothesis_family"] = result.apply(
            lambda row: (
                f"{family}_primary"
                if row["endpoint"] == ENDPOINTS[row["task"]][0]
                else f"{family}_secondary_{row['endpoint']}"
            ), axis=1,
        )
    else:
        result["hypothesis_family"] = family
    result["holm_corrected_p"] = np.nan
    result["bh_fdr_p"] = np.nan
    for _, indices in result.groupby("hypothesis_family").groups.items():
        result.loc[indices, "holm_corrected_p"] = holm_bonferroni(
            result.loc[indices, "p_value"]
        )
        result.loc[indices, "bh_fdr_p"] = benjamini_hochberg(
            result.loc[indices, "p_value"]
        )
    return result


def hypotheses_table() -> pd.DataFrame:
    rows = []
    descriptions = {
        "H1": "T-norm increment for sequence-level feature damage prediction",
        "H2": "T-norm increment for sequence-level feature recovery prediction",
        "H3": "Direction and ranking preserved across Stage 2 and Stage 1 best",
        "H4": "T-norm increment preserved across frozen scene-difficulty strata",
    }
    for hypothesis, description in descriptions.items():
        rows.append({
            "hypothesis": hypothesis,
            "description": description,
            "status": "pending",
            "statistical_unit": "sequence_id",
        })
    return pd.DataFrame(rows)


def run_analysis(
    data: pd.DataFrame,
    protocol: dict[str, Any],
    mode: str,
    iterations: int,
    output: Path,
) -> dict[str, Any]:
    data = add_endpoints(apply_normalization_aliases(data, mode))
    tables = output / "tables"
    statistics = output / "statistics"
    raw = output / "raw"
    for directory in (tables, statistics, raw):
        directory.mkdir(parents=True, exist_ok=True)
    hypotheses_table().to_csv(tables / "01_hypotheses_and_endpoints.csv", index=False)
    all_models: dict[str, list[pd.DataFrame]] = {"damage": [], "recovery": []}
    all_gains: list[pd.DataFrame] = []
    all_correlations: list[pd.DataFrame] = []
    all_deltas: list[pd.DataFrame] = []
    for task in ("damage", "recovery"):
        specifications = model_specifications(protocol, task)
        for endpoint in ENDPOINTS[task]:
            prepared = aggregate_task(data, task, endpoint, specifications)
            prepared.to_csv(
                raw / f"{task}_{endpoint}_sequence_aggregates.csv", index=False
            )
            for algorithm in protocol["algorithms"]["primary"]:
                models, predictions = out_of_fold_predictions(
                    prepared, endpoint, specifications, algorithm,
                    int(protocol["statistics"]["group_folds"]),
                )
                models.insert(0, "task", task)
                models.insert(1, "endpoint", endpoint)
                models.insert(2, "normalization", mode)
                all_models[task].append(models)
                predictions.to_csv(
                    statistics / f"{task}_{endpoint}_{algorithm}_oof_predictions.csv",
                    index=False,
                )
                gains = paired_model_bootstrap(
                    predictions, endpoint, MODEL_COMPARISONS[task], iterations,
                    int(protocol["random_seed"]),
                )
                gains.insert(0, "task", task)
                gains.insert(1, "endpoint", endpoint)
                gains.insert(2, "algorithm", algorithm)
                gains.insert(3, "normalization", mode)
                all_gains.append(gains)
            correlations, deltas = correlation_tables(
                data, task, endpoint, iterations, int(protocol["random_seed"])
            )
            correlations.insert(0, "normalization", mode)
            deltas.insert(0, "normalization", mode)
            all_correlations.append(correlations)
            all_deltas.append(deltas)
    damage_models = pd.concat(all_models["damage"], ignore_index=True)
    recovery_models = pd.concat(all_models["recovery"], ignore_index=True)
    damage_models.to_csv(tables / "05_damage_models_D0_D4.csv", index=False)
    recovery_models.to_csv(tables / "06_recovery_models_R0_R4.csv", index=False)
    correlations = add_corrections(
        pd.concat(all_correlations, ignore_index=True), "metric_correlations"
    )
    correlations.to_csv(tables / "03_baseline_metric_correlations.csv", index=False)
    deltas = add_corrections(
        pd.concat(all_deltas, ignore_index=True), "tnorm_vs_baseline"
    )
    deltas.to_csv(tables / "04_tnorm_vs_baseline_bootstrap.csv", index=False)
    gains = add_corrections(pd.concat(all_gains, ignore_index=True), "primary_model_gains")
    gains.to_csv(tables / "12_multiple_comparison_corrections.csv", index=False)
    return {
        "normalization": mode,
        "rows": len(data),
        "sequences": int(data["sequence_id"].nunique()),
        "bootstrap_iterations": iterations,
        "damage_models": len(damage_models),
        "recovery_models": len(recovery_models),
        "model_comparisons": len(gains),
    }


def main() -> None:
    protocol = load_protocol()
    parser = argparse.ArgumentParser(description="Frozen sequence-level Q1 analysis")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=output_root(protocol))
    parser.add_argument("--normalization", choices=MODES, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--bootstrap", type=int, default=int(protocol["bootstrap_iterations"]))
    args = parser.parse_args()
    if args.bootstrap < int(protocol["bootstrap_iterations"]):
        raise RuntimeError("Final Q1 analysis cannot reduce the frozen bootstrap count")
    assert_split_action("final_locked_evaluation", args.split)
    data = pd.read_csv(args.input)
    if data["split"].dropna().ne(args.split).any():
        raise RuntimeError(f"Input contains rows outside declared split {args.split}")
    summary = run_analysis(data, protocol, args.normalization, args.bootstrap, args.output)
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
