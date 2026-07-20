from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_INPUT = "outputs/final_practice/unified_diagnostics_raw.csv"
DEFAULT_OUTPUT = "outputs/final_practice/09_statistics"
BOOTSTRAP_ITERATIONS = 2000
SEED = 42


DAMAGE_MODELS = {
    "M0": ["epsilon", "attack", "perturbation_norm"],
    "M1": ["epsilon", "attack", "perturbation_norm", "cosine"],
    "M2": [
        "epsilon", "attack", "perturbation_norm", "cosine",
        "mse", "mae", "relative_l2", "mean_shift", "entropy",
    ],
    "M3": [
        "epsilon", "attack", "perturbation_norm", "cosine",
        "mse", "mae", "relative_l2", "mean_shift", "entropy",
        "product", "godel", "lukasiewicz",
    ],
    "M4": [
        "epsilon", "attack", "perturbation_norm", "cosine",
        "mse", "mae", "relative_l2", "mean_shift", "entropy",
        "product", "godel", "lukasiewicz",
        "c_sp_object", "c_dir", "c_atk_object",
    ],
}

RECOVERY_MODELS = {
    "M0": ["epsilon", "attack", "defense", "f1_attack"],
    "M1": ["epsilon", "attack", "defense", "f1_attack", "cosine_recovery"],
    "M2": [
        "epsilon", "attack", "defense", "f1_attack", "cosine_recovery",
        "mse_recovery", "mae_recovery", "relative_l2_recovery",
        "mean_shift_recovery", "entropy_recovery",
    ],
    "M3": [
        "epsilon", "attack", "defense", "f1_attack", "cosine_recovery",
        "mse_recovery", "mae_recovery", "relative_l2_recovery",
        "mean_shift_recovery", "entropy_recovery",
        "product_recovery", "godel_recovery", "lukasiewicz_recovery",
    ],
    "M4": [
        "epsilon", "attack", "defense", "f1_attack", "cosine_recovery",
        "mse_recovery", "mae_recovery", "relative_l2_recovery",
        "mean_shift_recovery", "entropy_recovery",
        "product_recovery", "godel_recovery", "lukasiewicz_recovery",
        "p_clean_preservation", "a_attacked_similarity", "r_restored_similarity",
        "g_recovery", "c_def",
    ],
}


def encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def pipeline(features: list[str], categorical: set[str]) -> Pipeline:
    numeric = [name for name in features if name not in categorical]
    categories = [name for name in features if name in categorical]
    transformer = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder()),
        ]), categories),
    ])
    return Pipeline([("preprocess", transformer), ("ridge", Ridge(alpha=1.0))])


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(spearmanr(left, right, nan_policy="omit").statistic)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "r2": float(r2_score(target, prediction)),
        "spearman": rank_correlation(target, prediction),
    }


def prepare_task(data: pd.DataFrame, task: str) -> tuple[pd.DataFrame, str, dict[str, list[str]]]:
    if task == "damage":
        subset = data[
            (data["attack"] != "clean")
            & (data["defense"] == "none")
            & (~data["adaptive"].astype(bool))
        ].copy()
        if "damage" not in subset:
            subset["damage"] = subset["f1_clean"] - subset["f1_attack"]
        group_keys = ["sequence_id", "attack", "epsilon", "seed"]
        target = "damage"
        model_sets = DAMAGE_MODELS
    else:
        subset = data[
            (data["attack"] != "clean")
            & (data["defense"] != "none")
            & (~data["adaptive"].astype(bool))
        ].copy()
        if "recovery" not in subset:
            subset["recovery"] = subset["f1_defended"] - subset["f1_attack"]
        group_keys = ["sequence_id", "attack", "epsilon", "defense", "seed"]
        target = "recovery"
        model_sets = RECOVERY_MODELS

    missing_keys = set(group_keys) - set(subset)
    if missing_keys:
        raise RuntimeError(f"{task}: missing analysis keys {sorted(missing_keys)}")
    needed = {target} | {column for columns in model_sets.values() for column in columns}
    missing = needed - set(subset)
    if missing:
        raise RuntimeError(f"{task}: incomplete final schema, missing {sorted(missing)}")

    categorical = {"attack", "defense", "perturbation_norm"}
    aggregations: dict[str, str] = {target: "mean"}
    for column in needed - {target}:
        aggregations[column] = "first" if column in categorical else "mean"
    aggregated = subset.groupby(group_keys, as_index=False, dropna=False).agg(aggregations)
    return aggregated, target, model_sets


def evaluate_task(data: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared, target_name, model_sets = prepare_task(data, task)
    sequence_count = prepared["sequence_id"].nunique()
    if sequence_count < 3:
        raise RuntimeError(f"{task}: at least 3 independent sequences are required")
    folds = min(5, sequence_count)
    cv = GroupKFold(n_splits=folds)
    target = prepared[target_name].to_numpy(float)
    predictions = prepared[["sequence_id", "attack", "epsilon", "seed", target_name]].copy()
    if "defense" in prepared:
        predictions["defense"] = prepared["defense"]
    rows = []
    categorical = {"attack", "defense", "perturbation_norm"}
    for name, features in model_sets.items():
        predicted = cross_val_predict(
            pipeline(features, categorical),
            prepared[features],
            target,
            groups=prepared["sequence_id"],
            cv=cv,
            n_jobs=-1,
        )
        predictions[f"prediction_{name}"] = predicted
        row = {
            "task": task,
            "model": name,
            **metrics(target, predicted),
            "rows": len(prepared),
            "sequences": sequence_count,
            "folds": folds,
            "grouping_unit": "sequence_id",
            "features": ",".join(features),
        }
        rows.append(row)
    return pd.DataFrame(rows), predictions


def bootstrap_predictions(predictions: pd.DataFrame, task: str, iterations: int, seed: int) -> pd.DataFrame:
    target_name = "damage" if task == "damage" else "recovery"
    models = [f"M{index}" for index in range(5)]
    observed = {
        model: metrics(
            predictions[target_name].to_numpy(float),
            predictions[f"prediction_{model}"].to_numpy(float),
        )
        for model in models
    }
    sequences = predictions["sequence_id"].drop_duplicates().to_numpy()
    indices = {
        sequence: np.flatnonzero(predictions["sequence_id"].to_numpy() == sequence)
        for sequence in sequences
    }
    rng = np.random.default_rng(seed)
    samples: dict[tuple[str, str, str], list[float]] = {}
    comparisons = list(zip(models[:-1], models[1:], strict=True))
    for _ in range(iterations):
        selected = rng.choice(sequences, size=len(sequences), replace=True)
        sampled_indices = np.concatenate([indices[value] for value in selected])
        target = predictions[target_name].to_numpy(float)[sampled_indices]
        sampled_metrics = {
            model: metrics(target, predictions[f"prediction_{model}"].to_numpy(float)[sampled_indices])
            for model in models
        }
        for left, right in comparisons:
            for metric_name in ("mae", "r2", "spearman"):
                gain = (
                    sampled_metrics[left][metric_name] - sampled_metrics[right][metric_name]
                    if metric_name == "mae"
                    else sampled_metrics[right][metric_name] - sampled_metrics[left][metric_name]
                )
                samples.setdefault((left, right, metric_name), []).append(gain)

    rows = []
    for left, right in comparisons:
        for metric_name in ("mae", "r2", "spearman"):
            observed_gain = (
                observed[left][metric_name] - observed[right][metric_name]
                if metric_name == "mae"
                else observed[right][metric_name] - observed[left][metric_name]
            )
            values = np.asarray(samples[(left, right, metric_name)])
            probability_low = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
            probability_high = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
            rows.append({
                "task": task,
                "comparison": f"{right}_minus_{left}",
                "metric": metric_name,
                "observed_gain": observed_gain,
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "p_value": min(1.0, 2.0 * min(probability_low, probability_high)),
                "iterations": iterations,
                "sequences": len(sequences),
            })
    return pd.DataFrame(rows)


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    array = values.to_numpy(float)
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], 1):
        rank = len(array) - reverse_rank + 1
        running = min(running, array[index] * len(array) / rank)
        adjusted[index] = running
    return pd.Series(adjusted, index=values.index)


def correlation_table(data: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "cosine", "mse", "mae", "relative_l2", "mean_shift", "entropy",
        "product", "godel", "lukasiewicz", "c_sp_object", "c_dir", "c_atk_object",
        "cosine_recovery", "product_recovery", "godel_recovery",
        "lukasiewicz_recovery", "g_recovery", "c_def",
    ]
    rows = []
    for target in ("damage", "recovery"):
        if target not in data:
            continue
        for layer in ["all", *sorted(data.get("layer", pd.Series(dtype=str)).dropna().unique())]:
            scope = data if layer == "all" else data[data["layer"] == layer]
            for metric_name in candidates:
                if metric_name not in scope:
                    continue
                pair = scope[[target, metric_name]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(pair) < 3 or pair[metric_name].nunique() < 2:
                    continue
                pearson = pearsonr(pair[metric_name], pair[target])
                spearman = spearmanr(pair[metric_name], pair[target])
                rows.append({
                    "target": target, "layer": layer, "metric": metric_name,
                    "pearson": pearson.statistic, "pearson_p": pearson.pvalue,
                    "spearman": spearman.statistic, "spearman_p": spearman.pvalue,
                    "rows": len(pair),
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["pearson_p_fdr"] = benjamini_hochberg(result["pearson_p"])
        result["spearman_p_fdr"] = benjamini_hochberg(result["spearman_p"])
    return result


def adaptive_comparison(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sequence_id", "attack", "epsilon", "steps", "seed", "defense",
        "adaptive", "f1_defended", "recall_defended", "false_negatives",
    }
    if required - set(data):
        return pd.DataFrame()
    pgd = data[(data["attack"] == "pgd") & (data["defense"] == "tnorm")].copy()
    if pgd.empty or pgd["adaptive"].nunique() < 2:
        return pd.DataFrame()
    keys = ["sequence_id", "epsilon", "steps", "seed"]
    aggregated = pgd.groupby([*keys, "adaptive"], as_index=False).agg({
        "f1_defended": "mean", "recall_defended": "mean", "false_negatives": "mean",
    })
    rows = []
    for metric_name in ("f1_defended", "recall_defended", "false_negatives"):
        pivot = aggregated.pivot(index=keys, columns="adaptive", values=metric_name)
        if False not in pivot or True not in pivot:
            continue
        table = pivot.dropna().reset_index()
        for _, row in table.iterrows():
            rows.append({
                **{key: row[key] for key in keys},
                "metric": metric_name,
                "nonadaptive": row[False],
                "adaptive": row[True],
                "adaptive_minus_nonadaptive": row[True] - row[False],
            })
    return pd.DataFrame(rows)


def interpretation(models: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {
        "practical_thresholds": {"delta_r2": 0.05, "mae_reduction_fraction": 0.05},
        "comparisons": {},
    }
    for task in models["task"].unique():
        table = models[models["task"] == task].set_index("model")
        delta_r2 = float(table.loc["M3", "r2"] - table.loc["M2", "r2"])
        mae_fraction = float((table.loc["M2", "mae"] - table.loc["M3", "mae"]) / table.loc["M2", "mae"])
        practically_significant = delta_r2 >= 0.05 or mae_fraction >= 0.05
        result["comparisons"][task] = {
            "comparison": "M3_minus_M2_tnorm_increment",
            "delta_r2": delta_r2,
            "mae_reduction_fraction": mae_fraction,
            "practically_significant": practically_significant,
            "allowed_claim": (
                "practical incremental value"
                if practically_significant
                else "small complementary diagnostic signal only"
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequence-level M0-M4 final statistics")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    if "sequence_id" not in data or data["sequence_id"].isna().any():
        raise RuntimeError("Final statistics require a complete sequence_id column")
    args.output.mkdir(parents=True, exist_ok=True)
    model_tables, prediction_tables, bootstrap_tables = [], [], []
    for task in ("damage", "recovery"):
        model_table, predictions = evaluate_task(data, task)
        model_tables.append(model_table)
        predictions.insert(0, "task", task)
        prediction_tables.append(predictions)
        bootstrap_tables.append(bootstrap_predictions(predictions, task, args.bootstrap, args.seed))
    models = pd.concat(model_tables, ignore_index=True)
    models.to_csv(args.output / "model_comparison_m0_m4.csv", index=False)
    pd.concat(prediction_tables, ignore_index=True).to_csv(args.output / "oof_predictions.csv", index=False)
    pd.concat(bootstrap_tables, ignore_index=True).to_csv(args.output / "sequence_bootstrap.csv", index=False)
    correlation_table(data).to_csv(args.output / "correlations_fdr.csv", index=False)
    adaptive_comparison(data).to_csv(
        args.output / "adaptive_vs_nonadaptive.csv", index=False
    )
    (args.output / "interpretation.json").write_text(
        json.dumps(interpretation(models), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
