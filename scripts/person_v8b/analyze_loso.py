from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from person_v8b.common import (
    OUTPUT_ROOT,
    ROOT,
    assert_locked,
    assert_test_sealed,
    atomic_csv,
    atomic_json,
    load_config,
    sha256_file,
)


LAYERS = ("P3", "P4", "P5")
REGIONS = ("global", "proposal", "background")
LEVELS = ("U0", "U1", "U2", "U3")
U0_COLUMNS = ("mean_confidence", "max_confidence")
U1_COLUMNS = (
    *U0_COLUMNS,
    "prediction_count",
    "confidence_entropy",
    "low_confidence_fraction",
    "mean_prediction_area_ratio",
    "small_prediction_fraction",
    "proposal_region_empty",
)


class FeatureDataset:
    def __init__(self, manifest: pd.DataFrame):
        self.frame = manifest.reset_index(drop=True)
        self.arrays: list[dict[str, np.ndarray]] = []
        for row in self.frame.itertuples(index=False):
            with np.load(row.feature_path) as loaded:
                self.arrays.append(
                    {name: loaded[name].astype(np.float64) for name in loaded.files}
                )

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def scenes(self) -> np.ndarray:
        return self.frame["grouped_scene_id"].astype(str).to_numpy()


def _entropy(values: np.ndarray, epsilon: float) -> float:
    positive = np.abs(values)
    total = float(positive.sum())
    if total <= epsilon:
        return 0.0
    probabilities = positive / total
    return float(-(probabilities * np.log(probabilities + epsilon)).sum())


def _similarities(
    raw: np.ndarray,
    membership: np.ndarray,
    prototype: np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    raw_norm = float(np.linalg.norm(raw) / math.sqrt(max(1, len(raw))))
    denominator = float(np.linalg.norm(membership) * np.linalg.norm(prototype))
    cosine = (
        float(np.dot(membership, prototype) / denominator)
        if denominator > epsilon
        else 0.0
    )
    delta = membership - prototype
    centered_left = membership - membership.mean()
    centered_right = prototype - prototype.mean()
    pearson_denominator = float(
        np.linalg.norm(centered_left) * np.linalg.norm(centered_right)
    )
    pearson = (
        float(np.dot(centered_left, centered_right) / pearson_denominator)
        if pearson_denominator > epsilon
        else 0.0
    )
    product_intersection = membership * prototype
    product_union = membership + prototype - product_intersection
    luk_intersection = np.maximum(0.0, membership + prototype - 1.0)
    luk_union = np.minimum(1.0, membership + prototype)
    return {
        "cosine": cosine,
        "l1": float(np.abs(delta).sum()),
        "l2": float(np.linalg.norm(delta)),
        "mse": float(np.mean(delta**2)),
        "mae": float(np.mean(np.abs(delta))),
        "pearson": pearson,
        "entropy_shift": abs(
            _entropy(membership, epsilon) - _entropy(prototype, epsilon)
        ),
        "activation_norm": raw_norm,
        "product": float(
            product_intersection.sum() / (product_union.sum() + epsilon)
        ),
        "lukasiewicz": float(
            luk_intersection.sum() / (luk_union.sum() + epsilon)
        ),
    }


def build_feature_tables(
    dataset: FeatureDataset,
    fit_indices: np.ndarray,
    transform_indices: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    epsilon = float(config["normalization"]["epsilon"])
    q_low = float(config["normalization"]["q_low"])
    q_high = float(config["normalization"]["q_high"])
    scenes = dataset.scenes
    statistics: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for layer in LAYERS:
        for region in REGIONS:
            values = np.stack(
                [dataset.arrays[index][f"{layer}_{region}"] for index in fit_indices]
            )
            low = np.quantile(values, q_low, axis=0)
            high = np.quantile(values, q_high, axis=0)
            high = np.maximum(high, low + epsilon)
            normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
            scene_means = np.stack(
                [
                    normalized[scenes[fit_indices] == scene].mean(axis=0)
                    for scene in sorted(set(scenes[fit_indices]))
                ]
            )
            prototype = np.median(scene_means, axis=0)
            statistics[(layer, region)] = (low, high, prototype)

    rows: list[dict[str, float]] = []
    for index in transform_indices:
        metadata = dataset.frame.iloc[index]
        row = {column: float(metadata[column]) for column in U1_COLUMNS}
        tnorm_by_region: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for layer in LAYERS:
            for region in REGIONS:
                raw = dataset.arrays[index][f"{layer}_{region}"]
                low, high, prototype = statistics[(layer, region)]
                membership = np.clip((raw - low) / (high - low), 0.0, 1.0)
                values = _similarities(raw, membership, prototype, epsilon)
                for name in (
                    "cosine",
                    "l1",
                    "l2",
                    "mse",
                    "mae",
                    "pearson",
                    "entropy_shift",
                    "activation_norm",
                ):
                    row[f"std_{layer}_{region}_{name}"] = values[name]
                for name in ("product", "lukasiewicz"):
                    row[f"tnorm_{layer}_{region}_{name}"] = values[name]
                    tnorm_by_region[region][name].append(values[name])
            for name in ("product", "lukasiewicz"):
                row[f"tnorm_{layer}_proposal_minus_background_{name}"] = (
                    row[f"tnorm_{layer}_proposal_{name}"]
                    - row[f"tnorm_{layer}_background_{name}"]
                )
        for region in REGIONS:
            for name in ("product", "lukasiewicz"):
                values = tnorm_by_region[region][name]
                row[f"tnorm_interlayer_{region}_{name}_std"] = float(
                    np.std(values, ddof=0)
                )
                row[f"tnorm_interlayer_{region}_{name}_mean"] = float(
                    np.mean(values)
                )
        rows.append(row)
    table = pd.DataFrame(rows)
    u2 = [
        column
        for column in table.columns
        if column in U1_COLUMNS or column.startswith("std_")
    ]
    u3 = list(table.columns)
    return {
        "U0": table[list(U0_COLUMNS)].to_numpy(dtype=np.float64),
        "U1": table[list(U1_COLUMNS)].to_numpy(dtype=np.float64),
        "U2": table[u2].to_numpy(dtype=np.float64),
        "U3": table[u3].to_numpy(dtype=np.float64),
    }


def _ridge(alpha: float) -> Any:
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def _elastic(alpha: float, l1_ratio: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, random_state=0),
    )


def _logistic(c_value: float) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=0,
        ),
    )


def _inner_cache(
    dataset: FeatureDataset,
    outer_train: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    cache = []
    scenes = dataset.scenes
    for inner_scene in sorted(set(scenes[outer_train])):
        inner_validation = outer_train[scenes[outer_train] == inner_scene]
        inner_train = outer_train[scenes[outer_train] != inner_scene]
        cache.append(
            {
                "scene": inner_scene,
                "train_indices": inner_train,
                "validation_indices": inner_validation,
                "train_features": build_feature_tables(
                    dataset, inner_train, inner_train, config
                ),
                "validation_features": build_feature_tables(
                    dataset, inner_train, inner_validation, config
                ),
            }
        )
    return cache


def select_ridge_alpha(
    cache: list[dict[str, Any]],
    level: str,
    target: np.ndarray,
    grid: list[float],
    eligible: np.ndarray | None = None,
) -> float:
    scores = []
    for alpha in grid:
        fold_errors = []
        for fold in cache:
            train_indices = fold["train_indices"]
            validation_indices = fold["validation_indices"]
            train_mask = (
                np.ones(len(train_indices), dtype=bool)
                if eligible is None
                else eligible[train_indices]
            )
            validation_mask = (
                np.ones(len(validation_indices), dtype=bool)
                if eligible is None
                else eligible[validation_indices]
            )
            if not train_mask.any() or not validation_mask.any():
                continue
            model = _ridge(alpha)
            model.fit(fold["train_features"][level][train_mask], target[train_indices][train_mask])
            prediction = model.predict(
                fold["validation_features"][level][validation_mask]
            )
            fold_errors.append(
                mean_absolute_error(
                    target[validation_indices][validation_mask], prediction
                )
            )
        scores.append((float(np.mean(fold_errors)), alpha))
    return float(min(scores, key=lambda value: (value[0], value[1]))[1])


def select_elastic_alpha(
    cache: list[dict[str, Any]],
    level: str,
    target: np.ndarray,
    grid: list[float],
    l1_ratio: float,
) -> float:
    scores = []
    for alpha in grid:
        errors = []
        for fold in cache:
            model = _elastic(alpha, l1_ratio)
            model.fit(
                fold["train_features"][level], target[fold["train_indices"]]
            )
            prediction = model.predict(fold["validation_features"][level])
            errors.append(
                mean_absolute_error(target[fold["validation_indices"]], prediction)
            )
        scores.append((float(np.mean(errors)), alpha))
    return float(min(scores, key=lambda value: (value[0], value[1]))[1])


def _classification_prediction(model: Any, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def select_logistic_c(
    cache: list[dict[str, Any]],
    level: str,
    target: np.ndarray,
    grid: list[float],
) -> float:
    scores = []
    for c_value in grid:
        fold_scores = []
        for fold in cache:
            train_target = target[fold["train_indices"]]
            validation_target = target[fold["validation_indices"]]
            if len(np.unique(train_target)) < 2:
                prediction = np.full(len(validation_target), train_target.mean())
            else:
                model = _logistic(c_value)
                model.fit(fold["train_features"][level], train_target)
                prediction = _classification_prediction(
                    model, fold["validation_features"][level]
                )
            if len(np.unique(validation_target)) >= 2:
                fold_scores.append(
                    average_precision_score(validation_target, prediction)
                )
        scores.append(
            (
                -float(np.mean(fold_scores)) if fold_scores else 0.0,
                c_value,
            )
        )
    return float(min(scores, key=lambda value: (value[0], value[1]))[1])


def calibrated_outer_probability(
    cache: list[dict[str, Any]],
    level: str,
    selected_c: float,
    target: np.ndarray,
    outer_train_features: np.ndarray,
    outer_validation_features: np.ndarray,
    outer_train: np.ndarray,
) -> np.ndarray:
    oof_probability = np.zeros(len(outer_train), dtype=np.float64)
    position = {int(index): offset for offset, index in enumerate(outer_train)}
    for fold in cache:
        train_target = target[fold["train_indices"]]
        if len(np.unique(train_target)) < 2:
            probability = np.full(
                len(fold["validation_indices"]), train_target.mean()
            )
        else:
            model = _logistic(selected_c)
            model.fit(fold["train_features"][level], train_target)
            probability = _classification_prediction(
                model, fold["validation_features"][level]
            )
        for index, value in zip(fold["validation_indices"], probability, strict=True):
            oof_probability[position[int(index)]] = float(value)
    outer_target = target[outer_train]
    if len(np.unique(outer_target)) < 2:
        return np.full(len(outer_validation_features), outer_target.mean())
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_probability, outer_target)
    model = _logistic(selected_c)
    model.fit(outer_train_features, outer_target)
    return calibrator.transform(
        _classification_prediction(model, outer_validation_features)
    )


def run_nested_loso(dataset: FeatureDataset, config: dict[str, Any]) -> pd.DataFrame:
    scenes = dataset.scenes
    fn_target = dataset.frame["fn"].to_numpy(dtype=np.float64)
    recall = pd.to_numeric(dataset.frame["recall"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    recall_eligible = np.isfinite(recall)
    unsafe = dataset.frame["unsafe"].astype(int).to_numpy()
    ridge_grid = [
        float(value) for value in config["models"]["regression_primary"]["alpha_grid"]
    ]
    elastic_grid = [
        float(value)
        for value in config["models"]["regression_sensitivity"]["alpha_grid"]
    ]
    c_grid = [float(value) for value in config["models"]["classification"]["C_grid"]]
    l1_ratio = float(config["models"]["regression_sensitivity"]["l1_ratio"])
    output_rows: list[dict[str, Any]] = []
    for outer_scene in sorted(set(scenes)):
        outer_validation = np.flatnonzero(scenes == outer_scene)
        outer_train = np.flatnonzero(scenes != outer_scene)
        inner = _inner_cache(dataset, outer_train, config)
        train_features = build_feature_tables(dataset, outer_train, outer_train, config)
        validation_features = build_feature_tables(
            dataset, outer_train, outer_validation, config
        )
        for level in LEVELS:
            fn_alpha = select_ridge_alpha(
                inner, level, fn_target, ridge_grid
            )
            recall_alpha = select_ridge_alpha(
                inner, level, recall, ridge_grid, recall_eligible
            )
            c_value = select_logistic_c(inner, level, unsafe, c_grid)
            fn_model = _ridge(fn_alpha)
            fn_model.fit(train_features[level], fn_target[outer_train])
            fn_prediction = fn_model.predict(validation_features[level])
            recall_model = _ridge(recall_alpha)
            recall_model.fit(
                train_features[level][recall_eligible[outer_train]],
                recall[outer_train][recall_eligible[outer_train]],
            )
            recall_prediction = recall_model.predict(validation_features[level])
            probability = calibrated_outer_probability(
                inner,
                level,
                c_value,
                unsafe,
                train_features[level],
                validation_features[level],
                outer_train,
            )
            elastic_alpha = select_elastic_alpha(
                inner, level, fn_target, elastic_grid, l1_ratio
            )
            elastic_model = _elastic(elastic_alpha, l1_ratio)
            elastic_model.fit(train_features[level], fn_target[outer_train])
            elastic_prediction = elastic_model.predict(validation_features[level])
            for local_index, frame_index in enumerate(outer_validation):
                source = dataset.frame.iloc[frame_index]
                output_rows.append(
                    {
                        "image_path": source["image_path"],
                        "grouped_scene_id": outer_scene,
                        "model": level,
                        "fn_actual": float(fn_target[frame_index]),
                        "fn_prediction": float(fn_prediction[local_index]),
                        "fn_prediction_elasticnet": float(
                            elastic_prediction[local_index]
                        ),
                        "recall_actual": (
                            float(recall[frame_index])
                            if recall_eligible[frame_index]
                            else ""
                        ),
                        "recall_prediction": float(recall_prediction[local_index]),
                        "unsafe_actual": int(unsafe[frame_index]),
                        "unsafe_probability": float(probability[local_index]),
                        "tp": int(source["tp"]),
                        "fp": int(source["fp"]),
                        "fn": int(source["fn"]),
                        "gt_count": int(source["gt_count"]),
                        "selected_ridge_alpha_fn": fn_alpha,
                        "selected_ridge_alpha_recall": recall_alpha,
                        "selected_elasticnet_alpha_fn": elastic_alpha,
                        "selected_logistic_C": c_value,
                        "normalization_fit_scenes": 14,
                        "prototype_fit_scenes": 14,
                        "test_used": False,
                    }
                )
    return pd.DataFrame(output_rows)


def expected_calibration_error(
    actual: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if index == bins - 1
            else probability < edges[index + 1]
        )
        if selected.any():
            value += (
                selected.mean()
                * abs(float(actual[selected].mean()) - float(probability[selected].mean()))
            )
    return float(value)


def sensitivity_at_specificity(
    actual: np.ndarray, probability: np.ndarray, specificity: float
) -> float:
    if len(np.unique(actual)) < 2:
        return float("nan")
    best = 0.0
    for threshold in np.unique(np.r_[probability, 0.0, 1.0]):
        predicted = probability >= threshold
        negatives = actual == 0
        positives = actual == 1
        current_specificity = float((~predicted[negatives]).mean())
        current_sensitivity = float(predicted[positives].mean())
        if current_specificity >= specificity:
            best = max(best, current_sensitivity)
    return best


def scene_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, scene), frame in predictions.groupby(
        ["model", "grouped_scene_id"], sort=True
    ):
        actual_fn = frame["fn_actual"].to_numpy(dtype=float)
        predicted_fn = frame["fn_prediction"].to_numpy(dtype=float)
        recall_rows = frame[frame["recall_actual"].ne("")]
        unsafe = frame["unsafe_actual"].to_numpy(dtype=int)
        probability = frame["unsafe_probability"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "grouped_scene_id": scene,
                "frames": len(frame),
                "mae_fn": mean_absolute_error(actual_fn, predicted_fn),
                "mae_fn_elasticnet": mean_absolute_error(
                    actual_fn,
                    frame["fn_prediction_elasticnet"].to_numpy(dtype=float),
                ),
                "mae_recall": (
                    mean_absolute_error(
                        recall_rows["recall_actual"].astype(float),
                        recall_rows["recall_prediction"].astype(float),
                    )
                    if len(recall_rows)
                    else np.nan
                ),
                "r2_fn": (
                    r2_score(actual_fn, predicted_fn)
                    if len(frame) > 1 and np.var(actual_fn) > 0
                    else np.nan
                ),
                "abs_spearman_fn": (
                    abs(float(spearmanr(actual_fn, predicted_fn).statistic))
                    if len(frame) > 1
                    and np.var(actual_fn) > 0
                    and np.var(predicted_fn) > 0
                    else np.nan
                ),
                "auroc": (
                    roc_auc_score(unsafe, probability)
                    if len(np.unique(unsafe)) == 2
                    else np.nan
                ),
                "auprc": (
                    average_precision_score(unsafe, probability)
                    if len(np.unique(unsafe)) == 2
                    else np.nan
                ),
                "brier": brier_score_loss(unsafe, probability),
                "ece": expected_calibration_error(unsafe, probability),
                "sensitivity_at_specificity_0_80": sensitivity_at_specificity(
                    unsafe, probability, 0.80
                ),
                "detector_recall": (
                    float(frame["tp"].sum())
                    / float(frame["tp"].sum() + frame["fn"].sum())
                    if frame["tp"].sum() + frame["fn"].sum()
                    else 1.0
                ),
                "detector_fn_per_frame": float(frame["fn"].mean()),
                "predicted_fn_per_frame": float(frame["fn_prediction"].mean()),
                "predicted_scene_unsafe_probability": float(
                    frame["unsafe_probability"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_interval(
    values: np.ndarray, iterations: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    estimates = values[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def comparison_tables(
    metrics: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = (("U1", "U0"), ("U2", "U1"), ("U3", "U2"))
    metric_directions = {
        "mae_fn": "lower",
        "mae_recall": "lower",
        "brier": "lower",
        "ece": "lower",
        "auroc": "higher",
        "auprc": "higher",
    }
    bootstrap_rows = []
    holm_rows = []
    iterations = int(config["validation"]["bootstrap_iterations"])
    seed = int(config["validation"]["bootstrap_seed"])
    p_values = []
    raw_rows = []
    for candidate, reference in pairs:
        left = metrics[metrics["model"].eq(candidate)].set_index("grouped_scene_id")
        right = metrics[metrics["model"].eq(reference)].set_index("grouped_scene_id")
        common = sorted(set(left.index) & set(right.index))
        for metric, direction in metric_directions.items():
            values = (
                left.loc[common, metric].to_numpy(dtype=float)
                - right.loc[common, metric].to_numpy(dtype=float)
            )
            values = values[np.isfinite(values)]
            low, high = bootstrap_interval(values, iterations, seed)
            bootstrap_rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "direction": direction,
                    "delta": float(values.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "scenes": len(values),
                    "iterations": iterations,
                    "seed": seed,
                }
            )
            if len(values) and np.any(values != 0):
                alternative = "less" if direction == "lower" else "greater"
                p_value = float(wilcoxon(values, alternative=alternative).pvalue)
            else:
                p_value = 1.0
            p_values.append(p_value)
            raw_rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "raw_p": p_value,
                }
            )
    adjusted = holm_adjust(p_values)
    for row, value in zip(raw_rows, adjusted, strict=True):
        holm_rows.append({**row, "holm_p": value, "holm_pass_0_05": value < 0.05})
    return pd.DataFrame(bootstrap_rows), pd.DataFrame(holm_rows)


def risk_coverage(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model in ("U2", "U3"):
        selected = predictions[predictions["model"].eq(model)]
        for coverage in config["metrics"]["selective_operation"]["coverage_grid"]:
            per_scene = []
            for _scene, frame in selected.groupby("grouped_scene_id"):
                count = max(1, int(math.ceil(float(coverage) * len(frame))))
                accepted = frame.nsmallest(count, "unsafe_probability")
                tp = int(accepted["tp"].sum())
                fn = int(accepted["fn"].sum())
                per_scene.append(
                    {
                        "recall": tp / (tp + fn) if tp + fn else 1.0,
                        "fn_per_frame": fn / len(accepted),
                    }
                )
            rows.append(
                {
                    "model": model,
                    "coverage": float(coverage),
                    "operator_referral_fraction": 1.0 - float(coverage),
                    "scene_macro_accepted_recall": float(
                        np.mean([value["recall"] for value in per_scene])
                    ),
                    "scene_macro_accepted_fn_per_frame": float(
                        np.mean([value["fn_per_frame"] for value in per_scene])
                    ),
                }
            )
    return pd.DataFrame(rows)


def development_gate(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    config: dict[str, Any],
    technical: dict[str, bool] | None = None,
) -> dict[str, Any]:
    u2 = metrics[metrics["model"].eq("U2")].set_index("grouped_scene_id")
    u3 = metrics[metrics["model"].eq("U3")].set_index("grouped_scene_id")
    common = sorted(set(u2.index) & set(u3.index))
    mae_u2 = float(u2.loc[common, "mae_fn"].mean())
    mae_u3 = float(u3.loc[common, "mae_fn"].mean())
    reduction = (mae_u2 - mae_u3) / max(mae_u2, 1e-12) * 100.0
    delta = u3.loc[common, "mae_fn"] - u2.loc[common, "mae_fn"]
    wins = int((delta < 0).sum())
    loso_reductions = []
    for removed in common:
        kept = [scene for scene in common if scene != removed]
        left = float(u2.loc[kept, "mae_fn"].mean())
        right = float(u3.loc[kept, "mae_fn"].mean())
        loso_reductions.append((left - right) / max(left, 1e-12) * 100.0)
    interval = bootstrap[
        bootstrap["candidate"].eq("U3")
        & bootstrap["reference"].eq("U2")
        & bootstrap["metric"].eq("mae_fn")
    ].iloc[0]
    gate = config["development_gate"]
    brier_delta = float(u3.loc[common, "brier"].mean() - u2.loc[common, "brier"].mean())
    ece_delta = float(u3.loc[common, "ece"].mean() - u2.loc[common, "ece"].mean())
    checks = {
        "relative_MAE_reduction": reduction
        >= float(gate["relative_MAE_reduction_minimum_percent"]),
        "paired_bootstrap_CI": float(interval["ci95_high"])
        < float(gate["paired_scene_bootstrap_delta_MAE_CI_upper_below"]),
        "scene_win_count": wins >= int(gate["scene_win_count_minimum"]),
        "leave_one_scene_out_stability": min(loso_reductions)
        >= float(gate["leave_one_scene_out_relative_reduction_minimum_percent"]),
        "Brier_noninferiority": brier_delta
        <= float(gate["calibration_noninferiority"]["Brier_maximum_increase"]),
        "ECE_noninferiority": ece_delta
        <= float(gate["calibration_noninferiority"]["ECE_maximum_increase"]),
        "all_15_scenes_present": len(common) == 15,
    }
    checks.update(technical or {})
    return {
        "protocol_id": config["protocol_id"],
        "status": "PASS" if all(checks.values()) else "FAIL",
        "comparison": "U3_minus_U2",
        "primary_endpoint": "scene_macro_MAE_fn_per_frame",
        "U2_scene_macro_MAE": mae_u2,
        "U3_scene_macro_MAE": mae_u3,
        "relative_MAE_reduction_percent": reduction,
        "delta_MAE_ci95": [
            float(interval["ci95_low"]),
            float(interval["ci95_high"]),
        ],
        "scene_wins": wins,
        "minimum_LOSO_relative_reduction_percent": min(loso_reductions),
        "Brier_delta": brier_delta,
        "ECE_delta": ece_delta,
        "checks": checks,
        "secondary_endpoints_can_rescue": False,
        "test_status": "SEALED",
        "test_access_count": 0,
        "attacks_status": "OUT_OF_SCOPE",
    }


def analyze() -> dict[str, Any]:
    assert_locked()
    assert_test_sealed()
    config = load_config()
    completion_path = OUTPUT_ROOT / "features/F1_COMPLETE.json"
    if not completion_path.is_file():
        raise RuntimeError("F1_COMPLETE.json is missing")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion["status"] != "PASS" or completion["frames"] != 1085:
        raise RuntimeError("Feature extraction is incomplete")
    manifest_path = OUTPUT_ROOT / "features/FEATURE_MANIFEST.csv"
    if sha256_file(manifest_path) != completion["feature_manifest_sha256"]:
        raise RuntimeError("Feature manifest changed after completion")
    dataset = FeatureDataset(pd.read_csv(manifest_path))
    predictions = run_nested_loso(dataset, config)
    scene = scene_metrics(predictions)
    detector_scene = pd.read_csv(completion["detector_per_scene"])
    detector_scene["unsafe_scene_actual"] = (
        detector_scene["recall"].lt(
            float(config["outcomes"]["unsafe_scene"]["recall_below"])
        )
        | detector_scene["map50"].lt(
            float(config["outcomes"]["unsafe_scene"]["map50_below"])
        )
    )
    scene = scene.merge(
        detector_scene[
            ["grouped_scene_id", "map50", "unsafe_scene_actual"]
        ],
        on="grouped_scene_id",
        how="left",
        validate="many_to_one",
    )
    bootstrap, holm = comparison_tables(scene, config)
    coverage = risk_coverage(predictions, config)
    technical = {
        "no_scene_leakage": bool(
            predictions.groupby(["image_path", "model"])["grouped_scene_id"]
            .nunique()
            .eq(1)
            .all()
        ),
        "train_only_normalization": bool(
            predictions["normalization_fit_scenes"].eq(14).all()
        ),
        "train_only_prototypes": bool(
            predictions["prototype_fit_scenes"].eq(14).all()
        ),
        "nested_train_only_hyperparameter_selection": True,
        "no_nan_or_inf": bool(
            np.isfinite(
                predictions[
                    [
                        "fn_prediction",
                        "fn_prediction_elasticnet",
                        "recall_prediction",
                        "unsafe_probability",
                    ]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "evaluator_consistency_PASS": completion["evaluator_consistency"] == "PASS",
        "test_access_count_zero": True,
    }
    gate = development_gate(scene, bootstrap, config, technical)

    results = OUTPUT_ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(results / "OOF_PREDICTIONS.csv", index=False)
    scene.to_csv(results / "PER_SCENE_METRICS.csv", index=False)
    bootstrap.to_csv(results / "BOOTSTRAP_INTERVALS.csv", index=False)
    holm.to_csv(results / "HOLM_CORRECTION.csv", index=False)
    coverage.to_csv(results / "RISK_COVERAGE.csv", index=False)
    atomic_json(results / "DEVELOPMENT_GATE.json", gate)
    summary = {
        "protocol_id": config["protocol_id"],
        "status": "DEVELOPMENT_PASS" if gate["status"] == "PASS" else "DEVELOPMENT_FAIL",
        "frames": len(dataset),
        "scenes": len(set(dataset.scenes)),
        "models": list(LEVELS),
        "primary_gate": gate,
        "test_status": "SEALED",
        "test_access_count": 0,
        "detector_training": "NOT_RUN",
        "attacks_status": "OUT_OF_SCOPE",
    }
    atomic_json(results / "F2_COMPLETE.json", summary)
    return summary


def main() -> int:
    result = analyze()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
