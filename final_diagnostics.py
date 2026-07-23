from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


SEED = 42
BOOTSTRAP_ITERATIONS = 2000

ANALYSIS_TEST = "outputs/diagnostics/analysis"
ANALYSIS_VAL = "outputs/diagnostics/analysis_val"

FEATURE_TEST = (
    "outputs/diagnostics/feature_consistency/"
    "feature_consistency_test.csv"
)

FEATURE_VAL = (
    "outputs/diagnostics/feature_consistency/"
    "feature_consistency_val.csv"
)

DETECTION_TEST = (
    "outputs/diagnostics/image_detection/"
    "image_detection_test.csv"
)

DETECTION_VAL = (
    "outputs/diagnostics/image_detection/"
    "image_detection_val.csv"
)

OUTPUT = "outputs/diagnostics/final_analysis"

SCENARIO_KEYS = [
    "image_path",
    "split",
    "attack",
    "epsilon_px",
]


def rho(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = spearmanr(
            left,
            right,
            nan_policy="omit",
        ).statistic

    return float(value)


def grouped_indices(
    groups: pd.Series,
) -> list[np.ndarray]:
    values = groups.astype(str).to_numpy()
    unique = pd.unique(values)

    return [
        np.flatnonzero(values == group)
        for group in unique
    ]


def bootstrap_statistics(
    values: list[float],
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if len(array) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    low = float(np.percentile(array, 2.5))
    high = float(np.percentile(array, 97.5))

    probability_low = (
        np.count_nonzero(array <= 0) + 1
    ) / (len(array) + 1)

    probability_high = (
        np.count_nonzero(array >= 0) + 1
    ) / (len(array) + 1)

    p_value = float(
        min(
            1.0,
            2.0 * min(
                probability_low,
                probability_high,
            ),
        )
    )

    return low, high, p_value


def model_gains(
    target: np.ndarray,
    prediction_c: np.ndarray,
    prediction_d: np.ndarray,
) -> dict[str, float]:
    return {
        "mae_gain": (
            mean_absolute_error(
                target,
                prediction_c,
            )
            - mean_absolute_error(
                target,
                prediction_d,
            )
        ),
        "r2_gain": (
            r2_score(
                target,
                prediction_d,
            )
            - r2_score(
                target,
                prediction_c,
            )
        ),
        "spearman_gain": (
            rho(target, prediction_d)
            - rho(target, prediction_c)
        ),
    }


def bootstrap_model_comparison(
    predictions_csv: Path,
    split: str,
    task: str,
    iterations: int,
    seed: int,
) -> list[dict[str, object]]:
    data = pd.read_csv(predictions_csv)

    target_column = (
        "f1_drop"
        if task == "damage"
        else "f1_recovery"
    )

    required = {
        "sequence_id",
        target_column,
        "prediction_C",
        "prediction_D",
    }

    missing = required - set(data.columns)

    if missing:
        raise RuntimeError(
            f"{predictions_csv}: missing "
            f"{sorted(missing)}"
        )

    data = data[
        [
            "sequence_id",
            target_column,
            "prediction_C",
            "prediction_D",
        ]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    target = data[target_column].to_numpy(float)
    prediction_c = data["prediction_C"].to_numpy(float)
    prediction_d = data["prediction_D"].to_numpy(float)

    groups = grouped_indices(data["sequence_id"])
    observed = model_gains(
        target,
        prediction_c,
        prediction_d,
    )

    rng = np.random.default_rng(seed)

    samples = {
        metric: []
        for metric in observed
    }

    for _ in range(iterations):
        selected_groups = rng.integers(
            0,
            len(groups),
            size=len(groups),
        )

        indices = np.concatenate([
            groups[index]
            for index in selected_groups
        ])

        gains = model_gains(
            target[indices],
            prediction_c[indices],
            prediction_d[indices],
        )

        for metric, value in gains.items():
            if np.isfinite(value):
                samples[metric].append(value)

    rows = []

    for metric, observed_value in observed.items():
        low, high, p_value = bootstrap_statistics(
            samples[metric]
        )

        rows.append({
            "split": split,
            "task": task,
            "comparison": "D_minus_C",
            "metric": metric,
            "observed_gain": observed_value,
            "ci_low": low,
            "ci_high": high,
            "p_value": p_value,
            "significant_95": bool(
                low > 0 or high < 0
            ),
            "iterations": iterations,
            "sequences": data["sequence_id"].nunique(),
            "grouping_unit": "sequence_id",
            "rows": len(data),
        })

    return rows


def correlation_gains(
    target: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    reference_rho = rho(
        reference,
        target,
    )

    candidate_rho = rho(
        candidate,
        target,
    )

    return {
        "reference_rho": reference_rho,
        "candidate_rho": candidate_rho,
        "signed_difference": (
            candidate_rho
            - reference_rho
        ),
        "absolute_gain": (
            abs(candidate_rho)
            - abs(reference_rho)
        ),
    }


def bootstrap_correlation_comparison(
    merged_csv: Path,
    split: str,
    task: str,
    reference_metric: str,
    candidate_metric: str,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    data = pd.read_csv(merged_csv)

    if task == "damage":
        data = data[
            (data["attack"] != "clean")
            & (data["defense"] == "none")
        ]

        target_column = "f1_drop"

    else:
        data = data[
            (data["attack"] != "clean")
            & (data["defense"] != "none")
        ]

        target_column = "f1_recovery"

    required = {
        "sequence_id",
        target_column,
        reference_metric,
        candidate_metric,
    }

    missing = required - set(data.columns)

    if missing:
        raise RuntimeError(
            f"{merged_csv}: missing "
            f"{sorted(missing)}"
        )

    data = data[
        [
            "sequence_id",
            target_column,
            reference_metric,
            candidate_metric,
        ]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    target = data[target_column].to_numpy(float)
    reference = data[reference_metric].to_numpy(float)
    candidate = data[candidate_metric].to_numpy(float)

    observed = correlation_gains(
        target,
        reference,
        candidate,
    )

    groups = grouped_indices(data["sequence_id"])
    rng = np.random.default_rng(seed)

    absolute_gains = []
    signed_differences = []

    for _ in range(iterations):
        selected_groups = rng.integers(
            0,
            len(groups),
            size=len(groups),
        )

        indices = np.concatenate([
            groups[index]
            for index in selected_groups
        ])

        gains = correlation_gains(
            target[indices],
            reference[indices],
            candidate[indices],
        )

        if np.isfinite(gains["absolute_gain"]):
            absolute_gains.append(
                gains["absolute_gain"]
            )

        if np.isfinite(
            gains["signed_difference"]
        ):
            signed_differences.append(
                gains["signed_difference"]
            )

    abs_low, abs_high, p_value = (
        bootstrap_statistics(
            absolute_gains
        )
    )

    signed_low = float(
        np.percentile(
            signed_differences,
            2.5,
        )
    )

    signed_high = float(
        np.percentile(
            signed_differences,
            97.5,
        )
    )

    return {
        "split": split,
        "task": task,
        "reference_metric": reference_metric,
        "candidate_metric": candidate_metric,
        "reference_rho": observed["reference_rho"],
        "candidate_rho": observed["candidate_rho"],
        "signed_difference": (
            observed["signed_difference"]
        ),
        "signed_ci_low": signed_low,
        "signed_ci_high": signed_high,
        "absolute_gain": observed["absolute_gain"],
        "absolute_gain_ci_low": abs_low,
        "absolute_gain_ci_high": abs_high,
        "p_value": p_value,
        "significant_95": bool(
            abs_low > 0 or abs_high < 0
        ),
        "iterations": iterations,
        "sequences": data["sequence_id"].nunique(),
        "grouping_unit": "sequence_id",
        "rows": len(data),
    }


def load_policy_table(
    feature_csv: Path,
    detection_csv: Path,
) -> pd.DataFrame:
    features = pd.read_csv(feature_csv)
    detections = pd.read_csv(detection_csv)

    feature_rows = features[
        (features["attack"] != "clean")
        & (features["defense"] == "none")
    ]

    scores = feature_rows.groupby(
        SCENARIO_KEYS,
        as_index=False,
    )["lukas_attack"].mean().rename(
        columns={
            "lukas_attack": "lukas_score",
        }
    )

    detection_rows = detections[
        detections["attack"] != "clean"
    ][
        SCENARIO_KEYS
        + [
            "defense",
            "f1",
        ]
    ]

    merged = detection_rows.merge(
        scores,
        on=SCENARIO_KEYS,
        how="inner",
        validate="many_to_one",
    )

    table = merged.pivot_table(
        index=(
            SCENARIO_KEYS
            + ["lukas_score"]
        ),
        columns="defense",
        values="f1",
        aggfunc="mean",
    ).reset_index()

    table.columns.name = None

    return table


def select_best_defense(
    data: pd.DataFrame,
    mask: np.ndarray,
    defenses: list[str],
) -> str:
    means = data.loc[
        mask,
        defenses,
    ].mean()

    means = means.dropna()

    if means.empty:
        raise RuntimeError(
            "No defense values in policy bin"
        )

    return str(means.idxmax())


def policy_predictions(
    data: pd.DataFrame,
    threshold_low: float,
    threshold_high: float,
    choices: dict[str, str],
) -> np.ndarray:
    scores = data["lukas_score"].to_numpy(float)

    low_mask = scores < threshold_low

    middle_mask = (
        (scores >= threshold_low)
        & (scores < threshold_high)
    )

    high_mask = scores >= threshold_high

    selected = np.full(
        len(data),
        np.nan,
        dtype=float,
    )

    for name, mask in [
        ("low", low_mask),
        ("middle", middle_mask),
        ("high", high_mask),
    ]:
        defense = choices[name]

        selected[mask] = data.loc[
            mask,
            defense,
        ].to_numpy(float)

    return selected


def fit_defense_policy(
    validation: pd.DataFrame,
    defenses: list[str],
) -> dict[str, object]:
    scores = validation[
        "lukas_score"
    ].dropna().to_numpy(float)

    thresholds = np.unique(
        np.quantile(
            scores,
            np.linspace(0.1, 0.9, 9),
        )
    )

    best_policy = None

    for low_index in range(
        len(thresholds) - 1
    ):
        for high_index in range(
            low_index + 1,
            len(thresholds),
        ):
            threshold_low = float(
                thresholds[low_index]
            )

            threshold_high = float(
                thresholds[high_index]
            )

            score_values = validation[
                "lukas_score"
            ].to_numpy(float)

            masks = {
                "low": (
                    score_values
                    < threshold_low
                ),
                "middle": (
                    (score_values >= threshold_low)
                    & (
                        score_values
                        < threshold_high
                    )
                ),
                "high": (
                    score_values
                    >= threshold_high
                ),
            }

            if any(
                not np.any(mask)
                for mask in masks.values()
            ):
                continue

            choices = {
                name: select_best_defense(
                    validation,
                    mask,
                    defenses,
                )
                for name, mask in masks.items()
            }

            predictions = policy_predictions(
                validation,
                threshold_low,
                threshold_high,
                choices,
            )

            mean_f1 = float(
                np.nanmean(predictions)
            )

            candidate = {
                "threshold_low": threshold_low,
                "threshold_high": threshold_high,
                "choices": choices,
                "validation_mean_f1": mean_f1,
            }

            if (
                best_policy is None
                or mean_f1
                > best_policy[
                    "validation_mean_f1"
                ]
            ):
                best_policy = candidate

    if best_policy is None:
        best_defense = str(
            validation[
                defenses
            ].mean().idxmax()
        )

        median = float(
            np.median(scores)
        )

        best_policy = {
            "threshold_low": median,
            "threshold_high": median,
            "choices": {
                "low": best_defense,
                "middle": best_defense,
                "high": best_defense,
            },
            "validation_mean_f1": float(
                validation[
                    best_defense
                ].mean()
            ),
        }

    return best_policy


def evaluate_defense_strategies(
    data: pd.DataFrame,
    split: str,
    defenses: list[str],
    policy: dict[str, object],
) -> pd.DataFrame:
    rows = []

    adaptive = policy_predictions(
        data,
        float(policy["threshold_low"]),
        float(policy["threshold_high"]),
        dict(policy["choices"]),
    )

    rows.append({
        "split": split,
        "strategy": "adaptive_lukas",
        "mean_f1": float(
            np.nanmean(adaptive)
        ),
        "median_f1": float(
            np.nanmedian(adaptive)
        ),
        "rows": len(data),
    })

    for defense in defenses:
        values = data[defense].to_numpy(float)

        rows.append({
            "split": split,
            "strategy": f"always_{defense}",
            "mean_f1": float(
                np.nanmean(values)
            ),
            "median_f1": float(
                np.nanmedian(values)
            ),
            "rows": len(data),
        })

    oracle = data[
        defenses
    ].max(axis=1).to_numpy(float)

    rows.append({
        "split": split,
        "strategy": "oracle",
        "mean_f1": float(
            np.nanmean(oracle)
        ),
        "median_f1": float(
            np.nanmedian(oracle)
        ),
        "rows": len(data),
    })

    result = pd.DataFrame(rows)

    fixed = result[
        result["strategy"].str.startswith(
            "always_"
        )
    ]

    best_fixed = float(
        fixed["mean_f1"].max()
    )

    result["gain_vs_best_fixed"] = (
        result["mean_f1"]
        - best_fixed
    )

    return result


def write_summary(
    output: Path,
    model_results: pd.DataFrame,
    correlation_results: pd.DataFrame,
    policy: dict[str, object],
    policy_results: pd.DataFrame,
) -> None:
    lines = [
        "# Final statistical analysis",
        "",
        "## Paired bootstrap: model D versus C",
        "",
    ]

    for _, row in model_results.iterrows():
        lines.append(
            f"- {row['split']} {row['task']} "
            f"{row['metric']}: "
            f"gain={row['observed_gain']:.6f}, "
            f"95% CI "
            f"[{row['ci_low']:.6f}, "
            f"{row['ci_high']:.6f}], "
            f"p={row['p_value']:.4f}"
        )

    lines += [
        "",
        "## Correlation comparisons",
        "",
    ]

    for _, row in correlation_results.iterrows():
        lines.append(
            f"- {row['split']} {row['task']} "
            f"{row['candidate_metric']} versus "
            f"{row['reference_metric']}: "
            f"absolute rho gain="
            f"{row['absolute_gain']:.6f}, "
            f"95% CI "
            f"[{row['absolute_gain_ci_low']:.6f}, "
            f"{row['absolute_gain_ci_high']:.6f}], "
            f"p={row['p_value']:.4f}"
        )

    lines += [
        "",
        "## Lukasiewicz defense policy",
        "",
        (
            f"Low threshold: "
            f"{policy['threshold_low']:.6f}"
        ),
        (
            f"High threshold: "
            f"{policy['threshold_high']:.6f}"
        ),
        (
            f"Low score defense: "
            f"{policy['choices']['low']}"
        ),
        (
            f"Middle score defense: "
            f"{policy['choices']['middle']}"
        ),
        (
            f"High score defense: "
            f"{policy['choices']['high']}"
        ),
        "",
    ]

    for _, row in policy_results.iterrows():
        lines.append(
            f"- {row['split']} "
            f"{row['strategy']}: "
            f"mean F1={row['mean_f1']:.6f}, "
            f"gain over best fixed="
            f"{row['gain_vs_best_fixed']:.6f}"
        )

    (
        output
        / "final_summary.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--analysis-test",
        default=ANALYSIS_TEST,
    )

    parser.add_argument(
        "--analysis-val",
        default=ANALYSIS_VAL,
    )

    parser.add_argument(
        "--features-test",
        default=FEATURE_TEST,
    )

    parser.add_argument(
        "--features-val",
        default=FEATURE_VAL,
    )

    parser.add_argument(
        "--detections-test",
        default=DETECTION_TEST,
    )

    parser.add_argument(
        "--detections-val",
        default=DETECTION_VAL,
    )

    parser.add_argument(
        "--output",
        default=OUTPUT,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    analysis_test = Path(
        args.analysis_test
    )

    analysis_val = Path(
        args.analysis_val
    )

    output = Path(args.output)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("FINAL STATISTICAL ANALYSIS")
    print(f"bootstrap: {args.bootstrap}")
    print(f"output:    {output}")
    print("=" * 80)

    model_rows = []

    for split, analysis_dir in [
        ("val", analysis_val),
        ("test", analysis_test),
    ]:
        for task in [
            "damage",
            "recovery",
        ]:
            predictions_csv = (
                analysis_dir
                / "tables"
                / f"{task}_predictions.csv"
            )

            model_rows.extend(
                bootstrap_model_comparison(
                    predictions_csv,
                    split,
                    task,
                    args.bootstrap,
                    SEED,
                )
            )

    model_results = pd.DataFrame(
        model_rows
    )

    model_results.to_csv(
        output
        / "bootstrap_model_comparison.csv",
        index=False,
    )

    correlation_rows = []

    comparisons = [
        (
            "damage",
            "cos_norm_attack",
            "product_attack",
        ),
        (
            "damage",
            "cos_norm_attack",
            "lukas_attack",
        ),
        (
            "recovery",
            "recovery_cos_norm",
            "recovery_product",
        ),
        (
            "recovery",
            "recovery_cos_norm",
            "recovery_lukas",
        ),
    ]

    for split, analysis_dir in [
        ("val", analysis_val),
        ("test", analysis_test),
    ]:
        merged_csv = (
            analysis_dir
            / "tables"
            / "merged_diagnostics.csv"
        )

        for (
            task,
            reference_metric,
            candidate_metric,
        ) in comparisons:
            correlation_rows.append(
                bootstrap_correlation_comparison(
                    merged_csv,
                    split,
                    task,
                    reference_metric,
                    candidate_metric,
                    args.bootstrap,
                    SEED,
                )
            )

    correlation_results = pd.DataFrame(
        correlation_rows
    )

    correlation_results.to_csv(
        output
        / "bootstrap_correlation_comparison.csv",
        index=False,
    )

    validation_policy_data = load_policy_table(
        Path(args.features_val),
        Path(args.detections_val),
    )

    test_policy_data = load_policy_table(
        Path(args.features_test),
        Path(args.detections_test),
    )

    metadata_columns = set(
        SCENARIO_KEYS
        + ["lukas_score"]
    )

    validation_defenses = {
        column
        for column in validation_policy_data.columns
        if column not in metadata_columns
    }

    test_defenses = {
        column
        for column in test_policy_data.columns
        if column not in metadata_columns
    }

    defenses = sorted(
        validation_defenses
        & test_defenses
    )

    if not defenses:
        raise RuntimeError(
            "No common defenses in val and test"
        )

    policy = fit_defense_policy(
        validation_policy_data,
        defenses,
    )

    validation_results = (
        evaluate_defense_strategies(
            validation_policy_data,
            "val",
            defenses,
            policy,
        )
    )

    test_results = (
        evaluate_defense_strategies(
            test_policy_data,
            "test",
            defenses,
            policy,
        )
    )

    policy_results = pd.concat(
        [
            validation_results,
            test_results,
        ],
        ignore_index=True,
    )

    policy_results.to_csv(
        output
        / "defense_policy_results.csv",
        index=False,
    )

    (
        output
        / "defense_policy.json"
    ).write_text(
        json.dumps(
            policy,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_summary(
        output,
        model_results,
        correlation_results,
        policy,
        policy_results,
    )

    print()
    print("MODEL BOOTSTRAP")
    print(
        model_results[
            [
                "split",
                "task",
                "metric",
                "observed_gain",
                "ci_low",
                "ci_high",
                "p_value",
            ]
        ].round(6).to_string(
            index=False
        )
    )

    print()
    print("DEFENSE POLICY")
    print(
        json.dumps(
            policy,
            indent=2,
        )
    )

    print()
    print(
        policy_results[
            [
                "split",
                "strategy",
                "mean_f1",
                "gain_vs_best_fixed",
            ]
        ].round(6).to_string(
            index=False
        )
    )

    print()
    print("DONE")
    print(
        output
        / "final_summary.md"
    )


if __name__ == "__main__":
    main()
