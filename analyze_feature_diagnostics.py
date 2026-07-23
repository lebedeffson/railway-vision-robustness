from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from audit_final_practice import canonical_path, load_manifest


FEATURE_CSV = "outputs/diagnostics/feature_consistency/feature_consistency_test.csv"
DETECTION_CSV = "outputs/diagnostics/image_detection/image_detection_test.csv"
OUTPUT = "outputs/diagnostics/analysis"
BOOTSTRAP_ITERATIONS = 2000
SEED = 42
MANIFEST_CSV = "data/yolo_osdar23/manifest.csv"

MERGE_KEYS = [
    "image_path",
    "split",
    "attack",
    "epsilon_px",
    "defense",
]


def spearman(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": x, "y": y}).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return float("nan")

    return float(data["x"].rank().corr(data["y"].rank()))


def bootstrap_spearman(
    data: pd.DataFrame,
    metric: str,
    target: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    data = data[
        [metric, target, "sequence_id"]
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    groups = data["sequence_id"].unique()

    if len(groups) < 2:
        return float("nan"), float("nan"), float("nan")

    indices = {
        group: data.index[data["sequence_id"] == group].to_numpy()
        for group in groups
    }

    rng = np.random.default_rng(seed)
    values = []

    for _ in range(iterations):
        sampled_groups = rng.choice(
            groups,
            size=len(groups),
            replace=True,
        )

        sampled_indices = np.concatenate([
            indices[group]
            for group in sampled_groups
        ])

        sampled = data.loc[sampled_indices]
        value = spearman(sampled[metric], sampled[target])

        if np.isfinite(value):
            values.append(value)

    if not values:
        return float("nan"), float("nan"), float("nan")

    return (
        spearman(data[metric], data[target]),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def load_data(
    feature_csv: Path,
    detection_csv: Path,
    manifest_csv: Path,
) -> pd.DataFrame:
    features = pd.read_csv(feature_csv)
    detections = pd.read_csv(detection_csv)

    required_features = {
        *MERGE_KEYS,
        "level",

        "cos_norm_attack",
        "mse_attack",
        "mae_attack",
        "relative_l2_attack",
        "mean_shift_attack",
        "entropy_change_attack",
        "product_attack",
        "godel_attack",
        "lukas_attack",

        "recovery_cos_norm",
        "recovery_mse",
        "recovery_mae",
        "recovery_relative_l2",
        "recovery_mean_shift",
        "recovery_entropy",
        "recovery_product",
        "recovery_godel",
        "recovery_lukas",
    }

    required_detections = {
        *MERGE_KEYS,
        "f1",
    }

    missing_features = required_features - set(features.columns)
    missing_detections = required_detections - set(detections.columns)

    if missing_features:
        raise RuntimeError(
            f"Missing feature columns: {sorted(missing_features)}"
        )

    if missing_detections:
        raise RuntimeError(
            f"Missing detection columns: {sorted(missing_detections)}"
        )

    detection_metrics = detections[
        MERGE_KEYS
        + [
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "fn",
        ]
    ]

    merged = features.merge(
        detection_metrics,
        on=MERGE_KEYS,
        how="left",
        validate="many_to_one",
    )

    clean = detections[
        (detections["attack"] == "clean")
        & (detections["defense"] == "none")
    ][
        ["image_path", "split", "f1"]
    ].rename(
        columns={"f1": "f1_clean"}
    )

    attack_baseline = detections[
        (detections["attack"] != "clean")
        & (detections["defense"] == "none")
    ][
        [
            "image_path",
            "split",
            "attack",
            "epsilon_px",
            "f1",
        ]
    ].rename(
        columns={"f1": "f1_attack"}
    )

    merged = merged.merge(
        clean,
        on=["image_path", "split"],
        how="left",
        validate="many_to_one",
    )

    merged = merged.merge(
        attack_baseline,
        on=[
            "image_path",
            "split",
            "attack",
            "epsilon_px",
        ],
        how="left",
        validate="many_to_one",
    )

    merged["f1_drop"] = (
        merged["f1_clean"]
        - merged["f1_attack"]
    )

    merged["f1_recovery"] = (
        merged["f1"]
        - merged["f1_attack"]
    )

    if "sequence_id" not in merged.columns:
        manifest_rows = load_manifest(manifest_csv)
        exact = {
            canonical_path(row["image_path"]): row["sequence_id"]
            for row in manifest_rows
        }
        basename_values: dict[str, set[str]] = {}
        for row in manifest_rows:
            basename_values.setdefault(
                Path(row["image_path"]).name, set()
            ).add(row["sequence_id"])
        basename = {
            name: next(iter(values))
            for name, values in basename_values.items()
            if len(values) == 1
        }
        merged["sequence_id"] = merged["image_path"].map(
            lambda value: exact.get(
                canonical_path(str(value)),
                basename.get(Path(str(value)).name),
            )
        )

    if merged["sequence_id"].isna().any():
        examples = merged.loc[
            merged["sequence_id"].isna(), "image_path"
        ].astype(str).drop_duplicates().head(5).tolist()
        raise RuntimeError(
            "Cannot map every row to sequence_id; examples: " + repr(examples)
        )

    return merged


def correlation_table(
    data: pd.DataFrame,
    metrics: list[str],
    target: str,
    iterations: int,
) -> pd.DataFrame:
    rows = []

    scopes = [("all", data)]

    scopes += [
        (
            level,
            data[data["level"] == level],
        )
        for level in sorted(
            data["level"].dropna().unique()
        )
    ]

    for scope_name, scope_data in scopes:
        for metric in metrics:
            value, low, high = bootstrap_spearman(
                scope_data,
                metric,
                target,
                iterations,
                SEED,
            )

            rows.append({
                "scope": scope_name,
                "metric": metric,
                "target": target,
                "spearman": value,
                "ci_low": low,
                "ci_high": high,
                "rows": len(scope_data),
                "images": scope_data["image_path"].nunique(),
                "sequences": scope_data["sequence_id"].nunique(),
            })

    return pd.DataFrame(rows)


def encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def prediction_model(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    numeric = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scale",
            StandardScaler(),
        ),
    ])

    categorical = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            encoder(),
        ),
    ])

    preprocessing = ColumnTransformer([
        (
            "numeric",
            numeric,
            numeric_columns,
        ),
        (
            "categorical",
            categorical,
            categorical_columns,
        ),
    ])

    return Pipeline([
        (
            "preprocessing",
            preprocessing,
        ),
        (
            "ridge",
            Ridge(alpha=1.0),
        ),
    ])


def evaluate_models(
    data: pd.DataFrame,
    target: str,
    feature_sets: dict[str, list[str]],
    categorical_columns: list[str],
    task: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=[target, "sequence_id"]
    )

    groups = data["sequence_id"]
    group_count = groups.nunique()

    if group_count < 2:
        raise RuntimeError(
            f"Not enough sequence groups for {task}"
        )

    folds = min(5, group_count)
    cv = GroupKFold(n_splits=folds)

    results = []

    predictions = data[
        [
            "image_path",
            "sequence_id",
            "attack",
            "epsilon_px",
            "defense",
            "level",
            target,
        ]
    ].copy()

    for model_name, columns in feature_sets.items():
        categorical = [
            column
            for column in columns
            if column in categorical_columns
        ]

        numeric = [
            column
            for column in columns
            if column not in categorical
        ]

        model = prediction_model(
            numeric,
            categorical,
        )

        predicted = cross_val_predict(
            model,
            data[columns],
            data[target],
            groups=groups,
            cv=cv,
            n_jobs=2,
        )

        predictions[
            f"prediction_{model_name}"
        ] = predicted

        results.append({
            "task": task,
            "model": model_name,
            "features": ", ".join(columns),
            "mae": mean_absolute_error(
                data[target],
                predicted,
            ),
            "r2": r2_score(
                data[target],
                predicted,
            ),
            "spearman": spearman(
                pd.Series(
                    predicted,
                    index=data.index,
                ),
                data[target],
            ),
            "rows": len(data),
            "images": data["image_path"].nunique(),
            "sequences": group_count,
            "grouping_unit": "sequence_id",
            "folds": folds,
        })

    results = pd.DataFrame(results)

    baseline_mae = float(
        results.loc[
            results["model"] == "A",
            "mae",
        ].iloc[0]
    )

    baseline_r2 = float(
        results.loc[
            results["model"] == "A",
            "r2",
        ].iloc[0]
    )

    results["mae_improvement_vs_A"] = (
        baseline_mae
        - results["mae"]
    )

    results["r2_improvement_vs_A"] = (
        results["r2"]
        - baseline_r2
    )

    return results, predictions


def save_plots(
    merged: pd.DataFrame,
    model_results: pd.DataFrame,
    figures: Path,
) -> None:
    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    attack_data = merged[
        (merged["attack"] != "clean")
        & (merged["defense"] == "none")
    ]

    product_by_budget = attack_data.groupby(
        [
            "attack",
            "epsilon_px",
            "level",
        ],
        as_index=False,
    )["product_attack"].mean()

    for attack_name in sorted(
        product_by_budget["attack"].unique()
    ):
        subset = product_by_budget[
            product_by_budget["attack"] == attack_name
        ]

        plt.figure(figsize=(8, 5))

        for level in sorted(
            subset["level"].unique()
        ):
            level_data = subset[
                subset["level"] == level
            ]

            plt.plot(
                level_data["epsilon_px"],
                level_data["product_attack"],
                marker="o",
                label=level,
            )

        plt.xlabel(
            "Attack budget, pixels / 255"
        )

        plt.ylabel(
            "Product consistency"
        )

        plt.title(
            f"Product consistency: {attack_name.upper()}"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            figures
            / f"product_consistency_{attack_name}.png",
            dpi=200,
        )

        plt.close()

    recovery_data = merged[
        (merged["attack"] != "clean")
        & (merged["defense"] != "none")
    ]

    recovery_summary = recovery_data.groupby(
        ["attack", "defense"]
    )["recovery_product"].mean().unstack()

    recovery_summary.plot(
        kind="bar",
        figsize=(10, 5),
    )

    plt.ylabel(
        "Mean Product recovery"
    )

    plt.xlabel("Attack")

    plt.title(
        "Product feature recovery by defense"
    )

    plt.tight_layout()

    plt.savefig(
        figures
        / "product_recovery_by_defense.png",
        dpi=200,
    )

    plt.close()

    pivot = model_results.pivot(
        index="model",
        columns="task",
        values="mae",
    )

    pivot.plot(
        kind="bar",
        figsize=(8, 5),
    )

    plt.ylabel(
        "Grouped-CV MAE"
    )

    plt.xlabel(
        "Feature set"
    )

    plt.title(
        "Predictive comparison"
    )

    plt.tight_layout()

    plt.savefig(
        figures
        / "model_comparison_mae.png",
        dpi=200,
    )

    plt.close()

    scatter = attack_data[
        [
            "product_attack",
            "f1_drop",
            "attack",
        ]
    ].dropna()

    plt.figure(figsize=(8, 5))

    for attack_name in sorted(
        scatter["attack"].unique()
    ):
        subset = scatter[
            scatter["attack"] == attack_name
        ]

        plt.scatter(
            subset["product_attack"],
            subset["f1_drop"],
            s=8,
            alpha=0.25,
            label=attack_name.upper(),
        )

    plt.xlabel(
        "Product consistency"
    )

    plt.ylabel(
        "Image F1 drop"
    )

    plt.title(
        "Product consistency and detection degradation"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figures
        / "product_vs_f1_drop.png",
        dpi=200,
    )

    plt.close()


def write_summary(
    path: Path,
    damage_correlations: pd.DataFrame,
    recovery_correlations: pd.DataFrame,
    model_results: pd.DataFrame,
) -> None:
    lines = [
        "# Feature diagnostic analysis",
        "",
        "A: budget and categorical controls.",
        "B: A plus cosine similarity.",
        "C: B plus MSE, MAE, relative L2 and mean shift.",
        "D: C plus Product, Godel and Lukasiewicz.",
        "",
    ]

    for task in [
        "damage",
        "recovery",
    ]:
        task_results = model_results[
            model_results["task"] == task
        ].sort_values("mae")

        best = task_results.iloc[0]

        model_c = task_results[
            task_results["model"] == "C"
        ].iloc[0]

        model_d = task_results[
            task_results["model"] == "D"
        ].iloc[0]

        lines += [
            f"## {task.capitalize()}",
            "",
            f"Best model: {best['model']}",
            f"Best MAE: {best['mae']:.6f}",
            f"Best R2: {best['r2']:.6f}",
            f"Best Spearman: {best['spearman']:.6f}",
            (
                "T-norm MAE improvement over classical metrics: "
                f"{model_c['mae'] - model_d['mae']:.6f}"
            ),
            (
                "T-norm R2 improvement over classical metrics: "
                f"{model_d['r2'] - model_c['r2']:.6f}"
            ),
            (
                "T-norm Spearman difference over classical metrics: "
                f"{model_d['spearman'] - model_c['spearman']:.6f}"
            ),
            "",
        ]

    for title, table in [
        (
            "Damage correlations",
            damage_correlations,
        ),
        (
            "Recovery correlations",
            recovery_correlations,
        ),
    ]:
        overall = table[
            table["scope"] == "all"
        ].sort_values(
            "spearman",
            key=lambda values: values.abs(),
            ascending=False,
        )

        lines += [
            f"## {title}",
            "",
        ]

        for _, row in overall.iterrows():
            lines.append(
                f"- {row['metric']}: "
                f"rho={row['spearman']:.4f}, "
                f"95% CI "
                f"[{row['ci_low']:.4f}, "
                f"{row['ci_high']:.4f}]"
            )

        lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        default=FEATURE_CSV,
    )

    parser.add_argument(
        "--detections",
        default=DETECTION_CSV,
    )

    parser.add_argument(
        "--output",
        default=OUTPUT,
    )

    parser.add_argument(
        "--manifest",
        default=MANIFEST_CSV,
        help="Dataset manifest containing the independent sequence_id/group",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=BOOTSTRAP_ITERATIONS,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_csv = Path(args.features)
    detection_csv = Path(args.detections)
    output = Path(args.output)

    tables = output / "tables"
    figures = output / "figures"

    tables.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("FEATURE DIAGNOSTIC ANALYSIS")
    print(f"features:   {feature_csv}")
    print(f"detections: {detection_csv}")
    print(f"output:     {output}")
    print("=" * 80)

    merged = load_data(
        feature_csv,
        detection_csv,
        Path(args.manifest),
    )

    merged.to_csv(
        tables / "merged_diagnostics.csv",
        index=False,
    )

    damage = merged[
        (merged["attack"] != "clean")
        & (merged["defense"] == "none")
    ].copy()

    recovery = merged[
        (merged["attack"] != "clean")
        & (merged["defense"] != "none")
    ].copy()

    damage_metrics = [
        "epsilon_px",
        "cos_norm_attack",
        "mse_attack",
        "mae_attack",
        "relative_l2_attack",
        "mean_shift_attack",
        "entropy_change_attack",
        "product_attack",
        "godel_attack",
        "lukas_attack",
    ]

    recovery_metrics = [
        "recovery_cos_norm",
        "recovery_mse",
        "recovery_mae",
        "recovery_relative_l2",
        "recovery_mean_shift",
        "recovery_entropy",
        "recovery_product",
        "recovery_godel",
        "recovery_lukas",
    ]

    damage_correlations = correlation_table(
        damage,
        damage_metrics,
        "f1_drop",
        args.bootstrap,
    )

    recovery_correlations = correlation_table(
        recovery,
        recovery_metrics,
        "f1_recovery",
        args.bootstrap,
    )

    damage_correlations.to_csv(
        tables / "damage_correlations.csv",
        index=False,
    )

    recovery_correlations.to_csv(
        tables / "recovery_correlations.csv",
        index=False,
    )

    damage_sets = {
        "A": [
            "epsilon_px",
            "attack",
            "level",
        ],

        "B": [
            "epsilon_px",
            "attack",
            "level",
            "cos_norm_attack",
        ],

        "C": [
            "epsilon_px",
            "attack",
            "level",
            "cos_norm_attack",
            "mse_attack",
            "mae_attack",
            "relative_l2_attack",
            "mean_shift_attack",
            "entropy_change_attack",
        ],

        "D": [
            "epsilon_px",
            "attack",
            "level",
            "cos_norm_attack",
            "mse_attack",
            "mae_attack",
            "relative_l2_attack",
            "mean_shift_attack",
            "entropy_change_attack",
            "product_attack",
            "godel_attack",
            "lukas_attack",
        ],
    }

    recovery_sets = {
        "A": [
            "epsilon_px",
            "attack",
            "defense",
            "level",
        ],

        "B": [
            "epsilon_px",
            "attack",
            "defense",
            "level",
            "recovery_cos_norm",
        ],

        "C": [
            "epsilon_px",
            "attack",
            "defense",
            "level",
            "recovery_cos_norm",
            "recovery_mse",
            "recovery_mae",
            "recovery_relative_l2",
            "recovery_mean_shift",
            "recovery_entropy",
        ],

        "D": [
            "epsilon_px",
            "attack",
            "defense",
            "level",
            "recovery_cos_norm",
            "recovery_mse",
            "recovery_mae",
            "recovery_relative_l2",
            "recovery_mean_shift",
            "recovery_entropy",
            "recovery_product",
            "recovery_godel",
            "recovery_lukas",
        ],
    }

    categorical = [
        "attack",
        "defense",
        "level",
    ]

    damage_results, damage_predictions = evaluate_models(
        damage,
        "f1_drop",
        damage_sets,
        categorical,
        "damage",
    )

    recovery_results, recovery_predictions = evaluate_models(
        recovery,
        "f1_recovery",
        recovery_sets,
        categorical,
        "recovery",
    )

    model_results = pd.concat(
        [
            damage_results,
            recovery_results,
        ],
        ignore_index=True,
    )

    model_results.to_csv(
        tables / "model_comparison.csv",
        index=False,
    )

    damage_predictions.to_csv(
        tables / "damage_predictions.csv",
        index=False,
    )

    recovery_predictions.to_csv(
        tables / "recovery_predictions.csv",
        index=False,
    )

    save_plots(
        merged,
        model_results,
        figures,
    )

    write_summary(
        output / "diagnostic_summary.md",
        damage_correlations,
        recovery_correlations,
        model_results,
    )

    config = {
        "feature_csv": str(feature_csv),
        "detection_csv": str(detection_csv),
        "bootstrap_iterations": args.bootstrap,
        "damage_rows": len(damage),
        "recovery_rows": len(recovery),
        "images": merged["image_path"].nunique(),
        "sequences": merged["sequence_id"].nunique(),
        "grouping_unit": "sequence_id",
        "manifest_csv": str(Path(args.manifest)),
    }

    (
        output / "analysis_config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        model_results[
            [
                "task",
                "model",
                "mae",
                "r2",
                "spearman",
                "mae_improvement_vs_A",
                "r2_improvement_vs_A",
            ]
        ].round(6).to_string(index=False)
    )

    print()
    print("DONE")
    print(
        output / "diagnostic_summary.md"
    )


if __name__ == "__main__":
    main()
