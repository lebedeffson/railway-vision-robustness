from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MERGED_CSV = (
    "outputs/diagnostics/analysis/"
    "tables/merged_diagnostics.csv"
)

OUTPUT = (
    "outputs/diagnostics/final_analysis/"
    "figures"
)


DAMAGE_METRICS = [
    "epsilon_px",
    "cos_norm_attack",
    "mse_attack",
    "mae_attack",
    "relative_l2_attack",
    "mean_shift_attack",
    "product_attack",
    "godel_attack",
    "lukas_attack",
    "f1_drop",
]


RECOVERY_METRICS = [
    "recovery_cos_norm",
    "recovery_mse",
    "recovery_mae",
    "recovery_relative_l2",
    "recovery_mean_shift",
    "recovery_product",
    "recovery_godel",
    "recovery_lukas",
    "f1_recovery",
]


BOX_METRICS = {
    "product_attack": "Product consistency",
    "godel_attack": "Godel consistency",
    "lukas_attack": "Lukasiewicz consistency",
}


def finite_data(
    data: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    return (
        data[columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )


def validate_columns(
    data: pd.DataFrame,
) -> None:
    required = {
        "image_path",
        "attack",
        "epsilon_px",
        "defense",
        *DAMAGE_METRICS,
        *RECOVERY_METRICS,
    }

    missing = required - set(data.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )


def save_damage_scatter(
    damage: pd.DataFrame,
    output: Path,
) -> None:
    scatter = finite_data(
        damage,
        [
            "lukas_attack",
            "f1_drop",
            "attack",
        ],
    )

    plt.figure(figsize=(8, 5))

    for attack in sorted(
        scatter["attack"].unique()
    ):
        subset = scatter[
            scatter["attack"] == attack
        ]

        plt.scatter(
            subset["lukas_attack"],
            subset["f1_drop"],
            s=12,
            alpha=0.25,
            label=str(attack).upper(),
        )

    plt.xlabel(
        "Lukasiewicz consistency"
    )

    plt.ylabel(
        "Image-level F1 drop"
    )

    plt.title(
        "Lukasiewicz consistency and detection damage"
    )

    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        output
        / "lukasiewicz_vs_f1_drop.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()


def save_recovery_scatter(
    recovery: pd.DataFrame,
    output: Path,
) -> None:
    scatter = finite_data(
        recovery,
        [
            "recovery_lukas",
            "f1_recovery",
            "defense",
        ],
    )

    plt.figure(figsize=(9, 5))

    for defense in sorted(
        scatter["defense"].unique()
    ):
        subset = scatter[
            scatter["defense"] == defense
        ]

        plt.scatter(
            subset["recovery_lukas"],
            subset["f1_recovery"],
            s=10,
            alpha=0.2,
            label=str(defense),
        )

    plt.xlabel(
        "Lukasiewicz feature recovery"
    )

    plt.ylabel(
        "Image-level F1 recovery"
    )

    plt.title(
        "Lukasiewicz recovery and detection recovery"
    )

    plt.legend(
        ncol=2,
        fontsize=8,
    )

    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        output
        / "lukasiewicz_recovery_vs_f1_recovery.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()


def scenario_label(
    attack: str,
    epsilon: float,
) -> str:
    return (
        f"{attack.upper()}\n"
        f"{epsilon:g}/255"
    )


def draw_boxplot(
    values: list[np.ndarray],
    labels: list[str],
) -> None:
    """
    Matplotlib >= 3.9 uses tick_labels.
    Older versions use labels.
    """

    try:
        plt.boxplot(
            values,
            tick_labels=labels,
            showfliers=False,
        )
    except TypeError:
        plt.boxplot(
            values,
            labels=labels,
            showfliers=False,
        )


def save_budget_boxplots(
    damage: pd.DataFrame,
    output: Path,
) -> None:
    scenarios = (
        damage[
            [
                "attack",
                "epsilon_px",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "attack",
                "epsilon_px",
            ]
        )
    )

    for metric, ylabel in BOX_METRICS.items():
        values: list[np.ndarray] = []
        labels: list[str] = []

        for row in scenarios.itertuples(
            index=False
        ):
            subset = damage[
                (
                    damage["attack"]
                    == row.attack
                )
                & (
                    damage["epsilon_px"]
                    == row.epsilon_px
                )
            ][metric]

            subset = (
                subset.replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
                .dropna()
            )

            if subset.empty:
                continue

            values.append(
                subset.to_numpy(
                    dtype=float
                )
            )

            labels.append(
                scenario_label(
                    str(row.attack),
                    float(row.epsilon_px),
                )
            )

        if not values:
            print(
                f"WARNING: no values for {metric}"
            )
            continue

        plt.figure(figsize=(11, 5))

        draw_boxplot(
            values,
            labels,
        )

        plt.xlabel(
            "Attack and perturbation budget"
        )

        plt.ylabel(ylabel)

        plt.title(
            f"{ylabel} by attack budget"
        )

        plt.grid(
            axis="y",
            alpha=0.2,
        )

        plt.tight_layout()

        plt.savefig(
            output
            / f"boxplot_{metric}_by_budget.png",
            dpi=250,
            bbox_inches="tight",
        )

        plt.close()


def short_label(
    column: str,
) -> str:
    labels = {
        "epsilon_px": "Epsilon",
        "cos_norm_attack": "Cosine",
        "mse_attack": "MSE",
        "mae_attack": "MAE",
        "relative_l2_attack": "Relative L2",
        "mean_shift_attack": "Mean shift",
        "product_attack": "Product",
        "godel_attack": "Godel",
        "lukas_attack": "Lukasiewicz",
        "f1_drop": "F1 drop",

        "recovery_cos_norm": "Cosine rec.",
        "recovery_mse": "MSE rec.",
        "recovery_mae": "MAE rec.",
        "recovery_relative_l2": "Relative L2 rec.",
        "recovery_mean_shift": "Mean shift rec.",
        "recovery_product": "Product rec.",
        "recovery_godel": "Godel rec.",
        "recovery_lukas": "Lukasiewicz rec.",
        "f1_recovery": "F1 recovery",
    }

    return labels.get(
        column,
        column,
    )


def save_heatmap(
    data: pd.DataFrame,
    columns: list[str],
    title: str,
    filename: str,
    output: Path,
) -> None:
    table = data[
        columns
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    correlation = table.corr(
        method="spearman"
    )

    labels = [
        short_label(column)
        for column in correlation.columns
    ]

    matrix = correlation.to_numpy(
        dtype=float
    )

    size = max(
        9.0,
        len(columns) * 0.95,
    )

    plt.figure(
        figsize=(
            size,
            size * 0.85,
        )
    )

    image = plt.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="Spearman correlation",
    )

    plt.xticks(
        np.arange(len(labels)),
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        np.arange(len(labels)),
        labels,
    )

    for row in range(
        matrix.shape[0]
    ):
        for column in range(
            matrix.shape[1]
        ):
            value = matrix[
                row,
                column,
            ]

            if np.isfinite(value):
                plt.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    plt.title(title)
    plt.tight_layout()

    plt.savefig(
        output / filename,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--merged",
        default=MERGED_CSV,
    )

    parser.add_argument(
        "--output",
        default=OUTPUT,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    merged_csv = Path(
        args.merged
    )

    output = Path(
        args.output
    )

    if not merged_csv.exists():
        raise FileNotFoundError(
            f"CSV not found: {merged_csv}"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(
        merged_csv
    )

    validate_columns(data)

    damage = data[
        (
            data["attack"] != "clean"
        )
        & (
            data["defense"] == "none"
        )
    ].copy()

    recovery = data[
        (
            data["attack"] != "clean"
        )
        & (
            data["defense"] != "none"
        )
    ].copy()

    print("=" * 80)
    print("FINAL FIGURES")
    print(f"merged: {merged_csv}")
    print(f"output: {output}")
    print(f"damage rows: {len(damage)}")
    print(f"recovery rows: {len(recovery)}")
    print("=" * 80)

    save_damage_scatter(
        damage,
        output,
    )

    save_recovery_scatter(
        recovery,
        output,
    )

    save_budget_boxplots(
        damage,
        output,
    )

    save_heatmap(
        damage,
        DAMAGE_METRICS,
        (
            "Spearman correlation matrix: "
            "feature damage"
        ),
        "damage_correlation_heatmap.png",
        output,
    )

    save_heatmap(
        recovery,
        RECOVERY_METRICS,
        (
            "Spearman correlation matrix: "
            "feature recovery"
        ),
        "recovery_correlation_heatmap.png",
        output,
    )

    print()
    print("CREATED:")

    created = sorted(
        output.glob("*.png")
    )

    for path in created:
        print(path)

    print()
    print(
        f"Total figures: {len(created)}"
    )
    print("DONE")


if __name__ == "__main__":
    main()