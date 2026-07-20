from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# ПУТИ И ПАРАМЕТРЫ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

FINAL_TEST_PATH = (
    PROJECT_DIR
    / "outputs"
    / "defenses"
    / "final_test"
    / "final_test_results.json"
)

ADAPTIVE_PATH = (
    PROJECT_DIR
    / "outputs"
    / "defenses"
    / "adaptive_attacks"
    / "adaptive_attacks_results.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "analysis"
    / "final_results"
)

TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
STATISTICS_DIR = OUTPUT_DIR / "statistics"

SCRIPT_VERSION = 1

RANDOM_SEED = 2026

BOOTSTRAP_ITERATIONS = 20_000

FIGURE_DPI = 220


METHOD_ORDER = [
    "none",
    "tnorm",
    "bilateral",
    "jpeg",
    "median",
    "gaussian",
]

METHOD_LABELS = {
    "none": "No defense",
    "tnorm": "Product T-norm",
    "bilateral": "Bilateral",
    "jpeg": "JPEG",
    "median": "Median",
    "gaussian": "Gaussian",
}

CLASS_NAMES = {
    0: "person",
    1: "signal",
    2: "road_vehicle",
    3: "train",
    4: "animal",
    5: "bicycle",
}

MAIN_CLASSES = [
    0,
    1,
    2,
    3,
]

ALL_CLASSES = [
    0,
    1,
    2,
    3,
    4,
    5,
]

ATTACK_ORDER = [
    "fgsm",
    "pgd",
]

EPSILON_ORDER = [
    1,
    2,
    4,
    8,
]


# ============================================================
# ОБЩИЕ ФУНКЦИИ
# ============================================================

def seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def write_text(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        text,
        encoding="utf-8",
    )


def write_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            f"Нет строк для записи: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def prepare_directories() -> None:
    for path in (
        OUTPUT_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        STATISTICS_DIR,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def check_input_files() -> None:
    missing = [
        path

        for path in (
            FINAL_TEST_PATH,
            ADAPTIVE_PATH,
        )

        if not path.is_file()
    ]

    if missing:
        message = "\n".join(
            str(path)
            for path in missing
        )

        raise FileNotFoundError(
            "Не найдены необходимые файлы результатов:\n"
            f"{message}"
        )


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    frame_name: str,
) -> None:
    missing = [
        column

        for column in columns

        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"{frame_name} не содержит столбцы: "
            + ", ".join(missing)
        )


def numeric_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    result = frame.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def method_sort_key(
    method: str,
) -> int:
    try:
        return METHOD_ORDER.index(
            method
        )

    except ValueError:
        return len(
            METHOD_ORDER
        )


def epsilon_label(
    epsilon: int | float,
) -> str:
    return f"{int(epsilon)}/255"


def safe_mean(
    values: pd.Series,
) -> float | None:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return None

    return float(
        numeric.mean()
    )


def safe_median(
    values: pd.Series,
) -> float | None:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return None

    return float(
        numeric.median()
    )


def safe_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        result = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        result
    ):
        return None

    return result


def format_float(
    value: float | None,
    digits: int = 4,
) -> str:
    if (
        value is None
        or not math.isfinite(value)
    ):
        return "n/a"

    return f"{value:.{digits}f}"


def format_percent(
    value: float | None,
    digits: int = 2,
) -> str:
    if (
        value is None
        or not math.isfinite(value)
    ):
        return "n/a"

    return (
        f"{value * 100.0:.{digits}f}%"
    )


def save_figure(
    figure: plt.Figure,
    filename_stem: str,
) -> None:
    png_path = (
        FIGURES_DIR
        / f"{filename_stem}.png"
    )

    pdf_path = (
        FIGURES_DIR
        / f"{filename_stem}.pdf"
    )

    figure.savefig(
        png_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# ЗАГРУЗКА И НОРМАЛИЗАЦИЯ
# ============================================================

def load_result_frames() -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    final_data = read_json(
        FINAL_TEST_PATH
    )

    adaptive_data = read_json(
        ADAPTIVE_PATH
    )

    final_summary = pd.DataFrame(
        final_data.get(
            "summary_rows",
            [],
        )
    )

    final_classes = pd.DataFrame(
        final_data.get(
            "class_rows",
            [],
        )
    )

    adaptive_summary = pd.DataFrame(
        adaptive_data.get(
            "summary_rows",
            [],
        )
    )

    adaptive_classes = pd.DataFrame(
        adaptive_data.get(
            "class_rows",
            [],
        )
    )

    if final_summary.empty:
        raise ValueError(
            "final_test_results.json "
            "не содержит summary_rows."
        )

    if final_classes.empty:
        raise ValueError(
            "final_test_results.json "
            "не содержит class_rows."
        )

    if adaptive_summary.empty:
        raise ValueError(
            "adaptive_attacks_results.json "
            "не содержит summary_rows."
        )

    if adaptive_classes.empty:
        raise ValueError(
            "adaptive_attacks_results.json "
            "не содержит class_rows."
        )

    require_columns(
        final_summary,
        (
            "method",
            "identifier",
            "attack",
            "epsilon_pixels",
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
            "clean_mAP50_relative_drop",
            "recovery_mAP50",
            "recovery_mAP50-95",
            "defense_ms_per_image",
        ),
        "final summary",
    )

    require_columns(
        final_classes,
        (
            "method",
            "attack",
            "epsilon_pixels",
            "class_id",
            "class_name",
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
        ),
        "final class results",
    )

    require_columns(
        adaptive_summary,
        (
            "attack",
            "epsilon_pixels",
            "no_defense_attacked_mAP50",
            "oblivious_tnorm_mAP50",
            "adaptive_tnorm_mAP50",
            "adaptive_minus_oblivious_mAP50",
            "oblivious_recovery_mAP50",
            "adaptive_recovery_mAP50",
            "adaptive_tnorm_mAP50-95",
        ),
        "adaptive summary",
    )

    require_columns(
        adaptive_classes,
        (
            "attack",
            "epsilon_pixels",
            "class_id",
            "class_name",
            "no_defense_mAP50",
            "oblivious_mAP50",
            "adaptive_mAP50",
            "no_defense_mAP50-95",
            "oblivious_mAP50-95",
            "adaptive_mAP50-95",
        ),
        "adaptive class results",
    )

    final_summary = numeric_columns(
        final_summary,
        (
            "epsilon_pixels",
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
            "fitness",
            "clean_reference_mAP50",
            "clean_reference_mAP50-95",
            "clean_defended_mAP50",
            "clean_defended_mAP50-95",
            "clean_mAP50_relative_drop",
            "clean_mAP50-95_relative_drop",
            "attacked_reference_mAP50",
            "attacked_reference_mAP50-95",
            "recovery_mAP50",
            "recovery_mAP50-95",
            "defense_ms_per_image",
            "defended_vs_clean_psnr",
            "defended_vs_clean_ssim",
            "defense_change_mae_pixels",
            "defense_change_linf_pixels",
            "defended_vs_clean_mae_pixels",
            "preprocess_ms",
            "inference_ms",
            "loss_ms",
            "postprocess_ms",
            "validator_total_ms",
        ),
    )

    final_classes = numeric_columns(
        final_classes,
        (
            "epsilon_pixels",
            "class_id",
            "images",
            "instances",
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
        ),
    )

    adaptive_summary = numeric_columns(
        adaptive_summary,
        tuple(
            column

            for column
            in adaptive_summary.columns

            if column
            not in {
                "attack",
                "proxy_target",
            }
        ),
    )

    adaptive_classes = numeric_columns(
        adaptive_classes,
        tuple(
            column

            for column
            in adaptive_classes.columns

            if column
            not in {
                "attack",
                "class_name",
            }
        ),
    )

    final_summary["method"] = (
        final_summary["method"]
        .astype(str)
        .str.lower()
    )

    final_summary["attack"] = (
        final_summary["attack"]
        .astype(str)
        .str.lower()
    )

    final_classes["method"] = (
        final_classes["method"]
        .astype(str)
        .str.lower()
    )

    final_classes["attack"] = (
        final_classes["attack"]
        .astype(str)
        .str.lower()
    )

    adaptive_summary["attack"] = (
        adaptive_summary["attack"]
        .astype(str)
        .str.lower()
    )

    adaptive_classes["attack"] = (
        adaptive_classes["attack"]
        .astype(str)
        .str.lower()
    )

    final_summary["method_label"] = (
        final_summary["method"]
        .map(METHOD_LABELS)
        .fillna(
            final_summary["method"]
        )
    )

    final_classes["method_label"] = (
        final_classes["method"]
        .map(METHOD_LABELS)
        .fillna(
            final_classes["method"]
        )
    )

    return (
        final_data,
        adaptive_data,
        final_summary,
        final_classes,
        adaptive_summary,
        adaptive_classes,
    )


# ============================================================
# ИТОГОВЫЕ ТАБЛИЦЫ
# ============================================================

def build_method_overview(
    final_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for method in METHOD_ORDER:
        method_rows = final_summary[
            final_summary["method"]
            == method
        ]

        if method_rows.empty:
            continue

        clean_rows = method_rows[
            method_rows["attack"]
            == "clean"
        ]

        fgsm_rows = method_rows[
            method_rows["attack"]
            == "fgsm"
        ]

        pgd_rows = method_rows[
            method_rows["attack"]
            == "pgd"
        ]

        if clean_rows.empty:
            raise ValueError(
                f"Для метода {method} "
                "отсутствует clean-результат."
            )

        clean_row = clean_rows.iloc[0]

        rows.append(
            {
                "method": (
                    method
                ),

                "method_label": (
                    METHOD_LABELS.get(
                        method,
                        method,
                    )
                ),

                "clean_precision": (
                    safe_float(
                        clean_row["precision"]
                    )
                ),

                "clean_recall": (
                    safe_float(
                        clean_row["recall"]
                    )
                ),

                "clean_mAP50": (
                    safe_float(
                        clean_row["mAP50"]
                    )
                ),

                "clean_mAP50-95": (
                    safe_float(
                        clean_row["mAP50-95"]
                    )
                ),

                "clean_mAP50_relative_drop": (
                    safe_float(
                        clean_row[
                            "clean_mAP50_relative_drop"
                        ]
                    )
                ),

                "clean_mAP50-95_relative_drop": (
                    safe_float(
                        clean_row.get(
                            "clean_mAP50-95_relative_drop"
                        )
                    )
                ),

                "mean_FGSM_mAP50": (
                    safe_mean(
                        fgsm_rows["mAP50"]
                    )
                ),

                "mean_FGSM_mAP50-95": (
                    safe_mean(
                        fgsm_rows["mAP50-95"]
                    )
                ),

                "mean_FGSM_recovery_mAP50": (
                    safe_mean(
                        fgsm_rows[
                            "recovery_mAP50"
                        ]
                    )
                ),

                "mean_FGSM_recovery_mAP50-95": (
                    safe_mean(
                        fgsm_rows[
                            "recovery_mAP50-95"
                        ]
                    )
                ),

                "mean_PGD_mAP50": (
                    safe_mean(
                        pgd_rows["mAP50"]
                    )
                ),

                "mean_PGD_mAP50-95": (
                    safe_mean(
                        pgd_rows["mAP50-95"]
                    )
                ),

                "mean_PGD_recovery_mAP50": (
                    safe_mean(
                        pgd_rows[
                            "recovery_mAP50"
                        ]
                    )
                ),

                "mean_PGD_recovery_mAP50-95": (
                    safe_mean(
                        pgd_rows[
                            "recovery_mAP50-95"
                        ]
                    )
                ),

                "median_defense_ms_per_image": (
                    safe_median(
                        method_rows[
                            "defense_ms_per_image"
                        ]
                    )
                ),

                "median_validator_total_ms": (
                    safe_median(
                        method_rows[
                            "validator_total_ms"
                        ]
                    )
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result["method_order"] = (
        result["method"].map(
            {
                method: index

                for index, method
                in enumerate(
                    METHOD_ORDER
                )
            }
        )
    )

    return (
        result
        .sort_values(
            "method_order"
        )
        .drop(
            columns=[
                "method_order"
            ]
        )
    )


def build_attack_table(
    final_summary: pd.DataFrame,
    attack: str,
) -> pd.DataFrame:
    table = final_summary[
        final_summary["attack"]
        == attack
    ].copy()

    table["method_order"] = (
        table["method"].map(
            {
                method: index

                for index, method
                in enumerate(
                    METHOD_ORDER
                )
            }
        )
    )

    table = table.sort_values(
        [
            "epsilon_pixels",
            "method_order",
        ]
    )

    columns = [
        "method",
        "method_label",
        "epsilon_pixels",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
        "clean_mAP50_relative_drop",
        "recovery_mAP50",
        "recovery_mAP50-95",
        "defense_ms_per_image",
        "defended_vs_clean_psnr",
        "defended_vs_clean_ssim",
        "validator_total_ms",
    ]

    return (
        table[columns]
        .reset_index(
            drop=True
        )
    )


def build_clean_table(
    final_summary: pd.DataFrame,
) -> pd.DataFrame:
    table = final_summary[
        final_summary["attack"]
        == "clean"
    ].copy()

    table["method_order"] = (
        table["method"].map(
            {
                method: index

                for index, method
                in enumerate(
                    METHOD_ORDER
                )
            }
        )
    )

    table = table.sort_values(
        "method_order"
    )

    columns = [
        "method",
        "method_label",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
        "clean_mAP50_relative_drop",
        "clean_mAP50-95_relative_drop",
        "defense_ms_per_image",
        "defended_vs_clean_psnr",
        "defended_vs_clean_ssim",
        "validator_total_ms",
    ]

    return (
        table[columns]
        .reset_index(
            drop=True
        )
    )


def build_adaptive_table(
    adaptive_summary: pd.DataFrame,
) -> pd.DataFrame:
    table = adaptive_summary.copy()

    table["attack_order"] = (
        table["attack"].map(
            {
                attack: index

                for index, attack
                in enumerate(
                    ATTACK_ORDER
                )
            }
        )
    )

    table = table.sort_values(
        [
            "attack_order",
            "epsilon_pixels",
        ]
    )

    columns = [
        "attack",
        "epsilon_pixels",
        "clean_no_defense_mAP50",
        "clean_tnorm_mAP50",
        "no_defense_attacked_mAP50",
        "oblivious_tnorm_mAP50",
        "adaptive_tnorm_mAP50",
        "adaptive_minus_oblivious_mAP50",
        "oblivious_recovery_mAP50",
        "adaptive_recovery_mAP50",
        "clean_no_defense_mAP50-95",
        "clean_tnorm_mAP50-95",
        "no_defense_attacked_mAP50-95",
        "oblivious_tnorm_mAP50-95",
        "adaptive_tnorm_mAP50-95",
        "adaptive_minus_oblivious_mAP50-95",
        "oblivious_recovery_mAP50-95",
        "adaptive_recovery_mAP50-95",
        "adaptive_precision",
        "adaptive_recall",
        "adaptive_attack_psnr",
        "adaptive_attack_ssim",
        "adaptive_attack_linf_pixels",
        "defense_ms_per_image",
        "validator_total_ms",
        "proxy_target",
    ]

    available_columns = [
        column

        for column in columns

        if column in table.columns
    ]

    return (
        table[
            available_columns
        ]
        .reset_index(
            drop=True
        )
    )


def build_class_table(
    final_classes: pd.DataFrame,
) -> pd.DataFrame:
    table = final_classes.copy()

    table["class_id"] = (
        table["class_id"]
        .astype(int)
    )

    table["class_name"] = (
        table.apply(
            lambda row: (
                CLASS_NAMES.get(
                    int(
                        row["class_id"]
                    ),
                    str(
                        row["class_name"]
                    ),
                )
            ),
            axis=1,
        )
    )

    table["method_order"] = (
        table["method"].map(
            {
                method: index

                for index, method
                in enumerate(
                    METHOD_ORDER
                )
            }
        )
    )

    table = table.sort_values(
        [
            "attack",
            "epsilon_pixels",
            "method_order",
            "class_id",
        ]
    )

    columns = [
        "method",
        "method_label",
        "attack",
        "epsilon_pixels",
        "class_id",
        "class_name",
        "images",
        "instances",
        "precision",
        "recall",
        "mAP50",
        "mAP50-95",
    ]

    available_columns = [
        column

        for column in columns

        if column in table.columns
    ]

    return (
        table[
            available_columns
        ]
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PARETO-АНАЛИЗ
# ============================================================

def mark_pareto_front(
    overview: pd.DataFrame,
) -> pd.DataFrame:
    result = overview.copy()

    result[
        "pareto_clean_vs_fgsm"
    ] = False

    candidates = result[
        result["method"]
        != "none"
    ].copy()

    candidates = (
        candidates.dropna(
            subset=[
                "clean_mAP50_relative_drop",
                "mean_FGSM_recovery_mAP50",
            ]
        )
    )

    for index, row in candidates.iterrows():
        dominated = False

        for (
            other_index,
            other,
        ) in candidates.iterrows():
            if index == other_index:
                continue

            no_worse_clean = (
                other[
                    "clean_mAP50_relative_drop"
                ]
                <= row[
                    "clean_mAP50_relative_drop"
                ]
            )

            no_worse_recovery = (
                other[
                    "mean_FGSM_recovery_mAP50"
                ]
                >= row[
                    "mean_FGSM_recovery_mAP50"
                ]
            )

            strictly_better = (
                other[
                    "clean_mAP50_relative_drop"
                ]
                < row[
                    "clean_mAP50_relative_drop"
                ]

                or

                other[
                    "mean_FGSM_recovery_mAP50"
                ]
                > row[
                    "mean_FGSM_recovery_mAP50"
                ]
            )

            if (
                no_worse_clean
                and no_worse_recovery
                and strictly_better
            ):
                dominated = True
                break

        if not dominated:
            result.loc[
                index,
                "pareto_clean_vs_fgsm",
            ] = True

    return result


# ============================================================
# ПОКЛАССОВЫЙ BOOTSTRAP
# ============================================================

def bootstrap_mean_difference(
    differences: np.ndarray,
    iterations: int,
    random_generator: np.random.Generator,
) -> dict[str, float | int | None]:
    values = np.asarray(
        differences,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return {
            "n_units": 0,
            "mean_difference": None,
            "median_difference": None,
            "ci95_low": None,
            "ci95_high": None,
            "probability_positive": None,
        }

    sample_indices = (
        random_generator.integers(
            low=0,
            high=values.size,
            size=(
                iterations,
                values.size,
            ),
        )
    )

    bootstrap_means = (
        values[
            sample_indices
        ].mean(
            axis=1
        )
    )

    return {
        "n_units": (
            int(
                values.size
            )
        ),

        "mean_difference": (
            float(
                values.mean()
            )
        ),

        "median_difference": (
            float(
                np.median(
                    values
                )
            )
        ),

        "ci95_low": (
            float(
                np.quantile(
                    bootstrap_means,
                    0.025,
                )
            )
        ),

        "ci95_high": (
            float(
                np.quantile(
                    bootstrap_means,
                    0.975,
                )
            )
        ),

        "probability_positive": (
            float(
                np.mean(
                    bootstrap_means
                    > 0.0
                )
            )
        ),
    }


def build_final_class_bootstrap(
    final_classes: pd.DataFrame,
) -> pd.DataFrame:
    random_generator = (
        np.random.default_rng(
            RANDOM_SEED
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for attack in ATTACK_ORDER:
        attack_rows = final_classes[
            final_classes["attack"]
            == attack
        ]

        for epsilon in EPSILON_ORDER:
            epsilon_rows = attack_rows[
                attack_rows[
                    "epsilon_pixels"
                ]
                == epsilon
            ]

            reference = epsilon_rows[
                epsilon_rows["method"]
                == "none"
            ][
                [
                    "class_id",
                    "mAP50",
                    "mAP50-95",
                ]
            ].copy()

            reference = reference.rename(
                columns={
                    "mAP50": (
                        "reference_mAP50"
                    ),

                    "mAP50-95": (
                        "reference_mAP50-95"
                    ),
                }
            )

            for method in METHOD_ORDER:
                if method == "none":
                    continue

                method_rows = epsilon_rows[
                    epsilon_rows["method"]
                    == method
                ][
                    [
                        "class_id",
                        "mAP50",
                        "mAP50-95",
                    ]
                ].copy()

                paired = reference.merge(
                    method_rows,
                    on="class_id",
                    how="inner",
                )

                for metric in (
                    "mAP50",
                    "mAP50-95",
                ):
                    differences = (
                        paired[metric]
                        - paired[
                            f"reference_{metric}"
                        ]
                    ).to_numpy(
                        dtype=np.float64
                    )

                    statistics = (
                        bootstrap_mean_difference(
                            differences=(
                                differences
                            ),

                            iterations=(
                                BOOTSTRAP_ITERATIONS
                            ),

                            random_generator=(
                                random_generator
                            ),
                        )
                    )

                    rows.append(
                        {
                            "bootstrap_unit": (
                                "class"
                            ),

                            "attack": (
                                attack
                            ),

                            "epsilon_pixels": (
                                epsilon
                            ),

                            "comparison": (
                                f"{method}"
                                "_minus_none"
                            ),

                            "method": (
                                method
                            ),

                            "reference_method": (
                                "none"
                            ),

                            "metric": (
                                metric
                            ),

                            **statistics,
                        }
                    )

    return pd.DataFrame(
        rows
    )


def build_adaptive_class_bootstrap(
    adaptive_classes: pd.DataFrame,
) -> pd.DataFrame:
    random_generator = (
        np.random.default_rng(
            RANDOM_SEED + 1
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    for attack in ATTACK_ORDER:
        attack_rows = adaptive_classes[
            adaptive_classes["attack"]
            == attack
        ]

        for epsilon in EPSILON_ORDER:
            condition = attack_rows[
                attack_rows[
                    "epsilon_pixels"
                ]
                == epsilon
            ].copy()

            for metric in (
                "mAP50",
                "mAP50-95",
            ):
                adaptive_column = (
                    f"adaptive_{metric}"
                )

                oblivious_column = (
                    f"oblivious_{metric}"
                )

                differences = (
                    condition[
                        adaptive_column
                    ]
                    - condition[
                        oblivious_column
                    ]
                ).to_numpy(
                    dtype=np.float64
                )

                statistics = (
                    bootstrap_mean_difference(
                        differences=(
                            differences
                        ),

                        iterations=(
                            BOOTSTRAP_ITERATIONS
                        ),

                        random_generator=(
                            random_generator
                        ),
                    )
                )

                rows.append(
                    {
                        "bootstrap_unit": (
                            "class"
                        ),

                        "attack": (
                            attack
                        ),

                        "epsilon_pixels": (
                            epsilon
                        ),

                        "comparison": (
                            "adaptive_minus_"
                            "oblivious_tnorm"
                        ),

                        "method": (
                            "adaptive_tnorm"
                        ),

                        "reference_method": (
                            "oblivious_tnorm"
                        ),

                        "metric": (
                            metric
                        ),

                        **statistics,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# НАСТРОЙКА ГРАФИКОВ
# ============================================================

def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": (
                "DejaVu Sans"
            ),

            "font.size": 10,

            "axes.titlesize": 12,

            "axes.labelsize": 11,

            "legend.fontsize": 9,

            "xtick.labelsize": 9,

            "ytick.labelsize": 9,

            "figure.dpi": (
                FIGURE_DPI
            ),
        }
    )


# ============================================================
# ГРАФИКИ CLEAN
# ============================================================

def plot_clean_map50(
    clean_table: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            8.2,
            4.8,
        )
    )

    table = clean_table.sort_values(
        "mAP50",
        ascending=False,
    )

    axis.bar(
        table["method_label"],
        table["mAP50"],
    )

    axis.set_title(
        "Clean test performance"
    )

    axis.set_ylabel(
        "mAP50"
    )

    axis.set_xlabel(
        "Defense"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.tick_params(
        axis="x",
        rotation=25,
    )

    for index, value in enumerate(
        table["mAP50"]
    ):
        axis.text(
            index,
            float(value),
            f"{float(value):.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(
        figure,
        "clean_map50",
    )


def plot_clean_drop(
    clean_table: pd.DataFrame,
) -> None:
    table = clean_table[
        clean_table["method"]
        != "none"
    ].copy()

    table["drop_percent"] = (
        table[
            "clean_mAP50_relative_drop"
        ]
        * 100.0
    )

    table = table.sort_values(
        "drop_percent"
    )

    figure, axis = plt.subplots(
        figsize=(
            8.2,
            4.8,
        )
    )

    axis.bar(
        table["method_label"],
        table["drop_percent"],
    )

    axis.axhline(
        3.0,
        linestyle="--",
        linewidth=1.0,
        label=(
            "3% primary threshold"
        ),
    )

    axis.axhline(
        5.0,
        linestyle=":",
        linewidth=1.0,
        label=(
            "5% relaxed threshold"
        ),
    )

    axis.set_title(
        "Clean mAP50 degradation"
    )

    axis.set_ylabel(
        "Relative drop, %"
    )

    axis.set_xlabel(
        "Defense"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.tick_params(
        axis="x",
        rotation=25,
    )

    axis.legend()

    save_figure(
        figure,
        "clean_map50_drop",
    )


def plot_latency(
    clean_table: pd.DataFrame,
) -> None:
    table = clean_table[
        clean_table["method"]
        != "none"
    ].copy()

    table = table.sort_values(
        "defense_ms_per_image"
    )

    figure, axis = plt.subplots(
        figsize=(
            8.2,
            4.8,
        )
    )

    axis.bar(
        table["method_label"],
        table[
            "defense_ms_per_image"
        ],
    )

    axis.set_title(
        "Defense latency on RTX 4090"
    )

    axis.set_ylabel(
        "Milliseconds per image"
    )

    axis.set_xlabel(
        "Defense"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.tick_params(
        axis="x",
        rotation=25,
    )

    save_figure(
        figure,
        "defense_latency",
    )


# ============================================================
# ГРАФИКИ АТАК
# ============================================================

def plot_attack_metric_curves(
    attack_table: pd.DataFrame,
    attack: str,
    metric: str,
    filename_stem: str,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            8.4,
            5.2,
        )
    )

    for method in METHOD_ORDER:
        method_rows = attack_table[
            attack_table["method"]
            == method
        ].sort_values(
            "epsilon_pixels"
        )

        if method_rows.empty:
            continue

        axis.plot(
            method_rows[
                "epsilon_pixels"
            ],

            method_rows[
                metric
            ],

            marker="o",

            linewidth=1.8,

            label=(
                METHOD_LABELS.get(
                    method,
                    method,
                )
            ),
        )

    axis.set_title(
        f"{attack.upper()} "
        "test performance"
    )

    axis.set_xlabel(
        "Perturbation budget, "
        "epsilon/255"
    )

    axis.set_ylabel(
        metric
    )

    axis.set_xticks(
        EPSILON_ORDER
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    save_figure(
        figure,
        filename_stem,
    )


def plot_recovery_curves(
    attack_table: pd.DataFrame,
    attack: str,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            8.4,
            5.2,
        )
    )

    for method in METHOD_ORDER:
        if method == "none":
            continue

        method_rows = attack_table[
            attack_table["method"]
            == method
        ].sort_values(
            "epsilon_pixels"
        )

        if method_rows.empty:
            continue

        axis.plot(
            method_rows[
                "epsilon_pixels"
            ],

            method_rows[
                "recovery_mAP50"
            ]
            * 100.0,

            marker="o",

            linewidth=1.8,

            label=(
                METHOD_LABELS.get(
                    method,
                    method,
                )
            ),
        )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_title(
        f"{attack.upper()} "
        "mAP50 recovery"
    )

    axis.set_xlabel(
        "Perturbation budget, "
        "epsilon/255"
    )

    axis.set_ylabel(
        "Recovery, %"
    )

    axis.set_xticks(
        EPSILON_ORDER
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    save_figure(
        figure,
        f"{attack}_map50_recovery",
    )


def plot_clean_recovery_tradeoff(
    overview: pd.DataFrame,
) -> None:
    table = overview[
        overview["method"]
        != "none"
    ].copy()

    figure, axis = plt.subplots(
        figsize=(
            8.4,
            5.4,
        )
    )

    axis.scatter(
        table[
            "clean_mAP50_relative_drop"
        ]
        * 100.0,

        table[
            "mean_FGSM_recovery_mAP50"
        ]
        * 100.0,

        s=70,
    )

    for _, row in table.iterrows():
        axis.annotate(
            row["method_label"],

            (
                row[
                    "clean_mAP50_relative_drop"
                ]
                * 100.0,

                row[
                    "mean_FGSM_recovery_mAP50"
                ]
                * 100.0,
            ),

            xytext=(
                5,
                5,
            ),

            textcoords=(
                "offset points"
            ),

            fontsize=9,
        )

    axis.axvline(
        3.0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.axvline(
        5.0,
        linestyle=":",
        linewidth=1.0,
    )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_title(
        "Clean-quality and "
        "FGSM-recovery trade-off"
    )

    axis.set_xlabel(
        "Clean mAP50 "
        "relative drop, %"
    )

    axis.set_ylabel(
        "Mean FGSM "
        "mAP50 recovery, %"
    )

    axis.grid(
        alpha=0.25
    )

    save_figure(
        figure,
        "clean_vs_fgsm_tradeoff",
    )


# ============================================================
# ГРАФИКИ ADAPTIVE
# ============================================================

def plot_adaptive_comparison(
    adaptive_table: pd.DataFrame,
    attack: str,
    metric: str,
) -> None:
    table = adaptive_table[
        adaptive_table["attack"]
        == attack
    ].sort_values(
        "epsilon_pixels"
    )

    if table.empty:
        return

    column_map = {
        "mAP50": (
            "no_defense_attacked_mAP50",
            "oblivious_tnorm_mAP50",
            "adaptive_tnorm_mAP50",
        ),

        "mAP50-95": (
            "no_defense_attacked_mAP50-95",
            "oblivious_tnorm_mAP50-95",
            "adaptive_tnorm_mAP50-95",
        ),
    }

    columns = column_map[
        metric
    ]

    labels = (
        "No defense",

        "Oblivious attack "
        "+ T-norm",

        "Adaptive attack "
        "+ T-norm",
    )

    figure, axis = plt.subplots(
        figsize=(
            8.4,
            5.2,
        )
    )

    for column, label in zip(
        columns,
        labels,
        strict=True,
    ):
        axis.plot(
            table[
                "epsilon_pixels"
            ],

            table[
                column
            ],

            marker="o",

            linewidth=1.8,

            label=label,
        )

    axis.set_title(
        "Adaptive versus oblivious "
        f"{attack.upper()}"
    )

    axis.set_xlabel(
        "Perturbation budget, "
        "epsilon/255"
    )

    axis.set_ylabel(
        metric
    )

    axis.set_xticks(
        EPSILON_ORDER
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    save_figure(
        figure,
        (
            f"adaptive_{attack}_"
            f"{metric.replace('-', '_')}"
        ),
    )


def plot_adaptive_delta(
    adaptive_table: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(
            8.4,
            5.2,
        )
    )

    for attack in ATTACK_ORDER:
        table = adaptive_table[
            adaptive_table["attack"]
            == attack
        ].sort_values(
            "epsilon_pixels"
        )

        if table.empty:
            continue

        axis.plot(
            table[
                "epsilon_pixels"
            ],

            table[
                "adaptive_minus_oblivious_mAP50"
            ],

            marker="o",

            linewidth=1.8,

            label=(
                attack.upper()
            ),
        )

    axis.axhline(
        0.0,
        linewidth=1.0,
    )

    axis.set_title(
        "Adaptive attack impact "
        "on defended mAP50"
    )

    axis.set_xlabel(
        "Perturbation budget, "
        "epsilon/255"
    )

    axis.set_ylabel(
        "Adaptive minus "
        "oblivious mAP50"
    )

    axis.set_xticks(
        EPSILON_ORDER
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    save_figure(
        figure,
        "adaptive_minus_oblivious_map50",
    )


# ============================================================
# ПОКЛАССОВЫЕ ГРАФИКИ
# ============================================================

def plot_class_heatmap(
    final_classes: pd.DataFrame,
    attack: str,
    epsilon: int,
    filename_stem: str,
) -> None:
    table = final_classes[
        (
            final_classes["attack"]
            == attack
        )
        &
        (
            final_classes[
                "epsilon_pixels"
            ]
            == epsilon
        )
    ].copy()

    if table.empty:
        return

    pivot = table.pivot_table(
        index="method",
        columns="class_id",
        values="mAP50",
        aggfunc="first",
    )

    pivot = pivot.reindex(
        index=[
            method

            for method in METHOD_ORDER

            if method
            in pivot.index
        ],

        columns=ALL_CLASSES,
    )

    values = pivot.to_numpy(
        dtype=np.float64
    )

    figure, axis = plt.subplots(
        figsize=(
            9.4,
            5.2,
        )
    )

    image = axis.imshow(
        values,
        aspect="auto",
    )

    axis.set_title(
        "Class-wise mAP50: "
        f"{attack.upper()} "
        f"{epsilon_label(epsilon)}"
    )

    axis.set_xlabel(
        "Class"
    )

    axis.set_ylabel(
        "Defense"
    )

    axis.set_xticks(
        np.arange(
            len(
                ALL_CLASSES
            )
        )
    )

    axis.set_xticklabels(
        [
            CLASS_NAMES[
                class_id
            ]

            for class_id
            in ALL_CLASSES
        ],

        rotation=30,

        ha="right",
    )

    axis.set_yticks(
        np.arange(
            len(
                pivot.index
            )
        )
    )

    axis.set_yticklabels(
        [
            METHOD_LABELS.get(
                method,
                method,
            )

            for method
            in pivot.index
        ]
    )

    for row_index in range(
        values.shape[0]
    ):
        for column_index in range(
            values.shape[1]
        ):
            value = values[
                row_index,
                column_index,
            ]

            if math.isfinite(
                value
            ):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    figure.colorbar(
        image,
        ax=axis,
        label="mAP50",
    )

    save_figure(
        figure,
        filename_stem,
    )


def plot_class_main_comparison(
    final_classes: pd.DataFrame,
) -> None:
    table = final_classes[
        (
            final_classes["attack"]
            == "fgsm"
        )
        &
        (
            final_classes[
                "epsilon_pixels"
            ]
            == 1
        )
        &
        (
            final_classes[
                "class_id"
            ].isin(
                MAIN_CLASSES
            )
        )
        &
        (
            final_classes[
                "method"
            ].isin(
                [
                    "none",
                    "tnorm",
                    "bilateral",
                ]
            )
        )
    ].copy()

    if table.empty:
        return

    pivot = table.pivot_table(
        index="class_id",
        columns="method",
        values="mAP50",
        aggfunc="first",
    ).reindex(
        index=MAIN_CLASSES
    )

    figure, axis = plt.subplots(
        figsize=(
            9.0,
            5.2,
        )
    )

    x_positions = np.arange(
        len(
            MAIN_CLASSES
        ),
        dtype=np.float64,
    )

    methods = [
        method

        for method in (
            "none",
            "tnorm",
            "bilateral",
        )

        if method
        in pivot.columns
    ]

    width = 0.24

    for (
        method_index,
        method,
    ) in enumerate(
        methods
    ):
        offsets = (
            method_index
            - (
                len(methods) - 1
            )
            / 2.0
        ) * width

        axis.bar(
            x_positions
            + offsets,

            pivot[
                method
            ].to_numpy(
                dtype=np.float64
            ),

            width=width,

            label=(
                METHOD_LABELS[
                    method
                ]
            ),
        )

    axis.set_title(
        "FGSM 1/255 "
        "class-wise mAP50"
    )

    axis.set_xlabel(
        "Class"
    )

    axis.set_ylabel(
        "mAP50"
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        [
            CLASS_NAMES[
                class_id
            ]

            for class_id
            in MAIN_CLASSES
        ]
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    save_figure(
        figure,
        "fgsm_1_main_classes",
    )


# ============================================================
# ЧИСЛА ДЛЯ СТАТЬИ
# ============================================================

def select_row(
    frame: pd.DataFrame,
    method: str,
) -> pd.Series:
    rows = frame[
        frame["method"]
        == method
    ]

    if rows.empty:
        raise KeyError(
            f"Метод не найден: {method}"
        )

    return rows.iloc[0]


def build_paper_numbers(
    overview: pd.DataFrame,
    final_summary: pd.DataFrame,
    adaptive_summary: pd.DataFrame,
) -> dict[str, Any]:
    no_defense = select_row(
        overview,
        "none",
    )

    tnorm = select_row(
        overview,
        "tnorm",
    )

    bilateral = select_row(
        overview,
        "bilateral",
    )

    jpeg = select_row(
        overview,
        "jpeg",
    )

    median = select_row(
        overview,
        "median",
    )

    gaussian = select_row(
        overview,
        "gaussian",
    )

    adaptive_fgsm = adaptive_summary[
        adaptive_summary["attack"]
        == "fgsm"
    ].copy()

    adaptive_pgd = adaptive_summary[
        adaptive_summary["attack"]
        == "pgd"
    ].copy()

    tnorm_fgsm = final_summary[
        (
            final_summary["method"]
            == "tnorm"
        )
        &
        (
            final_summary["attack"]
            == "fgsm"
        )
    ].sort_values(
        "epsilon_pixels"
    )

    bilateral_fgsm = final_summary[
        (
            final_summary["method"]
            == "bilateral"
        )
        &
        (
            final_summary["attack"]
            == "fgsm"
        )
    ].sort_values(
        "epsilon_pixels"
    )

    return {
        "clean": {
            "no_defense_mAP50": (
                safe_float(
                    no_defense[
                        "clean_mAP50"
                    ]
                )
            ),

            "no_defense_mAP50-95": (
                safe_float(
                    no_defense[
                        "clean_mAP50-95"
                    ]
                )
            ),

            "tnorm_mAP50": (
                safe_float(
                    tnorm[
                        "clean_mAP50"
                    ]
                )
            ),

            "tnorm_clean_drop": (
                safe_float(
                    tnorm[
                        "clean_mAP50_relative_drop"
                    ]
                )
            ),

            "bilateral_mAP50": (
                safe_float(
                    bilateral[
                        "clean_mAP50"
                    ]
                )
            ),

            "bilateral_clean_drop": (
                safe_float(
                    bilateral[
                        "clean_mAP50_relative_drop"
                    ]
                )
            ),

            "jpeg_clean_drop": (
                safe_float(
                    jpeg[
                        "clean_mAP50_relative_drop"
                    ]
                )
            ),

            "median_clean_drop": (
                safe_float(
                    median[
                        "clean_mAP50_relative_drop"
                    ]
                )
            ),

            "gaussian_clean_drop": (
                safe_float(
                    gaussian[
                        "clean_mAP50_relative_drop"
                    ]
                )
            ),
        },

        "latency_ms_per_image": {
            method: (
                safe_float(
                    select_row(
                        overview,
                        method,
                    )[
                        "median_defense_ms_per_image"
                    ]
                )
            )

            for method in METHOD_ORDER

            if method != "none"
        },

        "fgsm": {
            "tnorm_mean_recovery_mAP50": (
                safe_float(
                    tnorm[
                        "mean_FGSM_recovery_mAP50"
                    ]
                )
            ),

            "bilateral_mean_recovery_mAP50": (
                safe_float(
                    bilateral[
                        "mean_FGSM_recovery_mAP50"
                    ]
                )
            ),

            "tnorm_by_epsilon": [
                {
                    "epsilon_pixels": (
                        int(
                            row[
                                "epsilon_pixels"
                            ]
                        )
                    ),

                    "mAP50": (
                        safe_float(
                            row[
                                "mAP50"
                            ]
                        )
                    ),

                    "recovery_mAP50": (
                        safe_float(
                            row[
                                "recovery_mAP50"
                            ]
                        )
                    ),
                }

                for _, row
                in tnorm_fgsm.iterrows()
            ],

            "bilateral_by_epsilon": [
                {
                    "epsilon_pixels": (
                        int(
                            row[
                                "epsilon_pixels"
                            ]
                        )
                    ),

                    "mAP50": (
                        safe_float(
                            row[
                                "mAP50"
                            ]
                        )
                    ),

                    "recovery_mAP50": (
                        safe_float(
                            row[
                                "recovery_mAP50"
                            ]
                        )
                    ),
                }

                for _, row
                in bilateral_fgsm.iterrows()
            ],
        },

        "adaptive": {
            "fgsm_mean_adaptive_minus_oblivious_mAP50": (
                safe_mean(
                    adaptive_fgsm[
                        "adaptive_minus_oblivious_mAP50"
                    ]
                )
            ),

            "fgsm_mean_adaptive_recovery_mAP50": (
                safe_mean(
                    adaptive_fgsm[
                        "adaptive_recovery_mAP50"
                    ]
                )
            ),

            "pgd_mean_adaptive_minus_oblivious_mAP50": (
                safe_mean(
                    adaptive_pgd[
                        "adaptive_minus_oblivious_mAP50"
                    ]
                )
            ),

            "pgd_mean_adaptive_recovery_mAP50": (
                safe_mean(
                    adaptive_pgd[
                        "adaptive_recovery_mAP50"
                    ]
                )
            ),
        },
    }


# ============================================================
# АВТОМАТИЧЕСКАЯ СВОДКА
# ============================================================

def build_markdown_summary(
    overview: pd.DataFrame,
    adaptive_summary: pd.DataFrame,
    final_bootstrap: pd.DataFrame,
    adaptive_bootstrap: pd.DataFrame,
) -> str:
    no_defense = select_row(
        overview,
        "none",
    )

    tnorm = select_row(
        overview,
        "tnorm",
    )

    bilateral = select_row(
        overview,
        "bilateral",
    )

    jpeg = select_row(
        overview,
        "jpeg",
    )

    median = select_row(
        overview,
        "median",
    )

    gaussian = select_row(
        overview,
        "gaussian",
    )

    adaptive_fgsm = adaptive_summary[
        adaptive_summary["attack"]
        == "fgsm"
    ]

    adaptive_pgd = adaptive_summary[
        adaptive_summary["attack"]
        == "pgd"
    ]

    fgsm_adaptive_delta = safe_mean(
        adaptive_fgsm[
            "adaptive_minus_oblivious_mAP50"
        ]
    )

    fgsm_adaptive_recovery = safe_mean(
        adaptive_fgsm[
            "adaptive_recovery_mAP50"
        ]
    )

    pgd_adaptive_delta = safe_mean(
        adaptive_pgd[
            "adaptive_minus_oblivious_mAP50"
        ]
    )

    tnorm_bootstrap = final_bootstrap[
        (
            final_bootstrap["attack"]
            == "fgsm"
        )
        &
        (
            final_bootstrap[
                "epsilon_pixels"
            ]
            == 1
        )
        &
        (
            final_bootstrap["method"]
            == "tnorm"
        )
        &
        (
            final_bootstrap["metric"]
            == "mAP50"
        )
    ]

    adaptive_bootstrap_row = adaptive_bootstrap[
        (
            adaptive_bootstrap["attack"]
            == "fgsm"
        )
        &
        (
            adaptive_bootstrap[
                "epsilon_pixels"
            ]
            == 1
        )
        &
        (
            adaptive_bootstrap["metric"]
            == "mAP50"
        )
    ]

    if tnorm_bootstrap.empty:
        tnorm_ci = "n/a"

    else:
        row = tnorm_bootstrap.iloc[0]

        tnorm_ci = (
            f"[{float(row['ci95_low']):.4f}, "
            f"{float(row['ci95_high']):.4f}]"
        )

    if adaptive_bootstrap_row.empty:
        adaptive_ci = "n/a"

    else:
        row = (
            adaptive_bootstrap_row.iloc[0]
        )

        adaptive_ci = (
            f"[{float(row['ci95_low']):.4f}, "
            f"{float(row['ci95_high']):.4f}]"
        )

    lines = [
        "# Итоговый анализ результатов",
        "",
        "## Основные числа",
        "",
        (
            "Чистая модель получила mAP50 "
            f"{format_float(safe_float(no_defense['clean_mAP50']))} "
            "и mAP50-95 "
            f"{format_float(safe_float(no_defense['clean_mAP50-95']))}."
        ),
        (
            "Product T-норма снизила чистый mAP50 до "
            f"{format_float(safe_float(tnorm['clean_mAP50']))}, "
            "что соответствует относительному падению "
            f"{format_percent(safe_float(tnorm['clean_mAP50_relative_drop']))}."
        ),
        (
            "Bilateral сохранил чистый mAP50 на уровне "
            f"{format_float(safe_float(bilateral['clean_mAP50']))} "
            "при падении "
            f"{format_percent(safe_float(bilateral['clean_mAP50_relative_drop']))}."
        ),
        "",
        "## Неадаптивные атаки",
        "",
        (
            "Среднее восстановление mAP50 Product T-нормой "
            "по FGSM-бюджетам составило "
            f"{format_percent(safe_float(tnorm['mean_FGSM_recovery_mAP50']))}."
        ),
        (
            "Для Bilateral соответствующее среднее "
            "восстановление составило "
            f"{format_percent(safe_float(bilateral['mean_FGSM_recovery_mAP50']))}."
        ),
        (
            "JPEG, Median и Gaussian ухудшили чистый mAP50 "
            "соответственно на "
            f"{format_percent(safe_float(jpeg['clean_mAP50_relative_drop']))}, "
            f"{format_percent(safe_float(median['clean_mAP50_relative_drop']))} "
            "и "
            f"{format_percent(safe_float(gaussian['clean_mAP50_relative_drop']))}."
        ),
        "",
        "## Адаптивный white-box сценарий",
        "",
        (
            "Средняя разница adaptive minus oblivious "
            "для FGSM по mAP50 равна "
            f"{format_float(fgsm_adaptive_delta)}."
        ),
        (
            "Среднее adaptive-восстановление FGSM "
            "по mAP50 равно "
            f"{format_percent(fgsm_adaptive_recovery)}."
        ),
        (
            "Средняя разница adaptive minus oblivious "
            "для PGD по mAP50 равна "
            f"{format_float(pgd_adaptive_delta)}."
        ),
        "",
        "## Class-bootstrap",
        "",
        (
            "Bootstrap выполнен по классам, а не по изображениям. "
            "Он описывает устойчивость среднего поклассового эффекта "
            "и не является доверительным интервалом общего dataset-level mAP."
        ),
        (
            "Для T-нормы против FGSM 1/255 интервал среднего "
            "поклассового изменения mAP50 относительно модели "
            f"без защиты: {tnorm_ci}."
        ),
        (
            "Для adaptive minus oblivious FGSM 1/255 "
            f"соответствующий поклассовый интервал: {adaptive_ci}."
        ),
        "",
        "## Корректный вывод",
        "",
        (
            "Product T-нормовый фильтр сохраняет чистое качество "
            "значительно лучше JPEG, Median и Gaussian и даёт "
            "небольшое восстановление только при слабых неадаптивных "
            "FGSM-атаках. Bilateral показывает более выгодный компромисс "
            "между чистым качеством, восстановлением и задержкой. "
            "В адаптивном white-box сценарии преимущество T-нормы исчезает, "
            "поэтому фильтр нельзя интерпретировать как общую "
            "adversarial-защиту."
        ),
        "",
        "## Ограничение статистики",
        "",
        (
            "В сохранённых JSON находятся агрегированные метрики. "
            "Для корректного image-level bootstrap AP требуются "
            "сохранённые предсказания или пер-изображенческие результаты "
            "каждого условия. Скрипт намеренно не выдаёт class-bootstrap "
            "за dataset-level доверительный интервал."
        ),
    ]

    return (
        "\n".join(lines)
        + "\n"
    )


# ============================================================
# ВЫВОД В КОНСОЛЬ
# ============================================================

def print_overview(
    overview: pd.DataFrame,
) -> None:
    print(
        "\n"
        + "=" * 136
    )

    print(
        "FINAL METHOD OVERVIEW"
    )

    print(
        "=" * 136
    )

    print(
        f"{'Method':>18}"
        f"{'Clean mAP50':>15}"
        f"{'Clean drop %':>15}"
        f"{'Mean FGSM rec %':>18}"
        f"{'Mean PGD rec %':>17}"
        f"{'Defense ms':>14}"
        f"{'Pareto':>10}"
    )

    for _, row in overview.iterrows():
        clean_drop = safe_float(
            row[
                "clean_mAP50_relative_drop"
            ]
        )

        fgsm_recovery = safe_float(
            row[
                "mean_FGSM_recovery_mAP50"
            ]
        )

        pgd_recovery = safe_float(
            row[
                "mean_PGD_recovery_mAP50"
            ]
        )

        latency = safe_float(
            row[
                "median_defense_ms_per_image"
            ]
        )

        print(
            f"{str(row['method_label']):>18}"

            f"{format_float(safe_float(row['clean_mAP50'])):>15}"

            f"{format_percent(clean_drop):>15}"

            f"{format_percent(fgsm_recovery):>18}"

            f"{format_percent(pgd_recovery):>17}"

            f"{format_float(latency, 2):>14}"

            f"{str(bool(row['pareto_clean_vs_fgsm'])):>10}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    seed_everything(
        RANDOM_SEED
    )

    check_input_files()
    prepare_directories()
    configure_matplotlib()

    (
        final_data,
        adaptive_data,
        final_summary,
        final_classes,
        adaptive_summary,
        adaptive_classes,
    ) = load_result_frames()

    clean_table = build_clean_table(
        final_summary
    )

    fgsm_table = build_attack_table(
        final_summary,
        "fgsm",
    )

    pgd_table = build_attack_table(
        final_summary,
        "pgd",
    )

    adaptive_table = build_adaptive_table(
        adaptive_summary
    )

    class_table = build_class_table(
        final_classes
    )

    overview = build_method_overview(
        final_summary
    )

    overview = mark_pareto_front(
        overview
    )

    final_bootstrap = (
        build_final_class_bootstrap(
            final_classes
        )
    )

    adaptive_bootstrap = (
        build_adaptive_class_bootstrap(
            adaptive_classes
        )
    )

    # --------------------------------------------------------
    # ТАБЛИЦЫ
    # --------------------------------------------------------

    clean_table.to_csv(
        TABLES_DIR
        / "table_clean.csv",

        index=False,

        encoding="utf-8-sig",
    )

    fgsm_table.to_csv(
        TABLES_DIR
        / "table_fgsm.csv",

        index=False,

        encoding="utf-8-sig",
    )

    pgd_table.to_csv(
        TABLES_DIR
        / "table_pgd.csv",

        index=False,

        encoding="utf-8-sig",
    )

    adaptive_table.to_csv(
        TABLES_DIR
        / "table_adaptive.csv",

        index=False,

        encoding="utf-8-sig",
    )

    class_table.to_csv(
        TABLES_DIR
        / "table_classes_all.csv",

        index=False,

        encoding="utf-8-sig",
    )

    overview.to_csv(
        TABLES_DIR
        / "table_method_overview.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # СТАТИСТИКА
    # --------------------------------------------------------

    final_bootstrap.to_csv(
        STATISTICS_DIR
        / "class_bootstrap_defenses.csv",

        index=False,

        encoding="utf-8-sig",
    )

    adaptive_bootstrap.to_csv(
        STATISTICS_DIR
        / "class_bootstrap_adaptive.csv",

        index=False,

        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # CLEAN-ГРАФИКИ
    # --------------------------------------------------------

    plot_clean_map50(
        clean_table
    )

    plot_clean_drop(
        clean_table
    )

    plot_latency(
        clean_table
    )

    # --------------------------------------------------------
    # FGSM-ГРАФИКИ
    # --------------------------------------------------------

    plot_attack_metric_curves(
        fgsm_table,

        attack="fgsm",

        metric="mAP50",

        filename_stem=(
            "fgsm_map50_vs_epsilon"
        ),
    )

    plot_attack_metric_curves(
        fgsm_table,

        attack="fgsm",

        metric="mAP50-95",

        filename_stem=(
            "fgsm_map50_95_vs_epsilon"
        ),
    )

    plot_recovery_curves(
        fgsm_table,
        "fgsm",
    )

    # --------------------------------------------------------
    # PGD-ГРАФИКИ
    # --------------------------------------------------------

    plot_attack_metric_curves(
        pgd_table,

        attack="pgd",

        metric="mAP50",

        filename_stem=(
            "pgd_map50_vs_epsilon"
        ),
    )

    plot_attack_metric_curves(
        pgd_table,

        attack="pgd",

        metric="mAP50-95",

        filename_stem=(
            "pgd_map50_95_vs_epsilon"
        ),
    )

    plot_recovery_curves(
        pgd_table,
        "pgd",
    )

    # --------------------------------------------------------
    # КОМПРОМИСС CLEAN / ROBUSTNESS
    # --------------------------------------------------------

    plot_clean_recovery_tradeoff(
        overview
    )

    # --------------------------------------------------------
    # ADAPTIVE-ГРАФИКИ
    # --------------------------------------------------------

    plot_adaptive_comparison(
        adaptive_table,

        attack="fgsm",

        metric="mAP50",
    )

    plot_adaptive_comparison(
        adaptive_table,

        attack="fgsm",

        metric="mAP50-95",
    )

    plot_adaptive_comparison(
        adaptive_table,

        attack="pgd",

        metric="mAP50",
    )

    plot_adaptive_comparison(
        adaptive_table,

        attack="pgd",

        metric="mAP50-95",
    )

    plot_adaptive_delta(
        adaptive_table
    )

    # --------------------------------------------------------
    # ПОКЛАССОВЫЕ ГРАФИКИ
    # --------------------------------------------------------

    plot_class_heatmap(
        final_classes,

        attack="fgsm",

        epsilon=1,

        filename_stem=(
            "class_heatmap_fgsm_1"
        ),
    )

    plot_class_heatmap(
        final_classes,

        attack="pgd",

        epsilon=1,

        filename_stem=(
            "class_heatmap_pgd_1"
        ),
    )

    plot_class_main_comparison(
        final_classes
    )

    # --------------------------------------------------------
    # ЧИСЛА ДЛЯ СТАТЬИ
    # --------------------------------------------------------

    paper_numbers = (
        build_paper_numbers(
            overview=overview,

            final_summary=(
                final_summary
            ),

            adaptive_summary=(
                adaptive_summary
            ),
        )
    )

    write_json(
        OUTPUT_DIR
        / "paper_numbers.json",

        {
            "script_version": (
                SCRIPT_VERSION
            ),

            "random_seed": (
                RANDOM_SEED
            ),

            "bootstrap_iterations": (
                BOOTSTRAP_ITERATIONS
            ),

            "final_test_source": (
                str(
                    FINAL_TEST_PATH
                )
            ),

            "adaptive_source": (
                str(
                    ADAPTIVE_PATH
                )
            ),

            "numbers": (
                paper_numbers
            ),
        },
    )

    # --------------------------------------------------------
    # ГОТОВАЯ ТЕКСТОВАЯ СВОДКА
    # --------------------------------------------------------

    markdown_summary = (
        build_markdown_summary(
            overview=overview,

            adaptive_summary=(
                adaptive_summary
            ),

            final_bootstrap=(
                final_bootstrap
            ),

            adaptive_bootstrap=(
                adaptive_bootstrap
            ),
        )
    )

    write_text(
        OUTPUT_DIR
        / "results_summary.md",

        markdown_summary,
    )

    # --------------------------------------------------------
    # МЕТАДАННЫЕ СТАТИСТИКИ
    # --------------------------------------------------------

    write_json(
        STATISTICS_DIR
        / "statistics_metadata.json",

        {
            "script_version": (
                SCRIPT_VERSION
            ),

            "bootstrap_unit": (
                "class"
            ),

            "bootstrap_iterations": (
                BOOTSTRAP_ITERATIONS
            ),

            "random_seed": (
                RANDOM_SEED
            ),

            "important_limitation": (
                "These intervals are "
                "class-wise descriptive "
                "intervals, not image-level "
                "confidence intervals for "
                "dataset mAP."
            ),

            "all_classes": (
                CLASS_NAMES
            ),

            "main_detectable_classes": (
                MAIN_CLASSES
            ),
        },
    )

    # --------------------------------------------------------
    # МАНИФЕСТ
    # --------------------------------------------------------

    analysis_manifest = {
        "script_version": (
            SCRIPT_VERSION
        ),

        "input_files": {
            "final_test": (
                str(
                    FINAL_TEST_PATH
                )
            ),

            "adaptive_attacks": (
                str(
                    ADAPTIVE_PATH
                )
            ),
        },

        "output_directory": (
            str(
                OUTPUT_DIR
            )
        ),

        "tables": sorted(
            str(
                path.relative_to(
                    OUTPUT_DIR
                )
            )

            for path
            in TABLES_DIR.glob(
                "*.csv"
            )
        ),

        "statistics": sorted(
            str(
                path.relative_to(
                    OUTPUT_DIR
                )
            )

            for path
            in STATISTICS_DIR.glob(
                "*"
            )

            if path.is_file()
        ),

        "figures": sorted(
            str(
                path.relative_to(
                    OUTPUT_DIR
                )
            )

            for path
            in FIGURES_DIR.glob(
                "*"
            )

            if path.is_file()
        ),

        "source_metadata": {
            "final_test_split": (
                final_data.get(
                    "split"
                )
            ),

            "adaptive_split": (
                adaptive_data.get(
                    "split"
                )
            ),

            "model": (
                final_data.get(
                    "model"
                )
            ),

            "dataset": (
                final_data.get(
                    "dataset"
                )
            ),
        },
    }

    write_json(
        OUTPUT_DIR
        / "analysis_manifest.json",

        analysis_manifest,
    )

    print_overview(
        overview
    )

    print(
        "\n"
        + "=" * 136
    )

    print(
        "ANALYSIS COMPLETED"
    )

    print(
        "=" * 136
    )

    print(
        f"Таблицы:   {TABLES_DIR}"
    )

    print(
        f"Графики:   {FIGURES_DIR}"
    )

    print(
        f"Статистика: {STATISTICS_DIR}"
    )

    print(
        "Сводка:    "
        f"{OUTPUT_DIR / 'results_summary.md'}"
    )

    print(
        "Числа:     "
        f"{OUTPUT_DIR / 'paper_numbers.json'}"
    )

    print(
        "Манифест:  "
        f"{OUTPUT_DIR / 'analysis_manifest.json'}"
    )


if __name__ == "__main__":
    main()