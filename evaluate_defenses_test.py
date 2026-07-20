from __future__ import annotations

import csv
import gc
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Type

import numpy as np
import torch
from ultralytics import YOLO

import evaluate_pgd as pgd_module
from evaluate_fgsm import FGSMValidator, calculate_ssim_per_image
from evaluate_pgd import PGDValidator
from compare_defenses_val import (
    ComparisonDefenseValidator,
    DefenseConfig,
    apply_defense,
    calculate_psnr_sum,
    extract_class_metrics,
    extract_overall_metrics,
    value_to_float,
)


# ============================================================
# ПУТИ
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_YAML = (
    PROJECT_DIR
    / "data"
    / "yolo_osdar23"
    / "data.yaml"
)

MODEL_PATH = (
    PROJECT_DIR
    / "outputs"
    / "training"
    / "yolo11m_baseline_stage2"
    / "weights"
    / "best.pt"
)

SELECTION_PATH = (
    PROJECT_DIR
    / "outputs"
    / "defenses"
    / "defense_comparison_val"
    / "best_baseline_defenses.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "defenses"
    / "final_test"
)

REFERENCE_DIR = OUTPUT_DIR / "references"
RESULTS_DIR = OUTPUT_DIR / "results"
RUNS_DIR = OUTPUT_DIR / "runs"


# ============================================================
# ПАРАМЕТРЫ
# ============================================================

# Версию оставляем равной 2.
# Уже завершённые Clean и FGSM результаты будут взяты из кэша.
SCRIPT_VERSION = 2

IMAGE_SIZE = 1280
BATCH_SIZE = 2
DEVICE_INDEX = 0
WORKERS = 4

SEED = 2026

ATTACK_EPSILONS = [
    1,
    2,
    4,
    8,
]

PGD_STEPS = 20
PGD_ALPHA_RATIO = 0.25
PGD_RANDOM_START = True

METHOD_ORDER = [
    "tnorm",
    "bilateral",
    "jpeg",
    "median",
    "gaussian",
]


# ============================================================
# ОБЩИЕ ФУНКЦИИ
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_json(path: Path) -> dict[str, Any]:
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


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            f"Нет строк для сохранения в {path}"
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


def condition_key(
    attack: str,
    epsilon: int,
) -> str:
    if attack == "clean":
        return "clean"

    return f"{attack}_{epsilon}"


def condition_filename(
    attack: str,
    epsilon: int,
) -> str:
    if attack == "clean":
        return "clean.json"

    return (
        f"{attack}"
        f"_eps_{epsilon}"
        f"_255.json"
    )


def all_conditions() -> list[tuple[str, int]]:
    conditions = [
        ("clean", 0),
    ]

    for attack in (
        "fgsm",
        "pgd",
    ):
        for epsilon in ATTACK_EPSILONS:
            conditions.append(
                (
                    attack,
                    epsilon,
                )
            )

    return conditions


def run_seed(
    attack: str,
    epsilon: int,
) -> int:
    offsets = {
        "clean": 0,
        "fgsm": 10_000,
        "pgd": 20_000,
    }

    return (
        SEED
        + offsets[attack]
        + epsilon * 100
    )


def speed_dict(
    metrics: Any,
) -> dict[str, float]:
    result = {
        key: value_to_float(value)
        for key, value in metrics.speed.items()
    }

    result["total"] = float(
        sum(result.values())
    )

    return result


def recovery_rate(
    clean: float,
    attacked: float,
    defended: float,
) -> float | None:
    denominator = (
        clean
        - attacked
    )

    if denominator <= 1e-12:
        return None

    return (
        defended
        - attacked
    ) / denominator


def positive_relative_drop(
    reference: float,
    evaluated: float,
) -> float:
    if reference <= 0:
        return 0.0

    return max(
        0.0,
        (
            reference
            - evaluated
        )
        / reference,
    )


def attack_parameters(
    attack: str,
) -> dict[str, Any]:
    if attack != "pgd":
        return {
            "pgd_steps": None,
            "pgd_alpha_ratio": None,
            "pgd_random_start": None,
        }

    return {
        "pgd_steps": PGD_STEPS,
        "pgd_alpha_ratio": PGD_ALPHA_RATIO,
        "pgd_random_start": PGD_RANDOM_START,
    }


# ============================================================
# ПРОВЕРКА ОКРУЖЕНИЯ
# ============================================================

def prepare_directories() -> None:
    for path in (
        OUTPUT_DIR,
        REFERENCE_DIR,
        RESULTS_DIR,
        RUNS_DIR,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def check_environment() -> None:
    required_paths = [
        DATA_YAML,
        MODEL_PATH,
        SELECTION_PATH,
        PROJECT_DIR / "evaluate_fgsm.py",
        PROJECT_DIR / "evaluate_pgd.py",
        PROJECT_DIR / "compare_defenses_val.py",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                "Не найден необходимый файл:\n"
                f"{path}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна."
        )

    if DEVICE_INDEX >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU с индексом {DEVICE_INDEX} недоступна. "
            f"Количество GPU: {torch.cuda.device_count()}."
        )

    print("=" * 116)
    print("ФИНАЛЬНАЯ ОЦЕНКА ЗАЩИТ НА TEST")
    print("=" * 116)

    print(f"Модель: {MODEL_PATH}")
    print(f"Датасет: {DATA_YAML}")
    print(f"Конфигурации: {SELECTION_PATH}")

    print(
        "GPU: "
        f"{torch.cuda.get_device_name(DEVICE_INDEX)}"
    )

    print(
        f"Split: test, "
        f"imgsz={IMAGE_SIZE}, "
        f"batch={BATCH_SIZE}"
    )

    print(
        f"FGSM epsilon: "
        f"{ATTACK_EPSILONS}/255"
    )

    print(
        f"PGD: steps={PGD_STEPS}, "
        f"alpha ratio={PGD_ALPHA_RATIO}, "
        f"random start={PGD_RANDOM_START}"
    )


# ============================================================
# ЗАГРУЗКА КОНФИГУРАЦИЙ
# ============================================================

def load_selected_defenses() -> list[DefenseConfig]:
    data = read_json(
        SELECTION_PATH
    )

    selected = data.get(
        "selected_by_method"
    )

    if not isinstance(
        selected,
        dict,
    ):
        selected = {
            str(item["method"]): item
            for item in data.get(
                "selected_results",
                [],
            )
            if item.get("method")
        }

    missing = [
        method
        for method in METHOD_ORDER
        if method not in selected
    ]

    if missing:
        raise KeyError(
            "В best_baseline_defenses.json "
            "не найдены методы: "
            + ", ".join(missing)
        )

    configs: list[DefenseConfig] = []

    for method in METHOD_ORDER:
        method_result = selected[method]

        if not isinstance(
            method_result,
            dict,
        ):
            raise TypeError(
                f"Некорректный результат метода {method}."
            )

        config_data = method_result.get(
            "config"
        )

        if not isinstance(
            config_data,
            dict,
        ):
            raise KeyError(
                f"Для метода {method} отсутствует config."
            )

        config = DefenseConfig(
            **config_data
        )

        if config.method != method:
            raise ValueError(
                f"Ожидался метод {method}, "
                f"получен {config.method}."
            )

        configs.append(config)

    return configs


# ============================================================
# НАСТРОЙКА PGD
# ============================================================

def configure_pgd(
    validator_class: Type[Any],
    epsilon: int,
) -> None:
    validator_class.epsilon_pixels = epsilon

    # Устанавливаем распространённые варианты имён,
    # чтобы параметры точно совпали с evaluate_pgd.py.

    for name in (
        "pgd_steps",
        "steps",
        "num_steps",
        "iterations",
    ):
        setattr(
            validator_class,
            name,
            PGD_STEPS,
        )

    for name in (
        "alpha_ratio",
        "pgd_alpha_ratio",
        "step_size_ratio",
    ):
        setattr(
            validator_class,
            name,
            PGD_ALPHA_RATIO,
        )

    for name in (
        "random_start",
        "pgd_random_start",
        "use_random_start",
    ):
        setattr(
            validator_class,
            name,
            PGD_RANDOM_START,
        )

    # Модульные константы.
    pgd_module.PGD_STEPS = PGD_STEPS
    pgd_module.STEPS = PGD_STEPS
    pgd_module.NUM_STEPS = PGD_STEPS

    pgd_module.PGD_ALPHA_RATIO = PGD_ALPHA_RATIO
    pgd_module.ALPHA_RATIO = PGD_ALPHA_RATIO

    pgd_module.PGD_RANDOM_START = PGD_RANDOM_START
    pgd_module.RANDOM_START = PGD_RANDOM_START


# ============================================================
# PGD + ЗАЩИТА
# ============================================================

class PGDDefenseValidator(PGDValidator):
    defense_config = DefenseConfig(
        method="median",
        kernel_size=3,
    )

    last_defense_quality: dict[str, Any] = {}

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
    ) -> None:
        super().__init__(
            dataloader=dataloader,
            save_dir=save_dir,
            args=args,
            _callbacks=_callbacks,
        )

        self.defense_image_count = 0
        self.defense_seconds = 0.0

        self.psnr_sum = 0.0
        self.psnr_count = 0
        self.ssim_sum = 0.0

        self.change_mae_sum = 0.0
        self.change_linf_sum = 0.0
        self.clean_mae_sum = 0.0

    def create_clean_snapshot(
        self,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        """
        Создаёт чистое изображение до PGD.

        Вход от YOLO dataloader хранится в диапазоне 0..255.
        Здесь не используется self.args.half, поскольку такого
        поля в Ultralytics 8.4.99 может не быть.
        """

        raw_image = batch["img"]

        clean = raw_image.to(
            self.device,
            non_blocking=(
                self.device.type == "cuda"
            ),
        )

        clean = clean.to(
            dtype=torch.float32
        )

        clean = clean / 255.0

        return clean.detach().clone()

    def collect_defense_quality(
        self,
        clean: torch.Tensor,
        attacked: torch.Tensor,
        defended: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            clean_float = clean.float()
            attacked_float = attacked.float()
            defended_float = defended.float()

            batch_size = int(
                clean.shape[0]
            )

            (
                psnr_sum,
                psnr_count,
            ) = calculate_psnr_sum(
                clean_float,
                defended_float,
            )

            self.psnr_sum += psnr_sum
            self.psnr_count += psnr_count

            ssim = calculate_ssim_per_image(
                clean=clean_float,
                attacked=defended_float,
            )

            self.ssim_sum += float(
                ssim.sum().item()
            )

            defense_change = (
                defended_float
                - attacked_float
            ).abs().flatten(
                start_dim=1
            )

            clean_difference = (
                defended_float
                - clean_float
            ).abs().flatten(
                start_dim=1
            )

            change_mae = (
                defense_change.mean(
                    dim=1
                )
                * 255.0
            )

            change_linf = (
                defense_change.amax(
                    dim=1
                )
                * 255.0
            )

            clean_mae = (
                clean_difference.mean(
                    dim=1
                )
                * 255.0
            )

            self.change_mae_sum += float(
                change_mae.sum().item()
            )

            self.change_linf_sum += float(
                change_linf.sum().item()
            )

            self.clean_mae_sum += float(
                clean_mae.sum().item()
            )

            self.defense_image_count += (
                batch_size
            )

    def preprocess(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        # Сохраняем чистое изображение до изменения batch.
        clean = self.create_clean_snapshot(
            batch
        )

        # Используем полностью рабочий preprocess из
        # evaluate_pgd.py. Он самостоятельно создаёт PGD.
        batch = super().preprocess(
            batch
        )

        attacked = batch[
            "img"
        ].detach()

        if clean.shape != attacked.shape:
            raise RuntimeError(
                "Размеры clean и attacked не совпадают: "
                f"{tuple(clean.shape)} и "
                f"{tuple(attacked.shape)}."
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        started = time.perf_counter()

        defended = apply_defense(
            image=attacked,
            config=type(self).defense_config,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        self.defense_seconds += (
            time.perf_counter()
            - started
        )

        self.collect_defense_quality(
            clean=clean,
            attacked=attacked,
            defended=defended,
        )

        batch["img"] = defended

        return batch

    def finalize_metrics(self) -> None:
        super().finalize_metrics()

        if self.defense_image_count == 0:
            quality = {
                "defense_ms_per_image": None,
                "defended_vs_clean_psnr": None,
                "defended_vs_clean_ssim": None,
                "defense_change_mae_pixels": None,
                "defense_change_linf_pixels": None,
                "defended_vs_clean_mae_pixels": None,
            }

        else:
            mean_psnr = (
                self.psnr_sum
                / self.psnr_count
                if self.psnr_count > 0
                else None
            )

            quality = {
                "defense_ms_per_image": (
                    self.defense_seconds
                    / self.defense_image_count
                    * 1000.0
                ),

                "defended_vs_clean_psnr": (
                    mean_psnr
                ),

                "defended_vs_clean_ssim": (
                    self.ssim_sum
                    / self.defense_image_count
                ),

                "defense_change_mae_pixels": (
                    self.change_mae_sum
                    / self.defense_image_count
                ),

                "defense_change_linf_pixels": (
                    self.change_linf_sum
                    / self.defense_image_count
                ),

                "defended_vs_clean_mae_pixels": (
                    self.clean_mae_sum
                    / self.defense_image_count
                ),
            }

        type(self).last_defense_quality = quality


# ============================================================
# КЭШ
# ============================================================

def cached_result_is_current(
    path: Path,
    attack: str,
    epsilon: int,
    config: DefenseConfig | None,
) -> bool:
    if not path.is_file():
        return False

    try:
        data = read_json(path)

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return False

    expected_config = (
        asdict(config)
        if config is not None
        else None
    )

    return (
        data.get("script_version")
        == SCRIPT_VERSION
        and data.get("split")
        == "test"
        and data.get("attack")
        == attack
        and data.get("epsilon_pixels")
        == epsilon
        and data.get("config")
        == expected_config
        and data.get("attack_parameters")
        == attack_parameters(attack)
    )


# ============================================================
# REFERENCE БЕЗ ЗАЩИТЫ
# ============================================================

def evaluate_reference(
    model: YOLO,
    attack: str,
    epsilon: int,
) -> dict[str, Any]:
    result_path = (
        REFERENCE_DIR
        / condition_filename(
            attack,
            epsilon,
        )
    )

    if cached_result_is_current(
        path=result_path,
        attack=attack,
        epsilon=epsilon,
        config=None,
    ):
        print(
            "Загружен reference: "
            f"{condition_key(attack, epsilon)}"
        )

        return read_json(
            result_path
        )

    current_seed = run_seed(
        attack,
        epsilon,
    )

    seed_everything(
        current_seed
    )

    if attack in (
        "clean",
        "fgsm",
    ):
        validator_class: Type[Any] = (
            FGSMValidator
        )

        validator_class.epsilon_pixels = (
            epsilon
        )

    elif attack == "pgd":
        validator_class = PGDValidator

        configure_pgd(
            validator_class,
            epsilon,
        )

    else:
        raise ValueError(
            f"Неизвестная атака: {attack}"
        )

    validator_class.last_quality = {}

    print("\n" + "=" * 116)

    if attack == "clean":
        print(
            "REFERENCE TEST | CLEAN"
        )

    else:
        print(
            f"REFERENCE TEST | "
            f"{attack.upper()} | "
            f"EPSILON={epsilon}/255"
        )

    print("=" * 116)

    metrics = model.val(
        validator=validator_class,

        data=str(DATA_YAML),
        split="test",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE_INDEX,
        workers=WORKERS,

        rect=True,

        plots=False,
        save_json=False,

        project=str(RUNS_DIR),

        name=(
            "reference_"
            f"{condition_key(attack, epsilon)}"
        ),

        exist_ok=True,
        verbose=False,
    )

    result = {
        "script_version": SCRIPT_VERSION,
        "split": "test",

        "attack": attack,
        "epsilon_pixels": epsilon,

        "seed": current_seed,
        "config": None,

        "attack_parameters": (
            attack_parameters(attack)
        ),

        "overall": (
            extract_overall_metrics(
                metrics
            )
        ),

        "attack_quality": dict(
            validator_class.last_quality
        ),

        "defense_quality": None,

        "speed_ms_per_image": (
            speed_dict(metrics)
        ),

        "classes": (
            extract_class_metrics(
                metrics
            )
        ),
    }

    write_json(
        result_path,
        result,
    )

    cleanup_cuda()

    return result


# ============================================================
# ОЦЕНКА С ЗАЩИТОЙ
# ============================================================

def evaluate_defended(
    model: YOLO,
    config: DefenseConfig,
    attack: str,
    epsilon: int,
) -> dict[str, Any]:
    candidate_directory = (
        RESULTS_DIR
        / config.identifier
    )

    candidate_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        candidate_directory
        / condition_filename(
            attack,
            epsilon,
        )
    )

    if cached_result_is_current(
        path=result_path,
        attack=attack,
        epsilon=epsilon,
        config=config,
    ):
        print(
            f"Загружен: "
            f"{config.identifier} | "
            f"{condition_key(attack, epsilon)}"
        )

        return read_json(
            result_path
        )

    current_seed = run_seed(
        attack,
        epsilon,
    )

    seed_everything(
        current_seed
    )

    if attack in (
        "clean",
        "fgsm",
    ):
        validator_class: Type[Any] = (
            ComparisonDefenseValidator
        )

        validator_class.epsilon_pixels = (
            epsilon
        )

    elif attack == "pgd":
        validator_class = (
            PGDDefenseValidator
        )

        configure_pgd(
            validator_class,
            epsilon,
        )

    else:
        raise ValueError(
            f"Неизвестная атака: {attack}"
        )

    validator_class.defense_config = (
        config
    )

    validator_class.last_quality = {}

    validator_class.last_defense_quality = {}

    print("\n" + "-" * 116)

    if attack == "clean":
        print(
            f"{config.identifier} | CLEAN"
        )

    else:
        print(
            f"{config.identifier} | "
            f"{attack.upper()} | "
            f"EPSILON={epsilon}/255"
        )

    print("-" * 116)

    metrics = model.val(
        validator=validator_class,

        data=str(DATA_YAML),
        split="test",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE_INDEX,
        workers=WORKERS,

        rect=True,

        plots=False,
        save_json=False,

        project=str(RUNS_DIR),

        name=(
            f"{config.identifier}_"
            f"{condition_key(attack, epsilon)}"
        ),

        exist_ok=True,
        verbose=False,
    )

    result = {
        "script_version": SCRIPT_VERSION,
        "split": "test",

        "attack": attack,
        "epsilon_pixels": epsilon,

        "seed": current_seed,

        "config": asdict(
            config
        ),

        "attack_parameters": (
            attack_parameters(attack)
        ),

        "overall": (
            extract_overall_metrics(
                metrics
            )
        ),

        "attack_quality": dict(
            validator_class.last_quality
        ),

        "defense_quality": dict(
            validator_class
            .last_defense_quality
        ),

        "speed_ms_per_image": (
            speed_dict(metrics)
        ),

        "classes": (
            extract_class_metrics(
                metrics
            )
        ),
    }

    write_json(
        result_path,
        result,
    )

    cleanup_cuda()

    return result


# ============================================================
# ОБЩАЯ ТАБЛИЦА
# ============================================================

def build_summary_rows(
    references: dict[
        str,
        dict[str, Any],
    ],

    defended: dict[
        str,
        dict[
            str,
            dict[str, Any],
        ],
    ],

    configs: list[DefenseConfig],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    clean_reference = (
        references["clean"]["overall"]
    )

    def append_row(
        method: str,
        identifier: str,
        attack: str,
        epsilon: int,
        result: dict[str, Any],
        clean_defended: (
            dict[str, float]
            | None
        ),
    ) -> None:
        overall = result["overall"]

        speed = result.get(
            "speed_ms_per_image",
            {},
        )

        quality = (
            result.get(
                "defense_quality"
            )
            or {}
        )

        if attack == "clean":
            attacked_reference = None

        else:
            attacked_reference = (
                references[
                    condition_key(
                        attack,
                        epsilon,
                    )
                ]["overall"]
            )

        if clean_defended is None:
            clean_drop_map50 = None
            clean_drop_map50_95 = None

        else:
            clean_drop_map50 = (
                positive_relative_drop(
                    reference=(
                        clean_reference[
                            "mAP50"
                        ]
                    ),

                    evaluated=(
                        clean_defended[
                            "mAP50"
                        ]
                    ),
                )
            )

            clean_drop_map50_95 = (
                positive_relative_drop(
                    reference=(
                        clean_reference[
                            "mAP50-95"
                        ]
                    ),

                    evaluated=(
                        clean_defended[
                            "mAP50-95"
                        ]
                    ),
                )
            )

        if attacked_reference is None:
            recovery_map50 = None
            recovery_map50_95 = None

        else:
            recovery_map50 = (
                recovery_rate(
                    clean=(
                        clean_reference[
                            "mAP50"
                        ]
                    ),

                    attacked=(
                        attacked_reference[
                            "mAP50"
                        ]
                    ),

                    defended=(
                        overall[
                            "mAP50"
                        ]
                    ),
                )
            )

            recovery_map50_95 = (
                recovery_rate(
                    clean=(
                        clean_reference[
                            "mAP50-95"
                        ]
                    ),

                    attacked=(
                        attacked_reference[
                            "mAP50-95"
                        ]
                    ),

                    defended=(
                        overall[
                            "mAP50-95"
                        ]
                    ),
                )
            )

        rows.append(
            {
                "method": method,
                "identifier": identifier,

                "attack": attack,
                "epsilon_pixels": epsilon,

                "precision": (
                    overall["precision"]
                ),

                "recall": (
                    overall["recall"]
                ),

                "mAP50": (
                    overall["mAP50"]
                ),

                "mAP50-95": (
                    overall["mAP50-95"]
                ),

                "fitness": (
                    overall["fitness"]
                ),

                "clean_reference_mAP50": (
                    clean_reference[
                        "mAP50"
                    ]
                ),

                "clean_reference_mAP50-95": (
                    clean_reference[
                        "mAP50-95"
                    ]
                ),

                "clean_defended_mAP50": (
                    clean_defended[
                        "mAP50"
                    ]
                    if clean_defended
                    is not None
                    else None
                ),

                "clean_defended_mAP50-95": (
                    clean_defended[
                        "mAP50-95"
                    ]
                    if clean_defended
                    is not None
                    else None
                ),

                "clean_mAP50_relative_drop": (
                    clean_drop_map50
                ),

                "clean_mAP50-95_relative_drop": (
                    clean_drop_map50_95
                ),

                "attacked_reference_mAP50": (
                    attacked_reference[
                        "mAP50"
                    ]
                    if attacked_reference
                    is not None
                    else None
                ),

                "attacked_reference_mAP50-95": (
                    attacked_reference[
                        "mAP50-95"
                    ]
                    if attacked_reference
                    is not None
                    else None
                ),

                "recovery_mAP50": (
                    recovery_map50
                ),

                "recovery_mAP50-95": (
                    recovery_map50_95
                ),

                "defense_ms_per_image": (
                    quality.get(
                        "defense_ms_per_image"
                    )
                ),

                "defended_vs_clean_psnr": (
                    quality.get(
                        "defended_vs_clean_psnr"
                    )
                ),

                "defended_vs_clean_ssim": (
                    quality.get(
                        "defended_vs_clean_ssim"
                    )
                ),

                "defense_change_mae_pixels": (
                    quality.get(
                        "defense_change_mae_pixels"
                    )
                ),

                "defense_change_linf_pixels": (
                    quality.get(
                        "defense_change_linf_pixels"
                    )
                ),

                "defended_vs_clean_mae_pixels": (
                    quality.get(
                        "defended_vs_clean_mae_pixels"
                    )
                ),

                "preprocess_ms": (
                    speed.get("preprocess")
                ),

                "inference_ms": (
                    speed.get("inference")
                ),

                "loss_ms": (
                    speed.get("loss")
                ),

                "postprocess_ms": (
                    speed.get("postprocess")
                ),

                "validator_total_ms": (
                    speed.get("total")
                ),
            }
        )

    # Результаты без защиты.
    for attack, epsilon in all_conditions():
        append_row(
            method="none",
            identifier="none",
            attack=attack,
            epsilon=epsilon,

            result=references[
                condition_key(
                    attack,
                    epsilon,
                )
            ],

            clean_defended=None,
        )

    config_by_identifier = {
        config.identifier: config
        for config in configs
    }

    # Результаты с защитой.
    for (
        identifier,
        condition_results,
    ) in defended.items():
        config = (
            config_by_identifier[
                identifier
            ]
        )

        clean_defended = (
            condition_results[
                "clean"
            ]["overall"]
        )

        for attack, epsilon in all_conditions():
            append_row(
                method=config.method,
                identifier=identifier,
                attack=attack,
                epsilon=epsilon,

                result=(
                    condition_results[
                        condition_key(
                            attack,
                            epsilon,
                        )
                    ]
                ),

                clean_defended=(
                    clean_defended
                ),
            )

    return rows


# ============================================================
# ПОКЛАССОВАЯ ТАБЛИЦА
# ============================================================

def build_class_rows(
    references: dict[
        str,
        dict[str, Any],
    ],

    defended: dict[
        str,
        dict[
            str,
            dict[str, Any],
        ],
    ],

    configs: list[DefenseConfig],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for attack, epsilon in all_conditions():
        result = references[
            condition_key(
                attack,
                epsilon,
            )
        ]

        for class_row in result[
            "classes"
        ]:
            rows.append(
                {
                    "method": "none",
                    "identifier": "none",

                    "attack": attack,
                    "epsilon_pixels": epsilon,

                    **class_row,
                }
            )

    config_by_identifier = {
        config.identifier: config
        for config in configs
    }

    for (
        identifier,
        condition_results,
    ) in defended.items():
        method = (
            config_by_identifier[
                identifier
            ].method
        )

        for attack, epsilon in all_conditions():
            result = (
                condition_results[
                    condition_key(
                        attack,
                        epsilon,
                    )
                ]
            )

            for class_row in result[
                "classes"
            ]:
                rows.append(
                    {
                        "method": method,
                        "identifier": identifier,

                        "attack": attack,
                        "epsilon_pixels": epsilon,

                        **class_row,
                    }
                )

    return rows


# ============================================================
# ПЕЧАТЬ ИТОГОВ
# ============================================================

def print_condition_table(
    rows: list[dict[str, Any]],
    attack: str,
    epsilon: int,
) -> None:
    selected_rows = [
        row
        for row in rows
        if (
            row["attack"] == attack
            and row["epsilon_pixels"]
            == epsilon
        )
    ]

    order = {
        "none": 0,
        **{
            method: index + 1
            for index, method
            in enumerate(METHOD_ORDER)
        },
    }

    selected_rows.sort(
        key=lambda row: order.get(
            str(row["method"]),
            999,
        )
    )

    if attack == "clean":
        title = "TEST | CLEAN"

    else:
        title = (
            f"TEST | {attack.upper()} | "
            f"EPSILON={epsilon}/255"
        )

    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)

    print(
        f"{'Method':>12}"
        f"{'mAP50':>12}"
        f"{'mAP50-95':>13}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'Clean drop %':>15}"
        f"{'Recovery %':>14}"
        f"{'Defense ms':>13}"
    )

    for row in selected_rows:
        clean_drop = row[
            "clean_mAP50_relative_drop"
        ]

        recovery = row[
            "recovery_mAP50"
        ]

        defense_ms = row[
            "defense_ms_per_image"
        ]

        clean_drop_text = (
            "n/a"
            if clean_drop is None
            else (
                f"{float(clean_drop) * 100.0:.2f}"
            )
        )

        recovery_text = (
            "n/a"
            if (
                recovery is None
                or not math.isfinite(
                    float(recovery)
                )
            )
            else (
                f"{float(recovery) * 100.0:.2f}"
            )
        )

        defense_text = (
            "n/a"
            if defense_ms is None
            else f"{float(defense_ms):.2f}"
        )

        print(
            f"{str(row['method']):>12}"
            f"{float(row['mAP50']):>12.4f}"
            f"{float(row['mAP50-95']):>13.4f}"
            f"{float(row['precision']):>12.4f}"
            f"{float(row['recall']):>12.4f}"
            f"{clean_drop_text:>15}"
            f"{recovery_text:>14}"
            f"{defense_text:>13}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    check_environment()
    prepare_directories()
    seed_everything(SEED)

    configs = load_selected_defenses()

    print(
        "\nЗафиксированные конфигурации:"
    )

    for config in configs:
        print(
            f"  {config.method:>10}: "
            f"{config.identifier}"
        )

    evaluation_model = YOLO(
        str(MODEL_PATH),
        verbose=False,
    )

    # --------------------------------------------------------
    # REFERENCE БЕЗ ЗАЩИТЫ
    # --------------------------------------------------------

    references: dict[
        str,
        dict[str, Any],
    ] = {}

    for attack, epsilon in all_conditions():
        key = condition_key(
            attack,
            epsilon,
        )

        references[key] = evaluate_reference(
            model=evaluation_model,
            attack=attack,
            epsilon=epsilon,
        )

    # --------------------------------------------------------
    # ЗАЩИТЫ
    # --------------------------------------------------------

    defended: dict[
        str,
        dict[
            str,
            dict[str, Any],
        ],
    ] = {}

    for config_index, config in enumerate(
        configs,
        start=1,
    ):
        print("\n" + "#" * 116)

        print(
            f"ЗАЩИТА "
            f"{config_index}/"
            f"{len(configs)}: "
            f"{config.identifier}"
        )

        print("#" * 116)

        defended[
            config.identifier
        ] = {}

        for attack, epsilon in all_conditions():
            key = condition_key(
                attack,
                epsilon,
            )

            defended[
                config.identifier
            ][key] = evaluate_defended(
                model=evaluation_model,
                config=config,
                attack=attack,
                epsilon=epsilon,
            )

    # --------------------------------------------------------
    # СОХРАНЕНИЕ
    # --------------------------------------------------------

    summary_rows = build_summary_rows(
        references=references,
        defended=defended,
        configs=configs,
    )

    class_rows = build_class_rows(
        references=references,
        defended=defended,
        configs=configs,
    )

    summary_csv_path = (
        OUTPUT_DIR
        / "final_test_summary.csv"
    )

    class_csv_path = (
        OUTPUT_DIR
        / "final_test_classes.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "final_test_results.json"
    )

    write_csv(
        summary_csv_path,
        summary_rows,
    )

    write_csv(
        class_csv_path,
        class_rows,
    )

    write_json(
        json_path,
        {
            "script_version": SCRIPT_VERSION,
            "split": "test",

            "model": str(MODEL_PATH),
            "dataset": str(DATA_YAML),

            "selection_source": str(
                SELECTION_PATH
            ),

            "evaluation_parameters": {
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "device_index": DEVICE_INDEX,
                "workers": WORKERS,
                "seed": SEED,

                "attack_epsilons": (
                    ATTACK_EPSILONS
                ),

                "pgd_steps": PGD_STEPS,

                "pgd_alpha_ratio": (
                    PGD_ALPHA_RATIO
                ),

                "pgd_random_start": (
                    PGD_RANDOM_START
                ),
            },

            "selected_configs": [
                asdict(config)
                for config in configs
            ],

            "references": references,

            "defended_results": (
                defended
            ),

            "summary_rows": (
                summary_rows
            ),

            "class_rows": class_rows,
        },
    )

    # --------------------------------------------------------
    # ВЫВОД
    # --------------------------------------------------------

    for attack, epsilon in all_conditions():
        print_condition_table(
            rows=summary_rows,
            attack=attack,
            epsilon=epsilon,
        )

    print("\n" + "=" * 132)

    print(
        "ФИНАЛЬНАЯ ОЦЕНКА "
        "НА TEST ЗАВЕРШЕНА"
    )

    print("=" * 132)

    print(
        "Общая таблица:\n"
        f"{summary_csv_path}"
    )

    print(
        "\nМетрики по классам:\n"
        f"{class_csv_path}"
    )

    print(
        "\nПолный JSON:\n"
        f"{json_path}"
    )

    print(
        "\nПараметры защит загружены "
        "из validation-выбора "
        "и не изменялись."
    )


if __name__ == "__main__":
    main()