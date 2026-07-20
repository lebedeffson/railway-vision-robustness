from __future__ import annotations

import csv
import gc
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ultralytics import YOLO

import evaluate_pgd as pgd_module
from compare_defenses_val import (
    ComparisonDefenseValidator,
    DefenseConfig,
    apply_defense,
    extract_class_metrics,
    extract_overall_metrics,
    value_to_float,
)
from evaluate_defenses_test import PGDDefenseValidator


# ============================================================
# ПУТИ И ПАРАМЕТРЫ
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

FINAL_TEST_PATH = (
    PROJECT_DIR
    / "outputs"
    / "defenses"
    / "final_test"
    / "final_test_results.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "defenses"
    / "adaptive_attacks"
)

RESULTS_DIR = OUTPUT_DIR / "results"
RUNS_DIR = OUTPUT_DIR / "runs"

SCRIPT_VERSION = 1

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

# Фильтр пересчитывается при backward.
# Это уменьшает расход видеопамяти при 1280×1280.
USE_GRADIENT_CHECKPOINTING = True


# ============================================================
# ОБЩИЕ ФУНКЦИИ
# ============================================================

def seed_everything(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_cuda() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def speed_dict(
    metrics: Any,
) -> dict[str, float]:
    result = {
        key: value_to_float(value)
        for key, value
        in metrics.speed.items()
    }

    result["total"] = float(
        sum(result.values())
    )

    return result


def run_seed(
    attack: str,
    epsilon: int,
) -> int:
    offset = {
        "fgsm": 10_000,
        "pgd": 20_000,
    }[attack]

    return (
        SEED
        + offset
        + epsilon * 100
    )


def result_filename(
    attack: str,
    epsilon: int,
) -> str:
    return (
        f"adaptive_{attack}"
        f"_eps_{epsilon}"
        f"_255.json"
    )


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


def set_module_values(
    module: Any,
    names: tuple[str, ...],
    value: Any,
) -> None:
    for name in names:
        setattr(
            module,
            name,
            value,
        )


def configure_pgd(
    validator_class: Type[Any],
    epsilon: int,
) -> None:
    validator_class.epsilon_pixels = (
        epsilon
    )

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

    set_module_values(
        pgd_module,
        (
            "PGD_STEPS",
            "STEPS",
            "NUM_STEPS",
            "ITERATIONS",
        ),
        PGD_STEPS,
    )

    set_module_values(
        pgd_module,
        (
            "PGD_ALPHA_RATIO",
            "ALPHA_RATIO",
            "STEP_SIZE_RATIO",
        ),
        PGD_ALPHA_RATIO,
    )

    set_module_values(
        pgd_module,
        (
            "PGD_RANDOM_START",
            "RANDOM_START",
            "USE_RANDOM_START",
        ),
        PGD_RANDOM_START,
    )


# ============================================================
# ПРОВЕРКА ОКРУЖЕНИЯ
# ============================================================

def check_environment() -> None:
    required = [
        DATA_YAML,
        MODEL_PATH,
        SELECTION_PATH,
        FINAL_TEST_PATH,
        PROJECT_DIR / "evaluate_fgsm.py",
        PROJECT_DIR / "evaluate_pgd.py",
        PROJECT_DIR / "compare_defenses_val.py",
        PROJECT_DIR / "evaluate_defenses_test.py",
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                "Не найден необходимый файл:\n"
                f"{path}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна."
        )

    if (
        DEVICE_INDEX
        >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"GPU {DEVICE_INDEX} недоступна. "
            f"Найдено устройств: "
            f"{torch.cuda.device_count()}."
        )


def prepare_directories() -> None:
    for path in (
        OUTPUT_DIR,
        RESULTS_DIR,
        RUNS_DIR,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# ЗАГРУЗКА ЗАФИКСИРОВАННОЙ T-НОРМЫ
# ============================================================

def load_fixed_tnorm_config() -> DefenseConfig:
    data = read_json(
        SELECTION_PATH
    )

    selected = data.get(
        "selected_by_method"
    )

    if isinstance(
        selected,
        dict,
    ):
        tnorm_result = selected.get(
            "tnorm"
        )

    else:
        tnorm_result = next(
            (
                item
                for item
                in data.get(
                    "selected_results",
                    [],
                )
                if item.get("method")
                == "tnorm"
            ),
            None,
        )

    if not isinstance(
        tnorm_result,
        dict,
    ):
        raise KeyError(
            "В best_baseline_defenses.json "
            "не найдена выбранная T-норма."
        )

    config_data = tnorm_result.get(
        "config"
    )

    if not isinstance(
        config_data,
        dict,
    ):
        raise KeyError(
            "У выбранной T-нормы "
            "отсутствует config."
        )

    config = DefenseConfig(
        **config_data
    )

    if config.method != "tnorm":
        raise ValueError(
            f"Ожидался метод tnorm, "
            f"получен {config.method}."
        )

    if (
        str(config.tnorm).lower()
        != "product"
    ):
        raise ValueError(
            "Этот адаптивный градиентный тест "
            "рассчитан на Product T-норму, "
            f"но в JSON указано {config.tnorm}."
        )

    required_values = {
        "kernel_size": (
            config.kernel_size
        ),
        "sigma_color_pixels": (
            config.sigma_color_pixels
        ),
        "sigma_spatial": (
            config.sigma_spatial
        ),
        "strength": (
            config.strength
        ),
        "confidence_gamma": (
            config.confidence_gamma
        ),
    }

    missing = [
        name
        for name, value
        in required_values.items()
        if value is None
    ]

    if missing:
        raise ValueError(
            "В конфигурации T-нормы "
            "отсутствуют параметры: "
            + ", ".join(missing)
        )

    return config


# ============================================================
# ДИФФЕРЕНЦИРУЕМЫЙ PRODUCT T-NORM ФИЛЬТР
# ============================================================

def differentiable_product_tnorm_filter(
    image: torch.Tensor,
    config: DefenseConfig,
) -> torch.Tensor:
    """
    Дифференцируемый эквивалент Product T-нормового фильтра.

    torch.no_grad здесь намеренно не используется:
    адаптивная атака должна вычислять градиент через
    полный конвейер защита -> детектор.
    """

    kernel_size = int(
        config.kernel_size
    )

    if (
        kernel_size < 3
        or kernel_size % 2 == 0
    ):
        raise ValueError(
            "Размер окна T-нормы "
            "должен быть нечётным и >= 3."
        )

    sigma_color = (
        float(
            config.sigma_color_pixels
        )
        / 255.0
    )

    sigma_spatial = float(
        config.sigma_spatial
    )

    strength = float(
        config.strength
    )

    confidence_gamma = float(
        config.confidence_gamma
    )

    if (
        sigma_color <= 0
        or sigma_spatial <= 0
    ):
        raise ValueError(
            "Параметры sigma "
            "должны быть положительными."
        )

    original_dtype = image.dtype
    image_float = image.float()

    (
        batch_size,
        _,
        height,
        width,
    ) = image_float.shape

    radius = kernel_size // 2

    padded = F.pad(
        image_float,
        (
            radius,
            radius,
            radius,
            radius,
        ),
        mode="reflect",
    )

    weighted_sum = torch.zeros_like(
        image_float
    )

    weight_sum = torch.zeros(
        (
            batch_size,
            1,
            height,
            width,
        ),
        device=image_float.device,
        dtype=image_float.dtype,
    )

    maximum_spatial_sum = 0.0

    for offset_y in range(
        -radius,
        radius + 1,
    ):
        for offset_x in range(
            -radius,
            radius + 1,
        ):
            if (
                offset_x == 0
                and offset_y == 0
            ):
                continue

            start_y = (
                radius
                + offset_y
            )

            start_x = (
                radius
                + offset_x
            )

            neighbour = padded[
                :,
                :,
                start_y:start_y + height,
                start_x:start_x + width,
            ]

            color_distance = (
                image_float
                .sub(neighbour)
                .abs()
                .mean(
                    dim=1,
                    keepdim=True,
                )
            )

            color_membership = torch.exp(
                -color_distance
                / sigma_color
            )

            spatial_distance_squared = (
                offset_x**2
                + offset_y**2
            )

            spatial_membership = math.exp(
                -spatial_distance_squared
                / (
                    2.0
                    * sigma_spatial**2
                )
            )

            # Product T-норма.
            combined_weight = (
                color_membership
                * spatial_membership
            )

            # Без inplace-операций, чтобы не ломать autograd.
            weighted_sum = (
                weighted_sum
                + combined_weight
                * neighbour
            )

            weight_sum = (
                weight_sum
                + combined_weight
            )

            maximum_spatial_sum += (
                spatial_membership
            )

    local_estimate = (
        weighted_sum
        / weight_sum.clamp_min(
            1e-8
        )
    )

    confidence = torch.clamp(
        weight_sum
        / (
            maximum_spatial_sum
            + 1e-8
        ),
        min=0.0,
        max=1.0,
    )

    confidence = confidence.pow(
        confidence_gamma
    )

    filtered = (
        image_float
        + strength
        * confidence
        * (
            local_estimate
            - image_float
        )
    )

    return (
        filtered
        .clamp(
            0.0,
            1.0,
        )
        .to(original_dtype)
    )


def apply_differentiable_defense(
    image: torch.Tensor,
    config: DefenseConfig,
) -> torch.Tensor:
    if (
        USE_GRADIENT_CHECKPOINTING
        and image.requires_grad
    ):
        return checkpoint(
            lambda tensor: (
                differentiable_product_tnorm_filter(
                    tensor,
                    config,
                )
            ),
            image,
            use_reentrant=False,
        )

    return (
        differentiable_product_tnorm_filter(
            image,
            config,
        )
    )


def verify_filter_equivalence(
    config: DefenseConfig,
) -> None:
    seed_everything(SEED)

    device = torch.device(
        f"cuda:{DEVICE_INDEX}"
    )

    sample = torch.rand(
        (
            1,
            3,
            64,
            64,
        ),
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        expected = apply_defense(
            image=sample,
            config=config,
        )

        actual = (
            differentiable_product_tnorm_filter(
                image=sample,
                config=config,
            )
        )

    maximum_error = float(
        (
            expected
            - actual
        )
        .abs()
        .amax()
        .item()
    )

    print(
        "Проверка фильтра: "
        f"max_abs_error="
        f"{maximum_error:.10f}"
    )

    if maximum_error > 1e-6:
        raise RuntimeError(
            "Дифференцируемый фильтр "
            "не совпадает с фильтром final_test. "
            f"Максимальная ошибка: "
            f"{maximum_error}."
        )

    del sample
    del expected
    del actual

    cleanup_cuda()


# ============================================================
# ОБЁРТКА МОДЕЛИ ДЛЯ РАСЧЁТА LOSS
# ============================================================

class DefendedLossModel(
    torch.nn.Module
):
    """
    Перед каждым вычислением attack loss применяет
    дифференцируемый Product T-нормовый фильтр.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        config: DefenseConfig,
    ) -> None:
        super().__init__()

        self.base_model = (
            base_model
        )

        self.defense_config = (
            config
        )

    def transform_batch(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        if "img" not in batch:
            raise KeyError(
                "В batch отсутствует ключ img."
            )

        transformed = dict(
            batch
        )

        transformed["img"] = (
            apply_differentiable_defense(
                image=batch["img"],
                config=(
                    self.defense_config
                ),
            )
        )

        return transformed

    def forward(
        self,
        input_data: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if isinstance(
            input_data,
            dict,
        ):
            input_data = (
                self.transform_batch(
                    input_data
                )
            )

        elif torch.is_tensor(
            input_data
        ):
            input_data = (
                apply_differentiable_defense(
                    image=input_data,
                    config=(
                        self.defense_config
                    ),
                )
            )

        return self.base_model(
            input_data,
            *args,
            **kwargs,
        )

    def loss(
        self,
        batch: dict[str, Any],
        preds: Any = None,
    ) -> Any:
        transformed = (
            self.transform_batch(
                batch
            )
        )

        if preds is None:
            return self.base_model.loss(
                transformed
            )

        return self.base_model.loss(
            transformed,
            preds,
        )

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        try:
            return super().__getattr__(
                name
            )

        except AttributeError:
            base_model = (
                super().__getattr__(
                    "base_model"
                )
            )

            return getattr(
                base_model,
                name,
            )


def install_defended_loss_model(
    validator: Any,
    config: DefenseConfig,
) -> str:
    """
    Устанавливает дифференцируемую защиту в модель,
    которую существующий FGSM/PGD validator
    использует для расчёта loss.
    """

    if not hasattr(
        validator,
        "attack_model",
    ):
        available = sorted(
            name
            for name
            in vars(
                validator
            )
            if "attack"
            in name.lower()
        )

        raise AttributeError(
            "В валидаторе отсутствует "
            "attack_model. "
            "Найденные attack-поля: "
            f"{available}"
        )

    attack_holder = getattr(
        validator,
        "attack_model",
    )

    nested_model = getattr(
        attack_holder,
        "model",
        None,
    )

    # attack_model может быть высокоуровневым YOLO.
    # Тогда реальная DetectionModel лежит в .model.
    if (
        isinstance(
            nested_model,
            torch.nn.Module,
        )
        and hasattr(
            nested_model,
            "loss",
        )
    ):
        attack_holder.model = (
            DefendedLossModel(
                base_model=(
                    nested_model
                ),
                config=config,
            )
        )

        return (
            "attack_model.model"
        )

    # Либо attack_model уже является DetectionModel.
    if isinstance(
        attack_holder,
        torch.nn.Module,
    ):
        setattr(
            validator,
            "attack_model",
            DefendedLossModel(
                base_model=(
                    attack_holder
                ),
                config=config,
            ),
        )

        return "attack_model"

    raise TypeError(
        "Не удалось обернуть "
        "attack_model. "
        f"Тип: "
        f"{type(attack_holder).__name__}."
    )


# ============================================================
# АДАПТИВНЫЕ VALIDATOR-КЛАССЫ
# ============================================================

class AdaptiveFGSMValidator(
    ComparisonDefenseValidator
):
    proxy_target: str | None = (
        None
    )

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

        type(self).proxy_target = (
            install_defended_loss_model(
                validator=self,
                config=(
                    type(self)
                    .defense_config
                ),
            )
        )


class AdaptivePGDValidator(
    PGDDefenseValidator
):
    proxy_target: str | None = (
        None
    )

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

        type(self).proxy_target = (
            install_defended_loss_model(
                validator=self,
                config=(
                    type(self)
                    .defense_config
                ),
            )
        )


# ============================================================
# КЭШ И ОЦЕНКА
# ============================================================

def attack_parameters(
    attack: str,
) -> dict[str, Any]:
    return {
        "adaptive": True,

        "gradient_pipeline": (
            "product_tnorm_then_yolo"
        ),

        "gradient_checkpointing": (
            USE_GRADIENT_CHECKPOINTING
        ),

        "pgd_steps": (
            PGD_STEPS
            if attack == "pgd"
            else None
        ),

        "pgd_alpha_ratio": (
            PGD_ALPHA_RATIO
            if attack == "pgd"
            else None
        ),

        "pgd_random_start": (
            PGD_RANDOM_START
            if attack == "pgd"
            else None
        ),
    }


def cached_result_is_current(
    path: Path,
    attack: str,
    epsilon: int,
    config: DefenseConfig,
) -> bool:
    if not path.is_file():
        return False

    try:
        data = read_json(
            path
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return False

    return (
        data.get(
            "script_version"
        )
        == SCRIPT_VERSION

        and data.get(
            "split"
        )
        == "test"

        and data.get(
            "attack"
        )
        == attack

        and data.get(
            "epsilon_pixels"
        )
        == epsilon

        and data.get(
            "config"
        )
        == asdict(
            config
        )

        and data.get(
            "attack_parameters"
        )
        == attack_parameters(
            attack
        )
    )


def evaluate_adaptive_condition(
    model: YOLO,
    config: DefenseConfig,
    attack: str,
    epsilon: int,
) -> dict[str, Any]:
    result_path = (
        RESULTS_DIR
        / result_filename(
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
            "Загружен: adaptive "
            f"{attack.upper()} "
            f"{epsilon}/255"
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

    if attack == "fgsm":
        validator_class: Type[Any] = (
            AdaptiveFGSMValidator
        )

        validator_class.epsilon_pixels = (
            epsilon
        )

    elif attack == "pgd":
        validator_class = (
            AdaptivePGDValidator
        )

        configure_pgd(
            validator_class,
            epsilon,
        )

    else:
        raise ValueError(
            f"Неизвестная атака: "
            f"{attack}"
        )

    validator_class.defense_config = (
        config
    )

    validator_class.last_quality = {}

    validator_class.last_defense_quality = {}

    validator_class.proxy_target = (
        None
    )

    print(
        "\n"
        + "=" * 124
    )

    print(
        f"ADAPTIVE "
        f"{attack.upper()} | "
        "PRODUCT T-NORM -> YOLO | "
        f"EPSILON={epsilon}/255"
    )

    print(
        "=" * 124
    )

    metrics = model.val(
        validator=validator_class,

        data=str(
            DATA_YAML
        ),

        split="test",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE_INDEX,
        workers=WORKERS,

        rect=True,

        plots=False,
        save_json=False,

        project=str(
            RUNS_DIR
        ),

        name=(
            f"adaptive_{attack}"
            f"_eps_{epsilon}_255"
        ),

        exist_ok=True,
        verbose=False,
    )

    result = {
        "script_version": (
            SCRIPT_VERSION
        ),

        "split": "test",

        "attack": attack,

        "epsilon_pixels": (
            epsilon
        ),

        "seed": (
            current_seed
        ),

        "config": asdict(
            config
        ),

        "attack_parameters": (
            attack_parameters(
                attack
            )
        ),

        "proxy_target": (
            validator_class
            .proxy_target
        ),

        "overall": (
            extract_overall_metrics(
                metrics
            )
        ),

        "attack_quality": dict(
            validator_class
            .last_quality
        ),

        "defense_quality": dict(
            validator_class
            .last_defense_quality
        ),

        "speed_ms_per_image": (
            speed_dict(
                metrics
            )
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
# ЗАГРУЗКА FINAL TEST
# ============================================================

def load_final_test_context(
    config: DefenseConfig,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    str,
]:
    final_test = read_json(
        FINAL_TEST_PATH
    )

    references = final_test.get(
        "references"
    )

    defended_results = final_test.get(
        "defended_results"
    )

    if not isinstance(
        references,
        dict,
    ):
        raise KeyError(
            "В final_test_results.json "
            "отсутствует references."
        )

    if not isinstance(
        defended_results,
        dict,
    ):
        raise KeyError(
            "В final_test_results.json "
            "отсутствует defended_results."
        )

    identifier = (
        config.identifier
    )

    if identifier not in defended_results:
        possible = []

        for key, value in (
            defended_results.items()
        ):
            if not isinstance(
                value,
                dict,
            ):
                continue

            clean_config = (
                value
                .get(
                    "clean",
                    {},
                )
                .get(
                    "config",
                    {},
                )
            )

            if (
                clean_config.get(
                    "method"
                )
                == "tnorm"
            ):
                possible.append(
                    key
                )

        if len(possible) != 1:
            raise KeyError(
                "Не найден идентификатор "
                f"{config.identifier}. "
                "Доступные T-нормы: "
                f"{possible}"
            )

        identifier = possible[0]

    return (
        references,
        defended_results[
            identifier
        ],
        identifier,
    )


# ============================================================
# ИТОГОВАЯ ОБЩАЯ ТАБЛИЦА
# ============================================================

def build_summary_rows(
    adaptive_results: dict[
        str,
        dict[str, Any],
    ],
    references: dict[
        str,
        Any,
    ],
    oblivious_results: dict[
        str,
        Any,
    ],
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    clean_reference = (
        references[
            "clean"
        ][
            "overall"
        ]
    )

    clean_defended = (
        oblivious_results[
            "clean"
        ][
            "overall"
        ]
    )

    for attack in (
        "fgsm",
        "pgd",
    ):
        for epsilon in (
            ATTACK_EPSILONS
        ):
            key = (
                f"{attack}_"
                f"{epsilon}"
            )

            adaptive_result = (
                adaptive_results[
                    key
                ]
            )

            adaptive = (
                adaptive_result[
                    "overall"
                ]
            )

            no_defense = (
                references[
                    key
                ][
                    "overall"
                ]
            )

            oblivious = (
                oblivious_results[
                    key
                ][
                    "overall"
                ]
            )

            quality = (
                adaptive_result.get(
                    "attack_quality",
                    {},
                )
            )

            defense_quality = (
                adaptive_result.get(
                    "defense_quality",
                    {},
                )
            )

            rows.append(
                {
                    "attack": (
                        attack
                    ),

                    "epsilon_pixels": (
                        epsilon
                    ),

                    "clean_no_defense_mAP50": (
                        clean_reference[
                            "mAP50"
                        ]
                    ),

                    "clean_tnorm_mAP50": (
                        clean_defended[
                            "mAP50"
                        ]
                    ),

                    "no_defense_attacked_mAP50": (
                        no_defense[
                            "mAP50"
                        ]
                    ),

                    "oblivious_tnorm_mAP50": (
                        oblivious[
                            "mAP50"
                        ]
                    ),

                    "adaptive_tnorm_mAP50": (
                        adaptive[
                            "mAP50"
                        ]
                    ),

                    "adaptive_minus_oblivious_mAP50": (
                        adaptive[
                            "mAP50"
                        ]
                        - oblivious[
                            "mAP50"
                        ]
                    ),

                    "oblivious_recovery_mAP50": (
                        recovery_rate(
                            clean=(
                                clean_reference[
                                    "mAP50"
                                ]
                            ),

                            attacked=(
                                no_defense[
                                    "mAP50"
                                ]
                            ),

                            defended=(
                                oblivious[
                                    "mAP50"
                                ]
                            ),
                        )
                    ),

                    "adaptive_recovery_mAP50": (
                        recovery_rate(
                            clean=(
                                clean_reference[
                                    "mAP50"
                                ]
                            ),

                            attacked=(
                                no_defense[
                                    "mAP50"
                                ]
                            ),

                            defended=(
                                adaptive[
                                    "mAP50"
                                ]
                            ),
                        )
                    ),

                    "clean_no_defense_mAP50-95": (
                        clean_reference[
                            "mAP50-95"
                        ]
                    ),

                    "clean_tnorm_mAP50-95": (
                        clean_defended[
                            "mAP50-95"
                        ]
                    ),

                    "no_defense_attacked_mAP50-95": (
                        no_defense[
                            "mAP50-95"
                        ]
                    ),

                    "oblivious_tnorm_mAP50-95": (
                        oblivious[
                            "mAP50-95"
                        ]
                    ),

                    "adaptive_tnorm_mAP50-95": (
                        adaptive[
                            "mAP50-95"
                        ]
                    ),

                    "adaptive_minus_oblivious_mAP50-95": (
                        adaptive[
                            "mAP50-95"
                        ]
                        - oblivious[
                            "mAP50-95"
                        ]
                    ),

                    "oblivious_recovery_mAP50-95": (
                        recovery_rate(
                            clean=(
                                clean_reference[
                                    "mAP50-95"
                                ]
                            ),

                            attacked=(
                                no_defense[
                                    "mAP50-95"
                                ]
                            ),

                            defended=(
                                oblivious[
                                    "mAP50-95"
                                ]
                            ),
                        )
                    ),

                    "adaptive_recovery_mAP50-95": (
                        recovery_rate(
                            clean=(
                                clean_reference[
                                    "mAP50-95"
                                ]
                            ),

                            attacked=(
                                no_defense[
                                    "mAP50-95"
                                ]
                            ),

                            defended=(
                                adaptive[
                                    "mAP50-95"
                                ]
                            ),
                        )
                    ),

                    "adaptive_precision": (
                        adaptive[
                            "precision"
                        ]
                    ),

                    "adaptive_recall": (
                        adaptive[
                            "recall"
                        ]
                    ),

                    "adaptive_attack_psnr": (
                        quality.get(
                            "psnr"
                        )
                    ),

                    "adaptive_attack_ssim": (
                        quality.get(
                            "ssim"
                        )
                    ),

                    "adaptive_attack_linf_pixels": (
                        quality.get(
                            "linf_pixels"
                        )
                    ),

                    "defense_ms_per_image": (
                        defense_quality.get(
                            "defense_ms_per_image"
                        )
                    ),

                    "validator_total_ms": (
                        adaptive_result
                        .get(
                            "speed_ms_per_image",
                            {},
                        )
                        .get(
                            "total"
                        )
                    ),

                    "proxy_target": (
                        adaptive_result.get(
                            "proxy_target"
                        )
                    ),
                }
            )

    return rows


# ============================================================
# ПОКЛАССОВАЯ ТАБЛИЦА
# ============================================================

def class_lookup(
    result: dict[str, Any],
) -> dict[
    int,
    dict[str, Any],
]:
    return {
        int(
            row["class_id"]
        ): row

        for row
        in result.get(
            "classes",
            [],
        )
    }


def build_class_rows(
    adaptive_results: dict[
        str,
        dict[str, Any],
    ],
    references: dict[
        str,
        Any,
    ],
    oblivious_results: dict[
        str,
        Any,
    ],
) -> list[dict[str, Any]]:
    rows: list[
        dict[str, Any]
    ] = []

    for attack in (
        "fgsm",
        "pgd",
    ):
        for epsilon in (
            ATTACK_EPSILONS
        ):
            key = (
                f"{attack}_"
                f"{epsilon}"
            )

            no_defense_by_class = (
                class_lookup(
                    references[
                        key
                    ]
                )
            )

            oblivious_by_class = (
                class_lookup(
                    oblivious_results[
                        key
                    ]
                )
            )

            adaptive_by_class = (
                class_lookup(
                    adaptive_results[
                        key
                    ]
                )
            )

            class_ids = sorted(
                set(
                    no_defense_by_class
                )
                | set(
                    oblivious_by_class
                )
                | set(
                    adaptive_by_class
                )
            )

            for class_id in class_ids:
                no_defense = (
                    no_defense_by_class.get(
                        class_id,
                        {},
                    )
                )

                oblivious = (
                    oblivious_by_class.get(
                        class_id,
                        {},
                    )
                )

                adaptive = (
                    adaptive_by_class.get(
                        class_id,
                        {},
                    )
                )

                rows.append(
                    {
                        "attack": (
                            attack
                        ),

                        "epsilon_pixels": (
                            epsilon
                        ),

                        "class_id": (
                            class_id
                        ),

                        "class_name": (
                            adaptive.get(
                                "class_name"
                            )
                            or oblivious.get(
                                "class_name"
                            )
                            or no_defense.get(
                                "class_name"
                            )
                        ),

                        "no_defense_precision": (
                            no_defense.get(
                                "precision"
                            )
                        ),

                        "oblivious_precision": (
                            oblivious.get(
                                "precision"
                            )
                        ),

                        "adaptive_precision": (
                            adaptive.get(
                                "precision"
                            )
                        ),

                        "no_defense_recall": (
                            no_defense.get(
                                "recall"
                            )
                        ),

                        "oblivious_recall": (
                            oblivious.get(
                                "recall"
                            )
                        ),

                        "adaptive_recall": (
                            adaptive.get(
                                "recall"
                            )
                        ),

                        "no_defense_mAP50": (
                            no_defense.get(
                                "mAP50"
                            )
                        ),

                        "oblivious_mAP50": (
                            oblivious.get(
                                "mAP50"
                            )
                        ),

                        "adaptive_mAP50": (
                            adaptive.get(
                                "mAP50"
                            )
                        ),

                        "no_defense_mAP50-95": (
                            no_defense.get(
                                "mAP50-95"
                            )
                        ),

                        "oblivious_mAP50-95": (
                            oblivious.get(
                                "mAP50-95"
                            )
                        ),

                        "adaptive_mAP50-95": (
                            adaptive.get(
                                "mAP50-95"
                            )
                        ),
                    }
                )

    return rows


# ============================================================
# ПЕЧАТЬ ИТОГОВ
# ============================================================

def percent_text(
    value: float | None,
) -> str:
    if value is None:
        return "n/a"

    return (
        f"{value * 100.0:.2f}"
    )


def print_summary(
    rows: list[dict[str, Any]],
) -> None:
    print(
        "\n"
        + "=" * 150
    )

    print(
        "ADAPTIVE ATTACK RESULTS"
    )

    print(
        "=" * 150
    )

    print(
        f"{'Attack':>10}"
        f"{'Eps':>8}"
        f"{'No defense':>14}"
        f"{'Oblivious':>14}"
        f"{'Adaptive':>14}"
        f"{'Obl rec %':>12}"
        f"{'Adp rec %':>12}"
        f"{'Adp-Obl':>12}"
        f"{'mAP50-95':>13}"
    )

    for row in rows:
        print(
            f"{str(row['attack']).upper():>10}"

            f"{str(row['epsilon_pixels']) + '/255':>8}"

            f"{float(row['no_defense_attacked_mAP50']):>14.4f}"

            f"{float(row['oblivious_tnorm_mAP50']):>14.4f}"

            f"{float(row['adaptive_tnorm_mAP50']):>14.4f}"

            f"{percent_text(row['oblivious_recovery_mAP50']):>12}"

            f"{percent_text(row['adaptive_recovery_mAP50']):>12}"

            f"{float(row['adaptive_minus_oblivious_mAP50']):>12.4f}"

            f"{float(row['adaptive_tnorm_mAP50-95']):>13.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    check_environment()
    prepare_directories()
    seed_everything(SEED)

    config = (
        load_fixed_tnorm_config()
    )

    print(
        "=" * 124
    )

    print(
        "ADAPTIVE WHITE-BOX ATTACKS "
        "AGAINST PRODUCT T-NORM -> YOLO"
    )

    print(
        "=" * 124
    )

    print(
        f"Модель: {MODEL_PATH}"
    )

    print(
        f"Датасет: {DATA_YAML}"
    )

    print(
        "Split: test"
    )

    print(
        "GPU: "
        f"{torch.cuda.get_device_name(DEVICE_INDEX)}"
    )

    print(
        f"Защита: "
        f"{config.identifier}"
    )

    print(
        f"FGSM/PGD epsilon: "
        f"{ATTACK_EPSILONS}/255"
    )

    print(
        f"PGD: steps={PGD_STEPS}, "
        f"alpha_ratio={PGD_ALPHA_RATIO}, "
        f"random_start={PGD_RANDOM_START}"
    )

    print(
        "Gradient checkpointing: "
        f"{USE_GRADIENT_CHECKPOINTING}"
    )

    verify_filter_equivalence(
        config
    )

    (
        references,
        oblivious_results,
        final_identifier,
    ) = load_final_test_context(
        config
    )

    print(
        "Контрольная T-норма "
        "из final_test: "
        f"{final_identifier}"
    )

    evaluation_model = YOLO(
        str(MODEL_PATH),
        verbose=False,
    )

    adaptive_results: dict[
        str,
        dict[str, Any],
    ] = {}

    for attack in (
        "fgsm",
        "pgd",
    ):
        for epsilon in (
            ATTACK_EPSILONS
        ):
            key = (
                f"{attack}_"
                f"{epsilon}"
            )

            adaptive_results[key] = (
                evaluate_adaptive_condition(
                    model=(
                        evaluation_model
                    ),
                    config=config,
                    attack=attack,
                    epsilon=epsilon,
                )
            )

    summary_rows = (
        build_summary_rows(
            adaptive_results,
            references,
            oblivious_results,
        )
    )

    class_rows = (
        build_class_rows(
            adaptive_results,
            references,
            oblivious_results,
        )
    )

    summary_csv_path = (
        OUTPUT_DIR
        / "adaptive_attacks_summary.csv"
    )

    classes_csv_path = (
        OUTPUT_DIR
        / "adaptive_attacks_classes.csv"
    )

    results_json_path = (
        OUTPUT_DIR
        / "adaptive_attacks_results.json"
    )

    write_csv(
        summary_csv_path,
        summary_rows,
    )

    write_csv(
        classes_csv_path,
        class_rows,
    )

    write_json(
        results_json_path,
        {
            "script_version": (
                SCRIPT_VERSION
            ),

            "split": "test",

            "model": str(
                MODEL_PATH
            ),

            "dataset": str(
                DATA_YAML
            ),

            "selection_source": str(
                SELECTION_PATH
            ),

            "oblivious_source": str(
                FINAL_TEST_PATH
            ),

            "fixed_config": asdict(
                config
            ),

            "evaluation_parameters": {
                "image_size": (
                    IMAGE_SIZE
                ),

                "batch_size": (
                    BATCH_SIZE
                ),

                "device_index": (
                    DEVICE_INDEX
                ),

                "workers": (
                    WORKERS
                ),

                "seed": (
                    SEED
                ),

                "attack_epsilons": (
                    ATTACK_EPSILONS
                ),

                "pgd_steps": (
                    PGD_STEPS
                ),

                "pgd_alpha_ratio": (
                    PGD_ALPHA_RATIO
                ),

                "pgd_random_start": (
                    PGD_RANDOM_START
                ),

                "gradient_checkpointing": (
                    USE_GRADIENT_CHECKPOINTING
                ),
            },

            "adaptive_results": (
                adaptive_results
            ),

            "summary_rows": (
                summary_rows
            ),

            "class_rows": (
                class_rows
            ),
        },
    )

    print_summary(
        summary_rows
    )

    print(
        "\n"
        + "=" * 150
    )

    print(
        "ADAPTIVE ATTACK "
        "EVALUATION COMPLETED"
    )

    print(
        "=" * 150
    )

    print(
        "Общая таблица:\n"
        f"{summary_csv_path}"
    )

    print(
        "\nМетрики по классам:\n"
        f"{classes_csv_path}"
    )

    print(
        "\nПолный JSON:\n"
        f"{results_json_path}"
    )


if __name__ == "__main__":
    main()