from __future__ import annotations

import csv
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator

from evaluate_fgsm import (
    FGSMValidator,
    calculate_ssim_per_image,
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

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "defenses"
    / "tnorm_tuning_val"
)

REFERENCE_DIR = OUTPUT_DIR / "reference"

CANDIDATES_DIR = OUTPUT_DIR / "candidates"

ULTRALYTICS_RUNS_DIR = OUTPUT_DIR / "runs"


# ============================================================
# ПАРАМЕТРЫ ОЦЕНКИ
# ============================================================

IMAGE_SIZE = 1280
BATCH_SIZE = 2
DEVICE_INDEX = 0
WORKERS = 4

SEED = 2026

ATTACK_EPSILONS = [
    1,
    2,
]

# Максимально допустимое относительное падение
# качества на чистой validation-выборке.
MAX_CLEAN_RELATIVE_DROP = 0.03


# ============================================================
# ГРУБЫЙ ПОИСК
# ============================================================

COARSE_TNORMS = [
    "godel",
    "product",
    "lukasiewicz",
]

COARSE_WINDOW_SIZES = [
    3,
    5,
]

COARSE_SIGMA_COLOR_PIXELS = [
    6,
    10,
    16,
    24,
]

COARSE_STRENGTH = 0.70
COARSE_CONFIDENCE_GAMMA = 1.50
COARSE_SIGMA_SPATIAL = 1.00


# ============================================================
# УТОЧНЯЮЩИЙ ПОИСК
# ============================================================

REFINE_TOP_K = 2

REFINE_STRENGTHS = [
    0.45,
    0.70,
    0.90,
]

REFINE_CONFIDENCE_GAMMAS = [
    1.00,
    1.50,
    2.00,
]


# ============================================================
# СТРУКТУРА КОНФИГУРАЦИИ
# ============================================================

@dataclass(frozen=True)
class FilterConfig:
    tnorm: str
    window_size: int
    sigma_color_pixels: float
    sigma_spatial: float
    strength: float
    confidence_gamma: float

    @property
    def radius(self) -> int:
        return self.window_size // 2

    @property
    def sigma_color(self) -> float:
        return self.sigma_color_pixels / 255.0

    @property
    def identifier(self) -> str:
        def number_slug(value: float) -> str:
            text = f"{value:.4f}".rstrip("0").rstrip(".")
            return text.replace(".", "p")

        return (
            f"{self.tnorm}"
            f"_w{self.window_size}"
            f"_sc{number_slug(self.sigma_color_pixels)}"
            f"_ss{number_slug(self.sigma_spatial)}"
            f"_st{number_slug(self.strength)}"
            f"_g{number_slug(self.confidence_gamma)}"
        )


# ============================================================
# ПРОВЕРКИ
# ============================================================

def check_environment() -> None:
    required_paths = [
        DATA_YAML,
        MODEL_PATH,
        PROJECT_DIR / "evaluate_fgsm.py",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Не найден необходимый файл:\n{path}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна."
        )

    print("=" * 96)
    print("НАСТРОЙКА T-НОРМОВОЙ ЗАЩИТЫ НА VALIDATION")
    print("=" * 96)
    print(f"Модель: {MODEL_PATH}")
    print(f"Датасет: {DATA_YAML}")
    print("Split: val")
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(DEVICE_INDEX)}"
    )
    print(f"Размер изображения: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(
        f"FGSM epsilon: "
        f"{ATTACK_EPSILONS} / 255"
    )
    print(
        "Максимальное относительное падение "
        f"clean mAP: {MAX_CLEAN_RELATIVE_DROP * 100:.1f}%"
    )


def prepare_directories() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CANDIDATES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ULTRALYTICS_RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# JSON
# ============================================================

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


# ============================================================
# T-НОРМЫ
# ============================================================

def apply_tnorm(
    first: torch.Tensor,
    second: torch.Tensor,
    tnorm_name: str,
) -> torch.Tensor:
    if tnorm_name == "godel":
        return torch.minimum(
            first,
            second,
        )

    if tnorm_name == "product":
        return first * second

    if tnorm_name == "lukasiewicz":
        return torch.clamp(
            first + second - 1.0,
            min=0.0,
            max=1.0,
        )

    raise ValueError(
        f"Неизвестная T-норма: {tnorm_name}"
    )


# ============================================================
# НАСТРАИВАЕМЫЙ T-НОРМОВЫЙ ФИЛЬТР
# ============================================================

@torch.no_grad()
def tunable_tnorm_filter(
    image: torch.Tensor,
    config: FilterConfig,
) -> torch.Tensor:
    original_dtype = image.dtype

    image_float = image.float()

    batch_size, _, height, width = (
        image_float.shape
    )

    radius = config.radius

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
        device=image.device,
        dtype=torch.float32,
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

            start_y = radius + offset_y
            start_x = radius + offset_x

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
                / max(
                    config.sigma_color,
                    1e-8,
                )
            )

            squared_distance = (
                offset_x**2
                + offset_y**2
            )

            spatial_value = math.exp(
                -squared_distance
                / (
                    2.0
                    * config.sigma_spatial**2
                )
            )

            spatial_membership = (
                torch.full_like(
                    color_membership,
                    spatial_value,
                )
            )

            combined_weight = apply_tnorm(
                first=color_membership,
                second=spatial_membership,
                tnorm_name=config.tnorm,
            )

            weighted_sum += (
                combined_weight
                * neighbour
            )

            weight_sum += combined_weight

            maximum_spatial_sum += (
                spatial_value
            )

    local_estimate = (
        weighted_sum
        / weight_sum.clamp_min(1e-8)
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
        config.confidence_gamma
    )

    blend = (
        config.strength
        * confidence
    )

    filtered = (
        image_float
        + blend
        * (
            local_estimate
            - image_float
        )
    )

    return (
        filtered
        .clamp(0.0, 1.0)
        .to(original_dtype)
    )


# ============================================================
# МЕТРИКИ ИЗОБРАЖЕНИЯ
# ============================================================

def calculate_psnr_sum(
    clean: torch.Tensor,
    defended: torch.Tensor,
) -> tuple[float, int]:
    difference = (
        clean.float()
        - defended.float()
    )

    mse = (
        difference
        .pow(2)
        .flatten(start_dim=1)
        .mean(dim=1)
    )

    valid_mask = mse > 0

    if not valid_mask.any():
        return 0.0, 0

    psnr = (
        10.0
        * torch.log10(
            1.0
            / mse[valid_mask]
        )
    )

    return (
        float(psnr.sum().item()),
        int(psnr.numel()),
    )


# ============================================================
# VALIDATOR С НАСТРАИВАЕМЫМ ФИЛЬТРОМ
# ============================================================

class TunableTNormValidator(
    FGSMValidator,
):
    filter_config = FilterConfig(
        tnorm="godel",
        window_size=3,
        sigma_color_pixels=10.0,
        sigma_spatial=1.0,
        strength=0.7,
        confidence_gamma=1.5,
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

        self.defense_change_mae_sum = 0.0
        self.defense_change_linf_sum = 0.0

        self.defended_clean_mae_sum = 0.0

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
                current_psnr_sum,
                current_psnr_count,
            ) = calculate_psnr_sum(
                clean=clean_float,
                defended=defended_float,
            )

            self.psnr_sum += current_psnr_sum
            self.psnr_count += current_psnr_count

            ssim = calculate_ssim_per_image(
                clean=clean_float,
                attacked=defended_float,
            )

            defense_change = (
                defended_float
                - attacked_float
            )

            defended_clean_difference = (
                defended_float
                - clean_float
            )

            defense_change_mae = (
                defense_change
                .abs()
                .flatten(start_dim=1)
                .mean(dim=1)
                * 255.0
            )

            defense_change_linf = (
                defense_change
                .abs()
                .flatten(start_dim=1)
                .amax(dim=1)
                * 255.0
            )

            defended_clean_mae = (
                defended_clean_difference
                .abs()
                .flatten(start_dim=1)
                .mean(dim=1)
                * 255.0
            )

            self.ssim_sum += float(
                ssim.sum().item()
            )

            self.defense_change_mae_sum += float(
                defense_change_mae
                .sum()
                .item()
            )

            self.defense_change_linf_sum += float(
                defense_change_linf
                .sum()
                .item()
            )

            self.defended_clean_mae_sum += float(
                defended_clean_mae
                .sum()
                .item()
            )

            self.defense_image_count += (
                batch_size
            )

    def preprocess(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        # Вызывается базовый preprocess DetectionValidator,
        # а не FGSMValidator.preprocess, чтобы самостоятельно
        # разместить защиту между атакой и инференсом.
        batch = DetectionValidator.preprocess(
            self,
            batch,
        )

        if self.current_epsilon_pixels == 0:
            clean = batch["img"].detach()
            attacked = clean

            self.collect_image_quality(
                clean=clean,
                attacked=attacked,
            )

        else:
            (
                clean,
                attacked,
                clean_loss,
            ) = self.calculate_fgsm(batch)

            self.clean_loss_sum += clean_loss
            self.loss_batch_count += 1

            self.collect_image_quality(
                clean=clean,
                attacked=attacked,
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        start_time = time.perf_counter()

        defended = tunable_tnorm_filter(
            image=attacked,
            config=type(self).filter_config,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        self.defense_seconds += (
            time.perf_counter()
            - start_time
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
                    self.defense_change_mae_sum
                    / self.defense_image_count
                ),

                "defense_change_linf_pixels": (
                    self.defense_change_linf_sum
                    / self.defense_image_count
                ),

                "defended_vs_clean_mae_pixels": (
                    self.defended_clean_mae_sum
                    / self.defense_image_count
                ),
            }

        type(self).last_defense_quality = (
            quality
        )


# ============================================================
# ИЗВЛЕЧЕНИЕ МЕТРИК
# ============================================================

def value_to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())

    return float(value)


def extract_overall_metrics(
    metrics: Any,
) -> dict[str, float]:
    return {
        "precision": value_to_float(
            metrics.results_dict[
                "metrics/precision(B)"
            ]
        ),

        "recall": value_to_float(
            metrics.results_dict[
                "metrics/recall(B)"
            ]
        ),

        "mAP50": value_to_float(
            metrics.results_dict[
                "metrics/mAP50(B)"
            ]
        ),

        "mAP50-95": value_to_float(
            metrics.results_dict[
                "metrics/mAP50-95(B)"
            ]
        ),

        "fitness": value_to_float(
            metrics.results_dict[
                "fitness"
            ]
        ),
    }


def extract_class_metrics(
    metrics: Any,
) -> list[dict[str, Any]]:
    class_ids = np.asarray(
        metrics.box.ap_class_index,
        dtype=int,
    )

    precision = np.asarray(
        metrics.box.p,
        dtype=float,
    )

    recall = np.asarray(
        metrics.box.r,
        dtype=float,
    )

    f1 = np.asarray(
        metrics.box.f1,
        dtype=float,
    )

    ap50 = np.asarray(
        metrics.box.ap50,
        dtype=float,
    )

    ap50_95 = np.asarray(
        metrics.box.ap,
        dtype=float,
    )

    rows: list[dict[str, Any]] = []

    for metric_index, class_id in enumerate(
        class_ids
    ):
        rows.append(
            {
                "class_id": int(class_id),

                "class_name": (
                    metrics.names[
                        int(class_id)
                    ]
                ),

                "precision": float(
                    precision[metric_index]
                ),

                "recall": float(
                    recall[metric_index]
                ),

                "f1": float(
                    f1[metric_index]
                ),

                "mAP50": float(
                    ap50[metric_index]
                ),

                "mAP50-95": float(
                    ap50_95[metric_index]
                ),
            }
        )

    return rows


# ============================================================
# REFERENCE: БЕЗ ЗАЩИТЫ
# ============================================================

def evaluate_reference_condition(
    evaluation_model: YOLO,
    epsilon_pixels: int,
) -> dict[str, Any]:
    condition_name = (
        "clean"
        if epsilon_pixels == 0
        else f"fgsm_eps_{epsilon_pixels}_255"
    )

    result_path = (
        REFERENCE_DIR
        / f"{condition_name}.json"
    )

    if result_path.is_file():
        print(
            f"Reference загружен: {condition_name}"
        )

        return read_json(result_path)

    current_seed = (
        SEED
        + epsilon_pixels * 100
    )

    torch.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)
    np.random.seed(current_seed)

    FGSMValidator.epsilon_pixels = (
        epsilon_pixels
    )

    FGSMValidator.last_quality = {}

    print("\n" + "=" * 96)
    print(
        f"REFERENCE VAL: "
        f"{condition_name.upper()}"
    )
    print("=" * 96)

    metrics = evaluation_model.val(
        validator=FGSMValidator,

        data=str(DATA_YAML),
        split="val",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE_INDEX,
        workers=WORKERS,

        rect=True,

        plots=False,
        save_json=False,

        project=str(
            ULTRALYTICS_RUNS_DIR
        ),

        name=f"reference_{condition_name}",
        exist_ok=True,

        verbose=False,
    )

    result = {
        "condition": condition_name,
        "epsilon_pixels": epsilon_pixels,
        "seed": current_seed,

        "overall": extract_overall_metrics(
            metrics
        ),

        "attack_quality": dict(
            FGSMValidator.last_quality
        ),

        "classes": extract_class_metrics(
            metrics
        ),
    }

    write_json(
        result_path,
        result,
    )

    gc.collect()
    torch.cuda.empty_cache()

    return result


def load_or_create_reference_metrics(
    evaluation_model: YOLO,
) -> dict[str, Any]:
    references: dict[str, Any] = {}

    clean_result = (
        evaluate_reference_condition(
            evaluation_model=evaluation_model,
            epsilon_pixels=0,
        )
    )

    references["clean"] = clean_result

    for epsilon_pixels in ATTACK_EPSILONS:
        result = evaluate_reference_condition(
            evaluation_model=evaluation_model,
            epsilon_pixels=epsilon_pixels,
        )

        references[
            f"fgsm_{epsilon_pixels}"
        ] = result

    write_json(
        REFERENCE_DIR
        / "reference_results.json",
        references,
    )

    return references


# ============================================================
# ОЦЕНКА КОНФИГУРАЦИИ
# ============================================================

def evaluate_defended_condition(
    evaluation_model: YOLO,
    config: FilterConfig,
    epsilon_pixels: int,
) -> dict[str, Any]:
    condition_name = (
        "clean"
        if epsilon_pixels == 0
        else f"fgsm_eps_{epsilon_pixels}_255"
    )

    candidate_directory = (
        CANDIDATES_DIR
        / config.identifier
    )

    candidate_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        candidate_directory
        / f"{condition_name}.json"
    )

    if result_path.is_file():
        return read_json(result_path)

    current_seed = (
        SEED
        + epsilon_pixels * 100
    )

    torch.manual_seed(current_seed)
    torch.cuda.manual_seed_all(current_seed)
    np.random.seed(current_seed)

    TunableTNormValidator.epsilon_pixels = (
        epsilon_pixels
    )

    TunableTNormValidator.filter_config = (
        config
    )

    TunableTNormValidator.last_quality = {}
    TunableTNormValidator.last_defense_quality = {}

    print("\n" + "-" * 96)
    print(
        f"{config.identifier} | "
        f"{condition_name}"
    )
    print("-" * 96)

    metrics = evaluation_model.val(
        validator=TunableTNormValidator,

        data=str(DATA_YAML),
        split="val",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE_INDEX,
        workers=WORKERS,

        rect=True,

        plots=False,
        save_json=False,

        project=str(
            ULTRALYTICS_RUNS_DIR
        ),

        name=(
            f"{config.identifier}"
            f"_{condition_name}"
        ),

        exist_ok=True,

        verbose=False,
    )

    result = {
        "condition": condition_name,
        "epsilon_pixels": epsilon_pixels,
        "seed": current_seed,

        "config": asdict(config),

        "overall": extract_overall_metrics(
            metrics
        ),

        "attack_quality": dict(
            TunableTNormValidator.last_quality
        ),

        "defense_quality": dict(
            TunableTNormValidator
            .last_defense_quality
        ),

        "classes": extract_class_metrics(
            metrics
        ),
    }

    write_json(
        result_path,
        result,
    )

    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# ОЦЕНКА КАЧЕСТВА КОНФИГУРАЦИИ
# ============================================================

def relative_drop(
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


def recovery_rate(
    clean_value: float,
    attacked_value: float,
    defended_value: float,
) -> float | None:
    available_recovery = (
        clean_value
        - attacked_value
    )

    if available_recovery <= 1e-12:
        return None

    return (
        defended_value
        - attacked_value
    ) / available_recovery


def mean_valid(
    values: list[float | None],
) -> float:
    valid_values = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(value)
    ]

    if not valid_values:
        return float("-inf")

    return float(
        np.mean(valid_values)
    )


def evaluate_candidate(
    evaluation_model: YOLO,
    config: FilterConfig,
    references: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    candidate_directory = (
        CANDIDATES_DIR
        / config.identifier
    )

    candidate_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        candidate_directory
        / "candidate_summary.json"
    )

    if summary_path.is_file():
        return read_json(summary_path)

    clean_result = evaluate_defended_condition(
        evaluation_model=evaluation_model,
        config=config,
        epsilon_pixels=0,
    )

    clean_reference = (
        references["clean"]["overall"]
    )

    clean_defended = (
        clean_result["overall"]
    )

    clean_drop_map50 = relative_drop(
        reference=clean_reference["mAP50"],
        evaluated=clean_defended["mAP50"],
    )

    clean_drop_map50_95 = relative_drop(
        reference=clean_reference[
            "mAP50-95"
        ],
        evaluated=clean_defended[
            "mAP50-95"
        ],
    )

    feasible = (
        clean_drop_map50
        <= MAX_CLEAN_RELATIVE_DROP
        and clean_drop_map50_95
        <= MAX_CLEAN_RELATIVE_DROP
    )

    attack_results: dict[str, Any] = {}

    if feasible:
        for epsilon_pixels in ATTACK_EPSILONS:
            condition_result = (
                evaluate_defended_condition(
                    evaluation_model=(
                        evaluation_model
                    ),
                    config=config,
                    epsilon_pixels=(
                        epsilon_pixels
                    ),
                )
            )

            attack_results[
                str(epsilon_pixels)
            ] = condition_result

    recovery_map50_values: list[
        float | None
    ] = []

    recovery_map50_95_values: list[
        float | None
    ] = []

    if feasible:
        for epsilon_pixels in ATTACK_EPSILONS:
            reference_attacked = (
                references[
                    f"fgsm_{epsilon_pixels}"
                ]["overall"]
            )

            defended_attacked = (
                attack_results[
                    str(epsilon_pixels)
                ]["overall"]
            )

            recovery_map50_values.append(
                recovery_rate(
                    clean_value=(
                        clean_reference["mAP50"]
                    ),
                    attacked_value=(
                        reference_attacked["mAP50"]
                    ),
                    defended_value=(
                        defended_attacked["mAP50"]
                    ),
                )
            )

            recovery_map50_95_values.append(
                recovery_rate(
                    clean_value=(
                        clean_reference[
                            "mAP50-95"
                        ]
                    ),
                    attacked_value=(
                        reference_attacked[
                            "mAP50-95"
                        ]
                    ),
                    defended_value=(
                        defended_attacked[
                            "mAP50-95"
                        ]
                    ),
                )
            )

        mean_recovery_map50 = mean_valid(
            recovery_map50_values
        )

        mean_recovery_map50_95 = mean_valid(
            recovery_map50_95_values
        )

        clean_penalty = max(
            clean_drop_map50,
            clean_drop_map50_95,
        )

        score = (
            0.60
            * mean_recovery_map50
            + 0.40
            * mean_recovery_map50_95
            - 0.10
            * clean_penalty
        )

    else:
        mean_recovery_map50 = None
        mean_recovery_map50_95 = None
        score = float("-inf")

    summary = {
        "stage": stage,
        "identifier": config.identifier,
        "config": asdict(config),

        "feasible": feasible,

        "clean_reference": clean_reference,
        "clean_defended": clean_defended,

        "clean_relative_drop": {
            "mAP50": clean_drop_map50,
            "mAP50-95": (
                clean_drop_map50_95
            ),
        },

        "attack_results": attack_results,

        "recovery_rates": {
            "mAP50": (
                recovery_map50_values
            ),

            "mAP50-95": (
                recovery_map50_95_values
            ),

            "mean_mAP50": (
                mean_recovery_map50
            ),

            "mean_mAP50-95": (
                mean_recovery_map50_95
            ),
        },

        "score": (
            score
            if math.isfinite(score)
            else None
        ),
    }

    write_json(
        summary_path,
        summary,
    )

    print(
        f"\n{config.identifier}: "
        f"feasible={feasible}, "
        f"clean drop mAP50="
        f"{clean_drop_map50 * 100:.2f}%, "
        f"score="
        f"{score if math.isfinite(score) else 'reject'}"
    )

    return summary


# ============================================================
# ГЕНЕРАЦИЯ КОНФИГУРАЦИЙ
# ============================================================

def create_coarse_configs() -> list[FilterConfig]:
    configs: list[FilterConfig] = []

    for tnorm_name in COARSE_TNORMS:
        for window_size in (
            COARSE_WINDOW_SIZES
        ):
            for sigma_color_pixels in (
                COARSE_SIGMA_COLOR_PIXELS
            ):
                configs.append(
                    FilterConfig(
                        tnorm=tnorm_name,

                        window_size=(
                            window_size
                        ),

                        sigma_color_pixels=(
                            float(
                                sigma_color_pixels
                            )
                        ),

                        sigma_spatial=(
                            COARSE_SIGMA_SPATIAL
                        ),

                        strength=(
                            COARSE_STRENGTH
                        ),

                        confidence_gamma=(
                            COARSE_CONFIDENCE_GAMMA
                        ),
                    )
                )

    return configs


def create_refine_configs(
    best_coarse_results: list[
        dict[str, Any]
    ],
) -> list[FilterConfig]:
    configs: list[FilterConfig] = []

    seen_identifiers: set[str] = set()

    for result in best_coarse_results:
        base = result["config"]

        for strength in REFINE_STRENGTHS:
            for confidence_gamma in (
                REFINE_CONFIDENCE_GAMMAS
            ):
                config = FilterConfig(
                    tnorm=base["tnorm"],

                    window_size=int(
                        base["window_size"]
                    ),

                    sigma_color_pixels=float(
                        base[
                            "sigma_color_pixels"
                        ]
                    ),

                    sigma_spatial=float(
                        base["sigma_spatial"]
                    ),

                    strength=float(
                        strength
                    ),

                    confidence_gamma=float(
                        confidence_gamma
                    ),
                )

                if (
                    config.identifier
                    in seen_identifiers
                ):
                    continue

                seen_identifiers.add(
                    config.identifier
                )

                configs.append(config)

    return configs


# ============================================================
# РАНЖИРОВАНИЕ
# ============================================================

def result_score(
    result: dict[str, Any],
) -> float:
    score = result.get("score")

    if score is None:
        return float("-inf")

    return float(score)


def rank_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    feasible_results = [
        result
        for result in results
        if result.get("feasible", False)
        and result_score(result)
        != float("-inf")
    ]

    return sorted(
        feasible_results,
        key=result_score,
        reverse=True,
    )


# ============================================================
# СОХРАНЕНИЕ ТАБЛИЦ
# ============================================================

def save_results_csv(
    results: list[dict[str, Any]],
) -> None:
    csv_path = (
        OUTPUT_DIR
        / "tnorm_tuning_results.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "rank",
            "stage",
            "identifier",
            "tnorm",
            "window_size",
            "sigma_color_pixels",
            "sigma_spatial",
            "strength",
            "confidence_gamma",
            "feasible",
            "score",
            "clean_mAP50",
            "clean_mAP50-95",
            "clean_drop_mAP50",
            "clean_drop_mAP50-95",
            "fgsm1_mAP50",
            "fgsm1_mAP50-95",
            "fgsm2_mAP50",
            "fgsm2_mAP50-95",
            "mean_recovery_mAP50",
            "mean_recovery_mAP50-95",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        ranked = rank_results(results)

        rank_map = {
            result["identifier"]: index
            for index, result
            in enumerate(
                ranked,
                start=1,
            )
        }

        for result in results:
            config = result["config"]

            attack_results = result.get(
                "attack_results",
                {},
            )

            fgsm1 = attack_results.get("1")
            fgsm2 = attack_results.get("2")

            writer.writerow(
                {
                    "rank": rank_map.get(
                        result["identifier"]
                    ),

                    "stage": result["stage"],

                    "identifier": result[
                        "identifier"
                    ],

                    "tnorm": config["tnorm"],

                    "window_size": config[
                        "window_size"
                    ],

                    "sigma_color_pixels": (
                        config[
                            "sigma_color_pixels"
                        ]
                    ),

                    "sigma_spatial": config[
                        "sigma_spatial"
                    ],

                    "strength": config[
                        "strength"
                    ],

                    "confidence_gamma": (
                        config[
                            "confidence_gamma"
                        ]
                    ),

                    "feasible": result[
                        "feasible"
                    ],

                    "score": result.get(
                        "score"
                    ),

                    "clean_mAP50": result[
                        "clean_defended"
                    ]["mAP50"],

                    "clean_mAP50-95": result[
                        "clean_defended"
                    ]["mAP50-95"],

                    "clean_drop_mAP50": result[
                        "clean_relative_drop"
                    ]["mAP50"],

                    "clean_drop_mAP50-95": result[
                        "clean_relative_drop"
                    ]["mAP50-95"],

                    "fgsm1_mAP50": (
                        fgsm1["overall"]["mAP50"]
                        if fgsm1
                        else None
                    ),

                    "fgsm1_mAP50-95": (
                        fgsm1["overall"][
                            "mAP50-95"
                        ]
                        if fgsm1
                        else None
                    ),

                    "fgsm2_mAP50": (
                        fgsm2["overall"]["mAP50"]
                        if fgsm2
                        else None
                    ),

                    "fgsm2_mAP50-95": (
                        fgsm2["overall"][
                            "mAP50-95"
                        ]
                        if fgsm2
                        else None
                    ),

                    "mean_recovery_mAP50": (
                        result["recovery_rates"][
                            "mean_mAP50"
                        ]
                    ),

                    "mean_recovery_mAP50-95": (
                        result["recovery_rates"][
                            "mean_mAP50-95"
                        ]
                    ),
                }
            )


# ============================================================
# ВЫВОД
# ============================================================

def print_top_results(
    ranked_results: list[
        dict[str, Any]
    ],
    limit: int = 10,
) -> None:
    print("\n" + "=" * 120)
    print("ЛУЧШИЕ КОНФИГУРАЦИИ")
    print("=" * 120)

    print(
        f"{'Rank':>6}"
        f"{'T-norm':>15}"
        f"{'Win':>7}"
        f"{'Sigma':>9}"
        f"{'Strength':>11}"
        f"{'Gamma':>9}"
        f"{'Score':>11}"
        f"{'Clean AP50':>13}"
        f"{'Rec AP50':>12}"
        f"{'Rec AP':>12}"
    )

    for rank, result in enumerate(
        ranked_results[:limit],
        start=1,
    ):
        config = result["config"]

        print(
            f"{rank:>6}"
            f"{config['tnorm']:>15}"
            f"{config['window_size']:>7}"
            f"{config['sigma_color_pixels']:>9.1f}"
            f"{config['strength']:>11.2f}"
            f"{config['confidence_gamma']:>9.2f}"
            f"{result_score(result):>11.4f}"
            f"{result['clean_defended']['mAP50']:>13.4f}"
            f"{result['recovery_rates']['mean_mAP50']:>12.4f}"
            f"{result['recovery_rates']['mean_mAP50-95']:>12.4f}"
        )


# ============================================================
# ЗАПУСК
# ============================================================

def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    check_environment()
    prepare_directories()

    evaluation_model = YOLO(
        str(MODEL_PATH),
        verbose=False,
    )

    references = (
        load_or_create_reference_metrics(
            evaluation_model
        )
    )

    print("\nКонтрольные validation-метрики:")

    print(
        "clean: "
        f"mAP50="
        f"{references['clean']['overall']['mAP50']:.4f}, "
        f"mAP50-95="
        f"{references['clean']['overall']['mAP50-95']:.4f}"
    )

    for epsilon_pixels in ATTACK_EPSILONS:
        reference = references[
            f"fgsm_{epsilon_pixels}"
        ]["overall"]

        print(
            f"FGSM {epsilon_pixels}/255: "
            f"mAP50={reference['mAP50']:.4f}, "
            f"mAP50-95="
            f"{reference['mAP50-95']:.4f}"
        )

    all_results_by_identifier: dict[
        str,
        dict[str, Any],
    ] = {}

    print("\n" + "=" * 96)
    print("ЭТАП 1: ГРУБЫЙ ПОИСК")
    print("=" * 96)

    coarse_configs = (
        create_coarse_configs()
    )

    print(
        f"Количество грубых конфигураций: "
        f"{len(coarse_configs)}"
    )

    for index, config in enumerate(
        coarse_configs,
        start=1,
    ):
        print(
            f"\nГрубый поиск "
            f"{index}/{len(coarse_configs)}"
        )

        result = evaluate_candidate(
            evaluation_model=evaluation_model,
            config=config,
            references=references,
            stage="coarse",
        )

        all_results_by_identifier[
            config.identifier
        ] = result

    coarse_ranked = rank_results(
        list(
            all_results_by_identifier.values()
        )
    )

    if not coarse_ranked:
        raise RuntimeError(
            "Ни одна грубая конфигурация не прошла "
            "ограничение качества на clean validation."
        )

    best_coarse_results = (
        coarse_ranked[:REFINE_TOP_K]
    )

    print_top_results(
        coarse_ranked,
        limit=10,
    )

    print("\n" + "=" * 96)
    print("ЭТАП 2: УТОЧНЕНИЕ ЛУЧШИХ КОНФИГУРАЦИЙ")
    print("=" * 96)

    refine_configs = create_refine_configs(
        best_coarse_results
    )

    print(
        f"Количество уточняющих конфигураций: "
        f"{len(refine_configs)}"
    )

    for index, config in enumerate(
        refine_configs,
        start=1,
    ):
        print(
            f"\nУточнение "
            f"{index}/{len(refine_configs)}"
        )

        if (
            config.identifier
            in all_results_by_identifier
        ):
            print(
                "Конфигурация уже оценена "
                "на грубом этапе."
            )
            continue

        result = evaluate_candidate(
            evaluation_model=evaluation_model,
            config=config,
            references=references,
            stage="refine",
        )

        all_results_by_identifier[
            config.identifier
        ] = result

    all_results = list(
        all_results_by_identifier.values()
    )

    ranked_results = rank_results(
        all_results
    )

    if not ranked_results:
        raise RuntimeError(
            "Не удалось выбрать допустимую "
            "конфигурацию."
        )

    best_result = ranked_results[0]

    save_results_csv(all_results)

    write_json(
        OUTPUT_DIR
        / "tnorm_tuning_results.json",
        {
            "split": "val",

            "model": str(MODEL_PATH),

            "dataset": str(DATA_YAML),

            "max_clean_relative_drop": (
                MAX_CLEAN_RELATIVE_DROP
            ),

            "attack_epsilons": (
                ATTACK_EPSILONS
            ),

            "reference": references,

            "results": all_results,
        },
    )

    write_json(
        OUTPUT_DIR
        / "best_tnorm_config.json",
        {
            "selection_split": "val",

            "selection_metric": (
                "0.60 * mean recovery mAP50 "
                "+ 0.40 * mean recovery mAP50-95 "
                "- clean penalty"
            ),

            "best": best_result,
        },
    )

    print_top_results(
        ranked_results,
        limit=10,
    )

    print("\n" + "=" * 120)
    print("НАСТРОЙКА ЗАВЕРШЕНА")
    print("=" * 120)

    best_config = best_result["config"]

    print("Лучшая конфигурация:")
    print(
        f"T-норма: "
        f"{best_config['tnorm']}"
    )
    print(
        f"Окно: "
        f"{best_config['window_size']}x"
        f"{best_config['window_size']}"
    )
    print(
        f"Sigma color: "
        f"{best_config['sigma_color_pixels']}/255"
    )
    print(
        f"Sigma spatial: "
        f"{best_config['sigma_spatial']}"
    )
    print(
        f"Strength: "
        f"{best_config['strength']}"
    )
    print(
        f"Confidence gamma: "
        f"{best_config['confidence_gamma']}"
    )
    print(
        f"Score: "
        f"{result_score(best_result):.6f}"
    )

    print(
        "\nЛучшая конфигурация сохранена:\n"
        f"{OUTPUT_DIR / 'best_tnorm_config.json'}"
    )

    print(
        "\nПолная таблица:\n"
        f"{OUTPUT_DIR / 'tnorm_tuning_results.csv'}"
    )

    print(
        "\nСкрипт не запускал финальную оценку "
        "на test."
    )


if __name__ == "__main__":
    main()