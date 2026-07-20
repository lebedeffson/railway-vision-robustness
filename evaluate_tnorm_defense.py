from __future__ import annotations

import csv
import gc
import json
import math
import shutil
import time
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
from evaluate_pgd import (
    PGDValidator,
    PGD_STEPS,
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

CLEAN_RESULTS_PATH = (
    PROJECT_DIR
    / "outputs"
    / "evaluation"
    / "clean_test_baseline"
    / "clean_test_metrics.json"
)

FGSM_RESULTS_PATH = (
    PROJECT_DIR
    / "outputs"
    / "attacks"
    / "fgsm"
    / "fgsm_results.json"
)

PGD_RESULTS_PATH = (
    PROJECT_DIR
    / "outputs"
    / "attacks"
    / "pgd"
    / "pgd_results.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "defenses"
    / "tnorm_screening"
)


# ============================================================
# ОБЩИЕ ПАРАМЕТРЫ
# ============================================================

IMAGE_SIZE = 1280
BATCH_SIZE = 2
DEVICE_INDEX = 0
WORKERS = 4

SEED = 2026

RESET_OUTPUT = True

ATTACK_EPSILON_PIXELS = 1

ATTACK_TYPES = [
    "clean",
    "fgsm",
    "pgd",
]

TNORM_TYPES = [
    "godel",
    "product",
    "lukasiewicz",
]


# ============================================================
# ПАРАМЕТРЫ T-НОРМОВОГО ФИЛЬТРА
# ============================================================

# Радиус 1 означает окно 3x3.
FILTER_RADIUS = 1

# Чувствительность к различиям яркости.
# Значения изображений находятся в диапазоне [0, 1].
SIGMA_COLOR = 10.0 / 255.0

# Пространственная чувствительность.
SIGMA_SPATIAL = 1.0

# Максимальная сила смешивания с локальной оценкой.
FILTER_STRENGTH = 0.70

# Чем больше значение, тем сильнее сохраняются границы.
CONFIDENCE_GAMMA = 1.50

EPSILON = 1e-8


# ============================================================
# ПРОВЕРКИ
# ============================================================

def check_environment() -> None:
    required_paths = [
        DATA_YAML,
        MODEL_PATH,
        CLEAN_RESULTS_PATH,
        FGSM_RESULTS_PATH,
        PGD_RESULTS_PATH,
        PROJECT_DIR / "evaluate_fgsm.py",
        PROJECT_DIR / "evaluate_pgd.py",
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

    print("=" * 88)
    print("ПРОВЕРКА T-НОРМОВОЙ ЗАЩИТЫ")
    print("=" * 88)
    print(f"Модель: {MODEL_PATH}")
    print(f"Датасет: {DATA_YAML}")
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(DEVICE_INDEX)}"
    )
    print(f"Размер изображения: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(
        f"Атаки: FGSM и PGD-{PGD_STEPS}, "
        f"epsilon={ATTACK_EPSILON_PIXELS}/255"
    )
    print(f"T-нормы: {TNORM_TYPES}")
    print(
        f"Окно фильтра: "
        f"{2 * FILTER_RADIUS + 1}x"
        f"{2 * FILTER_RADIUS + 1}"
    )
    print(
        f"Sigma color: {SIGMA_COLOR:.6f}"
    )
    print(
        f"Sigma spatial: {SIGMA_SPATIAL:.4f}"
    )
    print(
        f"Filter strength: {FILTER_STRENGTH:.4f}"
    )


def prepare_output_directory() -> None:
    if OUTPUT_DIR.exists():
        if not RESET_OUTPUT:
            raise FileExistsError(
                "Папка результатов уже существует:\n"
                f"{OUTPUT_DIR}"
            )

        print(
            "\nУдаляется предыдущая папка защиты:\n"
            f"{OUTPUT_DIR}"
        )

        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


# ============================================================
# РАБОТА С JSON
# ============================================================

def read_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def find_attack_result(
    data: dict[str, Any],
    epsilon_pixels: int,
) -> dict[str, Any]:
    for result in data["results"]:
        if int(result["epsilon_pixels"]) == epsilon_pixels:
            return result

    raise KeyError(
        f"В файле не найден epsilon={epsilon_pixels}/255"
    )


def load_reference_metrics() -> dict[str, dict[str, float]]:
    clean_data = read_json(
        CLEAN_RESULTS_PATH
    )

    fgsm_data = read_json(
        FGSM_RESULTS_PATH
    )

    pgd_data = read_json(
        PGD_RESULTS_PATH
    )

    fgsm_result = find_attack_result(
        fgsm_data,
        ATTACK_EPSILON_PIXELS,
    )

    pgd_result = find_attack_result(
        pgd_data,
        ATTACK_EPSILON_PIXELS,
    )

    return {
        "clean": {
            key: float(value)
            for key, value
            in clean_data["overall"].items()
        },
        "fgsm": {
            key: float(value)
            for key, value
            in fgsm_result["overall"].items()
        },
        "pgd": {
            key: float(value)
            for key, value
            in pgd_result["overall"].items()
        },
    }


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
# T-НОРМОВЫЙ АДАПТИВНЫЙ ФИЛЬТР
# ============================================================

@torch.no_grad()
def tnorm_adaptive_filter(
    image: torch.Tensor,
    tnorm_name: str,
) -> torch.Tensor:
    """
    T-нормовый локальный фильтр.

    Для каждого пикселя вычисляются две степени принадлежности:

    1. Цветовое сходство центрального пикселя и соседа.
    2. Пространственная близость соседа.

    Они объединяются выбранной T-нормой.

    Полученные веса формируют локальную оценку пикселя.
    Степень фильтрации адаптивно уменьшается возле границ.
    """

    original_dtype = image.dtype

    image_float = image.float()

    batch_size, channels, height, width = (
        image_float.shape
    )

    radius = FILTER_RADIUS

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

    maximum_weight_sum = 0.0

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
                / SIGMA_COLOR
            )

            squared_distance = (
                offset_x**2
                + offset_y**2
            )

            spatial_value = math.exp(
                -squared_distance
                / (
                    2.0
                    * SIGMA_SPATIAL**2
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
                tnorm_name=tnorm_name,
            )

            weighted_sum += (
                combined_weight
                * neighbour
            )

            weight_sum += combined_weight

            maximum_weight_sum += spatial_value

    local_estimate = (
        weighted_sum
        / weight_sum.clamp_min(EPSILON)
    )

    confidence = torch.clamp(
        weight_sum
        / (
            maximum_weight_sum
            + EPSILON
        ),
        min=0.0,
        max=1.0,
    )

    confidence = confidence.pow(
        CONFIDENCE_GAMMA
    )

    blend = (
        FILTER_STRENGTH
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
# МЕТРИКИ ЗАЩИТЫ
# ============================================================

def calculate_psnr_values(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[float, int]:
    difference = (
        first.float()
        - second.float()
    )

    mse = (
        difference
        .pow(2)
        .flatten(start_dim=1)
        .mean(dim=1)
    )

    nonzero_mask = mse > 0

    if not nonzero_mask.any():
        return 0.0, 0

    psnr = (
        10.0
        * torch.log10(
            1.0
            / mse[nonzero_mask]
        )
    )

    return (
        float(psnr.sum().item()),
        int(psnr.numel()),
    )


class TNormDefenseMixin:
    tnorm_name: str = "product"

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

        self.defended_clean_psnr_sum = 0.0
        self.defended_clean_psnr_count = 0

        self.defended_clean_ssim_sum = 0.0

        self.defense_change_linf_sum = 0.0
        self.defense_change_mae_sum = 0.0

        self.defended_clean_mae_sum = 0.0

    def apply_defense_and_collect(
        self,
        clean: torch.Tensor,
        attacked: torch.Tensor,
    ) -> torch.Tensor:
        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        start_time = time.perf_counter()

        defended = tnorm_adaptive_filter(
            image=attacked,
            tnorm_name=type(self).tnorm_name,
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        self.defense_seconds += (
            time.perf_counter()
            - start_time
        )

        with torch.no_grad():
            clean_float = clean.float()
            attacked_float = attacked.float()
            defended_float = defended.float()

            batch_size = int(
                defended.shape[0]
            )

            (
                psnr_sum,
                psnr_count,
            ) = calculate_psnr_values(
                clean_float,
                defended_float,
            )

            self.defended_clean_psnr_sum += (
                psnr_sum
            )

            self.defended_clean_psnr_count += (
                psnr_count
            )

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

            defense_linf = (
                defense_change
                .abs()
                .flatten(start_dim=1)
                .amax(dim=1)
                * 255.0
            )

            defense_mae = (
                defense_change
                .abs()
                .flatten(start_dim=1)
                .mean(dim=1)
                * 255.0
            )

            defended_clean_mae = (
                defended_clean_difference
                .abs()
                .flatten(start_dim=1)
                .mean(dim=1)
                * 255.0
            )

            self.defended_clean_ssim_sum += float(
                ssim.sum().item()
            )

            self.defense_change_linf_sum += float(
                defense_linf.sum().item()
            )

            self.defense_change_mae_sum += float(
                defense_mae.sum().item()
            )

            self.defended_clean_mae_sum += float(
                defended_clean_mae
                .sum()
                .item()
            )

            self.defense_image_count += batch_size

        return defended

    def finalize_metrics(self) -> None:
        super().finalize_metrics()

        if self.defense_image_count == 0:
            quality = {
                "tnorm": type(self).tnorm_name,
                "defense_ms_per_image": None,
                "defended_vs_clean_psnr": None,
                "defended_vs_clean_ssim": None,
                "defense_change_linf_pixels": None,
                "defense_change_mae_pixels": None,
                "defended_vs_clean_mae_pixels": None,
            }

        else:
            mean_psnr = (
                self.defended_clean_psnr_sum
                / self.defended_clean_psnr_count
                if self.defended_clean_psnr_count > 0
                else None
            )

            quality = {
                "tnorm": type(self).tnorm_name,

                "defense_ms_per_image": (
                    self.defense_seconds
                    / self.defense_image_count
                    * 1000.0
                ),

                "defended_vs_clean_psnr": (
                    mean_psnr
                ),

                "defended_vs_clean_ssim": (
                    self.defended_clean_ssim_sum
                    / self.defense_image_count
                ),

                "defense_change_linf_pixels": (
                    self.defense_change_linf_sum
                    / self.defense_image_count
                ),

                "defense_change_mae_pixels": (
                    self.defense_change_mae_sum
                    / self.defense_image_count
                ),

                "defended_vs_clean_mae_pixels": (
                    self.defended_clean_mae_sum
                    / self.defense_image_count
                ),
            }

        type(self).last_defense_quality = quality


# ============================================================
# FGSM + T-NОРМА
# ============================================================

class TNormFGSMValidator(
    TNormDefenseMixin,
    FGSMValidator,
):
    def preprocess(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
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

        defended = (
            self.apply_defense_and_collect(
                clean=clean,
                attacked=attacked,
            )
        )

        batch["img"] = defended

        return batch


# ============================================================
# PGD + T-НОРМА
# ============================================================

class TNormPGDValidator(
    TNormDefenseMixin,
    PGDValidator,
):
    def preprocess(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
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
                initial_loss,
                final_loss,
            ) = self.calculate_pgd(batch)

            self.initial_loss_sum += (
                initial_loss
            )

            self.final_loss_sum += (
                final_loss
            )

            self.loss_batch_count += 1

            self.collect_image_quality(
                clean=clean,
                attacked=attacked,
            )

        defended = (
            self.apply_defense_and_collect(
                clean=clean,
                attacked=attacked,
            )
        )

        batch["img"] = defended

        return batch


# ============================================================
# ИЗВЛЕЧЕНИЕ МЕТРИК YOLO
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
# ВОССТАНОВЛЕНИЕ КАЧЕСТВА
# ============================================================

def calculate_recovery(
    clean_value: float,
    attacked_value: float,
    defended_value: float,
) -> dict[str, float | None]:
    absolute_recovery = (
        defended_value
        - attacked_value
    )

    available_recovery = (
        clean_value
        - attacked_value
    )

    if available_recovery <= 0:
        recovery_rate = None
    else:
        recovery_rate = (
            absolute_recovery
            / available_recovery
        )

    return {
        "absolute_recovery": (
            absolute_recovery
        ),

        "recovery_rate": (
            recovery_rate
        ),
    }


# ============================================================
# ЗАПУСК ОДНОГО ЭКСПЕРИМЕНТА
# ============================================================

def evaluate_case(
    evaluation_model: YOLO,
    attack_type: str,
    tnorm_name: str,
    reference_metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if attack_type == "clean":
        validator_class = (
            TNormFGSMValidator
        )

        epsilon_pixels = 0

    elif attack_type == "fgsm":
        validator_class = (
            TNormFGSMValidator
        )

        epsilon_pixels = (
            ATTACK_EPSILON_PIXELS
        )

    elif attack_type == "pgd":
        validator_class = (
            TNormPGDValidator
        )

        epsilon_pixels = (
            ATTACK_EPSILON_PIXELS
        )

    else:
        raise ValueError(
            f"Неизвестная атака: {attack_type}"
        )

    validator_class.epsilon_pixels = (
        epsilon_pixels
    )

    validator_class.tnorm_name = (
        tnorm_name
    )

    validator_class.last_quality = {}
    validator_class.last_defense_quality = {}

    run_name = (
        f"{attack_type}"
        f"_eps_{epsilon_pixels}_255"
        f"_{tnorm_name}"
    )

    print("\n" + "=" * 88)
    print(
        f"АТАКА: {attack_type.upper()} | "
        f"EPSILON: {epsilon_pixels}/255 | "
        f"T-NORMA: {tnorm_name.upper()}"
    )
    print("=" * 88)

    metrics = evaluation_model.val(
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

        project=str(OUTPUT_DIR),
        name=run_name,
        exist_ok=False,

        verbose=True,
    )

    overall = extract_overall_metrics(
        metrics
    )

    classes = extract_class_metrics(
        metrics
    )

    attack_quality = dict(
        validator_class.last_quality
    )

    defense_quality = dict(
        validator_class.last_defense_quality
    )

    clean_reference = (
        reference_metrics["clean"]
    )

    attacked_reference = (
        reference_metrics[attack_type]
    )

    recovery_map50 = calculate_recovery(
        clean_value=clean_reference["mAP50"],
        attacked_value=attacked_reference["mAP50"],
        defended_value=overall["mAP50"],
    )

    recovery_map50_95 = calculate_recovery(
        clean_value=clean_reference["mAP50-95"],
        attacked_value=attacked_reference["mAP50-95"],
        defended_value=overall["mAP50-95"],
    )

    result = {
        "attack_type": attack_type,

        "epsilon_pixels": epsilon_pixels,

        "epsilon_normalized": (
            epsilon_pixels / 255.0
        ),

        "tnorm": tnorm_name,

        "filter_parameters": {
            "window_size": (
                2 * FILTER_RADIUS + 1
            ),

            "sigma_color": SIGMA_COLOR,

            "sigma_spatial": (
                SIGMA_SPATIAL
            ),

            "strength": FILTER_STRENGTH,

            "confidence_gamma": (
                CONFIDENCE_GAMMA
            ),
        },

        "reference_clean": (
            clean_reference
        ),

        "reference_attacked": (
            attacked_reference
        ),

        "defended": overall,

        "clean_delta": {
            "mAP50": (
                overall["mAP50"]
                - clean_reference["mAP50"]
            ),

            "mAP50-95": (
                overall["mAP50-95"]
                - clean_reference["mAP50-95"]
            ),
        },

        "recovery": {
            "mAP50": recovery_map50,

            "mAP50-95": (
                recovery_map50_95
            ),
        },

        "attack_quality": (
            attack_quality
        ),

        "defense_quality": (
            defense_quality
        ),

        "speed_ms_per_image": {
            key: value_to_float(value)
            for key, value
            in metrics.speed.items()
        },

        "classes": classes,
    }

    run_directory = (
        OUTPUT_DIR
        / run_name
    )

    result_path = (
        run_directory
        / "tnorm_result.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    class_csv_path = (
        run_directory
        / "tnorm_metrics_by_class.csv"
    )

    with class_csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "attack_type",
            "epsilon_pixels",
            "tnorm",
            "class_id",
            "class_name",
            "precision",
            "recall",
            "f1",
            "mAP50",
            "mAP50-95",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in classes:
            writer.writerow(
                {
                    "attack_type": (
                        attack_type
                    ),

                    "epsilon_pixels": (
                        epsilon_pixels
                    ),

                    "tnorm": tnorm_name,

                    **row,
                }
            )

    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# СОХРАНЕНИЕ ОБЩЕЙ ТАБЛИЦЫ
# ============================================================

def save_combined_results(
    results: list[dict[str, Any]],
    reference_metrics: dict[str, dict[str, float]],
) -> None:
    summary_path = (
        OUTPUT_DIR
        / "tnorm_summary.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "attack_type",
            "epsilon_pixels",
            "tnorm",
            "reference_mAP50",
            "defended_mAP50",
            "absolute_recovery_mAP50",
            "recovery_rate_mAP50",
            "reference_mAP50-95",
            "defended_mAP50-95",
            "absolute_recovery_mAP50-95",
            "recovery_rate_mAP50-95",
            "precision",
            "recall",
            "defense_ms_per_image",
            "defended_vs_clean_psnr",
            "defended_vs_clean_ssim",
            "defense_change_mae_pixels",
            "defended_vs_clean_mae_pixels",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            defended = result["defended"]

            reference = (
                result["reference_attacked"]
            )

            recovery_map50 = (
                result["recovery"]["mAP50"]
            )

            recovery_map50_95 = (
                result["recovery"]["mAP50-95"]
            )

            defense_quality = (
                result["defense_quality"]
            )

            writer.writerow(
                {
                    "attack_type": result[
                        "attack_type"
                    ],

                    "epsilon_pixels": result[
                        "epsilon_pixels"
                    ],

                    "tnorm": result[
                        "tnorm"
                    ],

                    "reference_mAP50": (
                        reference["mAP50"]
                    ),

                    "defended_mAP50": (
                        defended["mAP50"]
                    ),

                    "absolute_recovery_mAP50": (
                        recovery_map50[
                            "absolute_recovery"
                        ]
                    ),

                    "recovery_rate_mAP50": (
                        recovery_map50[
                            "recovery_rate"
                        ]
                    ),

                    "reference_mAP50-95": (
                        reference["mAP50-95"]
                    ),

                    "defended_mAP50-95": (
                        defended["mAP50-95"]
                    ),

                    "absolute_recovery_mAP50-95": (
                        recovery_map50_95[
                            "absolute_recovery"
                        ]
                    ),

                    "recovery_rate_mAP50-95": (
                        recovery_map50_95[
                            "recovery_rate"
                        ]
                    ),

                    "precision": defended[
                        "precision"
                    ],

                    "recall": defended[
                        "recall"
                    ],

                    "defense_ms_per_image": (
                        defense_quality.get(
                            "defense_ms_per_image"
                        )
                    ),

                    "defended_vs_clean_psnr": (
                        defense_quality.get(
                            "defended_vs_clean_psnr"
                        )
                    ),

                    "defended_vs_clean_ssim": (
                        defense_quality.get(
                            "defended_vs_clean_ssim"
                        )
                    ),

                    "defense_change_mae_pixels": (
                        defense_quality.get(
                            "defense_change_mae_pixels"
                        )
                    ),

                    "defended_vs_clean_mae_pixels": (
                        defense_quality.get(
                            "defended_vs_clean_mae_pixels"
                        )
                    ),
                }
            )

    classes_path = (
        OUTPUT_DIR
        / "tnorm_all_classes.csv"
    )

    with classes_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "attack_type",
            "epsilon_pixels",
            "tnorm",
            "class_id",
            "class_name",
            "precision",
            "recall",
            "f1",
            "mAP50",
            "mAP50-95",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            for row in result["classes"]:
                writer.writerow(
                    {
                        "attack_type": result[
                            "attack_type"
                        ],

                        "epsilon_pixels": result[
                            "epsilon_pixels"
                        ],

                        "tnorm": result[
                            "tnorm"
                        ],

                        **row,
                    }
                )

    combined_path = (
        OUTPUT_DIR
        / "tnorm_results.json"
    )

    combined_data = {
        "model": str(MODEL_PATH),

        "dataset": str(DATA_YAML),

        "reference_metrics": (
            reference_metrics
        ),

        "attack_epsilon_pixels": (
            ATTACK_EPSILON_PIXELS
        ),

        "pgd_steps": PGD_STEPS,

        "tnorm_types": TNORM_TYPES,

        "filter_parameters": {
            "window_size": (
                2 * FILTER_RADIUS + 1
            ),

            "sigma_color": SIGMA_COLOR,

            "sigma_spatial": (
                SIGMA_SPATIAL
            ),

            "strength": FILTER_STRENGTH,

            "confidence_gamma": (
                CONFIDENCE_GAMMA
            ),
        },

        "results": results,
    }

    with combined_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            combined_data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# ВЫВОД СВОДКИ
# ============================================================

def percentage_text(
    value: float | None,
) -> str:
    if value is None:
        return "n/a"

    return f"{value * 100.0:.1f}%"


def print_summary(
    results: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 120)
    print("T-НОРМОВАЯ ЗАЩИТА ЗАВЕРШЕНА")
    print("=" * 120)

    print(
        f"{'Attack':>10}"
        f"{'T-norm':>14}"
        f"{'Before':>12}"
        f"{'After':>12}"
        f"{'Recovery':>14}"
        f"{'mAP50-95':>14}"
        f"{'SSIM':>12}"
        f"{'Time ms':>12}"
    )

    for result in results:
        reference = (
            result["reference_attacked"]
        )

        defended = result["defended"]

        recovery = (
            result["recovery"]
            ["mAP50"]
            ["recovery_rate"]
        )

        defense_quality = (
            result["defense_quality"]
        )

        print(
            f"{result['attack_type']:>10}"
            f"{result['tnorm']:>14}"
            f"{reference['mAP50']:>12.4f}"
            f"{defended['mAP50']:>12.4f}"
            f"{percentage_text(recovery):>14}"
            f"{defended['mAP50-95']:>14.4f}"
            f"{defense_quality.get('defended_vs_clean_ssim', 0):>12.4f}"
            f"{defense_quality.get('defense_ms_per_image', 0):>12.2f}"
        )

    print(
        "\nОбщая таблица:\n"
        f"{OUTPUT_DIR / 'tnorm_summary.csv'}"
    )

    print(
        "\nМетрики по классам:\n"
        f"{OUTPUT_DIR / 'tnorm_all_classes.csv'}"
    )

    print(
        "\nПолный JSON:\n"
        f"{OUTPUT_DIR / 'tnorm_results.json'}"
    )


# ============================================================
# ЗАПУСК
# ============================================================

def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    check_environment()
    prepare_output_directory()

    reference_metrics = (
        load_reference_metrics()
    )

    print("\nКонтрольные метрики:")

    for attack_name, values in (
        reference_metrics.items()
    ):
        print(
            f"{attack_name:8s}: "
            f"mAP50={values['mAP50']:.4f}, "
            f"mAP50-95={values['mAP50-95']:.4f}"
        )

    evaluation_model = YOLO(
        str(MODEL_PATH),
        verbose=False,
    )

    results: list[dict[str, Any]] = []

    for attack_type in ATTACK_TYPES:
        for tnorm_name in TNORM_TYPES:
            result = evaluate_case(
                evaluation_model=(
                    evaluation_model
                ),

                attack_type=attack_type,

                tnorm_name=tnorm_name,

                reference_metrics=(
                    reference_metrics
                ),
            )

            results.append(result)

    save_combined_results(
        results=results,
        reference_metrics=reference_metrics,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
    main()