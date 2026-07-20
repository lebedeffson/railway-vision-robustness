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
from torch import nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect.val import DetectionValidator


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
    / "attacks"
    / "fgsm"
)


# ============================================================
# ПАРАМЕТРЫ
# ============================================================

IMAGE_SIZE = 1280
BATCH_SIZE = 2
DEVICE_INDEX = 0
WORKERS = 4

EPSILON_VALUES = [
    0,
    1,
    2,
    4,
    8,
    16,
]

SEED = 2026

# При повторном запуске удаляется только:
# outputs/attacks/fgsm
RESET_OUTPUT = True


# ============================================================
# ПРОВЕРКИ
# ============================================================

def check_environment() -> None:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Не найден data.yaml:\n{DATA_YAML}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Не найдена модель:\n{MODEL_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна. FGSM на CPU не запускается."
        )

    print("=" * 76)
    print("WHITE-BOX FGSM ДЛЯ YOLO11m")
    print("=" * 76)
    print(f"Модель: {MODEL_PATH}")
    print(f"Датасет: {DATA_YAML}")
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(DEVICE_INDEX)}"
    )
    print(f"Размер входа: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epsilon: {EPSILON_VALUES} / 255")


def prepare_output_directory() -> None:
    if OUTPUT_DIR.exists():
        if not RESET_OUTPUT:
            raise FileExistsError(
                "Папка результатов уже существует:\n"
                f"{OUTPUT_DIR}"
            )

        print(
            "\nУдаляется предыдущая папка FGSM:\n"
            f"{OUTPUT_DIR}"
        )

        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def value_to_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())

    return float(value)


def prepare_model_configuration(
    model: nn.Module,
) -> None:
    """
    Преобразует model.args из dict в конфигурационный объект.

    YOLO loss обращается к параметрам через:
    model.args.box
    model.args.cls
    model.args.dfl
    """

    if isinstance(model.args, dict):
        configuration = get_cfg()

        for key, value in model.args.items():
            if hasattr(configuration, key):
                setattr(
                    configuration,
                    key,
                    value,
                )

        model.args = configuration

    # Функция потерь будет создана заново
    # с корректной конфигурацией.
    model.criterion = None


# ============================================================
# SSIM
# ============================================================

def create_gaussian_window(
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    coordinates = torch.arange(
        window_size,
        device=device,
        dtype=dtype,
    )

    coordinates = (
        coordinates
        - (window_size - 1) / 2
    )

    gaussian_1d = torch.exp(
        -(coordinates**2)
        / (2.0 * sigma**2)
    )

    gaussian_1d = (
        gaussian_1d
        / gaussian_1d.sum()
    )

    gaussian_2d = torch.outer(
        gaussian_1d,
        gaussian_1d,
    )

    window = gaussian_2d.view(
        1,
        1,
        window_size,
        window_size,
    )

    return window.expand(
        channels,
        1,
        window_size,
        window_size,
    ).contiguous()


@torch.no_grad()
def calculate_ssim_per_image(
    clean: torch.Tensor,
    attacked: torch.Tensor,
) -> torch.Tensor:
    clean = clean.float()
    attacked = attacked.float()

    channels = clean.shape[1]

    window = create_gaussian_window(
        channels=channels,
        device=clean.device,
        dtype=clean.dtype,
    )

    padding = 5

    mean_clean = F.conv2d(
        clean,
        window,
        padding=padding,
        groups=channels,
    )

    mean_attacked = F.conv2d(
        attacked,
        window,
        padding=padding,
        groups=channels,
    )

    mean_clean_squared = mean_clean**2
    mean_attacked_squared = mean_attacked**2

    mean_product = (
        mean_clean
        * mean_attacked
    )

    variance_clean = (
        F.conv2d(
            clean * clean,
            window,
            padding=padding,
            groups=channels,
        )
        - mean_clean_squared
    )

    variance_attacked = (
        F.conv2d(
            attacked * attacked,
            window,
            padding=padding,
            groups=channels,
        )
        - mean_attacked_squared
    )

    covariance = (
        F.conv2d(
            clean * attacked,
            window,
            padding=padding,
            groups=channels,
        )
        - mean_product
    )

    # Защита от небольших отрицательных значений
    # из-за погрешности float.
    variance_clean = torch.clamp(
        variance_clean,
        min=0.0,
    )

    variance_attacked = torch.clamp(
        variance_attacked,
        min=0.0,
    )

    c1 = 0.01**2
    c2 = 0.03**2

    numerator = (
        (2.0 * mean_product + c1)
        * (2.0 * covariance + c2)
    )

    denominator = (
        (
            mean_clean_squared
            + mean_attacked_squared
            + c1
        )
        * (
            variance_clean
            + variance_attacked
            + c2
        )
    )

    ssim_map = numerator / (
        denominator + 1e-12
    )

    return (
        ssim_map
        .flatten(start_dim=1)
        .mean(dim=1)
    )


# ============================================================
# FGSM-ВАЛИДАТОР
# ============================================================

class FGSMValidator(DetectionValidator):
    """
    Стандартный DetectionValidator Ultralytics,
    дополненный white-box FGSM.

    Атака выполняется после стандартной подготовки batch:
    - resize;
    - letterbox;
    - перенос на GPU;
    - нормализация [0, 1].

    Затем атакованное изображение передаётся
    обычному валидатору для расчёта P, R и mAP.
    """

    epsilon_pixels: int = 0
    last_quality: dict[str, Any] = {}

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

        self.current_epsilon_pixels = int(
            type(self).epsilon_pixels
        )

        self.current_epsilon = (
            self.current_epsilon_pixels
            / 255.0
        )

        self.attack_wrapper: YOLO | None = None
        self.attack_model: nn.Module | None = None

        self.image_count = 0

        self.psnr_sum = 0.0
        self.psnr_count = 0

        self.ssim_sum = 0.0

        self.linf_pixels_sum = 0.0
        self.mean_absolute_pixels_sum = 0.0

        self.clean_loss_sum = 0.0
        self.loss_batch_count = 0

        self.attack_seconds = 0.0

        if self.current_epsilon_pixels > 0:
            attack_device = torch.device(
                f"cuda:{DEVICE_INDEX}"
            )

            # Критически важно:
            # модель создаётся здесь, до вызова
            # BaseValidator.__call__, который работает
            # внутри torch.inference_mode.
            with torch.inference_mode(False):
                self.attack_wrapper = YOLO(
                    str(MODEL_PATH),
                    verbose=False,
                )

                self.attack_model = (
                    self.attack_wrapper
                    .model
                    .to(attack_device)
                    .float()
                )

            prepare_model_configuration(
                self.attack_model
            )

            # Веса модели фиксированы.
            # Градиент вычисляется только по изображению.
            for parameter in (
                self.attack_model.parameters()
            ):
                parameter.requires_grad_(False)

            inference_parameters = [
                name
                for name, parameter
                in self.attack_model.named_parameters()
                if torch.is_inference(parameter)
            ]

            if inference_parameters:
                raise RuntimeError(
                    "Параметры атакующей модели "
                    "ошибочно созданы как inference-тензоры:\n"
                    + "\n".join(
                        inference_parameters[:10]
                    )
                )

            # Training mode нужен для получения
            # выходов, используемых YOLO loss.
            self.attack_model.train()

            # Статистики BatchNorm изменять нельзя.
            for module in (
                self.attack_model.modules()
            ):
                if isinstance(
                    module,
                    nn.modules.batchnorm._BatchNorm,
                ):
                    module.eval()

    def init_metrics(
        self,
        model: nn.Module,
    ) -> None:
        super().init_metrics(model)

        if self.attack_model is None:
            return

        expected_device = torch.device(
            f"cuda:{DEVICE_INDEX}"
        )

        actual_device = next(
            self.attack_model.parameters()
        ).device

        if actual_device != expected_device:
            raise RuntimeError(
                "Атакующая модель находится "
                "на неправильном устройстве:\n"
                f"ожидалось: {expected_device}\n"
                f"получено: {actual_device}"
            )

        # Здесь запрещено выполнять .to(device),
        # потому что init_metrics уже вызывается
        # внутри inference_mode.

        self.attack_model.train()

        for module in (
            self.attack_model.modules()
        ):
            if isinstance(
                module,
                nn.modules.batchnorm._BatchNorm,
            ):
                module.eval()

    def create_normal_attack_batch(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Создаёт обычные тензоры из inference-тензоров
        валидационного batch.

        Клонируются все tensor-поля, поскольку некоторые
        из них используются YOLO loss при backward.
        """

        attack_batch: dict[str, Any] = {}

        for key, value in batch.items():
            if isinstance(
                value,
                torch.Tensor,
            ):
                cloned_value = (
                    value
                    .detach()
                    .clone()
                )

                if torch.is_inference(
                    cloned_value
                ):
                    raise RuntimeError(
                        "После clone тензор всё ещё "
                        f"является inference tensor: {key}"
                    )

                attack_batch[key] = cloned_value

            else:
                attack_batch[key] = value

        return attack_batch

    def calculate_fgsm(
        self,
        batch: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        float,
    ]:
        if self.attack_model is None:
            clean = batch["img"].detach()

            return (
                clean,
                clean,
                0.0,
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        attack_start = time.perf_counter()

        # Внешний валидатор находится в inference_mode.
        # Здесь явно создаётся локальная область autograd.
        with (
            torch.inference_mode(False),
            torch.enable_grad(),
        ):
            attack_batch = (
                self.create_normal_attack_batch(
                    batch
                )
            )

            clean = (
                attack_batch["img"]
                .detach()
                .clone()
                .float()
                .requires_grad_(True)
            )

            if torch.is_inference(clean):
                raise RuntimeError(
                    "Входное изображение осталось "
                    "inference-тензором."
                )

            attack_batch["img"] = clean

            self.attack_model.zero_grad(
                set_to_none=True
            )

            loss_output = self.attack_model(
                attack_batch
            )

            if isinstance(
                loss_output,
                (tuple, list),
            ):
                total_loss = loss_output[0]
            else:
                total_loss = loss_output

            if not isinstance(
                total_loss,
                torch.Tensor,
            ):
                raise TypeError(
                    "YOLO loss не вернул Tensor."
                )

            if total_loss.ndim > 0:
                total_loss = (
                    total_loss.sum()
                )

            gradient = torch.autograd.grad(
                outputs=total_loss,
                inputs=clean,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]

            if gradient is None:
                raise RuntimeError(
                    "Градиент по изображению не вычислен."
                )

            if not torch.isfinite(
                gradient
            ).all():
                raise RuntimeError(
                    "Градиент содержит NaN или Inf."
                )

            attacked = torch.clamp(
                clean
                + self.current_epsilon
                * gradient.sign(),
                min=0.0,
                max=1.0,
            )

            clean_detached = (
                clean
                .detach()
                .clone()
            )

            attacked_detached = (
                attacked
                .detach()
                .clone()
            )

            loss_value = float(
                total_loss
                .detach()
                .item()
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        self.attack_seconds += (
            time.perf_counter()
            - attack_start
        )

        return (
            clean_detached,
            attacked_detached,
            loss_value,
        )

    def collect_image_quality(
        self,
        clean: torch.Tensor,
        attacked: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            clean_float = clean.float()
            attacked_float = attacked.float()

            difference = (
                attacked_float
                - clean_float
            )

            batch_size = int(
                clean.shape[0]
            )

            mse = (
                difference
                .pow(2)
                .flatten(start_dim=1)
                .mean(dim=1)
            )

            finite_mask = mse > 0

            if finite_mask.any():
                psnr = (
                    10.0
                    * torch.log10(
                        1.0
                        / mse[finite_mask]
                    )
                )

                self.psnr_sum += float(
                    psnr.sum().item()
                )

                self.psnr_count += int(
                    psnr.numel()
                )

            ssim = calculate_ssim_per_image(
                clean=clean_float,
                attacked=attacked_float,
            )

            linf_pixels = (
                difference
                .abs()
                .flatten(start_dim=1)
                .amax(dim=1)
                * 255.0
            )

            mean_absolute_pixels = (
                difference
                .abs()
                .flatten(start_dim=1)
                .mean(dim=1)
                * 255.0
            )

            self.ssim_sum += float(
                ssim.sum().item()
            )

            self.linf_pixels_sum += float(
                linf_pixels.sum().item()
            )

            self.mean_absolute_pixels_sum += float(
                mean_absolute_pixels
                .sum()
                .item()
            )

            self.image_count += batch_size

    def preprocess(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        # Штатная подготовка Ultralytics:
        # перенос на GPU и нормализация /255.
        batch = super().preprocess(batch)

        if self.current_epsilon_pixels == 0:
            clean = batch["img"].detach()

            self.collect_image_quality(
                clean=clean,
                attacked=clean,
            )

            return batch

        (
            clean,
            attacked,
            clean_loss,
        ) = self.calculate_fgsm(batch)

        # Стандартная модель валидатора теперь
        # получает атакованные изображения.
        batch["img"] = attacked

        self.clean_loss_sum += clean_loss
        self.loss_batch_count += 1

        self.collect_image_quality(
            clean=clean,
            attacked=attacked,
        )

        return batch

    def finalize_metrics(self) -> None:
        super().finalize_metrics()

        if self.image_count == 0:
            quality = {
                "mean_psnr": None,
                "mean_ssim": None,
                "mean_linf_pixels": None,
                "mean_absolute_pixels": None,
                "mean_clean_loss": None,
                "attack_ms_per_image": None,
            }

        else:
            mean_psnr = (
                self.psnr_sum
                / self.psnr_count
                if self.psnr_count > 0
                else None
            )

            mean_clean_loss = (
                self.clean_loss_sum
                / self.loss_batch_count
                if self.loss_batch_count > 0
                else None
            )

            quality = {
                "mean_psnr": mean_psnr,

                "mean_ssim": (
                    self.ssim_sum
                    / self.image_count
                ),

                "mean_linf_pixels": (
                    self.linf_pixels_sum
                    / self.image_count
                ),

                "mean_absolute_pixels": (
                    self.mean_absolute_pixels_sum
                    / self.image_count
                ),

                "mean_clean_loss": (
                    mean_clean_loss
                ),

                "attack_ms_per_image": (
                    self.attack_seconds
                    / self.image_count
                    * 1000.0
                ),
            }

        type(self).last_quality = quality


# ============================================================
# ИЗВЛЕЧЕНИЕ МЕТРИК
# ============================================================

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
# ОЦЕНКА ОДНОГО EPSILON
# ============================================================

def evaluate_epsilon(
    evaluation_model: YOLO,
    epsilon_pixels: int,
) -> dict[str, Any]:
    FGSMValidator.epsilon_pixels = (
        epsilon_pixels
    )

    FGSMValidator.last_quality = {}

    run_name = (
        f"eps_{epsilon_pixels}_255"
    )

    print("\n" + "=" * 76)
    print(
        f"FGSM: EPSILON = "
        f"{epsilon_pixels}/255"
    )
    print("=" * 76)

    metrics = evaluation_model.val(
        validator=FGSMValidator,

        data=str(DATA_YAML),
        split="test",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE_INDEX,
        workers=WORKERS,

        rect=True,

        plots=True,
        save_json=False,

        project=str(OUTPUT_DIR),
        name=run_name,
        exist_ok=False,

        verbose=True,
    )

    overall = {
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

    class_metrics = extract_class_metrics(
        metrics
    )

    quality = dict(
        FGSMValidator.last_quality
    )

    result = {
        "attack": "FGSM",

        "epsilon_pixels": epsilon_pixels,

        "epsilon_normalized": (
            epsilon_pixels / 255.0
        ),

        "overall": overall,

        "image_quality": quality,

        "speed_ms_per_image": {
            key: value_to_float(value)
            for key, value
            in metrics.speed.items()
        },

        "classes": class_metrics,
    }

    run_directory = (
        OUTPUT_DIR
        / run_name
    )

    json_path = (
        run_directory
        / "fgsm_metrics.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = (
        run_directory
        / "fgsm_metrics_by_class.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "epsilon_pixels",
            "epsilon_normalized",
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

        for row in class_metrics:
            writer.writerow(
                {
                    "epsilon_pixels": (
                        epsilon_pixels
                    ),

                    "epsilon_normalized": (
                        epsilon_pixels
                        / 255.0
                    ),

                    **row,
                }
            )

    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# СОХРАНЕНИЕ ОБЩИХ РЕЗУЛЬТАТОВ
# ============================================================

def save_combined_results(
    results: list[dict[str, Any]],
) -> None:
    summary_path = (
        OUTPUT_DIR
        / "fgsm_summary.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "epsilon_pixels",
            "epsilon_normalized",
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
            "mean_psnr",
            "mean_ssim",
            "mean_linf_pixels",
            "mean_absolute_pixels",
            "attack_ms_per_image",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            overall = result["overall"]
            quality = result["image_quality"]
            speed = result["speed_ms_per_image"]

            writer.writerow(
                {
                    "epsilon_pixels": result[
                        "epsilon_pixels"
                    ],

                    "epsilon_normalized": result[
                        "epsilon_normalized"
                    ],

                    "precision": overall[
                        "precision"
                    ],

                    "recall": overall[
                        "recall"
                    ],

                    "mAP50": overall[
                        "mAP50"
                    ],

                    "mAP50-95": overall[
                        "mAP50-95"
                    ],

                    "mean_psnr": quality.get(
                        "mean_psnr"
                    ),

                    "mean_ssim": quality.get(
                        "mean_ssim"
                    ),

                    "mean_linf_pixels": (
                        quality.get(
                            "mean_linf_pixels"
                        )
                    ),

                    "mean_absolute_pixels": (
                        quality.get(
                            "mean_absolute_pixels"
                        )
                    ),

                    "attack_ms_per_image": (
                        quality.get(
                            "attack_ms_per_image"
                        )
                    ),

                    "preprocess_ms": speed.get(
                        "preprocess"
                    ),

                    "inference_ms": speed.get(
                        "inference"
                    ),

                    "postprocess_ms": speed.get(
                        "postprocess"
                    ),
                }
            )

    classes_path = (
        OUTPUT_DIR
        / "fgsm_all_classes.csv"
    )

    with classes_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "epsilon_pixels",
            "epsilon_normalized",
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
                        "epsilon_pixels": result[
                            "epsilon_pixels"
                        ],

                        "epsilon_normalized": result[
                            "epsilon_normalized"
                        ],

                        **row,
                    }
                )

    json_path = (
        OUTPUT_DIR
        / "fgsm_results.json"
    )

    combined_data = {
        "model": str(MODEL_PATH),

        "dataset": str(DATA_YAML),

        "split": "test",

        "image_size": IMAGE_SIZE,

        "batch_size": BATCH_SIZE,

        "epsilon_values_pixels": (
            EPSILON_VALUES
        ),

        "results": results,
    }

    with json_path.open(
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

def print_summary(
    results: list[dict[str, Any]],
) -> None:
    print("\n" + "=" * 100)
    print("FGSM ЗАВЕРШЁН")
    print("=" * 100)

    print(
        f"{'epsilon':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'mAP50':>12}"
        f"{'mAP50-95':>14}"
        f"{'PSNR':>12}"
        f"{'SSIM':>12}"
        f"{'Linf':>10}"
    )

    for result in results:
        overall = result["overall"]
        quality = result["image_quality"]

        psnr = quality.get(
            "mean_psnr"
        )

        if psnr is None:
            psnr_text = "inf"

        elif math.isfinite(psnr):
            psnr_text = f"{psnr:.2f}"

        else:
            psnr_text = "inf"

        mean_ssim = quality.get(
            "mean_ssim"
        )

        mean_linf = quality.get(
            "mean_linf_pixels"
        )

        print(
            f"{result['epsilon_pixels']:>8}/255"
            f"{overall['precision']:>12.4f}"
            f"{overall['recall']:>12.4f}"
            f"{overall['mAP50']:>12.4f}"
            f"{overall['mAP50-95']:>14.4f}"
            f"{psnr_text:>12}"
            f"{(mean_ssim or 0.0):>12.4f}"
            f"{(mean_linf or 0.0):>10.2f}"
        )

    print(
        "\nОбщая таблица:\n"
        f"{OUTPUT_DIR / 'fgsm_summary.csv'}"
    )

    print(
        "\nМетрики по классам:\n"
        f"{OUTPUT_DIR / 'fgsm_all_classes.csv'}"
    )

    print(
        "\nПолный JSON:\n"
        f"{OUTPUT_DIR / 'fgsm_results.json'}"
    )


# ============================================================
# ЗАПУСК
# ============================================================

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    check_environment()
    prepare_output_directory()

    evaluation_model = YOLO(
        str(MODEL_PATH),
        verbose=False,
    )

    results: list[dict[str, Any]] = []

    for epsilon_pixels in EPSILON_VALUES:
        result = evaluate_epsilon(
            evaluation_model=evaluation_model,
            epsilon_pixels=epsilon_pixels,
        )

        results.append(result)

    save_combined_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()