from __future__ import annotations

import csv
import gc
import io
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator

from evaluate_fgsm import FGSMValidator, calculate_ssim_per_image
from tune_tnorm_defense import FilterConfig, tunable_tnorm_filter


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

TNORM_TUNING_DIR = (
    PROJECT_DIR
    / "outputs"
    / "defenses"
    / "tnorm_tuning_val"
)

BEST_TNORM_PATH = (
    TNORM_TUNING_DIR
    / "best_tnorm_config.json"
)

TNORM_REFERENCE_PATH = (
    TNORM_TUNING_DIR
    / "reference"
    / "reference_results.json"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "defenses"
    / "defense_comparison_val"
)

RESULTS_DIR = OUTPUT_DIR / "results"
RUNS_DIR = OUTPUT_DIR / "runs"
REFERENCE_DIR = OUTPUT_DIR / "reference"


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

# Основной критерий допустимости.
PRIMARY_MAX_CLEAN_MAP50_RELATIVE_DROP = 0.03
PRIMARY_MAX_CLEAN_MAP50_95_ABSOLUTE_DROP = 0.005

# Резервный критерий используется только тогда,
# когда метод не имеет конфигурации основного уровня.
RELAXED_MAX_CLEAN_MAP50_RELATIVE_DROP = 0.05
RELAXED_MAX_CLEAN_MAP50_95_ABSOLUTE_DROP = 0.010

# Старые candidate_summary.json будут пересчитаны,
# но уже рассчитанные clean/FGSM JSON переиспользуются.
COMPARISON_VERSION = 2

# Median обрабатывается полосами, чтобы F.unfold
# не занимал почти всю видеопамять.
MEDIAN_CHUNK_ROWS = 128


# ============================================================
# СЕТКА БАЗОВЫХ ЗАЩИТ
# ============================================================

MEDIAN_KERNELS = [
    3,
    5,
]

GAUSSIAN_CONFIGS = [
    (3, 0.8),
    (3, 1.2),
    (5, 1.0),
    (5, 1.5),
]

JPEG_QUALITIES = [
    70,
    80,
    90,
    95,
]

BILATERAL_CONFIGS = [
    (3, 8.0, 1.0),
    (3, 16.0, 1.0),
    (3, 24.0, 1.0),
    (5, 16.0, 1.5),
    (5, 24.0, 1.5),
    (5, 32.0, 1.5),
]


# ============================================================
# КОНФИГУРАЦИЯ ЗАЩИТЫ
# ============================================================

@dataclass(frozen=True)
class DefenseConfig:
    method: str

    kernel_size: int | None = None
    sigma: float | None = None

    jpeg_quality: int | None = None

    sigma_color_pixels: float | None = None
    sigma_spatial: float | None = None

    tnorm: str | None = None
    strength: float | None = None
    confidence_gamma: float | None = None

    @staticmethod
    def number_slug(
        value: float,
    ) -> str:
        text = (
            f"{value:.4f}"
            .rstrip("0")
            .rstrip(".")
        )

        return text.replace(".", "p")

    @property
    def identifier(self) -> str:
        if self.method == "median":
            return (
                f"median_k"
                f"{self.kernel_size}"
            )

        if self.method == "gaussian":
            return (
                f"gaussian_k"
                f"{self.kernel_size}"
                f"_s"
                f"{self.number_slug(float(self.sigma))}"
            )

        if self.method == "jpeg":
            return (
                f"jpeg_q"
                f"{self.jpeg_quality}"
            )

        if self.method == "bilateral":
            return (
                f"bilateral_k"
                f"{self.kernel_size}"
                f"_sc"
                f"{self.number_slug(float(self.sigma_color_pixels))}"
                f"_ss"
                f"{self.number_slug(float(self.sigma_spatial))}"
            )

        if self.method == "tnorm":
            return (
                f"tnorm_"
                f"{self.tnorm}"
                f"_k"
                f"{self.kernel_size}"
                f"_sc"
                f"{self.number_slug(float(self.sigma_color_pixels))}"
                f"_ss"
                f"{self.number_slug(float(self.sigma_spatial))}"
                f"_st"
                f"{self.number_slug(float(self.strength))}"
                f"_g"
                f"{self.number_slug(float(self.confidence_gamma))}"
            )

        raise ValueError(
            f"Неизвестный метод защиты: "
            f"{self.method}"
        )


# ============================================================
# ПРОВЕРКИ И КАТАЛОГИ
# ============================================================

def check_environment() -> None:
    required_paths = [
        DATA_YAML,
        MODEL_PATH,
        BEST_TNORM_PATH,
        PROJECT_DIR / "evaluate_fgsm.py",
        PROJECT_DIR / "tune_tnorm_defense.py",
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

    print("=" * 108)
    print("СРАВНЕНИЕ ЗАЩИТ НА VALIDATION")
    print("=" * 108)
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
        "Основной clean-критерий: "
        f"mAP50 drop <= "
        f"{PRIMARY_MAX_CLEAN_MAP50_RELATIVE_DROP * 100:.1f}% "
        "и mAP50-95 absolute drop <= "
        f"{PRIMARY_MAX_CLEAN_MAP50_95_ABSOLUTE_DROP:.4f}"
    )

    print(
        "Резервный clean-критерий: "
        f"mAP50 drop <= "
        f"{RELAXED_MAX_CLEAN_MAP50_RELATIVE_DROP * 100:.1f}% "
        "и mAP50-95 absolute drop <= "
        f"{RELAXED_MAX_CLEAN_MAP50_95_ABSOLUTE_DROP:.4f}"
    )


def prepare_directories() -> None:
    for path in [
        OUTPUT_DIR,
        RESULTS_DIR,
        RUNS_DIR,
        REFERENCE_DIR,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# JSON
# ============================================================

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


# ============================================================
# ЗАФИКСИРОВАННАЯ T-НОРМА
# ============================================================

def load_fixed_tnorm_config() -> DefenseConfig:
    data = read_json(
        BEST_TNORM_PATH
    )

    try:
        best = data["best"]["config"]

    except KeyError as error:
        raise KeyError(
            "Некорректная структура "
            "best_tnorm_config.json. "
            "Ожидался путь best -> config."
        ) from error

    return DefenseConfig(
        method="tnorm",

        kernel_size=int(
            best["window_size"]
        ),

        sigma_color_pixels=float(
            best["sigma_color_pixels"]
        ),

        sigma_spatial=float(
            best["sigma_spatial"]
        ),

        tnorm=str(
            best["tnorm"]
        ),

        strength=float(
            best["strength"]
        ),

        confidence_gamma=float(
            best["confidence_gamma"]
        ),
    )


# ============================================================
# MEDIAN
# ============================================================

@torch.no_grad()
def median_filter(
    image: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    if (
        kernel_size < 3
        or kernel_size % 2 == 0
    ):
        raise ValueError(
            "Median kernel должен быть "
            "нечётным и >= 3."
        )

    original_dtype = image.dtype
    image_float = image.float()

    (
        batch_size,
        channels,
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

    output = torch.empty_like(
        image_float
    )

    for row_start in range(
        0,
        height,
        MEDIAN_CHUNK_ROWS,
    ):
        row_end = min(
            row_start + MEDIAN_CHUNK_ROWS,
            height,
        )

        chunk_height = (
            row_end
            - row_start
        )

        padded_chunk = padded[
            :,
            :,
            row_start:row_end + 2 * radius,
            :,
        ]

        patches = F.unfold(
            padded_chunk,
            kernel_size=kernel_size,
            stride=1,
        )

        patches = patches.view(
            batch_size,
            channels,
            kernel_size * kernel_size,
            chunk_height,
            width,
        )

        output[
            :,
            :,
            row_start:row_end,
            :,
        ] = patches.median(
            dim=2
        ).values

    return (
        output
        .clamp(0.0, 1.0)
        .to(original_dtype)
    )


# ============================================================
# GAUSSIAN
# ============================================================

@torch.no_grad()
def gaussian_filter(
    image: torch.Tensor,
    kernel_size: int,
    sigma: float,
) -> torch.Tensor:
    if (
        kernel_size < 3
        or kernel_size % 2 == 0
    ):
        raise ValueError(
            "Gaussian kernel должен быть "
            "нечётным и >= 3."
        )

    if sigma <= 0:
        raise ValueError(
            "Gaussian sigma должна быть "
            "положительной."
        )

    original_dtype = image.dtype
    image_float = image.float()

    channels = image_float.shape[1]
    radius = kernel_size // 2

    coordinates = (
        torch.arange(
            kernel_size,
            device=image.device,
            dtype=torch.float32,
        )
        - radius
    )

    kernel_1d = torch.exp(
        -(coordinates**2)
        / (2.0 * sigma**2)
    )

    kernel_1d = (
        kernel_1d
        / kernel_1d.sum()
    )

    kernel_2d = torch.outer(
        kernel_1d,
        kernel_1d,
    )

    kernel = kernel_2d.view(
        1,
        1,
        kernel_size,
        kernel_size,
    )

    kernel = kernel.expand(
        channels,
        1,
        kernel_size,
        kernel_size,
    )

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

    filtered = F.conv2d(
        padded,
        kernel,
        groups=channels,
    )

    return (
        filtered
        .clamp(0.0, 1.0)
        .to(original_dtype)
    )


# ============================================================
# BILATERAL
# ============================================================

@torch.no_grad()
def bilateral_filter(
    image: torch.Tensor,
    kernel_size: int,
    sigma_color_pixels: float,
    sigma_spatial: float,
) -> torch.Tensor:
    if (
        kernel_size < 3
        or kernel_size % 2 == 0
    ):
        raise ValueError(
            "Bilateral kernel должен быть "
            "нечётным и >= 3."
        )

    if (
        sigma_color_pixels <= 0
        or sigma_spatial <= 0
    ):
        raise ValueError(
            "Bilateral sigma должна быть "
            "положительной."
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

    sigma_color = (
        sigma_color_pixels
        / 255.0
    )

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

    for offset_y in range(
        -radius,
        radius + 1,
    ):
        for offset_x in range(
            -radius,
            radius + 1,
        ):
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

            color_distance_squared = (
                (
                    image_float
                    - neighbour
                )
                .pow(2)
                .mean(
                    dim=1,
                    keepdim=True,
                )
            )

            color_weight = torch.exp(
                -color_distance_squared
                / (
                    2.0
                    * sigma_color**2
                )
            )

            spatial_distance_squared = (
                offset_x**2
                + offset_y**2
            )

            spatial_weight = math.exp(
                -spatial_distance_squared
                / (
                    2.0
                    * sigma_spatial**2
                )
            )

            weight = (
                color_weight
                * spatial_weight
            )

            weighted_sum += (
                weight
                * neighbour
            )

            weight_sum += weight

    filtered = (
        weighted_sum
        / weight_sum.clamp_min(1e-8)
    )

    return (
        filtered
        .clamp(0.0, 1.0)
        .to(original_dtype)
    )


# ============================================================
# JPEG
# ============================================================

@torch.no_grad()
def jpeg_filter(
    image: torch.Tensor,
    quality: int,
) -> torch.Tensor:
    if not 1 <= quality <= 100:
        raise ValueError(
            "JPEG quality должна быть "
            "от 1 до 100."
        )

    original_device = image.device
    original_dtype = image.dtype

    image_uint8 = (
        image
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
    )

    decoded_images: list[
        torch.Tensor
    ] = []

    for sample in image_uint8:
        array = (
            sample
            .permute(1, 2, 0)
            .numpy()
        )

        pil_image = Image.fromarray(
            array
        )

        buffer = io.BytesIO()

        pil_image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=2,
            optimize=False,
        )

        buffer.seek(0)

        with Image.open(
            buffer
        ) as decoded:
            decoded_rgb = decoded.convert(
                "RGB"
            )

            decoded_array = np.array(
                decoded_rgb,
                dtype=np.uint8,
                copy=True,
            )

        decoded_tensor = (
            torch
            .from_numpy(decoded_array)
            .permute(2, 0, 1)
            .contiguous()
        )

        decoded_images.append(
            decoded_tensor
        )

    stacked = torch.stack(
        decoded_images,
        dim=0,
    )

    return (
        stacked
        .to(
            device=original_device,
            dtype=torch.float32,
        )
        .div(255.0)
        .to(original_dtype)
    )


# ============================================================
# ПРИМЕНЕНИЕ ЗАЩИТЫ
# ============================================================

@torch.no_grad()
def apply_defense(
    image: torch.Tensor,
    config: DefenseConfig,
) -> torch.Tensor:
    if config.method == "median":
        return median_filter(
            image=image,

            kernel_size=int(
                config.kernel_size
            ),
        )

    if config.method == "gaussian":
        return gaussian_filter(
            image=image,

            kernel_size=int(
                config.kernel_size
            ),

            sigma=float(
                config.sigma
            ),
        )

    if config.method == "jpeg":
        return jpeg_filter(
            image=image,

            quality=int(
                config.jpeg_quality
            ),
        )

    if config.method == "bilateral":
        return bilateral_filter(
            image=image,

            kernel_size=int(
                config.kernel_size
            ),

            sigma_color_pixels=float(
                config.sigma_color_pixels
            ),

            sigma_spatial=float(
                config.sigma_spatial
            ),
        )

    if config.method == "tnorm":
        tnorm_config = FilterConfig(
            tnorm=str(
                config.tnorm
            ),

            window_size=int(
                config.kernel_size
            ),

            sigma_color_pixels=float(
                config.sigma_color_pixels
            ),

            sigma_spatial=float(
                config.sigma_spatial
            ),

            strength=float(
                config.strength
            ),

            confidence_gamma=float(
                config.confidence_gamma
            ),
        )

        return tunable_tnorm_filter(
            image=image,
            config=tnorm_config,
        )

    raise ValueError(
        f"Неизвестный метод защиты: "
        f"{config.method}"
    )


# ============================================================
# КАЧЕСТВО ИЗОБРАЖЕНИЯ
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
        float(
            psnr.sum().item()
        ),

        int(
            psnr.numel()
        ),
    )


# ============================================================
# VALIDATOR
# ============================================================

class ComparisonDefenseValidator(
    FGSMValidator
):
    defense_config = DefenseConfig(
        method="median",
        kernel_size=3,
    )

    last_defense_quality: dict[
        str,
        Any,
    ] = {}

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

        self.defense_psnr_sum = 0.0
        self.defense_psnr_count = 0
        self.defense_ssim_sum = 0.0

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
                psnr_sum,
                psnr_count,
            ) = calculate_psnr_sum(
                clean=clean_float,
                defended=defended_float,
            )

            self.defense_psnr_sum += (
                psnr_sum
            )

            self.defense_psnr_count += (
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

            self.defense_ssim_sum += float(
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
            ) = self.calculate_fgsm(
                batch
            )

            self.clean_loss_sum += (
                clean_loss
            )

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
                self.defense_psnr_sum
                / self.defense_psnr_count
                if self.defense_psnr_count > 0
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
                    self.defense_ssim_sum
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
# МЕТРИКИ YOLO
# ============================================================

def value_to_float(
    value: Any,
) -> float:
    if hasattr(value, "item"):
        return float(
            value.item()
        )

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

    rows: list[
        dict[str, Any]
    ] = []

    for (
        metric_index,
        class_id,
    ) in enumerate(
        class_ids
    ):
        rows.append(
            {
                "class_id": int(
                    class_id
                ),

                "class_name": (
                    metrics.names[
                        int(class_id)
                    ]
                ),

                "precision": float(
                    precision[
                        metric_index
                    ]
                ),

                "recall": float(
                    recall[
                        metric_index
                    ]
                ),

                "f1": float(
                    f1[
                        metric_index
                    ]
                ),

                "mAP50": float(
                    ap50[
                        metric_index
                    ]
                ),

                "mAP50-95": float(
                    ap50_95[
                        metric_index
                    ]
                ),
            }
        )

    return rows


# ============================================================
# REFERENCE БЕЗ ЗАЩИТЫ
# ============================================================

def evaluate_reference_condition(
    evaluation_model: YOLO,
    epsilon_pixels: int,
) -> dict[str, Any]:
    condition_name = (
        "clean"
        if epsilon_pixels == 0
        else (
            f"fgsm_eps_"
            f"{epsilon_pixels}"
            f"_255"
        )
    )

    reference_path = (
        REFERENCE_DIR
        / f"{condition_name}.json"
    )

    if reference_path.is_file():
        return read_json(
            reference_path
        )

    current_seed = (
        SEED
        + epsilon_pixels * 100
    )

    torch.manual_seed(
        current_seed
    )

    torch.cuda.manual_seed_all(
        current_seed
    )

    np.random.seed(
        current_seed
    )

    FGSMValidator.epsilon_pixels = (
        epsilon_pixels
    )

    FGSMValidator.last_quality = {}

    print("\n" + "=" * 108)
    print(
        f"REFERENCE VAL: "
        f"{condition_name.upper()}"
    )
    print("=" * 108)

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
            RUNS_DIR
        ),

        name=(
            f"reference_"
            f"{condition_name}"
        ),

        exist_ok=True,
        verbose=False,
    )

    result = {
        "condition": condition_name,

        "epsilon_pixels": (
            epsilon_pixels
        ),

        "seed": current_seed,

        "overall": (
            extract_overall_metrics(
                metrics
            )
        ),

        "attack_quality": dict(
            FGSMValidator.last_quality
        ),

        "classes": (
            extract_class_metrics(
                metrics
            )
        ),
    }

    write_json(
        reference_path,
        result,
    )

    gc.collect()
    torch.cuda.empty_cache()

    return result


def load_or_create_references(
    evaluation_model: YOLO,
) -> dict[str, Any]:
    if TNORM_REFERENCE_PATH.is_file():
        references = read_json(
            TNORM_REFERENCE_PATH
        )

        required_keys = {
            "clean"
        } | {
            f"fgsm_{epsilon}"
            for epsilon
            in ATTACK_EPSILONS
        }

        if required_keys.issubset(
            references.keys()
        ):
            print(
                "Используются reference-метрики "
                "из настройки T-нормы:\n"
                f"{TNORM_REFERENCE_PATH}"
            )

            return references

    references: dict[
        str,
        Any,
    ] = {
        "clean": (
            evaluate_reference_condition(
                evaluation_model=(
                    evaluation_model
                ),

                epsilon_pixels=0,
            )
        )
    }

    for epsilon_pixels in (
        ATTACK_EPSILONS
    ):
        references[
            f"fgsm_{epsilon_pixels}"
        ] = evaluate_reference_condition(
            evaluation_model=(
                evaluation_model
            ),

            epsilon_pixels=(
                epsilon_pixels
            ),
        )

    write_json(
        REFERENCE_DIR
        / "reference_results.json",

        references,
    )

    return references


# ============================================================
# ОДНО УСЛОВИЕ С ЗАЩИТОЙ
# ============================================================

def evaluate_defended_condition(
    evaluation_model: YOLO,
    config: DefenseConfig,
    epsilon_pixels: int,
) -> dict[str, Any]:
    condition_name = (
        "clean"
        if epsilon_pixels == 0
        else (
            f"fgsm_eps_"
            f"{epsilon_pixels}"
            f"_255"
        )
    )

    candidate_dir = (
        RESULTS_DIR
        / config.identifier
    )

    candidate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_path = (
        candidate_dir
        / f"{condition_name}.json"
    )

    if result_path.is_file():
        stored = read_json(
            result_path
        )

        if (
            stored.get(
                "epsilon_pixels"
            ) == epsilon_pixels

            and stored.get(
                "config"
            ) == asdict(config)
        ):
            return stored

    current_seed = (
        SEED
        + epsilon_pixels * 100
    )

    torch.manual_seed(
        current_seed
    )

    torch.cuda.manual_seed_all(
        current_seed
    )

    np.random.seed(
        current_seed
    )

    ComparisonDefenseValidator.epsilon_pixels = (
        epsilon_pixels
    )

    ComparisonDefenseValidator.defense_config = (
        config
    )

    ComparisonDefenseValidator.last_quality = {}

    ComparisonDefenseValidator.last_defense_quality = {}

    print("\n" + "-" * 108)
    print(
        f"{config.identifier} | "
        f"{condition_name}"
    )
    print("-" * 108)

    metrics = evaluation_model.val(
        validator=(
            ComparisonDefenseValidator
        ),

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
            RUNS_DIR
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

        "epsilon_pixels": (
            epsilon_pixels
        ),

        "seed": current_seed,

        "config": asdict(
            config
        ),

        "overall": (
            extract_overall_metrics(
                metrics
            )
        ),

        "attack_quality": dict(
            ComparisonDefenseValidator
            .last_quality
        ),

        "defense_quality": dict(
            ComparisonDefenseValidator
            .last_defense_quality
        ),

        "speed_ms_per_image": {
            key: value_to_float(value)

            for key, value
            in metrics.speed.items()
        },

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

    gc.collect()
    torch.cuda.empty_cache()

    return result


# ============================================================
# ОЦЕНКА КАНДИДАТА
# ============================================================

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


def positive_absolute_drop(
    reference: float,
    evaluated: float,
) -> float:
    return max(
        0.0,
        reference - evaluated,
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
    values: list[
        float | None
    ],
) -> float:
    valid_values = [
        float(value)

        for value in values

        if (
            value is not None
            and math.isfinite(value)
        )
    ]

    if not valid_values:
        return float("-inf")

    return float(
        np.mean(
            valid_values
        )
    )


def candidate_summary_is_current(
    summary: dict[str, Any],
) -> bool:
    if (
        summary.get(
            "comparison_version"
        )
        != COMPARISON_VERSION
    ):
        return False

    attack_results = summary.get(
        "attack_results",
        {},
    )

    return all(
        str(epsilon)
        in attack_results

        for epsilon
        in ATTACK_EPSILONS
    )


def evaluate_candidate(
    evaluation_model: YOLO,
    config: DefenseConfig,
    references: dict[str, Any],
) -> dict[str, Any]:
    candidate_dir = (
        RESULTS_DIR
        / config.identifier
    )

    candidate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        candidate_dir
        / "candidate_summary.json"
    )

    if summary_path.is_file():
        stored_summary = read_json(
            summary_path
        )

        if candidate_summary_is_current(
            stored_summary
        ):
            return stored_summary

    clean_result = (
        evaluate_defended_condition(
            evaluation_model=(
                evaluation_model
            ),

            config=config,

            epsilon_pixels=0,
        )
    )

    attack_results: dict[
        str,
        Any,
    ] = {}

    # Все методы оцениваются под атакой,
    # даже когда clean-критерий не выполнен.
    for epsilon_pixels in (
        ATTACK_EPSILONS
    ):
        attack_results[
            str(epsilon_pixels)
        ] = evaluate_defended_condition(
            evaluation_model=(
                evaluation_model
            ),

            config=config,

            epsilon_pixels=(
                epsilon_pixels
            ),
        )

    clean_reference = (
        references["clean"]["overall"]
    )

    clean_defended = (
        clean_result["overall"]
    )

    clean_drop_map50_relative = (
        positive_relative_drop(
            reference=(
                clean_reference["mAP50"]
            ),

            evaluated=(
                clean_defended["mAP50"]
            ),
        )
    )

    clean_drop_map50_95_absolute = (
        positive_absolute_drop(
            reference=(
                clean_reference["mAP50-95"]
            ),

            evaluated=(
                clean_defended["mAP50-95"]
            ),
        )
    )

    clean_drop_map50_95_relative = (
        positive_relative_drop(
            reference=(
                clean_reference["mAP50-95"]
            ),

            evaluated=(
                clean_defended["mAP50-95"]
            ),
        )
    )

    primary_feasible = (
        clean_drop_map50_relative
        <= PRIMARY_MAX_CLEAN_MAP50_RELATIVE_DROP

        and

        clean_drop_map50_95_absolute
        <= PRIMARY_MAX_CLEAN_MAP50_95_ABSOLUTE_DROP
    )

    relaxed_feasible = (
        clean_drop_map50_relative
        <= RELAXED_MAX_CLEAN_MAP50_RELATIVE_DROP

        and

        clean_drop_map50_95_absolute
        <= RELAXED_MAX_CLEAN_MAP50_95_ABSOLUTE_DROP
    )

    if primary_feasible:
        selection_tier = "primary"

    elif relaxed_feasible:
        selection_tier = "relaxed"

    else:
        selection_tier = "fallback"

    recovery_map50_values: list[
        float | None
    ] = []

    recovery_map50_95_values: list[
        float | None
    ] = []

    for epsilon_pixels in (
        ATTACK_EPSILONS
    ):
        attacked_reference = (
            references[
                f"fgsm_{epsilon_pixels}"
            ]["overall"]
        )

        defended_result = (
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
                    attacked_reference["mAP50"]
                ),

                defended_value=(
                    defended_result["mAP50"]
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
                    attacked_reference[
                        "mAP50-95"
                    ]
                ),

                defended_value=(
                    defended_result[
                        "mAP50-95"
                    ]
                ),
            )
        )

    mean_recovery_map50 = (
        mean_valid(
            recovery_map50_values
        )
    )

    mean_recovery_map50_95 = (
        mean_valid(
            recovery_map50_95_values
        )
    )

    # Score применяется внутри одного tier.
    score = (
        0.60
        * mean_recovery_map50

        + 0.40
        * mean_recovery_map50_95

        - 0.10
        * clean_drop_map50_relative

        - 0.05
        * clean_drop_map50_95_relative
    )

    clean_defense_quality = (
        clean_result.get(
            "defense_quality",
            {},
        )
    )

    summary = {
        "comparison_version": (
            COMPARISON_VERSION
        ),

        "identifier": (
            config.identifier
        ),

        "method": config.method,

        "config": asdict(
            config
        ),

        "primary_feasible": (
            primary_feasible
        ),

        "relaxed_feasible": (
            relaxed_feasible
        ),

        "selection_tier": (
            selection_tier
        ),

        "clean_reference": (
            clean_reference
        ),

        "clean_defended": (
            clean_defended
        ),

        "clean_drop": {
            "mAP50_relative": (
                clean_drop_map50_relative
            ),

            "mAP50-95_absolute": (
                clean_drop_map50_95_absolute
            ),

            "mAP50-95_relative": (
                clean_drop_map50_95_relative
            ),
        },

        "attack_results": (
            attack_results
        ),

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

        "clean_defense_quality": (
            clean_defense_quality
        ),

        "score": score,
    }

    write_json(
        summary_path,
        summary,
    )

    print(
        f"\n{config.identifier}: "
        f"tier={selection_tier}, "
        f"clean AP50 drop="
        f"{clean_drop_map50_relative * 100.0:.2f}%, "
        f"clean AP50-95 abs drop="
        f"{clean_drop_map50_95_absolute:.4f}, "
        f"score={score:.6f}"
    )

    return summary


# ============================================================
# ГЕНЕРАЦИЯ КОНФИГУРАЦИЙ
# ============================================================

def create_baseline_configs() -> list[
    DefenseConfig
]:
    configs: list[
        DefenseConfig
    ] = []

    for kernel_size in (
        MEDIAN_KERNELS
    ):
        configs.append(
            DefenseConfig(
                method="median",
                kernel_size=kernel_size,
            )
        )

    for (
        kernel_size,
        sigma,
    ) in GAUSSIAN_CONFIGS:
        configs.append(
            DefenseConfig(
                method="gaussian",
                kernel_size=kernel_size,
                sigma=sigma,
            )
        )

    for quality in (
        JPEG_QUALITIES
    ):
        configs.append(
            DefenseConfig(
                method="jpeg",
                jpeg_quality=quality,
            )
        )

    for (
        kernel_size,
        sigma_color_pixels,
        sigma_spatial,
    ) in BILATERAL_CONFIGS:
        configs.append(
            DefenseConfig(
                method="bilateral",

                kernel_size=(
                    kernel_size
                ),

                sigma_color_pixels=(
                    sigma_color_pixels
                ),

                sigma_spatial=(
                    sigma_spatial
                ),
            )
        )

    return configs


# ============================================================
# РАНЖИРОВАНИЕ И ВЫБОР
# ============================================================

def result_score(
    result: dict[str, Any],
) -> float:
    score = result.get(
        "score"
    )

    if score is None:
        return float("-inf")

    return float(score)


def selection_tier_priority(
    result: dict[str, Any],
) -> int:
    tier = result.get(
        "selection_tier",
        "fallback",
    )

    priorities = {
        "primary": 0,
        "relaxed": 1,
        "fallback": 2,
    }

    return priorities.get(
        tier,
        2,
    )


def rank_for_selection(
    results: list[
        dict[str, Any]
    ],
) -> list[dict[str, Any]]:
    return sorted(
        results,

        key=lambda result: (
            selection_tier_priority(
                result
            ),

            -result_score(
                result
            ),
        ),
    )


def select_best_per_method(
    results: list[
        dict[str, Any]
    ],
) -> dict[
    str,
    dict[str, Any],
]:
    selected: dict[
        str,
        dict[str, Any],
    ] = {}

    methods = [
        "median",
        "gaussian",
        "jpeg",
        "bilateral",
        "tnorm",
    ]

    for method in methods:
        method_results = [
            result

            for result in results

            if result["method"] == method
        ]

        if not method_results:
            continue

        selected[method] = (
            rank_for_selection(
                method_results
            )[0]
        )

    return selected


def calculate_pareto_identifiers(
    results: list[
        dict[str, Any]
    ],
) -> set[str]:
    """
    Pareto-критерии:
    - меньше clean mAP50 relative drop;
    - больше mean recovery mAP50;
    - меньше время защиты.
    """

    pareto_identifiers: set[
        str
    ] = set()

    for candidate in results:
        candidate_drop = float(
            candidate[
                "clean_drop"
            ][
                "mAP50_relative"
            ]
        )

        candidate_recovery = float(
            candidate[
                "recovery_rates"
            ][
                "mean_mAP50"
            ]
        )

        candidate_time = (
            candidate.get(
                "clean_defense_quality",
                {},
            )
            .get(
                "defense_ms_per_image"
            )
        )

        if candidate_time is None:
            candidate_time = float(
                "inf"
            )

        else:
            candidate_time = float(
                candidate_time
            )

        dominated = False

        for other in results:
            if (
                other["identifier"]
                == candidate["identifier"]
            ):
                continue

            other_drop = float(
                other[
                    "clean_drop"
                ][
                    "mAP50_relative"
                ]
            )

            other_recovery = float(
                other[
                    "recovery_rates"
                ][
                    "mean_mAP50"
                ]
            )

            other_time = (
                other.get(
                    "clean_defense_quality",
                    {},
                )
                .get(
                    "defense_ms_per_image"
                )
            )

            if other_time is None:
                other_time = float(
                    "inf"
                )

            else:
                other_time = float(
                    other_time
                )

            no_worse = (
                other_drop
                <= candidate_drop

                and

                other_recovery
                >= candidate_recovery

                and

                other_time
                <= candidate_time
            )

            strictly_better = (
                other_drop
                < candidate_drop

                or

                other_recovery
                > candidate_recovery

                or

                other_time
                < candidate_time
            )

            if (
                no_worse
                and strictly_better
            ):
                dominated = True
                break

        if not dominated:
            pareto_identifiers.add(
                candidate["identifier"]
            )

    return pareto_identifiers


# ============================================================
# CSV
# ============================================================

def save_results_csv(
    results: list[
        dict[str, Any]
    ],

    selected_identifiers: set[str],

    pareto_identifiers: set[str],
) -> None:
    csv_path = (
        OUTPUT_DIR
        / "defense_comparison_val.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        fieldnames = [
            "selected",
            "pareto_optimal",
            "selection_tier",
            "primary_feasible",
            "relaxed_feasible",
            "method",
            "identifier",
            "score",
            "kernel_size",
            "sigma",
            "jpeg_quality",
            "sigma_color_pixels",
            "sigma_spatial",
            "tnorm",
            "strength",
            "confidence_gamma",
            "clean_mAP50",
            "clean_mAP50-95",
            "clean_drop_mAP50_relative",
            "clean_drop_mAP50-95_absolute",
            "clean_drop_mAP50-95_relative",
            "fgsm1_mAP50",
            "fgsm1_mAP50-95",
            "fgsm2_mAP50",
            "fgsm2_mAP50-95",
            "mean_recovery_mAP50",
            "mean_recovery_mAP50-95",
            "clean_defense_ms",
            "clean_defended_vs_clean_psnr",
            "clean_defended_vs_clean_ssim",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        ordered_results = sorted(
            results,

            key=lambda item: (
                item["method"],

                selection_tier_priority(
                    item
                ),

                -result_score(
                    item
                ),
            ),
        )

        for result in ordered_results:
            config = result["config"]
            attacks = result[
                "attack_results"
            ]

            fgsm1 = attacks["1"]
            fgsm2 = attacks["2"]

            defense_quality = (
                result.get(
                    "clean_defense_quality",
                    {},
                )
            )

            writer.writerow(
                {
                    "selected": (
                        result["identifier"]
                        in selected_identifiers
                    ),

                    "pareto_optimal": (
                        result["identifier"]
                        in pareto_identifiers
                    ),

                    "selection_tier": (
                        result["selection_tier"]
                    ),

                    "primary_feasible": (
                        result[
                            "primary_feasible"
                        ]
                    ),

                    "relaxed_feasible": (
                        result[
                            "relaxed_feasible"
                        ]
                    ),

                    "method": (
                        result["method"]
                    ),

                    "identifier": (
                        result["identifier"]
                    ),

                    "score": (
                        result["score"]
                    ),

                    "kernel_size": (
                        config.get(
                            "kernel_size"
                        )
                    ),

                    "sigma": (
                        config.get(
                            "sigma"
                        )
                    ),

                    "jpeg_quality": (
                        config.get(
                            "jpeg_quality"
                        )
                    ),

                    "sigma_color_pixels": (
                        config.get(
                            "sigma_color_pixels"
                        )
                    ),

                    "sigma_spatial": (
                        config.get(
                            "sigma_spatial"
                        )
                    ),

                    "tnorm": (
                        config.get(
                            "tnorm"
                        )
                    ),

                    "strength": (
                        config.get(
                            "strength"
                        )
                    ),

                    "confidence_gamma": (
                        config.get(
                            "confidence_gamma"
                        )
                    ),

                    "clean_mAP50": (
                        result[
                            "clean_defended"
                        ]["mAP50"]
                    ),

                    "clean_mAP50-95": (
                        result[
                            "clean_defended"
                        ]["mAP50-95"]
                    ),

                    "clean_drop_mAP50_relative": (
                        result[
                            "clean_drop"
                        ][
                            "mAP50_relative"
                        ]
                    ),

                    "clean_drop_mAP50-95_absolute": (
                        result[
                            "clean_drop"
                        ][
                            "mAP50-95_absolute"
                        ]
                    ),

                    "clean_drop_mAP50-95_relative": (
                        result[
                            "clean_drop"
                        ][
                            "mAP50-95_relative"
                        ]
                    ),

                    "fgsm1_mAP50": (
                        fgsm1[
                            "overall"
                        ]["mAP50"]
                    ),

                    "fgsm1_mAP50-95": (
                        fgsm1[
                            "overall"
                        ]["mAP50-95"]
                    ),

                    "fgsm2_mAP50": (
                        fgsm2[
                            "overall"
                        ]["mAP50"]
                    ),

                    "fgsm2_mAP50-95": (
                        fgsm2[
                            "overall"
                        ]["mAP50-95"]
                    ),

                    "mean_recovery_mAP50": (
                        result[
                            "recovery_rates"
                        ][
                            "mean_mAP50"
                        ]
                    ),

                    "mean_recovery_mAP50-95": (
                        result[
                            "recovery_rates"
                        ][
                            "mean_mAP50-95"
                        ]
                    ),

                    "clean_defense_ms": (
                        defense_quality.get(
                            "defense_ms_per_image"
                        )
                    ),

                    "clean_defended_vs_clean_psnr": (
                        defense_quality.get(
                            "defended_vs_clean_psnr"
                        )
                    ),

                    "clean_defended_vs_clean_ssim": (
                        defense_quality.get(
                            "defended_vs_clean_ssim"
                        )
                    ),
                }
            )


# ============================================================
# ВЫВОД
# ============================================================

def print_selected_results(
    selected_results: list[
        dict[str, Any]
    ],
) -> None:
    print("\n" + "=" * 156)
    print("ЛУЧШИЕ ЗАЩИТЫ НА VALIDATION")
    print("=" * 156)

    print(
        f"{'Method':>12}"
        f"{'Tier':>11}"
        f"{'Identifier':>42}"
        f"{'Score':>11}"
        f"{'Clean AP50':>13}"
        f"{'Drop %':>10}"
        f"{'FGSM1 AP50':>13}"
        f"{'FGSM2 AP50':>13}"
        f"{'Rec AP50':>12}"
        f"{'Rec AP':>12}"
    )

    for result in selected_results:
        attacks = result[
            "attack_results"
        ]

        fgsm1 = attacks["1"]
        fgsm2 = attacks["2"]

        print(
            f"{result['method']:>12}"

            f"{result['selection_tier']:>11}"

            f"{result['identifier']:>42}"

            f"{result_score(result):>11.4f}"

            f"{result['clean_defended']['mAP50']:>13.4f}"

            f"{result['clean_drop']['mAP50_relative'] * 100.0:>10.2f}"

            f"{fgsm1['overall']['mAP50']:>13.4f}"

            f"{fgsm2['overall']['mAP50']:>13.4f}"

            f"{result['recovery_rates']['mean_mAP50']:>12.4f}"

            f"{result['recovery_rates']['mean_mAP50-95']:>12.4f}"
        )


# ============================================================
# ЗАПУСК
# ============================================================

def main() -> None:
    torch.manual_seed(
        SEED
    )

    torch.cuda.manual_seed_all(
        SEED
    )

    np.random.seed(
        SEED
    )

    check_environment()
    prepare_directories()

    fixed_tnorm_config = (
        load_fixed_tnorm_config()
    )

    print(
        "\nЗафиксированная T-норма:"
    )

    print(
        f"  "
        f"{fixed_tnorm_config.identifier}"
    )

    evaluation_model = YOLO(
        str(MODEL_PATH),
        verbose=False,
    )

    references = (
        load_or_create_references(
            evaluation_model
        )
    )

    print(
        "\nКонтрольные validation-метрики:"
    )

    print(
        "clean: "
        f"mAP50="
        f"{references['clean']['overall']['mAP50']:.4f}, "
        f"mAP50-95="
        f"{references['clean']['overall']['mAP50-95']:.4f}"
    )

    for epsilon_pixels in (
        ATTACK_EPSILONS
    ):
        reference = references[
            f"fgsm_{epsilon_pixels}"
        ]["overall"]

        print(
            f"FGSM {epsilon_pixels}/255: "
            f"mAP50="
            f"{reference['mAP50']:.4f}, "
            f"mAP50-95="
            f"{reference['mAP50-95']:.4f}"
        )

    configs = (
        create_baseline_configs()
        + [fixed_tnorm_config]
    )

    print("\n" + "=" * 108)
    print("ОЦЕНКА ВСЕХ ЗАЩИТ")
    print("=" * 108)

    print(
        f"Количество конфигураций: "
        f"{len(configs)}"
    )

    all_results: list[
        dict[str, Any]
    ] = []

    for (
        index,
        config,
    ) in enumerate(
        configs,
        start=1,
    ):
        print(
            f"\nКонфигурация "
            f"{index}/"
            f"{len(configs)}: "
            f"{config.identifier}"
        )

        result = evaluate_candidate(
            evaluation_model=(
                evaluation_model
            ),

            config=config,

            references=references,
        )

        all_results.append(
            result
        )

    selected_by_method = (
        select_best_per_method(
            all_results
        )
    )

    selected_results = [
        selected_by_method[method]

        for method in [
            "median",
            "gaussian",
            "jpeg",
            "bilateral",
            "tnorm",
        ]

        if method in selected_by_method
    ]

    selected_results = sorted(
        selected_results,

        key=lambda result: (
            selection_tier_priority(
                result
            ),

            -result_score(
                result
            ),
        ),
    )

    selected_identifiers = {
        result["identifier"]

        for result
        in selected_results
    }

    pareto_identifiers = (
        calculate_pareto_identifiers(
            all_results
        )
    )

    save_results_csv(
        results=all_results,

        selected_identifiers=(
            selected_identifiers
        ),

        pareto_identifiers=(
            pareto_identifiers
        ),
    )

    comparison_data = {
        "comparison_version": (
            COMPARISON_VERSION
        ),

        "split": "val",

        "model": str(
            MODEL_PATH
        ),

        "dataset": str(
            DATA_YAML
        ),

        "attack": "FGSM",

        "attack_epsilons": (
            ATTACK_EPSILONS
        ),

        "selection_constraints": {
            "primary": {
                "max_clean_mAP50_relative_drop": (
                    PRIMARY_MAX_CLEAN_MAP50_RELATIVE_DROP
                ),

                "max_clean_mAP50-95_absolute_drop": (
                    PRIMARY_MAX_CLEAN_MAP50_95_ABSOLUTE_DROP
                ),
            },

            "relaxed": {
                "max_clean_mAP50_relative_drop": (
                    RELAXED_MAX_CLEAN_MAP50_RELATIVE_DROP
                ),

                "max_clean_mAP50-95_absolute_drop": (
                    RELAXED_MAX_CLEAN_MAP50_95_ABSOLUTE_DROP
                ),
            },
        },

        "selection_score": (
            "0.60 * mean recovery mAP50 + "
            "0.40 * mean recovery mAP50-95 - "
            "0.10 * clean mAP50 relative drop - "
            "0.05 * clean mAP50-95 relative drop"
        ),

        "references": references,

        "all_results": (
            all_results
        ),

        "selected_results": (
            selected_results
        ),

        "pareto_identifiers": sorted(
            pareto_identifiers
        ),
    }

    write_json(
        OUTPUT_DIR
        / "defense_comparison_val.json",

        comparison_data,
    )

    write_json(
        OUTPUT_DIR
        / "best_baseline_defenses.json",

        {
            "comparison_version": (
                COMPARISON_VERSION
            ),

            "selection_split": "val",

            "selection_rule": (
                "Для каждого метода сначала "
                "выбирается лучший primary-кандидат. "
                "Если его нет, лучший relaxed. "
                "Если нет и relaxed, используется "
                "fallback с наибольшим penalized score."
            ),

            "selected_by_method": (
                selected_by_method
            ),

            "selected_results": (
                selected_results
            ),
        },
    )

    write_json(
        OUTPUT_DIR
        / "pareto_frontier.json",

        {
            "criteria": {
                "minimize": [
                    "clean mAP50 relative drop",
                    "clean defense ms per image",
                ],

                "maximize": [
                    "mean recovery mAP50",
                ],
            },

            "identifiers": sorted(
                pareto_identifiers
            ),

            "results": [
                result

                for result in all_results

                if (
                    result["identifier"]
                    in pareto_identifiers
                )
            ],
        },
    )

    print_selected_results(
        selected_results
    )

    print("\n" + "=" * 156)
    print("СРАВНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 156)

    print(
        "Выбранные конфигурации:\n"
        f"{OUTPUT_DIR / 'best_baseline_defenses.json'}"
    )

    print(
        "\nПолная таблица:\n"
        f"{OUTPUT_DIR / 'defense_comparison_val.csv'}"
    )

    print(
        "\nПолный JSON:\n"
        f"{OUTPUT_DIR / 'defense_comparison_val.json'}"
    )

    print(
        "\nPareto-фронт:\n"
        f"{OUTPUT_DIR / 'pareto_frontier.json'}"
    )

    print(
        "\nФинальная оценка на test "
        "не запускалась."
    )


if __name__ == "__main__":
    main()