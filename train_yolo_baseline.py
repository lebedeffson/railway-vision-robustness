from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO


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

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "training"
)

PRETRAINED_MODEL = "yolo11m.pt"

STAGE1_NAME = "yolo11m_baseline_stage1"
STAGE2_NAME = "yolo11m_baseline_stage2"


# ============================================================
# ОБЩИЕ ПАРАМЕТРЫ
# ============================================================

IMAGE_SIZE = 1280
# RTX 4060 Laptop has 8 GiB; batch=2 at 1280 px can exceed VRAM in the
# unfrozen stage. Keep the effective protocol explicit and reproducible.
BATCH_SIZE = 1
DEVICE = 0
WORKERS = 2
SEED = 2026
EPOCH_COOLDOWN_SECONDS = int(os.environ.get("TNORM_EPOCH_COOLDOWN_SECONDS", "0"))


def low_impact_epoch_cooldown(_trainer: object) -> None:
    """Give the desktop a short idle window between training epochs."""
    if EPOCH_COOLDOWN_SECONDS > 0:
        time.sleep(EPOCH_COOLDOWN_SECONDS)


def configure_low_impact_runtime() -> None:
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def check_environment() -> None:
    configure_low_impact_runtime()
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Не найден файл датасета:\n{DATA_YAML}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна. Обучение на CPU не запускается."
        )

    gpu_properties = torch.cuda.get_device_properties(DEVICE)
    gpu_memory_gb = gpu_properties.total_memory / 1024**3

    print("=" * 72)
    print("ДВУХЭТАПНОЕ ДООБУЧЕНИЕ YOLO11m")
    print("=" * 72)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"VRAM: {gpu_memory_gb:.2f} ГБ")
    print(f"Датасет: {DATA_YAML}")
    print(f"Размер изображения: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")


def check_output_directories() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage1_directory = OUTPUT_DIR / STAGE1_NAME
    stage2_directory = OUTPUT_DIR / STAGE2_NAME

    for directory in (stage1_directory, stage2_directory):
        if not directory.exists():
            continue
        weights = directory / "weights"
        if not (weights / "best.pt").is_file() and not (weights / "last.pt").is_file():
            raise FileExistsError(
                "Найдена незавершённая папка без checkpoint:\n"
                f"{directory}\n"
                "Автоматическое удаление запрещено."
            )


def completed_or_resumable_model(stage_name: str) -> tuple[Path | None, bool]:
    stage_directory = OUTPUT_DIR / stage_name
    weights = stage_directory / "weights"
    best = weights / "best.pt"
    last = weights / "last.pt"
    marker = stage_directory / "TRAINING_COMPLETE"
    if marker.is_file() and best.is_file():
        return best, False
    if last.is_file():
        return last, True
    return None, False


def mark_stage_complete(stage_name: str, best_model_path: Path) -> None:
    marker = OUTPUT_DIR / stage_name / "TRAINING_COMPLETE"
    marker.write_text(f"best_model={best_model_path}\n", encoding="utf-8")


def train_stage1() -> Path:
    """
    Первый этап.

    Замораживаем ранние слои YOLO и обучаем преимущественно
    детектирующую голову и поздние слои. Это снижает риск быстро
    разрушить предобученные признаки.
    """

    print("\n" + "=" * 72)
    print("ЭТАП 1: ОБУЧЕНИЕ С ЗАМОРОЖЕННЫМ BACKBONE")
    print("=" * 72)

    checkpoint, should_resume = completed_or_resumable_model(STAGE1_NAME)
    if checkpoint is not None and not should_resume:
        print(f"Этап 1 уже завершён: {checkpoint}")
        return checkpoint

    model = YOLO(str(checkpoint) if should_resume else PRETRAINED_MODEL)
    model.add_callback("on_train_epoch_end", low_impact_epoch_cooldown)

    if should_resume:
        print(f"Продолжаю этап 1 из checkpoint: {checkpoint}")
        model.train(resume=True)
        best_model_path = OUTPUT_DIR / STAGE1_NAME / "weights" / "best.pt"
        if not best_model_path.is_file():
            raise FileNotFoundError(f"После resume не найден {best_model_path}")
        mark_stage_complete(STAGE1_NAME, best_model_path)
        return best_model_path

    model.train(
        data=str(DATA_YAML),
        task="detect",

        epochs=20,
        patience=20,

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,
        workers=WORKERS,

        optimizer="AdamW",

        lr0=0.0003,
        lrf=0.10,

        momentum=0.937,
        weight_decay=0.0005,

        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.05,

        # Замораживаются первые 10 модулей модели.
        freeze=10,

        pretrained=True,
        amp=True,
        cos_lr=True,

        # Умеренные аугментации.
        hsv_h=0.010,
        hsv_s=0.35,
        hsv_v=0.25,

        degrees=0.0,
        translate=0.05,
        scale=0.20,
        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.5,

        mosaic=0.30,
        mixup=0.0,
        cutmix=0.0,

        close_mosaic=5,

        cache=False,

        seed=SEED,
        deterministic=True,

        val=True,
        plots=True,

        save=True,
        save_period=5,

        project=str(OUTPUT_DIR),
        name=STAGE1_NAME,
        exist_ok=False,

        verbose=True,
    )

    best_model_path = (
        OUTPUT_DIR
        / STAGE1_NAME
        / "weights"
        / "best.pt"
    )

    if not best_model_path.is_file():
        raise FileNotFoundError(
            "После первого этапа не найден файл:\n"
            f"{best_model_path}"
        )

    mark_stage_complete(STAGE1_NAME, best_model_path)

    print("\nПервый этап завершён.")
    print(f"Лучшая модель этапа 1:\n{best_model_path}")

    return best_model_path


def train_stage2(stage1_best_model: Path) -> Path:
    """
    Второй этап.

    Загружаем лучшую модель первого этапа, размораживаем все слои
    и выполняем осторожное полное дообучение.
    """

    print("\n" + "=" * 72)
    print("ЭТАП 2: ПОЛНОЕ ДООБУЧЕНИЕ МОДЕЛИ")
    print("=" * 72)

    checkpoint, should_resume = completed_or_resumable_model(STAGE2_NAME)
    if checkpoint is not None and not should_resume:
        print(f"Этап 2 уже завершён: {checkpoint}")
        return checkpoint

    model = YOLO(str(checkpoint) if should_resume else str(stage1_best_model))
    model.add_callback("on_train_epoch_end", low_impact_epoch_cooldown)

    if should_resume:
        print(f"Продолжаю этап 2 из checkpoint: {checkpoint}")
        model.train(resume=True)
        best_model_path = OUTPUT_DIR / STAGE2_NAME / "weights" / "best.pt"
        if not best_model_path.is_file():
            raise FileNotFoundError(f"После resume не найден {best_model_path}")
        mark_stage_complete(STAGE2_NAME, best_model_path)
        return best_model_path

    model.train(
        data=str(DATA_YAML),
        task="detect",

        epochs=150,
        patience=50,

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,
        workers=WORKERS,

        optimizer="AdamW",

        # Learning rate ниже, чем на первом этапе.
        lr0=0.0001,
        lrf=0.10,

        momentum=0.937,
        weight_decay=0.0005,

        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.02,

        # Ноль означает, что замороженных слоёв нет.
        freeze=0,

        amp=True,
        cos_lr=True,

        # Более слабые аугментации для точной настройки.
        hsv_h=0.008,
        hsv_s=0.25,
        hsv_v=0.20,

        degrees=0.0,
        translate=0.04,
        scale=0.15,
        shear=0.0,
        perspective=0.0,

        flipud=0.0,
        fliplr=0.5,

        mosaic=0.20,
        mixup=0.0,
        cutmix=0.0,

        close_mosaic=15,

        cache=False,

        seed=SEED,
        deterministic=True,

        val=True,
        plots=True,

        save=True,
        save_period=10,

        project=str(OUTPUT_DIR),
        name=STAGE2_NAME,
        exist_ok=False,

        verbose=True,
    )

    best_model_path = (
        OUTPUT_DIR
        / STAGE2_NAME
        / "weights"
        / "best.pt"
    )

    if not best_model_path.is_file():
        raise FileNotFoundError(
            "После второго этапа не найден файл:\n"
            f"{best_model_path}"
        )

    mark_stage_complete(STAGE2_NAME, best_model_path)

    return best_model_path


def main() -> None:
    check_environment()
    check_output_directories()

    stage1_best_model = train_stage1()
    final_best_model = train_stage2(stage1_best_model)

    print("\n" + "=" * 72)
    print("ДВУХЭТАПНОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 72)

    print(
        "Итоговая лучшая модель:\n"
        f"{final_best_model}"
    )

    print(
        "\nРезультаты первого этапа:\n"
        f"{OUTPUT_DIR / STAGE1_NAME}"
    )

    print(
        "\nРезультаты второго этапа:\n"
        f"{OUTPUT_DIR / STAGE2_NAME}"
    )


if __name__ == "__main__":
    main()
