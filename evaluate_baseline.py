from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


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
    / "evaluation"
)

RUN_NAME = "clean_test_baseline"

IMAGE_SIZE = 1280
BATCH_SIZE = 2
DEVICE = 0
WORKERS = 4


def check_environment() -> None:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Не найден data.yaml:\n{DATA_YAML}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Не найдена итоговая модель:\n{MODEL_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA недоступна."
        )

    run_directory = OUTPUT_DIR / RUN_NAME

    if run_directory.exists():
        raise FileExistsError(
            "Папка результатов уже существует:\n"
            f"{run_directory}\n\n"
            "Удалите только эту папку перед повторным запуском."
        )

    print("=" * 72)
    print("ОЦЕНКА BASELINE НА ЧИСТОЙ TEST-ВЫБОРКЕ")
    print("=" * 72)
    print(f"Модель: {MODEL_PATH}")
    print(f"Датасет: {DATA_YAML}")
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    print(f"Размер изображения: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")


def to_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())

    return float(value)


def main() -> None:
    check_environment()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",

        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,

        device=DEVICE,
        workers=WORKERS,

        plots=True,
        save_json=False,

        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=False,

        verbose=True,
    )

    run_directory = OUTPUT_DIR / RUN_NAME

    class_names = metrics.names

    precision = np.asarray(metrics.box.p, dtype=float)
    recall = np.asarray(metrics.box.r, dtype=float)
    ap50 = np.asarray(metrics.box.ap50, dtype=float)
    ap50_95 = np.asarray(metrics.box.ap, dtype=float)
    f1 = np.asarray(metrics.box.f1, dtype=float)

    rows = []

    for class_id in range(len(class_names)):
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "mAP50": float(ap50[class_id]),
                "mAP50-95": float(ap50_95[class_id]),
            }
        )

    csv_path = run_directory / "clean_test_metrics_by_class.csv"

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "class_id",
                "class_name",
                "precision",
                "recall",
                "f1",
                "mAP50",
                "mAP50-95",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": str(MODEL_PATH),
        "dataset": str(DATA_YAML),
        "split": "test",
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "overall": {
            "precision": to_float(
                metrics.results_dict["metrics/precision(B)"]
            ),
            "recall": to_float(
                metrics.results_dict["metrics/recall(B)"]
            ),
            "mAP50": to_float(
                metrics.results_dict["metrics/mAP50(B)"]
            ),
            "mAP50-95": to_float(
                metrics.results_dict["metrics/mAP50-95(B)"]
            ),
            "fitness": to_float(
                metrics.results_dict["fitness"]
            ),
        },
        "speed_ms_per_image": {
            key: to_float(value)
            for key, value in metrics.speed.items()
        },
        "classes": rows,
    }

    json_path = run_directory / "clean_test_metrics.json"

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("ОЦЕНКА TEST-ВЫБОРКИ ЗАВЕРШЕНА")
    print("=" * 72)

    print(
        f"Precision:  "
        f"{summary['overall']['precision']:.4f}"
    )
    print(
        f"Recall:     "
        f"{summary['overall']['recall']:.4f}"
    )
    print(
        f"mAP50:      "
        f"{summary['overall']['mAP50']:.4f}"
    )
    print(
        f"mAP50-95:   "
        f"{summary['overall']['mAP50-95']:.4f}"
    )

    print("\nМетрики по классам:")

    for row in rows:
        print(
            f"{row['class_name']:15s} "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} "
            f"mAP50={row['mAP50']:.4f} "
            f"mAP50-95={row['mAP50-95']:.4f}"
        )

    print(f"\nРезультаты:\n{run_directory}")
    print(f"CSV:\n{csv_path}")
    print(f"JSON:\n{json_path}")


if __name__ == "__main__":
    main()