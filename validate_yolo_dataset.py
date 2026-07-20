from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "data" / "yolo_osdar23"
DATA_YAML = DATASET_DIR / "data.yaml"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "label_preview"

SPLITS = ("train", "val", "test")

# Количество изображений для визуальной проверки из каждого split.
PREVIEWS_PER_SPLIT = 4

RANDOM_SEED = 2026

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}


def load_class_names() -> list[str]:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Не найден data.yaml:\n{DATA_YAML}"
        )

    with DATA_YAML.open(
        "r",
        encoding="utf-8",
    ) as file:
        config: dict[str, Any] = yaml.safe_load(file)

    names = config.get("names")

    if isinstance(names, list):
        return [str(name) for name in names]

    if isinstance(names, dict):
        normalized_names: dict[int, str] = {}

        for class_id, class_name in names.items():
            normalized_names[int(class_id)] = str(class_name)

        return [
            normalized_names[index]
            for index in sorted(normalized_names)
        ]

    raise ValueError(
        "В data.yaml отсутствует корректный раздел names."
    )


def find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Папка изображений не найдена:\n{directory}"
        )

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_label_file(
    label_path: Path,
    class_names: list[str],
) -> tuple[
    list[tuple[int, float, float, float, float]],
    list[str],
]:
    annotations: list[
        tuple[int, float, float, float, float]
    ] = []

    errors: list[str] = []

    if not label_path.is_file():
        errors.append(
            f"Отсутствует файл разметки: {label_path}"
        )
        return annotations, errors

    text = label_path.read_text(
        encoding="utf-8",
    ).strip()

    if not text:
        return annotations, errors

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"{label_path}, строка {line_number}: "
                f"ожидалось 5 значений, найдено {len(parts)}"
            )
            continue

        try:
            class_id = int(parts[0])

            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            errors.append(
                f"{label_path}, строка {line_number}: "
                "не удалось прочитать числовые значения"
            )
            continue

        if not 0 <= class_id < len(class_names):
            errors.append(
                f"{label_path}, строка {line_number}: "
                f"неизвестный class_id={class_id}"
            )
            continue

        values = {
            "center_x": center_x,
            "center_y": center_y,
            "width": width,
            "height": height,
        }

        for value_name, value in values.items():
            if not 0.0 <= value <= 1.0:
                errors.append(
                    f"{label_path}, строка {line_number}: "
                    f"{value_name}={value} находится вне [0, 1]"
                )

        if width <= 0.0 or height <= 0.0:
            errors.append(
                f"{label_path}, строка {line_number}: "
                f"неположительный размер рамки "
                f"{width} x {height}"
            )

        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0

        tolerance = 1e-5

        if (
            x1 < -tolerance
            or y1 < -tolerance
            or x2 > 1.0 + tolerance
            or y2 > 1.0 + tolerance
        ):
            errors.append(
                f"{label_path}, строка {line_number}: "
                "границы рамки выходят за изображение"
            )

        annotations.append(
            (
                class_id,
                center_x,
                center_y,
                width,
                height,
            )
        )

    return annotations, errors


def validate_split(
    split_name: str,
    class_names: list[str],
) -> tuple[
    dict[str, Any],
    dict[Path, list[tuple[int, float, float, float, float]]],
    list[str],
]:
    image_dir = DATASET_DIR / "images" / split_name
    label_dir = DATASET_DIR / "labels" / split_name

    images = find_images(image_dir)

    all_annotations: dict[
        Path,
        list[tuple[int, float, float, float, float]]
    ] = {}

    errors: list[str] = []

    class_counter: Counter[str] = Counter()

    empty_images = 0
    total_annotations = 0

    image_stems = {
        image_path.stem
        for image_path in images
    }

    label_files = sorted(label_dir.glob("*.txt"))

    label_stems = {
        label_path.stem
        for label_path in label_files
    }

    for missing_label_stem in sorted(
        image_stems - label_stems
    ):
        errors.append(
            f"{split_name}: отсутствует разметка для "
            f"{missing_label_stem}"
        )

    for missing_image_stem in sorted(
        label_stems - image_stems
    ):
        errors.append(
            f"{split_name}: отсутствует изображение для "
            f"{missing_image_stem}.txt"
        )

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        annotations, label_errors = read_label_file(
            label_path=label_path,
            class_names=class_names,
        )

        errors.extend(label_errors)
        all_annotations[image_path] = annotations

        if not annotations:
            empty_images += 1

        total_annotations += len(annotations)

        for class_id, *_ in annotations:
            class_counter[
                class_names[class_id]
            ] += 1

    stats = {
        "images": len(images),
        "labels": len(label_files),
        "empty_images": empty_images,
        "annotations": total_annotations,
        "class_counter": class_counter,
    }

    return stats, all_annotations, errors


def select_preview_images(
    annotations_by_image: dict[
        Path,
        list[tuple[int, float, float, float, float]]
    ],
    number_to_select: int,
) -> list[Path]:
    """
    Выбирает изображения так, чтобы показать
    как можно больше различных классов.
    """

    random_generator = random.Random(
        RANDOM_SEED
    )

    candidates = list(
        annotations_by_image.keys()
    )

    random_generator.shuffle(candidates)

    selected: list[Path] = []
    covered_classes: set[int] = set()

    while (
        candidates
        and len(selected) < number_to_select
    ):
        best_image: Path | None = None
        best_new_classes = -1
        best_annotation_count = -1

        for image_path in candidates:
            annotations = annotations_by_image[
                image_path
            ]

            image_classes = {
                annotation[0]
                for annotation in annotations
            }

            new_classes = len(
                image_classes - covered_classes
            )

            annotation_count = len(annotations)

            if (
                new_classes > best_new_classes
                or (
                    new_classes == best_new_classes
                    and annotation_count
                    > best_annotation_count
                )
            ):
                best_image = image_path
                best_new_classes = new_classes
                best_annotation_count = annotation_count

        if best_image is None:
            break

        selected.append(best_image)

        covered_classes.update(
            annotation[0]
            for annotation
            in annotations_by_image[best_image]
        )

        candidates.remove(best_image)

    return selected


def class_color(class_id: int) -> tuple[int, int, int]:
    colors = [
        (255, 64, 64),
        (64, 255, 64),
        (64, 128, 255),
        (255, 192, 64),
        (192, 64, 255),
        (64, 255, 255),
    ]

    return colors[class_id % len(colors)]


def draw_preview(
    image_path: Path,
    annotations: list[
        tuple[int, float, float, float, float]
    ],
    class_names: list[str],
    output_path: Path,
) -> None:
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    image_width, image_height = image.size

    line_width = max(
        2,
        round(min(image_width, image_height) / 500),
    )

    for (
        class_id,
        center_x,
        center_y,
        width,
        height,
    ) in annotations:
        x1 = int(
            (center_x - width / 2.0)
            * image_width
        )

        y1 = int(
            (center_y - height / 2.0)
            * image_height
        )

        x2 = int(
            (center_x + width / 2.0)
            * image_width
        )

        y2 = int(
            (center_y + height / 2.0)
            * image_height
        )

        color = class_color(class_id)
        label = class_names[class_id]

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=line_width,
        )

        text_box = draw.textbbox(
            (x1, y1),
            label,
            font=font,
        )

        text_width = (
            text_box[2] - text_box[0]
        )

        text_height = (
            text_box[3] - text_box[1]
        )

        text_y = max(
            0,
            y1 - text_height - 6,
        )

        draw.rectangle(
            (
                x1,
                text_y,
                x1 + text_width + 8,
                text_y + text_height + 6,
            ),
            fill=color,
        )

        draw.text(
            (x1 + 4, text_y + 3),
            label,
            fill=(0, 0, 0),
            font=font,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        output_path,
        quality=95,
    )


def main() -> None:
    print("=" * 72)
    print("ПРОВЕРКА YOLO-ДАТАСЕТА")
    print("=" * 72)

    class_names = load_class_names()

    print("\nКлассы:")

    for class_id, class_name in enumerate(
        class_names
    ):
        print(f"  {class_id}: {class_name}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_errors: list[str] = []

    for split_name in SPLITS:
        print("\n" + "-" * 72)
        print(split_name.upper())
        print("-" * 72)

        (
            stats,
            annotations_by_image,
            split_errors,
        ) = validate_split(
            split_name=split_name,
            class_names=class_names,
        )

        all_errors.extend(split_errors)

        print(
            f"Изображений:       {stats['images']}"
        )

        print(
            f"Файлов разметки:   {stats['labels']}"
        )

        print(
            f"Пустых изображений:{stats['empty_images']:8}"
        )

        print(
            f"Аннотаций:         {stats['annotations']}"
        )

        print("Классы:")

        class_counter: Counter[str] = stats[
            "class_counter"
        ]

        for class_name in class_names:
            print(
                f"  {class_name:18} "
                f"{class_counter[class_name]}"
            )

        selected_images = select_preview_images(
            annotations_by_image=annotations_by_image,
            number_to_select=PREVIEWS_PER_SPLIT,
        )

        split_output_dir = (
            OUTPUT_DIR / split_name
        )

        for preview_index, image_path in enumerate(
            selected_images,
            start=1,
        ):
            output_path = (
                split_output_dir
                / (
                    f"{preview_index:02d}_"
                    f"{image_path.stem}.jpg"
                )
            )

            draw_preview(
                image_path=image_path,
                annotations=annotations_by_image[
                    image_path
                ],
                class_names=class_names,
                output_path=output_path,
            )

            print(
                f"Создан пример: {output_path.name}"
            )

    print("\n" + "=" * 72)
    print("РЕЗУЛЬТАТ ПРОВЕРКИ")
    print("=" * 72)

    if all_errors:
        error_path = OUTPUT_DIR / "errors.txt"

        error_path.write_text(
            "\n".join(all_errors) + "\n",
            encoding="utf-8",
        )

        print(
            f"Обнаружено ошибок: {len(all_errors)}"
        )

        print(
            f"Список ошибок:\n{error_path}"
        )

        raise RuntimeError(
            "Датасет содержит ошибки."
        )

    print("Ошибок разметки не обнаружено.")

    print(
        "\nИзображения с нарисованными рамками:"
    )

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()