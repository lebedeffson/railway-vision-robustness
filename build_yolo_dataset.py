from __future__ import annotations

import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from download_osdar23_direct import RAW_EXCLUSIONS_PATH, load_frame_exclusions


# =====================================================================
# НАСТРОЙКИ
# =====================================================================

PROJECT_DIR = Path(__file__).resolve().parent

RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "yolo_osdar23"
RAW_AUDIT_DIR = PROJECT_DIR / "outputs" / "final_practice" / "00_audit"

STREAM_NAME = "rgb_highres_center"

CANDIDATE_CLASSES = [
    "person",
    "signal",
    "road_vehicle",
    "train",
    "animal",
    "bicycle",
]

# Класс должен встречаться хотя бы в трёх независимых сценариях,
# чтобы его можно было представить в train, val и test.
MIN_GROUPS_PER_CLASS = 3

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 2026
SEARCH_ITERATIONS = 100_000

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
}

# На Windows hardlink не занимает дополнительное место.
# Если создать ссылку не получится, изображение будет скопировано.
USE_HARDLINKS = True


# =====================================================================
# ОБЩИЕ ФУНКЦИИ
# =====================================================================

def get_group_name(sequence_name: str) -> str:
    """
    3_fire_site_3.1 -> 3_fire_site_3

    Все части одного сценария должны находиться
    только в одном split.
    """

    if "." not in sequence_name:
        return sequence_name

    return sequence_name.rsplit(".", 1)[0]


def find_labels_file(sequence_dir: Path) -> Path:
    expected = sequence_dir / f"{sequence_dir.name}_labels.json"

    if expected.is_file():
        return expected

    candidates = sorted(
        sequence_dir.glob("*_labels.json")
    )

    if len(candidates) == 1:
        return candidates[0]

    raise FileNotFoundError(
        f"Не найден однозначный файл разметки:\n"
        f"{sequence_dir}"
    )


def discover_sequences() -> list[Path]:
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"Папка data/raw не найдена:\n{RAW_DIR}"
        )

    sequences: list[Path] = []

    for path in RAW_DIR.iterdir():
        if not path.is_dir():
            continue

        camera_dir = path / STREAM_NAME

        if not camera_dir.is_dir():
            continue

        try:
            find_labels_file(path)
        except FileNotFoundError:
            continue

        sequences.append(path)

    return sorted(
        sequences,
        key=lambda path: path.name,
    )


def load_openlabel(sequence_dir: Path) -> dict[str, Any]:
    labels_file = find_labels_file(sequence_dir)

    with labels_file.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        payload = json.load(file)

    openlabel = payload.get("openlabel")

    if not isinstance(openlabel, dict):
        raise ValueError(
            f"В JSON отсутствует раздел openlabel:\n"
            f"{labels_file}"
        )

    return openlabel


def get_object_types(
    openlabel: dict[str, Any],
) -> dict[str, str]:
    result: dict[str, str] = {}

    objects = openlabel.get("objects", {})

    if not isinstance(objects, dict):
        return result

    for object_id, object_data in objects.items():
        if not isinstance(object_data, dict):
            continue

        object_type = object_data.get("type")

        if isinstance(object_type, str):
            result[str(object_id)] = object_type

    return result


def normalize_shapes(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]

    if isinstance(value, list):
        return [
            item
            for item in value
            if isinstance(item, dict)
        ]

    return []


def belongs_to_stream(
    shape: dict[str, Any],
) -> bool:
    coordinate_system = shape.get(
        "coordinate_system"
    )

    if coordinate_system == STREAM_NAME:
        return True

    name = shape.get("name")

    return (
        isinstance(name, str)
        and name.startswith(f"{STREAM_NAME}__")
    )


def resolve_image_path(
    sequence_dir: Path,
    uri: str,
) -> Path:
    clean_uri = uri.replace("\\", "/").lstrip("/")

    return sequence_dir / Path(clean_uri)


# =====================================================================
# АНАЛИЗ КЛАССОВ ПО СЦЕНАРИЯМ
# =====================================================================

def count_sequence_classes(
    sequence_dir: Path,
) -> tuple[int, Counter[str]]:
    openlabel = load_openlabel(sequence_dir)
    object_types = get_object_types(openlabel)

    frames = openlabel.get("frames", {})

    if not isinstance(frames, dict):
        return 0, Counter()

    image_count = 0
    class_counter: Counter[str] = Counter()

    for frame_data in frames.values():
        if not isinstance(frame_data, dict):
            continue

        frame_properties = frame_data.get(
            "frame_properties",
            {},
        )

        stream_data = (
            frame_properties
            .get("streams", {})
            .get(STREAM_NAME)
        )

        if not isinstance(stream_data, dict):
            continue

        uri = stream_data.get("uri")

        if not isinstance(uri, str):
            continue

        image_path = resolve_image_path(
            sequence_dir,
            uri,
        )

        if not image_path.is_file():
            continue

        image_count += 1

        frame_objects = frame_data.get(
            "objects",
            {},
        )

        if not isinstance(frame_objects, dict):
            continue

        for object_id, frame_object in frame_objects.items():
            if not isinstance(frame_object, dict):
                continue

            object_type = object_types.get(
                str(object_id)
            )

            if object_type not in CANDIDATE_CLASSES:
                continue

            object_data = frame_object.get(
                "object_data",
                {},
            )

            if not isinstance(object_data, dict):
                continue

            has_geometry = False

            for geometry_type in ("bbox", "poly2d"):
                shapes = normalize_shapes(
                    object_data.get(geometry_type)
                )

                if any(
                    belongs_to_stream(shape)
                    for shape in shapes
                ):
                    has_geometry = True
                    break

            if has_geometry:
                class_counter[object_type] += 1

    return image_count, class_counter


def analyze_groups(
    sequences: list[Path],
) -> tuple[
    dict[str, dict[str, Any]],
    list[str],
]:
    group_stats: dict[str, dict[str, Any]] = {}
    class_groups: dict[str, set[str]] = defaultdict(set)

    for sequence_dir in sequences:
        sequence_name = sequence_dir.name
        group_name = get_group_name(sequence_name)

        image_count, class_counter = (
            count_sequence_classes(sequence_dir)
        )

        if group_name not in group_stats:
            group_stats[group_name] = {
                "sequences": [],
                "images": 0,
                "classes": Counter(),
            }

        group_stats[group_name]["sequences"].append(
            sequence_name
        )

        group_stats[group_name]["images"] += (
            image_count
        )

        group_stats[group_name]["classes"].update(
            class_counter
        )

        for class_name, count in class_counter.items():
            if count > 0:
                class_groups[class_name].add(
                    group_name
                )

    print("\nПокрытие классов независимыми сценариями:")

    selected_classes: list[str] = []

    for class_name in CANDIDATE_CLASSES:
        group_count = len(
            class_groups.get(class_name, set())
        )

        status = (
            "ВКЛЮЧЁН"
            if group_count >= MIN_GROUPS_PER_CLASS
            else "ИСКЛЮЧЁН"
        )

        print(
            f"  {class_name:20} "
            f"сценариев: {group_count:2} "
            f"{status}"
        )

        if group_count >= MIN_GROUPS_PER_CLASS:
            selected_classes.append(class_name)

    if not selected_classes:
        raise RuntimeError(
            "Ни один класс не подходит для "
            "разделения train/val/test."
        )

    return group_stats, selected_classes


# =====================================================================
# ПОИСК РАЗДЕЛЕНИЯ TRAIN / VAL / TEST
# =====================================================================

def calculate_split_score(
    split_groups: dict[str, list[str]],
    group_stats: dict[str, dict[str, Any]],
    selected_classes: list[str],
) -> float:
    ratios = {
        "train": TRAIN_RATIO,
        "val": VAL_RATIO,
        "test": TEST_RATIO,
    }

    total_images = sum(
        stats["images"]
        for stats in group_stats.values()
    )

    total_classes: Counter[str] = Counter()

    for stats in group_stats.values():
        total_classes.update(
            stats["classes"]
        )

    score = 0.0

    for split_name, groups in split_groups.items():
        ratio = ratios[split_name]

        split_images = sum(
            group_stats[group]["images"]
            for group in groups
        )

        target_images = total_images * ratio

        if target_images > 0:
            image_error = (
                split_images - target_images
            ) / target_images

            score += 2.0 * image_error ** 2

        split_classes: Counter[str] = Counter()

        for group in groups:
            split_classes.update(
                group_stats[group]["classes"]
            )

        for class_name in selected_classes:
            total_count = total_classes[class_name]
            target_count = total_count * ratio
            actual_count = split_classes[class_name]

            if actual_count == 0:
                score += 1000.0
                continue

            if target_count > 0:
                class_error = (
                    actual_count - target_count
                ) / target_count

                score += class_error ** 2

    return score


def find_best_split(
    group_stats: dict[str, dict[str, Any]],
    selected_classes: list[str],
) -> dict[str, list[str]]:
    groups = sorted(group_stats)

    group_count = len(groups)

    val_count = max(
        1,
        round(group_count * VAL_RATIO),
    )

    test_count = max(
        1,
        round(group_count * TEST_RATIO),
    )

    train_count = (
        group_count
        - val_count
        - test_count
    )

    if train_count < 1:
        raise RuntimeError(
            "Недостаточно сценариев для разделения."
        )

    print(
        "\nИщем сбалансированное разделение:"
    )

    print(
        f"  train: {train_count} сценариев"
    )

    print(
        f"  val:   {val_count} сценариев"
    )

    print(
        f"  test:  {test_count} сценариев"
    )

    random_generator = random.Random(
        RANDOM_SEED
    )

    best_score = float("inf")
    best_split: dict[str, list[str]] | None = None

    for _ in range(SEARCH_ITERATIONS):
        shuffled = groups.copy()
        random_generator.shuffle(shuffled)

        candidate = {
            "train": shuffled[:train_count],
            "val": shuffled[
                train_count:
                train_count + val_count
            ],
            "test": shuffled[
                train_count + val_count:
            ],
        }

        score = calculate_split_score(
            split_groups=candidate,
            group_stats=group_stats,
            selected_classes=selected_classes,
        )

        if score < best_score:
            best_score = score

            best_split = {
                split_name: sorted(split_groups)
                for split_name, split_groups
                in candidate.items()
            }

    if best_split is None:
        raise RuntimeError(
            "Не удалось построить разделение."
        )

    print(
        f"Лучший показатель дисбаланса: "
        f"{best_score:.6f}"
    )

    return best_split


# =====================================================================
# КОНВЕРТАЦИЯ ГЕОМЕТРИИ
# =====================================================================

def clip(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(
        minimum,
        min(value, maximum),
    )


def xyxy_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    x1 = clip(x1, 0.0, float(image_width))
    y1 = clip(y1, 0.0, float(image_height))
    x2 = clip(x2, 0.0, float(image_width))
    y2 = clip(y2, 0.0, float(image_height))

    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1

    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0

    return (
        center_x / image_width,
        center_y / image_height,
        width / image_width,
        height / image_height,
    )


def bbox_to_yolo(
    bbox: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    values = bbox.get("val")

    if (
        not isinstance(values, list)
        or len(values) != 4
    ):
        return None

    center_x, center_y, width, height = map(
        float,
        values,
    )

    return xyxy_to_yolo(
        x1=center_x - width / 2.0,
        y1=center_y - height / 2.0,
        x2=center_x + width / 2.0,
        y2=center_y + height / 2.0,
        image_width=image_width,
        image_height=image_height,
    )


def polygon_to_yolo(
    polygon: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    values = polygon.get("val")

    if (
        not isinstance(values, list)
        or len(values) < 6
        or len(values) % 2 != 0
    ):
        return None

    x_values = [
        float(values[index])
        for index in range(0, len(values), 2)
    ]

    y_values = [
        float(values[index])
        for index in range(1, len(values), 2)
    ]

    return xyxy_to_yolo(
        x1=min(x_values),
        y1=min(y_values),
        x2=max(x_values),
        y2=max(y_values),
        image_width=image_width,
        image_height=image_height,
    )


def get_yolo_annotations(
    frame_data: dict[str, Any],
    object_types: dict[str, str],
    class_to_id: dict[str, int],
    image_width: int,
    image_height: int,
) -> tuple[list[str], Counter[str]]:
    """
    Создаёт не более одной YOLO-рамки
    для одного объекта на одном изображении.

    Если объект содержит несколько bbox для камеры,
    они объединяются в одну охватывающую рамку.
    """

    lines: list[str] = []
    counter: Counter[str] = Counter()

    frame_objects = frame_data.get(
        "objects",
        {},
    )

    if not isinstance(frame_objects, dict):
        return lines, counter

    for object_id, frame_object in frame_objects.items():
        if not isinstance(frame_object, dict):
            continue

        object_type = object_types.get(
            str(object_id)
        )

        if object_type not in class_to_id:
            continue

        object_data = frame_object.get(
            "object_data",
            {},
        )

        if not isinstance(object_data, dict):
            continue

        converted_boxes: list[
            tuple[float, float, float, float]
        ] = []

        # Сначала используем bbox выбранной камеры.
        for bbox in normalize_shapes(
            object_data.get("bbox")
        ):
            if not belongs_to_stream(bbox):
                continue

            converted = bbox_to_yolo(
                bbox=bbox,
                image_width=image_width,
                image_height=image_height,
            )

            if converted is not None:
                converted_boxes.append(converted)

        # Если bbox нет, используем рамки вокруг poly2d.
        if not converted_boxes:
            for polygon in normalize_shapes(
                object_data.get("poly2d")
            ):
                if not belongs_to_stream(polygon):
                    continue

                converted = polygon_to_yolo(
                    polygon=polygon,
                    image_width=image_width,
                    image_height=image_height,
                )

                if converted is not None:
                    converted_boxes.append(converted)

        if not converted_boxes:
            continue

        # Переводим все нормализованные рамки в xyxy.
        x1_values: list[float] = []
        y1_values: list[float] = []
        x2_values: list[float] = []
        y2_values: list[float] = []

        for center_x, center_y, width, height in converted_boxes:
            x1_values.append(center_x - width / 2.0)
            y1_values.append(center_y - height / 2.0)
            x2_values.append(center_x + width / 2.0)
            y2_values.append(center_y + height / 2.0)

        # Одна охватывающая рамка для одного объекта.
        x1 = max(0.0, min(x1_values))
        y1 = max(0.0, min(y1_values))
        x2 = min(1.0, max(x2_values))
        y2 = min(1.0, max(y2_values))

        if x2 <= x1 or y2 <= y1:
            continue

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        class_id = class_to_id[object_type]

        lines.append(
            f"{class_id} "
            f"{center_x:.8f} "
            f"{center_y:.8f} "
            f"{width:.8f} "
            f"{height:.8f}"
        )

        counter[object_type] += 1

    return lines, counter

# =====================================================================
# СОЗДАНИЕ YOLO-ДАТАСЕТА
# =====================================================================

def prepare_output_directories() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split_name in ("train", "val", "test"):
        (
            OUTPUT_DIR
            / "images"
            / split_name
        ).mkdir(parents=True, exist_ok=True)

        (
            OUTPUT_DIR
            / "labels"
            / split_name
        ).mkdir(parents=True, exist_ok=True)


def link_or_copy(
    source: Path,
    destination: Path,
) -> str:
    if USE_HARDLINKS:
        try:
            os.link(source, destination)
            return "hardlink"

        except OSError:
            pass

    shutil.copy2(source, destination)
    return "copy"


def build_dataset(
    sequences: list[Path],
    split_definition: dict[str, list[str]],
    selected_classes: list[str],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Counter[str]],
    Counter[str],
    list[dict[str, Any]],
]:
    group_to_split: dict[str, str] = {}

    for split_name, groups in split_definition.items():
        for group_name in groups:
            group_to_split[group_name] = split_name

    class_to_id = {
        class_name: class_id
        for class_id, class_name
        in enumerate(selected_classes)
    }

    manifest: list[dict[str, Any]] = []

    split_stats: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    global_class_stats: Counter[str] = Counter()
    frame_exclusions = load_frame_exclusions()
    excluded_rows: list[dict[str, Any]] = []

    for sequence_index, sequence_dir in enumerate(
        sequences,
        start=1,
    ):
        sequence_name = sequence_dir.name
        group_name = get_group_name(sequence_name)
        split_name = group_to_split[group_name]

        print(
            f"[{sequence_index:02}/{len(sequences):02}] "
            f"{split_name:5} "
            f"{sequence_name}"
        )

        openlabel = load_openlabel(sequence_dir)
        object_types = get_object_types(openlabel)

        frames = openlabel.get("frames", {})

        if not isinstance(frames, dict):
            continue

        for frame_id, frame_data in frames.items():
            if not isinstance(frame_data, dict):
                continue

            stream_data = (
                frame_data
                .get("frame_properties", {})
                .get("streams", {})
                .get(STREAM_NAME)
            )

            if not isinstance(stream_data, dict):
                continue

            uri = stream_data.get("uri")

            if not isinstance(uri, str):
                continue

            source_image = resolve_image_path(
                sequence_dir,
                uri,
            )

            source_relative = source_image.resolve().relative_to(
                RAW_DIR.resolve()
            ).as_posix()
            if source_relative in frame_exclusions:
                excluded_rows.append({
                    "sequence_id": group_name,
                    "sequence": sequence_name,
                    "frame_id": str(frame_id),
                    "relative_path": source_relative,
                    "file_exists": source_image.is_file(),
                    "disposition": "EXCLUDED_BEFORE_SPLIT",
                })
                continue

            if not source_image.is_file():
                raise FileNotFoundError(
                    f"Изображение не найдено:\n"
                    f"{source_image}"
                )

            with Image.open(source_image) as image:
                image_width, image_height = image.size

            lines, class_counter = (
                get_yolo_annotations(
                    frame_data=frame_data,
                    object_types=object_types,
                    class_to_id=class_to_id,
                    image_width=image_width,
                    image_height=image_height,
                )
            )

            output_stem = (
                f"{sequence_name}__"
                f"{source_image.stem}"
            )

            output_image = (
                OUTPUT_DIR
                / "images"
                / split_name
                / f"{output_stem}{source_image.suffix.lower()}"
            )

            output_label = (
                OUTPUT_DIR
                / "labels"
                / split_name
                / f"{output_stem}.txt"
            )

            transfer_method = link_or_copy(
                source=source_image,
                destination=output_image,
            )

            output_label.write_text(
                "\n".join(lines)
                + ("\n" if lines else ""),
                encoding="utf-8",
            )

            split_stats[split_name]["images"] += 1
            split_stats[split_name]["annotations"] += len(lines)

            if not lines:
                split_stats[split_name]["empty_images"] += 1

            split_stats[split_name].update(
                {
                    f"class:{class_name}": count
                    for class_name, count
                    in class_counter.items()
                }
            )

            global_class_stats.update(
                class_counter
            )

            manifest.append(
                {
                    "split": split_name,
                    "group": group_name,
                    # Statistical unit used by all downstream CV/bootstrap.
                    # Parts such as ``3_fire_site_3.1`` and ``.2`` deliberately
                    # share one sequence_id so adjacent frames cannot leak.
                    "sequence_id": group_name,
                    "sequence": sequence_name,
                    "frame_id": str(frame_id),
                    "source_image": str(source_image),
                    "output_image": str(output_image),
                    "annotations": len(lines),
                    "transfer_method": transfer_method,
                }
            )

    return (
        manifest,
        split_stats,
        global_class_stats,
        excluded_rows,
    )


# =====================================================================
# СОХРАНЕНИЕ ОТЧЁТОВ
# =====================================================================

def write_data_yaml(
    selected_classes: list[str],
) -> None:
    yaml_lines = [
        f"path: {OUTPUT_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(selected_classes)}",
        "names:",
    ]

    for class_id, class_name in enumerate(
        selected_classes
    ):
        yaml_lines.append(
            f"  {class_id}: {class_name}"
        )

    (
        OUTPUT_DIR / "data.yaml"
    ).write_text(
        "\n".join(yaml_lines) + "\n",
        encoding="utf-8",
    )


def write_manifest(
    manifest: list[dict[str, Any]],
) -> None:
    path = OUTPUT_DIR / "manifest.csv"

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "group",
                "sequence_id",
                "sequence",
                "frame_id",
                "source_image",
                "output_image",
                "annotations",
                "transfer_method",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest)


def write_raw_exclusion_audit(
    excluded_rows: list[dict[str, Any]],
    manifest_rows: int,
) -> None:
    configured = load_frame_exclusions()
    observed = {str(row["relative_path"]) for row in excluded_rows}
    if observed != configured:
        missing_from_labels = sorted(configured - observed)
        unexpected = sorted(observed - configured)
        raise RuntimeError(
            "Raw exclusion policy does not match OpenLABEL references: "
            f"missing_from_labels={missing_from_labels}, unexpected={unexpected}"
        )
    RAW_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_AUDIT_DIR / "raw_frame_exclusions.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sequence_id", "sequence", "frame_id", "relative_path",
                "file_exists", "disposition",
            ],
        )
        writer.writeheader()
        writer.writerows(excluded_rows)
    policy = json.loads(RAW_EXCLUSIONS_PATH.read_text(encoding="utf-8"))
    computed_expected = manifest_rows + len(excluded_rows)
    declared_expected = policy.get("expected_frames")
    declared_usable = policy.get("usable_frames")
    if declared_expected != computed_expected or declared_usable != manifest_rows:
        raise RuntimeError(
            "Raw exclusion policy counts do not match the built dataset: "
            f"declared expected/usable={declared_expected}/{declared_usable}, "
            f"computed={computed_expected}/{manifest_rows}"
        )
    summary = {
        "status": "PASS",
        "policy_id": policy.get("policy_id"),
        "reason": policy.get("reason"),
        "policy_path": str(RAW_EXCLUSIONS_PATH),
        "configured_exclusions": len(configured),
        "observed_exclusions": len(observed),
        "expected_openlabel_frames": computed_expected,
        "usable_manifest_frames": manifest_rows,
        "exclusions_applied_before_split": True,
    }
    (RAW_AUDIT_DIR / "raw_frame_exclusions.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_split_definition(
    split_definition: dict[str, list[str]],
) -> None:
    lines: list[str] = []

    for split_name in ("train", "val", "test"):
        lines.append(
            f"[{split_name.upper()}]"
        )

        for group_name in split_definition[split_name]:
            lines.append(group_name)

        lines.append("")

    (
        OUTPUT_DIR / "split_groups.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 72)
    print("СОЗДАНИЕ YOLO-ДАТАСЕТА OSDaR23")
    print("=" * 72)

    sequences = discover_sequences()

    print(
        f"Последовательностей найдено: "
        f"{len(sequences)}"
    )

    if len(sequences) < 10:
        raise RuntimeError(
            "Слишком мало последовательностей."
        )

    group_stats, selected_classes = (
        analyze_groups(sequences)
    )

    print("\nИтоговые классы:")

    for class_id, class_name in enumerate(
        selected_classes
    ):
        print(
            f"  {class_id}: {class_name}"
        )

    split_definition = find_best_split(
        group_stats=group_stats,
        selected_classes=selected_classes,
    )

    print("\nРазделение сценариев:")

    for split_name in ("train", "val", "test"):
        print(
            f"\n{split_name.upper()}:"
        )

        for group_name in split_definition[split_name]:
            print(f"  {group_name}")

    prepare_output_directories()

    (
        manifest,
        split_stats,
        global_class_stats,
        excluded_rows,
    ) = build_dataset(
        sequences=sequences,
        split_definition=split_definition,
        selected_classes=selected_classes,
    )

    write_data_yaml(selected_classes)
    write_manifest(manifest)
    write_raw_exclusion_audit(excluded_rows, len(manifest))
    write_split_definition(split_definition)

    print("\n" + "=" * 72)
    print("ИТОГ")
    print("=" * 72)

    for split_name in ("train", "val", "test"):
        stats = split_stats[split_name]

        print(
            f"\n{split_name.upper()}:"
        )

        print(
            f"  изображений: "
            f"{stats['images']}"
        )

        print(
            f"  пустых изображений: "
            f"{stats['empty_images']}"
        )

        print(
            f"  аннотаций: "
            f"{stats['annotations']}"
        )

        for class_name in selected_classes:
            print(
                f"  {class_name:18} "
                f"{stats[f'class:{class_name}']}"
            )

    print("\nВСЕ КЛАССЫ:")

    for class_name in selected_classes:
        print(
            f"  {class_name:18} "
            f"{global_class_stats[class_name]}"
        )

    print("\nДатасет создан:")
    print(OUTPUT_DIR)

    print("\nКонфигурация YOLO:")
    print(OUTPUT_DIR / "data.yaml")

    print("\nРазделение сценариев:")
    print(OUTPUT_DIR / "split_groups.txt")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(f"\nОШИБКА:\n{error}")
        raise
