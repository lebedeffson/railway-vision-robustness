from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "dataset_inventory"

STREAM_NAME = "rgb_highres_center"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def sequence_sort_key(name: str) -> tuple[int, int, int, str]:
    """
    Сортировка последовательностей по номеру сценария и части.
    """

    first_match = re.match(r"^(\d+)_", name)
    last_match = re.search(r"_(\d+)\.(\d+)$", name)

    scenario_number = (
        int(first_match.group(1))
        if first_match
        else 9999
    )

    group_number = (
        int(last_match.group(1))
        if last_match
        else 9999
    )

    part_number = (
        int(last_match.group(2))
        if last_match
        else 9999
    )

    return scenario_number, group_number, part_number, name


def get_group_name(sequence_name: str) -> str:
    """
    3_fire_site_3.1 -> 3_fire_site_3

    Все части одного сценария затем должны попадать
    только в один split.
    """

    if "." not in sequence_name:
        return sequence_name

    return sequence_name.rsplit(".", 1)[0]


def find_labels_file(sequence_dir: Path) -> Path | None:
    expected = sequence_dir / f"{sequence_dir.name}_labels.json"

    if expected.is_file():
        return expected

    candidates = sorted(sequence_dir.glob("*_labels.json"))

    if len(candidates) == 1:
        return candidates[0]

    return None


def discover_sequences() -> list[Path]:
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(
            f"Папка датасета не найдена:\n{RAW_DIR}"
        )

    sequences: list[Path] = []

    for directory in RAW_DIR.iterdir():
        if not directory.is_dir():
            continue

        labels_file = find_labels_file(directory)
        camera_dir = directory / STREAM_NAME

        if labels_file is None:
            continue

        if not camera_dir.is_dir():
            continue

        sequences.append(directory)

    return sorted(
        sequences,
        key=lambda path: sequence_sort_key(path.name),
    )


def load_openlabel(labels_file: Path) -> dict[str, Any]:
    with labels_file.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        payload = json.load(file)

    openlabel = payload.get("openlabel")

    if not isinstance(openlabel, dict):
        raise ValueError(
            f"В файле отсутствует openlabel:\n{labels_file}"
        )

    return openlabel


def resolve_image_path(
    sequence_dir: Path,
    uri: str,
) -> Path:
    clean_uri = uri.replace("\\", "/").lstrip("/")
    return sequence_dir / Path(clean_uri)


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


def shape_belongs_to_stream(
    shape: dict[str, Any],
    stream_name: str,
) -> bool:
    coordinate_system = shape.get("coordinate_system")

    if coordinate_system == stream_name:
        return True

    shape_name = shape.get("name")

    if (
        isinstance(shape_name, str)
        and shape_name.startswith(f"{stream_name}__")
    ):
        return True

    return False


def object_has_2d_annotation(
    object_data: dict[str, Any],
    stream_name: str,
) -> bool:
    """
    Объект считается размеченным для камеры,
    если у него есть bbox или poly2d этой камеры.
    """

    for geometry_type in ("bbox", "poly2d"):
        shapes = normalize_shapes(
            object_data.get(geometry_type)
        )

        for shape in shapes:
            if shape_belongs_to_stream(
                shape,
                stream_name,
            ):
                return True

    return False


def count_actual_images(camera_dir: Path) -> int:
    return sum(
        1
        for path in camera_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def analyze_sequence(
    sequence_dir: Path,
) -> tuple[
    dict[str, Any],
    Counter[str],
    list[dict[str, str]],
]:
    sequence_name = sequence_dir.name
    group_name = get_group_name(sequence_name)

    labels_file = find_labels_file(sequence_dir)

    if labels_file is None:
        raise FileNotFoundError(
            f"JSON-разметка не найдена:\n{sequence_dir}"
        )

    openlabel = load_openlabel(labels_file)

    global_objects = openlabel.get("objects", {})
    frames = openlabel.get("frames", {})

    object_types: dict[str, str] = {}

    for object_id, object_info in global_objects.items():
        if not isinstance(object_info, dict):
            continue

        object_type = object_info.get("type")

        if isinstance(object_type, str):
            object_types[str(object_id)] = object_type

    class_counter: Counter[str] = Counter()
    missing_images: list[dict[str, str]] = []

    referenced_images = 0
    existing_images = 0
    annotated_images = 0
    empty_images = 0
    total_annotations = 0
    frames_without_stream = 0

    for frame_id, frame_data in frames.items():
        if not isinstance(frame_data, dict):
            continue

        frame_properties = frame_data.get(
            "frame_properties",
            {},
        )

        streams = frame_properties.get(
            "streams",
            {},
        )

        stream_data = streams.get(STREAM_NAME)

        if not isinstance(stream_data, dict):
            frames_without_stream += 1
            continue

        uri = stream_data.get("uri")

        if not isinstance(uri, str):
            frames_without_stream += 1
            continue

        referenced_images += 1

        image_path = resolve_image_path(
            sequence_dir,
            uri,
        )

        if image_path.is_file():
            existing_images += 1
        else:
            missing_images.append(
                {
                    "sequence": sequence_name,
                    "frame_id": str(frame_id),
                    "uri": uri,
                    "expected_path": str(image_path),
                }
            )

        frame_annotation_count = 0

        frame_objects = frame_data.get(
            "objects",
            {},
        )

        if not isinstance(frame_objects, dict):
            frame_objects = {}

        for object_id, frame_object in frame_objects.items():
            if not isinstance(frame_object, dict):
                continue

            object_type = object_types.get(
                str(object_id),
                "unknown",
            )

            object_data = frame_object.get(
                "object_data",
                {},
            )

            if not isinstance(object_data, dict):
                continue

            if not object_has_2d_annotation(
                object_data,
                STREAM_NAME,
            ):
                continue

            frame_annotation_count += 1
            class_counter[object_type] += 1

        total_annotations += frame_annotation_count

        if frame_annotation_count > 0:
            annotated_images += 1
        else:
            empty_images += 1

    camera_dir = sequence_dir / STREAM_NAME

    sequence_stats = {
        "sequence": sequence_name,
        "group": group_name,
        "global_objects": len(global_objects),
        "json_frames": len(frames),
        "camera_files": count_actual_images(camera_dir),
        "referenced_images": referenced_images,
        "existing_images": existing_images,
        "annotated_images": annotated_images,
        "empty_images": empty_images,
        "annotations": total_annotations,
        "frames_without_stream": frames_without_stream,
        "missing_images": len(missing_images),
    }

    return sequence_stats, class_counter, missing_images


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    print("=" * 72)
    print("АНАЛИЗ ПОЛНОГО OSDaR23")
    print("=" * 72)

    sequences = discover_sequences()

    if not sequences:
        raise RuntimeError(
            f"Подготовленные последовательности не найдены:\n"
            f"{RAW_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Найдено последовательностей: {len(sequences)}")
    print(f"Используемая камера: {STREAM_NAME}\n")

    sequence_rows: list[dict[str, Any]] = []
    per_sequence_classes: list[dict[str, Any]] = []
    missing_image_rows: list[dict[str, str]] = []

    total_classes: Counter[str] = Counter()

    group_numeric_stats: dict[str, Counter[str]] = defaultdict(
        Counter
    )

    group_sequences: dict[str, set[str]] = defaultdict(set)
    group_classes: dict[str, Counter[str]] = defaultdict(Counter)

    for index, sequence_dir in enumerate(
        sequences,
        start=1,
    ):
        print(
            f"[{index:02}/{len(sequences):02}] "
            f"{sequence_dir.name}"
        )

        try:
            (
                sequence_stats,
                class_counter,
                missing_images,
            ) = analyze_sequence(sequence_dir)

        except Exception as error:
            print(f"  ОШИБКА: {error}")
            continue

        sequence_rows.append(sequence_stats)
        missing_image_rows.extend(missing_images)
        total_classes.update(class_counter)

        group_name = str(sequence_stats["group"])
        group_sequences[group_name].add(
            str(sequence_stats["sequence"])
        )

        for numeric_field in (
            "global_objects",
            "json_frames",
            "camera_files",
            "referenced_images",
            "existing_images",
            "annotated_images",
            "empty_images",
            "annotations",
            "frames_without_stream",
            "missing_images",
        ):
            group_numeric_stats[group_name][numeric_field] += int(
                sequence_stats[numeric_field]
            )

        group_classes[group_name].update(
            class_counter
        )

        if class_counter:
            class_text = ", ".join(
                f"{class_name}={count}"
                for class_name, count
                in class_counter.most_common()
            )
        else:
            class_text = "нет 2D-аннотаций"

        print(
            f"  кадры={sequence_stats['existing_images']}, "
            f"аннотации={sequence_stats['annotations']}, "
            f"классы: {class_text}"
        )

        for class_name, count in sorted(
            class_counter.items()
        ):
            per_sequence_classes.append(
                {
                    "sequence": sequence_stats["sequence"],
                    "group": group_name,
                    "class_name": class_name,
                    "annotations": count,
                }
            )

    if not sequence_rows:
        raise RuntimeError(
            "Ни одна последовательность не была прочитана."
        )

    sequence_rows.sort(
        key=lambda row: sequence_sort_key(
            str(row["sequence"])
        )
    )

    sequence_fieldnames = [
        "sequence",
        "group",
        "global_objects",
        "json_frames",
        "camera_files",
        "referenced_images",
        "existing_images",
        "annotated_images",
        "empty_images",
        "annotations",
        "frames_without_stream",
        "missing_images",
    ]

    write_csv(
        OUTPUT_DIR / "sequence_stats.csv",
        sequence_rows,
        sequence_fieldnames,
    )

    write_csv(
        OUTPUT_DIR / "sequence_class_stats.csv",
        per_sequence_classes,
        [
            "sequence",
            "group",
            "class_name",
            "annotations",
        ],
    )

    class_rows = [
        {
            "class_name": class_name,
            "annotations": count,
        }
        for class_name, count
        in total_classes.most_common()
    ]

    write_csv(
        OUTPUT_DIR / "class_stats.csv",
        class_rows,
        [
            "class_name",
            "annotations",
        ],
    )

    group_rows: list[dict[str, Any]] = []

    for group_name in sorted(
        group_numeric_stats,
        key=sequence_sort_key,
    ):
        numeric = group_numeric_stats[group_name]

        group_rows.append(
            {
                "group": group_name,
                "sequences": len(
                    group_sequences[group_name]
                ),
                "json_frames": numeric["json_frames"],
                "existing_images": numeric["existing_images"],
                "annotated_images": numeric["annotated_images"],
                "empty_images": numeric["empty_images"],
                "annotations": numeric["annotations"],
                "missing_images": numeric["missing_images"],
            }
        )

    write_csv(
        OUTPUT_DIR / "group_stats.csv",
        group_rows,
        [
            "group",
            "sequences",
            "json_frames",
            "existing_images",
            "annotated_images",
            "empty_images",
            "annotations",
            "missing_images",
        ],
    )

    group_class_rows: list[dict[str, Any]] = []

    for group_name in sorted(
        group_classes,
        key=sequence_sort_key,
    ):
        for class_name, count in sorted(
            group_classes[group_name].items()
        ):
            group_class_rows.append(
                {
                    "group": group_name,
                    "class_name": class_name,
                    "annotations": count,
                }
            )

    write_csv(
        OUTPUT_DIR / "group_class_stats.csv",
        group_class_rows,
        [
            "group",
            "class_name",
            "annotations",
        ],
    )

    if missing_image_rows:
        write_csv(
            OUTPUT_DIR / "missing_images.csv",
            missing_image_rows,
            [
                "sequence",
                "frame_id",
                "uri",
                "expected_path",
            ],
        )

    totals = Counter()

    for row in sequence_rows:
        for field in (
            "json_frames",
            "camera_files",
            "referenced_images",
            "existing_images",
            "annotated_images",
            "empty_images",
            "annotations",
            "missing_images",
        ):
            totals[field] += int(row[field])

    summary_lines = [
        "OSDaR23 DATASET SUMMARY",
        "=" * 50,
        f"Sequences: {len(sequence_rows)}",
        f"Scenario groups: {len(group_rows)}",
        f"JSON frames: {totals['json_frames']}",
        f"Camera files: {totals['camera_files']}",
        f"Referenced images: {totals['referenced_images']}",
        f"Existing images: {totals['existing_images']}",
        f"Annotated images: {totals['annotated_images']}",
        f"Empty images: {totals['empty_images']}",
        f"2D annotations: {totals['annotations']}",
        f"Missing images: {totals['missing_images']}",
        "",
        "CLASS DISTRIBUTION",
        "=" * 50,
    ]

    for class_name, count in total_classes.most_common():
        summary_lines.append(
            f"{class_name}: {count}"
        )

    summary_path = OUTPUT_DIR / "summary.txt"

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("ИТОГ")
    print("=" * 72)

    print(f"Последовательностей: {len(sequence_rows)}")
    print(f"Групп сценариев: {len(group_rows)}")
    print(f"Кадров в JSON: {totals['json_frames']}")
    print(f"Файлов камеры: {totals['camera_files']}")
    print(f"Найдено изображений: {totals['existing_images']}")
    print(f"Размеченных изображений: {totals['annotated_images']}")
    print(f"Пустых изображений: {totals['empty_images']}")
    print(f"Всего 2D-аннотаций: {totals['annotations']}")
    print(f"Отсутствующих изображений: {totals['missing_images']}")

    print("\nРаспределение классов:")

    for class_name, count in total_classes.most_common():
        print(f"  {class_name:25} {count}")

    print("\nОтчёты сохранены в:")
    print(OUTPUT_DIR)

    print("\nОсновной отчёт:")
    print(summary_path)


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(f"\nОШИБКА:\n{error}")
        raise