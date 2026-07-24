from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from ultralytics import YOLO

from scripts.person_v5.common import OUTPUT, PROJECT, atomic_json, atomic_text


PERSON_V3_DATA = PROJECT / "outputs/person_v3/dataset/folds"
RUNTIME = PROJECT / "configs/canonical_v5_person_data_first_runtime.yaml"


def label_path(image: Path) -> Path:
    value = str(image)
    if "/images/" not in value:
        raise RuntimeError(f"Cannot derive label path: {image}")
    return Path(value.replace("/images/", "/labels/")).with_suffix(".txt")


def labels(image: Path) -> list[list[float]]:
    path = label_path(image)
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if int(values[0]) != 0:
            raise RuntimeError(f"Non-person label in v5 dataset: {path}")
        result.append(values[1:5])
    return result


def xywh_to_xyxy(
    box: list[float], width: int, height: int
) -> list[float]:
    x, y, w, h = box
    return [
        (x - w / 2) * width,
        (y - h / 2) * height,
        (x + w / 2) * width,
        (y + h / 2) * height,
    ]


def iou(left: list[float], right: list[float]) -> float:
    intersection = max(
        0.0, min(left[2], right[2]) - max(left[0], right[0])
    ) * max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    left_area = max(0.0, left[2] - left[0]) * max(
        0.0, left[3] - left[1]
    )
    right_area = max(0.0, right[2] - right[0]) * max(
        0.0, right[3] - right[1]
    )
    return intersection / max(left_area + right_area - intersection, 1e-12)


def mine(fold: int, checkpoint: Path) -> Path:
    config = yaml.safe_load(RUNTIME.read_text(encoding="utf-8"))[
        "hard_mining"
    ]
    output = OUTPUT / f"railway/fold_{fold}/hard_mining"
    marker = output / "HARD_MINING_COMPLETE.json"
    data_path = PERSON_V3_DATA / f"fold_{fold}/data.yaml"
    source = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    train_list = Path(source["train"])
    validation_list = Path(source["val"])
    images = [
        Path(line) for line in train_list.read_text().splitlines() if line
    ]
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        mined_data = Path(payload["data_yaml"])
        if mined_data.is_file():
            return mined_data
    model = YOLO(str(checkpoint))
    rows: list[dict[str, Any]] = []
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        targets = labels(image_path)
        target_boxes = [
            xywh_to_xyxy(box, width, height) for box in targets
        ]
        scale = min(640.0 / width, 640.0 / height)
        small = any(
            min(box[2] * width * scale, box[3] * height * scale)
            < float(config["small_person_min_side_after_letterbox_px"])
            for box in targets
        )
        prediction = model.predict(
            source=str(image_path),
            imgsz=640,
            conf=float(config["inference_confidence"]),
            iou=0.70,
            max_det=300,
            device=0,
            verbose=False,
        )[0]
        predicted = (
            [
                {
                    "box": list(map(float, box)),
                    "confidence": float(confidence),
                }
                for box, confidence in zip(
                    prediction.boxes.xyxy.cpu().tolist(),
                    prediction.boxes.conf.cpu().tolist(),
                )
            ]
            if prediction.boxes is not None
            else []
        )
        missed = any(
            not any(
                candidate["confidence"]
                >= float(config["missed_match_confidence"])
                and iou(target, candidate["box"])
                >= float(config["missed_match_IoU"])
                for candidate in predicted
            )
            for target in target_boxes
        )
        maximum_confidence = max(
            (candidate["confidence"] for candidate in predicted),
            default=0.0,
        )
        hard_negative = (
            not targets
            and maximum_confidence
            >= float(config["hard_negative_confidence"])
        )
        rows.append(
            {
                "image": str(image_path),
                "person_GT": len(targets),
                "small_person": small,
                "missed_person": missed,
                "hard_positive": bool(targets) and (small or missed),
                "hard_negative": hard_negative,
                "maximum_prediction_confidence": maximum_confidence,
            }
        )
    positive_rows = []
    positive = [row for row in rows if row["person_GT"]]
    for row in positive:
        weight = (
            int(config["hard_positive_weight"])
            if row["hard_positive"]
            else int(config["regular_positive_weight"])
        )
        positive_rows.extend([row["image"]] * weight)
    target_background = max(
        1,
        int(
            round(
                len(positive_rows)
                * float(config["target_background_fraction"])
                / (1.0 - float(config["target_background_fraction"]))
            )
        ),
    )
    background = sorted(
        [row for row in rows if not row["person_GT"]],
        key=lambda row: (
            not row["hard_negative"],
            -row["maximum_prediction_confidence"],
            row["image"],
        ),
    )
    background_rows = [row["image"] for row in background[:target_background]]
    hard = [
        row["image"]
        for row in background
        if row["hard_negative"] and row["image"] in background_rows
    ]
    for image in hard:
        if len(background_rows) >= target_background:
            break
        background_rows.append(image)
    emitted = positive_rows + background_rows
    output.mkdir(parents=True, exist_ok=True)
    with (output / "hard_mining.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mined_train = output / "train.txt"
    atomic_text(mined_train, "\n".join(emitted) + "\n")
    data = {
        "path": source["path"],
        "train": str(mined_train.resolve()),
        "val": str(validation_list.resolve()),
        "nc": 1,
        "names": {0: "person"},
    }
    mined_data = output / "data.yaml"
    atomic_text(mined_data, yaml.safe_dump(data, sort_keys=False))
    background_fraction = len(background_rows) / max(len(emitted), 1)
    if not 0.20 <= background_fraction <= 0.30:
        raise RuntimeError(
            f"Hard-mined background fraction invalid: {background_fraction}"
        )
    atomic_json(
        marker,
        {
            "status": "PASS",
            "fold": fold,
            "source_tiles": len(rows),
            "hard_positive_tiles": sum(
                int(row["hard_positive"]) for row in rows
            ),
            "hard_negative_tiles": sum(
                int(row["hard_negative"]) for row in rows
            ),
            "emitted_rows": len(emitted),
            "background_fraction": background_fraction,
            "maximum_frame_weight": int(
                config["hard_positive_weight"]
            ),
            "data_yaml": str(mined_data.resolve()),
            "heldout_mining": False,
            "test_used": False,
        },
    )
    return mined_data
