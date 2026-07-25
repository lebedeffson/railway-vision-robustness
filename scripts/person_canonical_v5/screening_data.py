from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from PIL import Image
from ultralytics import YOLO

from scripts.person_canonical_v5.common import PROJECT, atomic_json
from scripts.person_canonical_v5.materialize_pasting import (
    audit_person_only_labels,
    materialize,
)
from scripts.person_canonical_v5.screening_common import ROOT, config


PERSON_FOLDS = PROJECT / "outputs/person_v3/dataset/folds"
PASTING = PROJECT / "outputs/person_canonical_v5/instance_pasting"


def label_path(image: Path) -> Path:
    value = str(image)
    if "/images/" not in value:
        raise RuntimeError(f"Cannot derive label path: {image}")
    return Path(value.replace("/images/", "/labels/")).with_suffix(".txt")


def person_count(image: Path) -> int:
    path = label_path(image)
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        if int(float(values[0])) != 0:
            raise RuntimeError(f"Non-person label in screening dataset: {path}")
        count += 1
    return count


def _ensure_pasting(fold: int) -> Path:
    root = PASTING / f"fold_{fold}/fraction_25"
    summary = root / "pasting_summary.json"
    if not summary.is_file():
        materialize(fold, 0.25)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    label_audit = payload.get("person_only_label_audit", {})
    if (
        label_audit.get("invalid_class_rows") != 0
        or label_audit.get("invalid_geometry_rows") != 0
    ):
        raise RuntimeError("Pasting person-only label audit failed")
    data = root / "dataset/data.yaml"
    if not data.is_file():
        raise RuntimeError("Pasting dataset did not materialize")
    return data


def _hard_negative_data(fold: int, source_data: Path) -> Path:
    destination = ROOT / f"data/C2/fold_{fold}"
    marker = destination / "HARD_NEGATIVE_COMPLETE.json"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        result = Path(payload["data_yaml"])
        if result.is_file():
            return result
    source = yaml.safe_load(source_data.read_text(encoding="utf-8"))
    train_list = Path(source["train"])
    validation_list = Path(source["val"])
    images = [
        Path(line)
        for line in train_list.read_text(encoding="utf-8").splitlines()
        if line
    ]
    positive = [image for image in images if person_count(image) > 0]
    background = [image for image in images if person_count(image) == 0]
    model = YOLO(str((PROJECT / config()["training"]["initialization"]).resolve()))
    scored: list[tuple[float, str]] = []
    inference_confidence = float(
        config()["hard_negative_sampler"]["inference_confidence"]
    )
    for image in background:
        prediction = model.predict(
            source=str(image),
            imgsz=int(config()["training"]["imgsz"]),
            conf=inference_confidence,
            iou=0.70,
            max_det=300,
            classes=[0],
            device=0,
            verbose=False,
        )[0]
        confidence = (
            max(map(float, prediction.boxes.conf.cpu().tolist()))
            if prediction.boxes is not None and len(prediction.boxes)
            else 0.0
        )
        scored.append((confidence, str(image)))
    fraction = float(
        config()["hard_negative_sampler"]["target_background_fraction"]
    )
    target = min(
        len(scored),
        max(1, int(round(len(positive) * fraction / (1.0 - fraction)))),
    )
    selected_background = [
        image for _, image in sorted(scored, key=lambda row: (-row[0], row[1]))[:target]
    ]
    emitted = [str(image) for image in positive] + selected_background
    destination.mkdir(parents=True, exist_ok=True)
    train = destination / "train.txt"
    train.write_text("\n".join(emitted) + "\n", encoding="utf-8")
    data = {
        "path": source["path"],
        "train": str(train.resolve()),
        "val": str(validation_list.resolve()),
        "nc": 1,
        "names": {0: "person"},
    }
    data_path = destination / "data.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    background_fraction = len(selected_background) / max(len(emitted), 1)
    atomic_json(
        marker,
        {
            "status": "PASS",
            "fold": fold,
            "source_data": str(source_data.resolve()),
            "source_tiles": len(images),
            "positive_tiles": len(positive),
            "background_tiles": len(background),
            "selected_hard_negative_tiles": len(selected_background),
            "background_fraction": background_fraction,
            "maximum_frame_weight": 1,
            "data_yaml": str(data_path.resolve()),
            "heldout_mining": False,
            "test_used": False,
        },
    )
    return data_path


def data_for(candidate: str, fold: int) -> Path:
    if candidate == "C0":
        return PERSON_FOLDS / f"fold_{fold}/data.yaml"
    pasting = _ensure_pasting(fold)
    if candidate == "C1":
        return pasting
    if candidate == "C2":
        return _hard_negative_data(fold, pasting)
    raise ValueError(f"Unknown screening candidate: {candidate}")


def audit_data(candidate: str, fold: int) -> dict[str, Any]:
    data = data_for(candidate, fold)
    payload = yaml.safe_load(data.read_text(encoding="utf-8"))
    images = [
        Path(line)
        for line in Path(payload["train"]).read_text(encoding="utf-8").splitlines()
        if line
    ]
    labels = {label_path(image).parent for image in images}
    audit = {
        "candidate": candidate,
        "fold": fold,
        "data_yaml": str(data.resolve()),
        "train_rows": len(images),
        "missing_images": sum(not image.is_file() for image in images),
        "missing_labels": sum(not label_path(image).is_file() for image in images),
        "non_person_rows": 0,
        "invalid_geometry_rows": 0,
        "test_used": False,
    }
    for root in labels:
        result = audit_person_only_labels(root)
        audit["non_person_rows"] += result["invalid_class_rows"]
        audit["invalid_geometry_rows"] += result["invalid_geometry_rows"]
    if any(
        audit[key]
        for key in (
            "missing_images",
            "missing_labels",
            "non_person_rows",
            "invalid_geometry_rows",
        )
    ):
        raise RuntimeError(f"Screening data audit failed: {audit}")
    return audit

