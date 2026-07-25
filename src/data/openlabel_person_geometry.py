from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from src.augmentation.person_pasting import Box
from src.data.range_assignment import RangeAssignment, assign_range_or_scale


RGB_PERSON_BBOX = "rgb_highres_center__bbox__person"
LIDAR_PERSON_CUBOID = "lidar__cuboid__person"


@dataclass(frozen=True)
class PersonGeometry:
    object_uuid: str
    frame_id: str
    box: Box
    image_width: int
    image_height: int
    assignment: RangeAssignment
    occlusion: float
    visibility: float


def _named(values: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((value for value in values if value.get("name") == name), None)


def _text_attribute(value: dict[str, Any], name: str) -> str | None:
    attributes = value.get("attributes", {})
    for item in attributes.get("text", []):
        if item.get("name") == name:
            return str(item.get("val"))
    return None


def parse_occlusion(value: str | None) -> float:
    if not value:
        return 0.0
    normalized = value.strip().lower().replace(" ", "")
    if normalized in {"none", "notoccluded", "0%"}:
        return 0.0
    numbers = [
        float(token)
        for token in normalized.replace("%", "").replace(">", "-").split("-")
        if token.replace(".", "", 1).isdigit()
    ]
    if not numbers:
        return 0.0
    return min(1.0, max(0.0, sum(numbers) / len(numbers) / 100.0))


def bbox_from_center(value: list[float]) -> Box:
    if len(value) != 4:
        raise ValueError("OpenLABEL bbox must contain cx, cy, width, height")
    center_x, center_y, width, height = map(float, value)
    if width <= 0 or height <= 0:
        raise ValueError("OpenLABEL bbox dimensions must be positive")
    return Box(
        center_x - width / 2,
        center_y - height / 2,
        center_x + width / 2,
        center_y + height / 2,
    )


def lidar_distance(cuboid: dict[str, Any] | None) -> float | None:
    if cuboid is None:
        return None
    value = cuboid.get("val", [])
    if len(value) < 3:
        return None
    coordinates = tuple(map(float, value[:3]))
    if not all(math.isfinite(item) for item in coordinates):
        return None
    return math.sqrt(sum(item * item for item in coordinates))


def iter_person_geometry(
    label_path: Path,
    *,
    image_width: int,
    image_height: int,
) -> Iterator[PersonGeometry]:
    document = json.loads(label_path.read_text(encoding="utf-8"))["openlabel"]
    for frame_id, frame in document.get("frames", {}).items():
        for object_uuid, instance in frame.get("objects", {}).items():
            data = instance.get("object_data", {})
            bbox = _named(data.get("bbox", []), RGB_PERSON_BBOX)
            if bbox is None:
                continue
            box = bbox_from_center(bbox["val"])
            cuboid = _named(data.get("cuboid", []), LIDAR_PERSON_CUBOID)
            distance = lidar_distance(cuboid)
            area_ratio = box.area / float(image_width * image_height)
            assignment = assign_range_or_scale(
                area_ratio=area_ratio,
                lidar_distance_m=distance,
                verified_rgb_object_link=cuboid is not None,
            )
            occlusion = parse_occlusion(_text_attribute(bbox, "occlusion"))
            yield PersonGeometry(
                object_uuid=str(object_uuid),
                frame_id=str(frame_id),
                box=box,
                image_width=image_width,
                image_height=image_height,
                assignment=assignment,
                occlusion=occlusion,
                visibility=1.0 - occlusion,
            )

