from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageFilter


class AuditFlag(StrEnum):
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    INVALID_SURFACE = "INVALID_SURFACE"
    IMPLAUSIBLE_SCALE = "IMPLAUSIBLE_SCALE"
    EXCESSIVE_OCCLUSION = "EXCESSIVE_OCCLUSION"
    MASK_HALO = "MASK_HALO"
    DUPLICATE_INSTANCE = "DUPLICATE_INSTANCE"
    SOURCE_LEAKAGE = "SOURCE_LEAKAGE"
    GT_OVERWRITE = "GT_OVERWRITE"


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clipped(self, width: int, height: int) -> "Box":
        return Box(
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
            min(max(self.x2, 0.0), float(width)),
            min(max(self.y2, 0.0), float(height)),
        )


@dataclass(frozen=True)
class InstanceRecord:
    instance_id: str
    source_scene_id: str
    source_frame_id: str
    image_path: Path
    mask_path: Path
    box: Box
    original_width: int
    original_height: int
    estimated_range_m: float | None
    visibility: float
    occlusion: float
    quality_status: str


@dataclass(frozen=True)
class PerspectiveModel:
    log_height_intercept: float
    log_height_bottom_slope: float
    minimum_height: float = 4.0

    def expected_height(self, bottom_y_ratio: float) -> float:
        value = np.exp(
            self.log_height_intercept
            + self.log_height_bottom_slope * float(bottom_y_ratio)
        )
        return max(float(self.minimum_height), float(value))


@dataclass
class PasteResult:
    accepted: bool
    image: Image.Image
    box: Box | None
    flags: list[AuditFlag]
    scale: float | None


def validate_instance_bank(
    records: Iterable[InstanceRecord],
    *,
    train_scene_ids: set[str],
    heldout_scene_ids: set[str],
    test_scene_ids: set[str],
) -> None:
    identifiers: set[str] = set()
    for record in records:
        if record.instance_id in identifiers:
            raise ValueError(f"Duplicate instance ID: {record.instance_id}")
        identifiers.add(record.instance_id)
        if record.source_scene_id not in train_scene_ids:
            raise ValueError(
                f"Instance source is outside fold train scenes: {record.instance_id}"
            )
        if record.source_scene_id in heldout_scene_ids | test_scene_ids:
            raise ValueError(
                f"Instance source leakage: {record.instance_id}"
            )
        if record.quality_status != "PASS":
            raise ValueError(
                f"Unapproved instance entered bank: {record.instance_id}"
            )


def intersection_over_union(first: Box, second: Box) -> float:
    intersection = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection *= max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0


def mask_quality(
    mask: np.ndarray,
    *,
    minimum_foreground_fraction: float,
    maximum_foreground_fraction: float,
    maximum_border_fraction: float,
) -> tuple[bool, dict[str, float]]:
    binary = np.asarray(mask) > 0
    if binary.ndim != 2 or binary.size == 0:
        return False, {"foreground_fraction": 0.0, "border_fraction": 1.0}
    foreground = int(binary.sum())
    fraction = foreground / binary.size
    border = np.zeros_like(binary)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    border_fraction = (
        int((binary & border).sum()) / foreground if foreground else 1.0
    )
    passed = (
        minimum_foreground_fraction <= fraction <= maximum_foreground_fraction
        and border_fraction <= maximum_border_fraction
    )
    return passed, {
        "foreground_fraction": float(fraction),
        "border_fraction": float(border_fraction),
    }


def grabcut_mask(image: Image.Image, box: Box) -> np.ndarray:
    """Prepare a mask candidate; the caller must still pass mask_quality."""
    array = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
    rectangle = (
        max(0, x1),
        max(0, y1),
        max(1, min(array.shape[1], x2) - max(0, x1)),
        max(1, min(array.shape[0], y2) - max(0, y1)),
    )
    mask = np.zeros(array.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        array,
        mask,
        rectangle,
        background,
        foreground,
        5,
        cv2.GC_INIT_WITH_RECT,
    )
    return np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)


def source_instance_id(scene: str, frame: str, box: Box) -> str:
    payload = (
        f"{scene}|{frame}|{box.x1:.3f}|{box.y1:.3f}|"
        f"{box.x2:.3f}|{box.y2:.3f}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def paste_person(
    target: Image.Image,
    instance: Image.Image,
    mask: Image.Image,
    *,
    target_scene_id: str,
    train_scene_ids: set[str],
    bottom_center: tuple[int, int],
    surface_mask: np.ndarray,
    perspective: PerspectiveModel,
    existing_boxes: Iterable[Box],
    used_instance_ids: set[str],
    maximum_existing_iou: float = 0.30,
    minimum_box_side_px: int = 4,
    blur_radius: float = 0.0,
) -> PasteResult:
    flags: list[AuditFlag] = []
    if instance.width <= 0 or instance.height <= 0:
        flags.append(AuditFlag.IMPLAUSIBLE_SCALE)
        return PasteResult(False, target, None, flags, None)
    if (
        instance.info.get("instance_id") in used_instance_ids
        or not instance.info.get("instance_id")
    ):
        flags.append(AuditFlag.DUPLICATE_INSTANCE)
    source_scene = str(instance.info.get("source_scene_id", ""))
    if source_scene not in train_scene_ids or target_scene_id not in train_scene_ids:
        flags.append(AuditFlag.SOURCE_LEAKAGE)
    bottom_x, bottom_y = map(int, bottom_center)
    if (
        bottom_x < 0
        or bottom_y < 0
        or bottom_x >= target.width
        or bottom_y >= target.height
    ):
        flags.append(AuditFlag.OUT_OF_BOUNDS)
    elif (
        surface_mask.shape != (target.height, target.width)
        or not bool(surface_mask[bottom_y, bottom_x])
    ):
        flags.append(AuditFlag.INVALID_SURFACE)
    expected_height = perspective.expected_height(bottom_y / target.height)
    scale = expected_height / instance.height
    width = max(1, int(round(instance.width * scale)))
    height = max(1, int(round(instance.height * scale)))
    if min(width, height) < minimum_box_side_px:
        flags.append(AuditFlag.IMPLAUSIBLE_SCALE)
    box = Box(
        bottom_x - width / 2,
        bottom_y - height,
        bottom_x + width / 2,
        bottom_y,
    )
    if box.clipped(target.width, target.height).area != box.area:
        flags.append(AuditFlag.OUT_OF_BOUNDS)
    if any(
        intersection_over_union(box, existing) > maximum_existing_iou
        for existing in existing_boxes
    ):
        flags.append(AuditFlag.GT_OVERWRITE)
    quality, _ = mask_quality(
        np.asarray(mask),
        minimum_foreground_fraction=0.10,
        maximum_foreground_fraction=0.85,
        maximum_border_fraction=0.25,
    )
    if not quality:
        flags.append(AuditFlag.MASK_HALO)
    if flags:
        return PasteResult(False, target, None, sorted(set(flags)), scale)
    resized = instance.resize((width, height), Image.Resampling.LANCZOS)
    alpha = mask.resize((width, height), Image.Resampling.LANCZOS).convert("L")
    if blur_radius > 0:
        resized = resized.filter(ImageFilter.GaussianBlur(blur_radius))
        alpha = alpha.filter(ImageFilter.GaussianBlur(blur_radius / 2))
    output = target.copy()
    output.paste(resized, (int(box.x1), int(box.y1)), alpha)
    return PasteResult(True, output, box, [], scale)
