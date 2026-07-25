from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RangeAssignment:
    source: str
    group: str
    distance_m: float | None
    area_ratio: float


def scale_group(area_ratio: float) -> str:
    if not math.isfinite(area_ratio) or area_ratio < 0:
        raise ValueError("area_ratio must be finite and non-negative")
    if area_ratio < 0.001:
        return "small"
    if area_ratio < 0.01:
        return "medium"
    return "large"


def geometric_group(distance_m: float) -> str:
    if not math.isfinite(distance_m) or distance_m < 0:
        raise ValueError("distance_m must be finite and non-negative")
    if distance_m < 25:
        return "near"
    if distance_m < 60:
        return "middle"
    return "far"


def assign_range_or_scale(
    *,
    area_ratio: float,
    lidar_distance_m: float | None,
    verified_rgb_object_link: bool,
) -> RangeAssignment:
    if lidar_distance_m is not None and verified_rgb_object_link:
        return RangeAssignment(
            source="verified_lidar_geometry",
            group=geometric_group(lidar_distance_m),
            distance_m=float(lidar_distance_m),
            area_ratio=float(area_ratio),
        )
    return RangeAssignment(
        source="bbox_area_scale_fallback",
        group=scale_group(area_ratio),
        distance_m=None,
        area_ratio=float(area_ratio),
    )


def small_object_weight(
    area_px: float, *, lambda_s: float, tau_s: float
) -> float:
    if area_px < 0 or tau_s <= 0:
        raise ValueError("area_px must be non-negative and tau_s positive")
    return 1.0 + float(lambda_s) * max(
        0.0, 1.0 - math.sqrt(float(area_px)) / float(tau_s)
    )


def preferred_levels(assignment: RangeAssignment) -> tuple[str, ...]:
    if assignment.group in {"small", "far"}:
        return ("P2", "P3")
    if assignment.group in {"medium", "middle"}:
        return ("P3", "P4")
    return ("P4", "P5")

