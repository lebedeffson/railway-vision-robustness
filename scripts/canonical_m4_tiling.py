from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from audit_evaluator import box_iou


@dataclass(frozen=True)
class Tile:
    tile_id: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def frozen_tiles(protocol: dict[str, Any]) -> list[Tile]:
    config = protocol["tiling"]
    width = int(protocol["dataset"]["image_width"])
    height = int(protocol["dataset"]["image_height"])
    tile_width = int(config["tile_width"])
    tile_height = int(config["tile_height"])
    x_starts = [0, width - tile_width]
    y_starts = [0, height - tile_height]
    tiles = [
        Tile(
            tile_id=f"tile_{row}_{column}",
            left=left,
            top=top,
            right=left + tile_width,
            bottom=top + tile_height,
        )
        for row, top in enumerate(y_starts)
        for column, left in enumerate(x_starts)
    ]
    if tile_width - (x_starts[1] - x_starts[0]) != int(config["overlap_x"]):
        raise RuntimeError("Frozen horizontal tile overlap is inconsistent")
    if tile_height - (y_starts[1] - y_starts[0]) != int(config["overlap_y"]):
        raise RuntimeError("Frozen vertical tile overlap is inconsistent")
    return tiles


def tile_coverage(protocol: dict[str, Any]) -> set[tuple[int, int]]:
    return {
        (x, y)
        for tile in frozen_tiles(protocol)
        for x in range(tile.left, tile.right)
        for y in range(tile.top, tile.bottom)
    }


def clip_ground_truth(
    label: dict[str, Any], tile: Tile, minimum_visible_fraction: float
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = map(float, label["box"])
    clipped = [
        max(x1, tile.left) - tile.left,
        max(y1, tile.top) - tile.top,
        min(x2, tile.right) - tile.left,
        min(y2, tile.bottom) - tile.top,
    ]
    intersection = max(0.0, clipped[2] - clipped[0]) * max(
        0.0, clipped[3] - clipped[1]
    )
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0 or intersection / area < minimum_visible_fraction:
        return None
    return {
        "class_id": int(label["class_id"]),
        "box": clipped,
        "visible_fraction": intersection / area,
        "source_gt_id": label.get("source_gt_id"),
    }


def assign_ground_truth_to_tiles(
    labels: list[dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    tiles = frozen_tiles(protocol)
    threshold = float(protocol["tiling"]["min_visible_fraction"])
    assigned: dict[str, list[dict[str, Any]]] = {tile.tile_id: [] for tile in tiles}
    retained: defaultdict[int, int] = defaultdict(int)
    for index, label in enumerate(labels):
        source = {**label, "source_gt_id": index}
        for tile in tiles:
            clipped = clip_ground_truth(source, tile, threshold)
            if clipped is not None:
                assigned[tile.tile_id].append(clipped)
                retained[index] += 1
    if protocol["tiling"]["require_each_ground_truth_in_at_least_one_tile"]:
        missing = [index for index in range(len(labels)) if retained[index] == 0]
        if missing:
            raise RuntimeError(f"Ground-truth objects lost at tile boundaries: {missing}")
    return assigned


def restore_global_box(box: list[float], tile: Tile) -> list[float]:
    return [
        float(box[0] + tile.left),
        float(box[1] + tile.top),
        float(box[2] + tile.left),
        float(box[3] + tile.top),
    ]


def local_box(box: list[float], tile: Tile) -> list[float]:
    return [
        float(box[0] - tile.left),
        float(box[1] - tile.top),
        float(box[2] - tile.left),
        float(box[3] - tile.top),
    ]


def fuse_predictions(
    predictions: list[dict[str, Any]], protocol: dict[str, Any]
) -> list[dict[str, Any]]:
    config = protocol["fusion"]
    threshold = float(config["confidence_prefilter"])
    ordered = sorted(
        [row for row in predictions if float(row["confidence"]) >= threshold],
        key=lambda row: (
            -float(row["confidence"]),
            int(row["class_id"]),
            *map(float, row["box"]),
        ),
    )
    kept: list[dict[str, Any]] = []
    for prediction in ordered:
        duplicate = any(
            int(prediction["class_id"]) == int(other["class_id"])
            and box_iou(prediction["box"], other["box"])
            > float(config["iou_threshold"])
            for other in kept
        )
        if not duplicate:
            kept.append(prediction)
        if len(kept) >= int(config["global_max_detections"]):
            break
    return kept


def validate_scene_folds(
    scene_by_image: dict[str, str],
    tile_source_image: dict[str, str],
    fold_by_scene: dict[str, int],
) -> None:
    for tile_path, source_path in tile_source_image.items():
        scene = scene_by_image[source_path]
        if scene not in fold_by_scene:
            raise RuntimeError(f"Tile scene is absent from frozen folds: {tile_path}")
    seen: defaultdict[str, set[int]] = defaultdict(set)
    for source_path, scene in scene_by_image.items():
        if source_path in tile_source_image.values():
            seen[scene].add(fold_by_scene[scene])
    crossing = {scene: folds for scene, folds in seen.items() if len(folds) != 1}
    if crossing:
        raise RuntimeError(f"Tiles cross scene folds: {crossing}")
