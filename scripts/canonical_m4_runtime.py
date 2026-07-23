from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from canonical_m4_tiling import (
    assign_ground_truth_to_tiles,
    frozen_tiles,
    fuse_predictions,
    restore_global_box,
)
from evaluate_image_level_detection import detection_metrics, predict_batch
from extract_attack_consistency import AttackResult
from extract_feature_consistency import FeatureHook, tnorm_filter, yolo_loss
from run_micro_view_candidate_v2 import read_yolo_labels


@dataclass(frozen=True)
class FrameInput:
    image_path: Path
    label_path: Path
    grouped_scene_id: str
    subsequence_id: str


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    model = YOLO(str(checkpoint)).model.to(device).float().eval()
    overrides = model.args if isinstance(model.args, dict) else vars(model.args)
    model.args = get_cfg(overrides=overrides)
    model.criterion = None
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_image(path: Path, device: torch.device) -> Tensor:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    return (
        torch.from_numpy(np.ascontiguousarray(array))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def tile_geometry(protocol: dict[str, Any]) -> tuple[float, int, int, int, int]:
    size = int(protocol["model"]["input_size"])
    tile = frozen_tiles(protocol)[0]
    scale = min(size / tile.width, size / tile.height)
    resized_width = int(round(tile.width * scale))
    resized_height = int(round(tile.height * scale))
    left = int(round((size - resized_width) / 2 - 0.1))
    top = int(round((size - resized_height) / 2 - 0.1))
    return scale, resized_width, resized_height, left, top


def image_to_tile_batch(image: Tensor, protocol: dict[str, Any]) -> Tensor:
    _scale, resized_width, resized_height, left, top = tile_geometry(protocol)
    size = int(protocol["model"]["input_size"])
    rows = []
    for tile in frozen_tiles(protocol):
        crop = image[..., tile.top:tile.bottom, tile.left:tile.right]
        resized = F.interpolate(
            crop,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        right = size - resized_width - left
        bottom = size - resized_height - top
        rows.append(F.pad(resized, (left, right, top, bottom), value=114.0 / 255.0))
    return torch.cat(rows, dim=0)


def tile_training_batch(
    labels: list[dict[str, Any]],
    protocol: dict[str, Any],
    device: torch.device,
) -> dict[str, Tensor]:
    assigned = assign_ground_truth_to_tiles(labels, protocol)
    scale, _width, _height, left, top = tile_geometry(protocol)
    size = float(protocol["model"]["input_size"])
    classes = []
    boxes = []
    batch_indices = []
    for tile_index, tile in enumerate(frozen_tiles(protocol)):
        for label in assigned[tile.tile_id]:
            x1, y1, x2, y2 = map(float, label["box"])
            x1, x2 = x1 * scale + left, x2 * scale + left
            y1, y2 = y1 * scale + top, y2 * scale + top
            classes.append([int(label["class_id"])])
            boxes.append([
                (x1 + x2) / (2 * size),
                (y1 + y2) / (2 * size),
                (x2 - x1) / size,
                (y2 - y1) / size,
            ])
            batch_indices.append(tile_index)
    return {
        "cls": torch.tensor(classes, dtype=torch.float32, device=device).reshape(-1, 1),
        "bboxes": torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
        "batch_idx": torch.tensor(batch_indices, dtype=torch.long, device=device),
    }


def frame_labels(frame: FrameInput, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return read_yolo_labels(
        frame.label_path,
        int(protocol["dataset"]["image_width"]),
        int(protocol["dataset"]["image_height"]),
    )


def defended_tiles(tiles: Tensor, defense: str, differentiable: bool = False) -> Tensor:
    if defense == "none":
        return tiles if differentiable else tiles.detach()
    if defense in {"Product_preprocessing", "tnorm", "product"}:
        return tnorm_filter(tiles) if differentiable else tnorm_filter(tiles.detach())
    if differentiable:
        raise RuntimeError(f"Defense {defense} is not differentiable")
    output = []
    for tile in tiles:
        rgb = (
            tile.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
        ).round().astype(np.uint8)
        if defense == "bilateral":
            filtered = cv2.bilateralFilter(rgb, 3, 8.0, 1.0)
        elif defense == "median":
            filtered = cv2.medianBlur(rgb, 3)
        elif defense == "jpeg":
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if not ok:
                raise RuntimeError("JPEG encode failed")
            filtered = cv2.cvtColor(
                cv2.imdecode(encoded, cv2.IMREAD_COLOR),
                cv2.COLOR_BGR2RGB,
            )
        else:
            raise ValueError(defense)
        output.append(
            torch.from_numpy(np.ascontiguousarray(filtered))
            .permute(2, 0, 1)
            .float()
            .to(tiles.device)
            / 255.0
        )
    return torch.stack(output)


def attack_loss(
    model: nn.Module,
    image: Tensor,
    labels: list[dict[str, Any]],
    protocol: dict[str, Any],
    adaptive_product: bool,
) -> Tensor:
    tiles = image_to_tile_batch(image, protocol)
    if adaptive_product:
        tiles = checkpoint(tnorm_filter, tiles, use_reentrant=False)
    batch = tile_training_batch(labels, protocol, image.device)
    return yolo_loss(model, batch, tiles)


def loss_gradient(
    model: nn.Module,
    image: Tensor,
    labels: list[dict[str, Any]],
    protocol: dict[str, Any],
    adaptive_product: bool,
) -> tuple[Tensor, float]:
    differentiable = image.detach().requires_grad_(True)
    loss = attack_loss(model, differentiable, labels, protocol, adaptive_product)
    gradient = torch.autograd.grad(loss, differentiable)[0]
    if not torch.isfinite(gradient).all() or float(gradient.abs().max()) <= 0:
        raise RuntimeError("Canonical M4 attack gradient is invalid")
    return gradient.detach(), float(loss.detach())


def fgsm(
    model: nn.Module,
    clean: Tensor,
    labels: list[dict[str, Any]],
    protocol: dict[str, Any],
    epsilon_px: float,
    adaptive_product: bool,
) -> AttackResult:
    gradient, clean_loss = loss_gradient(
        model, clean, labels, protocol, adaptive_product
    )
    adversarial = (
        clean + epsilon_px / 255.0 * gradient.sign()
    ).clamp(0, 1).detach()
    final = float(
        attack_loss(model, adversarial, labels, protocol, adaptive_product).detach()
    )
    return AttackResult(adversarial, gradient, gradient, clean_loss, final)


def pgd(
    model: nn.Module,
    clean: Tensor,
    labels: list[dict[str, Any]],
    protocol: dict[str, Any],
    epsilon_px: float,
    steps: int,
    seed: int,
    adaptive_product: bool,
) -> AttackResult:
    epsilon = epsilon_px / 255.0
    alpha = epsilon / 4.0
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    delta = torch.empty_like(clean).uniform_(-epsilon, epsilon, generator=generator)
    adversarial = (clean + delta).clamp(0, 1).detach()
    clean_gradient, clean_loss = loss_gradient(
        model, clean, labels, protocol, adaptive_product
    )
    path_gradient = torch.zeros_like(clean)
    best = adversarial.clone()
    best_loss = float(
        attack_loss(model, adversarial, labels, protocol, adaptive_product).detach()
    )
    for _ in range(steps):
        gradient, _ = loss_gradient(
            model, adversarial, labels, protocol, adaptive_product
        )
        path_gradient.add_(gradient)
        candidate = adversarial + alpha * gradient.sign()
        delta = (candidate - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0, 1).detach()
        value = float(
            attack_loss(
                model, adversarial, labels, protocol, adaptive_product
            ).detach()
        )
        if value > best_loss:
            best, best_loss = adversarial.clone(), value
    return AttackResult(
        best, clean_gradient, path_gradient / steps, clean_loss, best_loss
    )


def prediction_tiles_to_global(
    predictions: list[Tensor],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    scale, _width, _height, left, top = tile_geometry(protocol)
    rows = []
    for tile, prediction in zip(frozen_tiles(protocol), predictions, strict=True):
        for value in prediction:
            local = [
                float((value[0] - left) / scale),
                float((value[1] - top) / scale),
                float((value[2] - left) / scale),
                float((value[3] - top) / scale),
            ]
            local[0] = min(max(local[0], 0.0), tile.width)
            local[2] = min(max(local[2], 0.0), tile.width)
            local[1] = min(max(local[1], 0.0), tile.height)
            local[3] = min(max(local[3], 0.0), tile.height)
            if local[2] <= local[0] or local[3] <= local[1]:
                continue
            rows.append({
                "class_id": int(value[5]),
                "confidence": float(value[4]),
                "box": restore_global_box(local, tile),
            })
    return fuse_predictions(rows, protocol)


def global_prediction_tensor(
    rows: list[dict[str, Any]], device: torch.device
) -> Tensor:
    if not rows:
        return torch.empty((0, 6), device=device)
    return torch.tensor(
        [
            [*map(float, row["box"]), float(row["confidence"]), float(row["class_id"])]
            for row in rows
        ],
        device=device,
    )


def detection(
    model: nn.Module,
    image: Tensor,
    labels: list[dict[str, Any]],
    protocol: dict[str, Any],
    threshold: float,
    defense: str = "none",
) -> tuple[dict[str, Any], list[dict[str, Any]], Tensor]:
    tiles = defended_tiles(image_to_tile_batch(image, protocol), defense)
    predictions, nms = predict_batch(
        model, tiles, max_time_img=10.0, return_diagnostics=True
    )
    rows = prediction_tiles_to_global(predictions, protocol)
    prediction = global_prediction_tensor(rows, image.device)
    gt_boxes = torch.tensor(
        [row["box"] for row in labels],
        dtype=torch.float32,
        device=image.device,
    ).reshape(-1, 4)
    gt_classes = torch.tensor(
        [int(row["class_id"]) for row in labels],
        dtype=torch.long,
        device=image.device,
    )
    metrics = detection_metrics(prediction, gt_boxes, gt_classes, threshold)
    filtered = prediction[prediction[:, 4] >= threshold]
    metrics.update({
        "f2": 5 * metrics["precision"] * metrics["recall"] / max(
            4 * metrics["precision"] + metrics["recall"], 1e-12
        ),
        "mean_confidence": (
            float(filtered[:, 4].mean()) if len(filtered) else 0.0
        ),
        "nms_timeout": any(bool(value["nms_timeout"]) for value in nms),
        "nms_runtime_ms": sum(float(value["nms_runtime_ms"]) for value in nms),
        "nms_candidates_before": sum(
            int(value["nms_candidates_before"]) for value in nms
        ),
        "predictions_after_nms": len(rows),
        # The shared helper conservatively marks every multi-image batch as
        # incomplete. For the fixed four-tile batch, completeness is known
        # whenever the batch-level NMS timer did not fire.
        "nms_output_complete": not any(
            bool(value["nms_timeout"]) for value in nms
        ),
    })
    return metrics, rows, tiles


def feature_sets(
    hook: FeatureHook,
    model: nn.Module,
    image: Tensor,
    protocol: dict[str, Any],
    defense: str = "none",
) -> list[Tensor]:
    return hook.extract(
        model,
        defended_tiles(image_to_tile_batch(image, protocol), defense),
    )


def object_mask(
    labels: list[dict[str, Any]], height: int, width: int, device: torch.device
) -> Tensor:
    result = torch.zeros((height, width), dtype=torch.bool, device=device)
    for label in labels:
        x1, y1, x2, y2 = map(float, label["box"])
        result[
            max(0, math.floor(y1)):min(height, math.ceil(y2)),
            max(0, math.floor(x1)):min(width, math.ceil(x2)),
        ] = True
    return result
