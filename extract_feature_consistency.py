from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from checkpoint_selection import selected_checkpoint


MODEL = str(selected_checkpoint())
DATA = "data/yolo_osdar23/data.yaml"
OUTDIR = "outputs/diagnostics/feature_consistency"

EPS_LIST = [1, 2, 4, 8]
ATTACKS = ["fgsm", "pgd"]
DEFENSES = ["none", "tnorm", "bilateral", "jpeg", "median", "gaussian"]

PGD_STEPS = 20
PGD_ALPHA_RATIO = 0.25
EPS = 1e-8

NORM_Q_LOW = 0.05
NORM_Q_HIGH = 0.95
NORM_SAMPLES_PER_IMAGE = 128

ACTIVE_THR = 0.05
GODEL_TOL = 1e-4

TNORM_K = 5
TNORM_SIGMA_COLOR = 24.0 / 255.0
TNORM_SIGMA_SPATIAL = 1.0
TNORM_STRENGTH = 0.9
TNORM_GAMMA = 2.0

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# DATA
# =============================================================================

def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def split_images(data_yaml: Path, split: str) -> list[Path]:
    cfg = read_yaml(data_yaml)
    root = resolve(data_yaml.parent, cfg.get("path", data_yaml.parent))
    value = cfg[split]
    entries = value if isinstance(value, list) else [value]

    images: list[Path] = []

    for entry in entries:
        path = resolve(root, entry)

        if path.is_dir():
            images.extend(
                sorted(
                    item
                    for item in path.rglob("*")
                    if item.is_file() and item.suffix.lower() in IMG_EXT
                )
            )

        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue

                candidate = Path(line.strip())

                if not candidate.is_absolute():
                    from_root = resolve(root, candidate)
                    from_list = resolve(path.parent, candidate)
                    candidate = from_root if from_root.exists() else from_list

                if candidate.suffix.lower() in IMG_EXT:
                    images.append(candidate)

        elif path.is_file() and path.suffix.lower() in IMG_EXT:
            images.append(path)

        else:
            raise FileNotFoundError(path)

    images = sorted(dict.fromkeys(images))

    if not images:
        raise RuntimeError(f"No images found for split={split}")

    return images


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)

    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")

    return image_path.with_suffix(".txt")


def read_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        return np.empty((0,), np.float32), np.empty((0, 4), np.float32)

    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.strip().split()

        if len(values) >= 5:
            rows.append([float(value) for value in values[:5]])

    if not rows:
        return np.empty((0,), np.float32), np.empty((0, 4), np.float32)

    array = np.asarray(rows, np.float32)
    return array[:, 0], array[:, 1:5]


def letterbox(
    image: np.ndarray,
    boxes: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = min(size / height, size / width)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    padding_width = size - new_width
    padding_height = size - new_height

    left = int(round(padding_width / 2 - 0.1))
    top = int(round(padding_height / 2 - 0.1))
    right = size - new_width - left
    bottom = size - new_height - top

    resized = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    if boxes.size == 0:
        return resized, boxes.astype(np.float32)

    result = boxes.astype(np.float32).copy()

    center_x = result[:, 0] * width
    center_y = result[:, 1] * height
    box_width = result[:, 2] * width
    box_height = result[:, 3] * height

    x1 = (center_x - box_width / 2) * scale + left
    y1 = (center_y - box_height / 2) * scale + top
    x2 = (center_x + box_width / 2) * scale + left
    y2 = (center_y + box_height / 2) * scale + top

    x1 = np.clip(x1, 0, size)
    y1 = np.clip(y1, 0, size)
    x2 = np.clip(x2, 0, size)
    y2 = np.clip(y2, 0, size)

    result[:, 0] = ((x1 + x2) / 2) / size
    result[:, 1] = ((y1 + y2) / 2) / size
    result[:, 2] = (x2 - x1) / size
    result[:, 3] = (y2 - y1) / size

    valid = (result[:, 2] > 0) & (result[:, 3] > 0)
    return resized, result[valid]


class YoloDataset(Dataset):
    def __init__(self, data_yaml: Path, split: str, imgsz: int):
        self.images = split_images(data_yaml, split)
        self.imgsz = imgsz

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.images[index]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if bgr is None:
            raise RuntimeError(f"Cannot read image: {path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        classes, boxes = read_labels(label_path(path))
        rgb, boxes = letterbox(rgb, boxes, self.imgsz)
        classes = classes[:len(boxes)]

        image = torch.from_numpy(np.ascontiguousarray(rgb))
        image = image.permute(2, 0, 1).float() / 255.0

        return {
            "img": image,
            "cls": torch.from_numpy(classes).float().view(-1, 1),
            "bboxes": torch.from_numpy(boxes).float().view(-1, 4),
            "im_file": str(path),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["img"] for item in batch])
    classes, boxes, batch_indices = [], [], []

    for index, item in enumerate(batch):
        count = len(item["cls"])

        if count:
            classes.append(item["cls"])
            boxes.append(item["bboxes"])
            batch_indices.append(torch.full((count,), index, dtype=torch.long))

    return {
        "img": images,
        "cls": torch.cat(classes, 0) if classes else torch.empty((0, 1)),
        "bboxes": torch.cat(boxes, 0) if boxes else torch.empty((0, 4)),
        "batch_idx": (
            torch.cat(batch_indices, 0)
            if batch_indices
            else torch.empty((0,), dtype=torch.long)
        ),
        "im_file": [item["im_file"] for item in batch],
    }


def loader(
    data_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    workers: int,
    cuda: bool,
) -> DataLoader:
    return DataLoader(
        YoloDataset(data_yaml, split, imgsz),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=cuda,
        collate_fn=collate,
        persistent_workers=workers > 0,
    )


# =============================================================================
# FEATURES
# =============================================================================

def find_detect(model: nn.Module) -> nn.Module:
    found = [
        module
        for module in model.modules()
        if module.__class__.__name__.lower().endswith("detect")
    ]

    if not found:
        raise RuntimeError("Detect head not found")

    return found[-1]


class FeatureHook:
    def __init__(self, model: nn.Module):
        self.features: list[Tensor] | None = None
        self.handle = find_detect(model).register_forward_pre_hook(self._hook)

    def _hook(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        features = inputs[0]

        if not isinstance(features, (list, tuple)):
            raise RuntimeError("Detect input is not a P3/P4/P5 list")

        self.features = [tensor.detach().clone() for tensor in features]

    def extract(self, model: nn.Module, images: Tensor) -> list[Tensor]:
        self.features = None

        with torch.no_grad():
            _ = model(images)

        if self.features is None or len(self.features) != 3:
            raise RuntimeError("Failed to capture P3/P4/P5")

        return self.features

    def close(self) -> None:
        self.handle.remove()


# =============================================================================
# NORMALIZATION AND METRICS
# =============================================================================

def collect_norm(
    model: nn.Module,
    hook: FeatureHook,
    data_loader: DataLoader,
    device: torch.device,
    output: Path,
) -> dict[str, dict[str, Tensor]]:
    samples = [[], [], []]
    generator = torch.Generator(device="cpu").manual_seed(42)

    for batch in tqdm(data_loader, desc="norm val", unit="batch"):
        images = batch["img"].to(device, non_blocking=True)
        features = hook.extract(model, images)

        for level_index, feature in enumerate(features):
            feature = feature.detach().float().cpu()
            batch_size, channels, height, width = feature.shape
            flattened = feature.view(batch_size, channels, height * width)
            count = min(NORM_SAMPLES_PER_IMAGE, height * width)

            for image_index in range(batch_size):
                indices = torch.randint(
                    0,
                    height * width,
                    (count,),
                    generator=generator,
                )
                samples[level_index].append(flattened[image_index, :, indices])

    stats = {}

    for level_index, level_samples in enumerate(samples):
        values = torch.cat(level_samples, dim=1)
        low = torch.quantile(values, NORM_Q_LOW, dim=1)
        high = torch.quantile(values, NORM_Q_HIGH, dim=1)
        high = torch.maximum(high, low + torch.full_like(high, 1e-6))

        stats[f"P{level_index + 3}"] = {
            "low": low,
            "high": high,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stats": stats}, output)

    return stats


def load_norm(path: Path) -> dict[str, dict[str, Tensor]]:
    return torch.load(path, map_location="cpu")["stats"]


def norm_feat(feature: Tensor, stats: dict[str, Tensor]) -> Tensor:
    low = stats["low"].to(feature.device, feature.dtype).view(1, -1, 1, 1)
    high = stats["high"].to(feature.device, feature.dtype).view(1, -1, 1, 1)
    return ((feature - low) / (high - low + EPS)).clamp(0, 1)


def cosine(left: Tensor, right: Tensor) -> float:
    left = left.flatten().float()
    right = right.flatten().float()

    left_norm = left.norm()
    right_norm = right.norm()

    if left_norm <= EPS and right_norm <= EPS:
        return 1.0

    if left_norm <= EPS or right_norm <= EPS:
        return 0.0

    return float(torch.dot(left, right) / (left_norm * right_norm + EPS))


def activation_entropy(values: Tensor, bins: int = 64) -> float:
    """Normalized histogram entropy on the validation-normalized activations."""
    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return 0.0
    histogram = torch.histc(flat, bins=bins, min=0.0, max=1.0)
    probabilities = histogram / histogram.sum().clamp_min(EPS)
    probabilities = probabilities[probabilities > 0]
    if probabilities.numel() <= 1:
        return 0.0
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy / math.log(bins))


def metrics(
    clean: Tensor,
    other: Tensor,
    stats: dict[str, Tensor],
) -> list[dict[str, float]]:
    clean_norm = norm_feat(clean, stats)
    other_norm = norm_feat(other, stats)
    rows = []

    for index in range(clean.shape[0]):
        clean_image = clean_norm[index]
        other_image = other_norm[index]
        mask = torch.maximum(clean_image, other_image) > ACTIVE_THR
        active = float(mask.float().mean())

        if not bool(mask.any()):
            rows.append({
                "active": active,
                "cos_raw": cosine(clean[index], other[index]),
                "cos_norm": 1.0,
                "product": 1.0,
                "godel": 1.0,
                "lukas": 1.0,
                "mse": 0.0,
                "mae": 0.0,
                "relative_l2": 0.0,
                "mean_shift": 0.0,
                "entropy_clean": activation_entropy(clean_image),
                "entropy_other": activation_entropy(other_image),
                "entropy_change": 0.0,
            })
            continue

        left = clean_image[mask].float()
        right = other_image[mask].float()
        difference = left - right

        minimum = torch.minimum(left, right)
        maximum = torch.maximum(left, right)

        product = torch.where(
            maximum <= EPS,
            torch.ones_like(maximum),
            minimum / (maximum + EPS),
        )

        godel = torch.where(
            difference.abs() <= GODEL_TOL,
            torch.ones_like(left),
            minimum,
        )

        rows.append({
            "active": active,
            "cos_raw": cosine(clean[index], other[index]),
            "cos_norm": cosine(left, right),
            "product": float(product.mean()),
            "godel": float(godel.mean()),
            "lukas": float((1.0 - difference.abs()).mean()),
            "mse": float(difference.square().mean()),
            "mae": float(difference.abs().mean()),
            "relative_l2": float(difference.norm() / (left.norm() + EPS)),
            "mean_shift": float((left.mean() - right.mean()).abs()),
            "entropy_clean": activation_entropy(clean_image),
            "entropy_other": activation_entropy(other_image),
            "entropy_change": abs(
                activation_entropy(clean_image)
                - activation_entropy(other_image)
            ),
        })

    return rows


def recovery(before: float, after: float) -> float:
    denominator = 1.0 - before
    return math.nan if abs(denominator) <= EPS else (after - before) / denominator


def distance_recovery(before: float, after: float) -> float:
    return math.nan if before <= EPS else (before - after) / (before + EPS)


def clipped_recovery(before: float, after: float) -> float:
    value = recovery(before, after)
    return math.nan if math.isnan(value) else min(1.0, max(0.0, value))


def defense_consistency(operator: str, preservation: float, gain: float) -> float:
    if math.isnan(preservation) or math.isnan(gain):
        return math.nan
    preservation = min(1.0, max(0.0, preservation))
    gain = min(1.0, max(0.0, gain))
    if operator == "product":
        return preservation * gain
    if operator == "godel":
        return min(preservation, gain)
    if operator == "lukas":
        return max(0.0, preservation + gain - 1.0)
    raise ValueError(operator)


# =============================================================================
# ATTACKS
# =============================================================================

def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "img": batch["img"].to(device, non_blocking=True),
        "cls": batch["cls"].to(device, non_blocking=True),
        "bboxes": batch["bboxes"].to(device, non_blocking=True),
        "batch_idx": batch["batch_idx"].to(device, non_blocking=True),
        "im_file": batch["im_file"],
    }


def yolo_loss(
    model: nn.Module,
    batch: dict[str, Any],
    images: Tensor,
) -> Tensor:
    output = model({
        "img": images,
        "cls": batch["cls"],
        "bboxes": batch["bboxes"],
        "batch_idx": batch["batch_idx"],
    })

    return (output[0] if isinstance(output, tuple) else output).sum()


def fgsm(
    model: nn.Module,
    batch: dict[str, Any],
    epsilon_pixels: int,
) -> Tensor:
    clean = batch["img"].detach()
    adversarial = clean.clone().requires_grad_(True)

    loss = yolo_loss(model, batch, adversarial)
    gradient = torch.autograd.grad(loss, adversarial)[0]

    return (
        clean + (epsilon_pixels / 255.0) * gradient.sign()
    ).clamp(0, 1).detach()


def pgd(
    model: nn.Module,
    batch: dict[str, Any],
    epsilon_pixels: int,
) -> Tensor:
    clean = batch["img"].detach()
    epsilon = epsilon_pixels / 255.0
    alpha = epsilon * PGD_ALPHA_RATIO

    adversarial = (
        clean + torch.empty_like(clean).uniform_(-epsilon, epsilon)
    ).clamp(0, 1).detach()

    for _ in range(PGD_STEPS):
        adversarial.requires_grad_(True)

        loss = yolo_loss(model, batch, adversarial)
        gradient = torch.autograd.grad(loss, adversarial)[0]

        adversarial = adversarial.detach() + alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0, 1).detach()

    return adversarial


def attack(
    model: nn.Module,
    batch: dict[str, Any],
    name: str,
    epsilon_pixels: int,
) -> Tensor:
    if name == "fgsm":
        return fgsm(model, batch, epsilon_pixels)

    if name == "pgd":
        return pgd(model, batch, epsilon_pixels)

    raise ValueError(name)


# =============================================================================
# DEFENSES
# =============================================================================

def tnorm_filter_one(image: Tensor) -> Tensor:
    padding = TNORM_K // 2
    _, channels, height, width = image.shape

    padded = F.pad(
        image,
        (padding, padding, padding, padding),
        mode="reflect",
    )

    patches = F.unfold(
        padded,
        kernel_size=TNORM_K,
    ).view(
        1,
        channels,
        TNORM_K * TNORM_K,
        height,
        width,
    )

    center = image.unsqueeze(2)

    color = torch.exp(
        -((patches - center).abs().mean(1, keepdim=True))
        / max(TNORM_SIGMA_COLOR, EPS)
    )

    coordinates = torch.arange(
        -padding,
        padding + 1,
        device=image.device,
        dtype=image.dtype,
    )

    grid_y, grid_x = torch.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )

    spatial = torch.exp(
        -(grid_x.square() + grid_y.square())
        / (2 * TNORM_SIGMA_SPATIAL ** 2)
    ).reshape(
        1,
        1,
        TNORM_K * TNORM_K,
        1,
        1,
    )

    weights = color * spatial
    weight_sum = weights.sum(2).clamp_min(EPS)
    filtered = (patches * weights).sum(2) / weight_sum

    confidence = (weight_sum / spatial.sum(2)).clamp(0, 1)
    mixing = TNORM_STRENGTH * confidence.pow(TNORM_GAMMA)

    return (image + mixing * (filtered - image)).clamp(0, 1)


def tnorm_filter(images: Tensor) -> Tensor:
    return torch.cat(
        [
            tnorm_filter_one(images[index:index + 1])
            for index in range(images.shape[0])
        ],
        dim=0,
    )


def cv_defense(images: Tensor, name: str) -> Tensor:
    outputs = []

    for image in images:
        rgb = (
            image.detach()
            .clamp(0, 1)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            * 255
        ).round().astype(np.uint8)

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if name == "bilateral":
            bgr = cv2.bilateralFilter(bgr, 3, 8.0, 1.0)

        elif name == "jpeg":
            success, encoded = cv2.imencode(
                ".jpg",
                bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 70],
            )

            if not success:
                raise RuntimeError("JPEG encode failed")

            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        elif name == "median":
            bgr = cv2.medianBlur(bgr, 3)

        elif name == "gaussian":
            bgr = cv2.GaussianBlur(bgr, (5, 5), 1.5)

        else:
            raise ValueError(name)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(
            np.ascontiguousarray(rgb)
        ).to(images.device)

        tensor = tensor.permute(2, 0, 1).float() / 255.0
        outputs.append(tensor)

    return torch.stack(outputs)


def defend(images: Tensor, name: str) -> Tensor:
    if name == "none":
        return images.detach()

    if name == "tnorm":
        return tnorm_filter(images.detach())

    if name in {"bilateral", "jpeg", "median", "gaussian"}:
        return cv_defense(images.detach(), name)

    raise ValueError(name)


# =============================================================================
# OUTPUT
# =============================================================================

COLUMNS = [
    "image",
    "image_path",
    "split",
    "attack",
    "epsilon_px",
    "epsilon",
    "defense",
    "level",

    "active_attack",
    "cos_raw_attack",
    "cos_norm_attack",
    "product_attack",
    "godel_attack",
    "lukas_attack",
    "mse_attack",
    "mae_attack",
    "relative_l2_attack",
    "mean_shift_attack",
    "entropy_clean",
    "entropy_attack",
    "entropy_change_attack",

    "active_defended",
    "cos_raw_defended",
    "cos_norm_defended",
    "product_defended",
    "godel_defended",
    "lukas_defended",
    "mse_defended",
    "mae_defended",
    "relative_l2_defended",
    "mean_shift_defended",
    "entropy_defended",
    "entropy_change_defended",

    "damage_product_attack",
    "damage_godel_attack",
    "damage_lukas_attack",
    "damage_product_defended",
    "damage_godel_defended",
    "damage_lukas_defended",

    "recovery_cos_raw",
    "recovery_cos_norm",
    "recovery_product",
    "recovery_godel",
    "recovery_lukas",
    "recovery_mse",
    "recovery_mae",
    "recovery_relative_l2",
    "recovery_mean_shift",
    "recovery_entropy",

    "p_cosine", "a_cosine", "r_cosine", "g_cosine",
    "p_product", "a_product", "r_product", "g_product", "c_def_product",
    "p_godel", "a_godel", "r_godel", "g_godel", "c_def_godel",
    "p_lukas", "a_lukas", "r_lukas", "g_lukas", "c_def_lukas",
]


def write_rows(
    writer: csv.DictWriter,
    clean_features: list[Tensor],
    attack_features: list[Tensor],
    defended_features: list[Tensor],
    clean_filtered_features: list[Tensor],
    paths: list[str],
    split: str,
    attack_name: str,
    epsilon_pixels: int,
    defense: str,
    stats: dict[str, dict[str, Tensor]],
    clean_case: bool = False,
) -> None:
    for level_index, level in enumerate(["P3", "P4", "P5"]):
        attack_metrics = metrics(
            clean_features[level_index],
            attack_features[level_index],
            stats[level],
        )

        defended_metrics = metrics(
            clean_features[level_index],
            defended_features[level_index],
            stats[level],
        )

        preservation_metrics = metrics(
            clean_features[level_index],
            clean_filtered_features[level_index],
            stats[level],
        )

        for image_index, path in enumerate(paths):
            attacked = attack_metrics[image_index]
            defended = defended_metrics[image_index]
            preservation = preservation_metrics[image_index]
            gains = {
                name: (
                    math.nan
                    if clean_case
                    else clipped_recovery(attacked[name], defended[name])
                )
                for name in ("cos_norm", "product", "godel", "lukas")
            }

            writer.writerow({
                "image": Path(path).name,
                "image_path": path,
                "split": split,
                "attack": attack_name,
                "epsilon_px": epsilon_pixels,
                "epsilon": epsilon_pixels / 255.0,
                "defense": defense,
                "level": level,

                "active_attack": attacked["active"],
                "cos_raw_attack": attacked["cos_raw"],
                "cos_norm_attack": attacked["cos_norm"],
                "product_attack": attacked["product"],
                "godel_attack": attacked["godel"],
                "lukas_attack": attacked["lukas"],
                "mse_attack": attacked["mse"],
                "mae_attack": attacked["mae"],
                "relative_l2_attack": attacked["relative_l2"],
                "mean_shift_attack": attacked["mean_shift"],
                "entropy_clean": attacked["entropy_clean"],
                "entropy_attack": attacked["entropy_other"],
                "entropy_change_attack": attacked["entropy_change"],

                "active_defended": defended["active"],
                "cos_raw_defended": defended["cos_raw"],
                "cos_norm_defended": defended["cos_norm"],
                "product_defended": defended["product"],
                "godel_defended": defended["godel"],
                "lukas_defended": defended["lukas"],
                "mse_defended": defended["mse"],
                "mae_defended": defended["mae"],
                "relative_l2_defended": defended["relative_l2"],
                "mean_shift_defended": defended["mean_shift"],
                "entropy_defended": defended["entropy_other"],
                "entropy_change_defended": defended["entropy_change"],

                "damage_product_attack": 1 - attacked["product"],
                "damage_godel_attack": 1 - attacked["godel"],
                "damage_lukas_attack": 1 - attacked["lukas"],

                "damage_product_defended": 1 - defended["product"],
                "damage_godel_defended": 1 - defended["godel"],
                "damage_lukas_defended": 1 - defended["lukas"],

                "recovery_cos_raw": (
                    math.nan
                    if clean_case
                    else recovery(attacked["cos_raw"], defended["cos_raw"])
                ),
                "recovery_cos_norm": (
                    math.nan
                    if clean_case
                    else recovery(attacked["cos_norm"], defended["cos_norm"])
                ),
                "recovery_product": (
                    math.nan
                    if clean_case
                    else recovery(attacked["product"], defended["product"])
                ),
                "recovery_godel": (
                    math.nan
                    if clean_case
                    else recovery(attacked["godel"], defended["godel"])
                ),
                "recovery_lukas": (
                    math.nan
                    if clean_case
                    else recovery(attacked["lukas"], defended["lukas"])
                ),
                "recovery_mse": (
                    math.nan
                    if clean_case
                    else distance_recovery(attacked["mse"], defended["mse"])
                ),
                "recovery_mae": (
                    math.nan
                    if clean_case
                    else distance_recovery(attacked["mae"], defended["mae"])
                ),
                "recovery_relative_l2": (
                    math.nan
                    if clean_case
                    else distance_recovery(
                        attacked["relative_l2"],
                        defended["relative_l2"],
                    )
                ),
                "recovery_mean_shift": (
                    math.nan
                    if clean_case
                    else distance_recovery(
                        attacked["mean_shift"],
                        defended["mean_shift"],
                    )
                ),
                "recovery_entropy": (
                    math.nan
                    if clean_case
                    else distance_recovery(
                        attacked["entropy_change"],
                        defended["entropy_change"],
                    )
                ),

                "p_cosine": preservation["cos_norm"],
                "a_cosine": attacked["cos_norm"],
                "r_cosine": defended["cos_norm"],
                "g_cosine": gains["cos_norm"],
                "p_product": preservation["product"],
                "a_product": attacked["product"],
                "r_product": defended["product"],
                "g_product": gains["product"],
                "c_def_product": defense_consistency(
                    "product", preservation["product"], gains["product"]
                ),
                "p_godel": preservation["godel"],
                "a_godel": attacked["godel"],
                "r_godel": defended["godel"],
                "g_godel": gains["godel"],
                "c_def_godel": defense_consistency(
                    "godel", preservation["godel"], gains["godel"]
                ),
                "p_lukas": preservation["lukas"],
                "a_lukas": attacked["lukas"],
                "r_lukas": defended["lukas"],
                "g_lukas": gains["lukas"],
                "c_def_lukas": defense_consistency(
                    "lukas", preservation["lukas"], gains["lukas"]
                ),
            })


def run(
    model: nn.Module,
    hook: FeatureHook,
    data_loader: DataLoader,
    device: torch.device,
    split: str,
    attacks: list[str],
    epsilon_list: list[int],
    defenses: list[str],
    stats: dict[str, dict[str, Tensor]],
    csv_path: Path,
    skip_clean: bool,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()

        progress = tqdm(
            data_loader,
            desc=f"diagnostics {split}",
            unit="batch",
        )

        for raw_batch in progress:
            batch = to_device(raw_batch, device)
            clean = batch["img"].detach()
            paths = batch["im_file"]
            clean_features = hook.extract(model, clean)

            clean_defended_by_defense: dict[str, list[Tensor]] = {
                "none": clean_features,
            }
            for defense_name in defenses:
                if defense_name != "none":
                    clean_defended_by_defense[defense_name] = hook.extract(
                        model, defend(clean, defense_name)
                    )

            if not skip_clean:
                for defense_name in defenses:
                    defended_features = clean_defended_by_defense[defense_name]

                    write_rows(
                        writer,
                        clean_features,
                        clean_features,
                        defended_features,
                        clean_defended_by_defense[defense_name],
                        paths,
                        split,
                        "clean",
                        0,
                        defense_name,
                        stats,
                        True,
                    )

            for attack_name in attacks:
                for epsilon_pixels in epsilon_list:
                    progress.set_postfix(
                        attack=attack_name,
                        eps=epsilon_pixels,
                    )

                    with torch.enable_grad():
                        adversarial = attack(
                            model,
                            batch,
                            attack_name,
                            epsilon_pixels,
                        )

                    adversarial_features = hook.extract(model, adversarial)

                    for defense_name in defenses:
                        defended_features = (
                            adversarial_features
                            if defense_name == "none"
                            else hook.extract(
                                model,
                                defend(adversarial, defense_name),
                            )
                        )

                        write_rows(
                            writer,
                            clean_features,
                            adversarial_features,
                            defended_features,
                            clean_defended_by_defense[defense_name],
                            paths,
                            split,
                            attack_name,
                            epsilon_pixels,
                            defense_name,
                            stats,
                        )

            file.flush()


# =============================================================================
# CLI
# =============================================================================

def parse_list(value: str) -> list[str]:
    return [
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    ]


def parse_ints(value: str) -> list[int]:
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--output", default=OUTDIR)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--stats-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--attacks", type=parse_list, default=ATTACKS)
    parser.add_argument("--epsilons", type=parse_ints, default=EPS_LIST)
    parser.add_argument("--defenses", type=parse_list, default=DEFENSES)
    parser.add_argument("--rebuild-stats", action="store_true")
    parser.add_argument("--skip-clean-defenses", action="store_true")
    parser.add_argument("--quick", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device.isdigit():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output)
    stats_path = output_dir / "feature_normalization_val.pt"
    csv_path = output_dir / f"feature_consistency_{args.eval_split}.csv"
    config_path = output_dir / "feature_consistency_config.json"

    attacks = ["fgsm"] if args.quick else args.attacks
    epsilon_list = [1] if args.quick else args.epsilons
    defenses = ["none", "tnorm"] if args.quick else args.defenses

    print("=" * 80)
    print("FEATURE CONSISTENCY")
    print(f"model:   {args.model}")
    print(f"data:    {args.data}")
    print(f"device:  {device}")
    print(f"attacks: {attacks}")
    print(f"eps:     {epsilon_list}")
    print(f"defense: {defenses}")
    print(f"out:     {csv_path}")
    print("=" * 80)

    yolo = YOLO(args.model)
    model = yolo.model.to(device).float().eval()

    overrides = model.args if isinstance(model.args, dict) else vars(model.args)
    model.args = get_cfg(overrides=overrides)
    model.criterion = None

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    hook = FeatureHook(model)

    try:
        if args.rebuild_stats or not stats_path.exists():
            validation_loader = loader(
                Path(args.data),
                args.stats_split,
                args.imgsz,
                args.batch,
                args.workers,
                device.type == "cuda",
            )

            stats = collect_norm(
                model,
                hook,
                validation_loader,
                device,
                stats_path,
            )
        else:
            stats = load_norm(stats_path)

        evaluation_loader = loader(
            Path(args.data),
            args.eval_split,
            args.imgsz,
            args.batch,
            args.workers,
            device.type == "cuda",
        )

        run(
            model,
            hook,
            evaluation_loader,
            device,
            args.eval_split,
            attacks,
            epsilon_list,
            defenses,
            stats,
            csv_path,
            args.skip_clean_defenses,
        )

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "data": args.data,
                    "eval_split": args.eval_split,
                    "attacks": attacks,
                    "epsilons": epsilon_list,
                    "defenses": defenses,
                    "pgd_steps": PGD_STEPS,
                    "active_threshold": ACTIVE_THR,
                    "norm_quantiles": [NORM_Q_LOW, NORM_Q_HIGH],
                    "additional_metrics": [
                        "mse",
                        "mae",
                        "relative_l2",
                        "mean_shift",
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    finally:
        hook.close()

    print("DONE")
    print(csv_path)


if __name__ == "__main__":
    main()
