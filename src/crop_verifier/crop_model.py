from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


def select_crop_rows(group: pd.DataFrame) -> pd.DataFrame:
    detected = group[group["is_detection"].astype(bool)].copy()
    candidates = detected if not detected.empty else group.copy()
    candidates = candidates.sort_values(
        ["detector_confidence", "frame_order", "image_path"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    indices = [0, len(candidates) // 2, len(candidates) - 1]
    return candidates.iloc[indices].reset_index(drop=True)


def context_crop(
    image: np.ndarray,
    box: list[float],
    multiplier: float,
    size: int,
    padding_value: int,
) -> np.ndarray:
    height, width = image.shape[:2]
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    side = max(box[2] - box[0], box[3] - box[1], 2.0) * multiplier
    x1, y1 = int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2))
    x2, y2 = int(np.ceil(cx + side / 2)), int(np.ceil(cy + side / 2))
    canvas_side = max(x2 - x1, y2 - y1, 2)
    canvas = np.full((canvas_side, canvas_side, 3), padding_value, dtype=np.uint8)
    sx1, sy1 = max(x1, 0), max(y1, 0)
    sx2, sy2 = min(x2, width), min(y2, height)
    if sx2 > sx1 and sy2 > sy1:
        dx1, dy1 = sx1 - x1, sy1 - y1
        canvas[dy1 : dy1 + (sy2 - sy1), dx1 : dx1 + (sx2 - sx1)] = image[
            sy1:sy2, sx1:sx2
        ]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)


class FrozenYoloCropEncoder:
    def __init__(
        self,
        checkpoint: Path,
        last_layer_index: int,
        input_size: int,
        context_multiplier: float,
        padding_value: int,
        batch_size: int,
        device: str | None = None,
    ) -> None:
        self.input_size = input_size
        self.context_multiplier = context_multiplier
        self.padding_value = padding_value
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        wrapper = YOLO(str(checkpoint))
        self.layers = list(wrapper.model.model[: last_layer_index + 1])
        for layer in self.layers:
            layer.eval()
            layer.to(self.device)
            for parameter in layer.parameters():
                parameter.requires_grad_(False)

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        value: Any = batch
        with torch.inference_mode():
            for layer in self.layers:
                value = layer(value)
        if not isinstance(value, torch.Tensor) or value.ndim != 4:
            raise RuntimeError("Frozen crop backbone did not emit a feature map")
        return value.mean(dim=(2, 3)).cpu()

    def encode(self, observations: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        metadata = []
        crops = []
        for track_key, group in observations.groupby("track_key", sort=True):
            selected = select_crop_rows(group)
            for crop_rank, row in enumerate(selected.itertuples(index=False)):
                image = cv2.imread(str(row.image_path))
                if image is None:
                    raise RuntimeError(f"Unreadable crop source: {row.image_path}")
                crop = context_crop(
                    image,
                    [float(row.x1), float(row.y1), float(row.x2), float(row.y2)],
                    self.context_multiplier,
                    self.input_size,
                    self.padding_value,
                )
                crops.append(crop)
                metadata.append(
                    {
                        "track_key": str(track_key),
                        "grouped_scene_id": str(row.grouped_scene_id),
                        "crop_rank": crop_rank,
                        "image_path": str(row.image_path),
                    }
                )
        vectors = []
        for start in range(0, len(crops), self.batch_size):
            batch = np.stack(crops[start : start + self.batch_size])
            tensor = torch.from_numpy(
                batch[:, :, :, ::-1].copy().transpose(0, 3, 1, 2)
            ).float().div_(255.0).to(self.device)
            vectors.append(self._forward(tensor).numpy())
        encoded = np.concatenate(vectors, axis=0)
        meta = pd.DataFrame(metadata)
        rows = []
        matrices = []
        for track_key, group in meta.groupby("track_key", sort=True):
            indices = group.sort_values("crop_rank").index.to_numpy()
            matrix = encoded[indices]
            if matrix.shape[0] != 3:
                raise RuntimeError("Each tracklet must have three crop features")
            rows.append(
                {
                    "track_key": str(track_key),
                    "grouped_scene_id": str(group["grouped_scene_id"].iloc[0]),
                }
            )
            matrices.append(matrix.reshape(-1))
        result = np.stack(matrices).astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("Crop encoder produced NaN/Inf")
        return pd.DataFrame(rows), result


@dataclass
class CropMLPClassifier:
    hidden_units: int = 64
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ranking_weight: float = 0.25
    maximum_pairs_per_epoch: int = 4096
    seed: int = 20260725

    def fit(self, x: np.ndarray, y: np.ndarray) -> "CropMLPClassifier":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if set(np.unique(y)) != {0.0, 1.0}:
            raise ValueError("CropMLPClassifier requires both classes")
        self.classes_ = np.asarray([0, 1])
        self.mean_ = x.mean(axis=0).astype(np.float32)
        self.scale_ = np.maximum(x.std(axis=0), 1e-6).astype(np.float32)
        normalized = torch.tensor((x - self.mean_) / self.scale_)
        target = torch.tensor(y)
        torch.manual_seed(self.seed)
        network = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], self.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_units, 1),
        )
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        positive = torch.where(target == 1)[0]
        negative = torch.where(target == 0)[0]
        positive_weight = torch.tensor(
            max(float(len(negative)) / max(len(positive), 1), 1.0)
        )
        generator = torch.Generator().manual_seed(self.seed)
        network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = network(normalized).squeeze(1)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, pos_weight=positive_weight
            )
            pair_count = min(
                self.maximum_pairs_per_epoch,
                max(len(positive), len(negative)),
            )
            pos_index = positive[
                torch.randint(len(positive), (pair_count,), generator=generator)
            ]
            neg_index = negative[
                torch.randint(len(negative), (pair_count,), generator=generator)
            ]
            ranking = torch.nn.functional.softplus(
                -(logits[pos_index] - logits[neg_index])
            ).mean()
            (bce + self.ranking_weight * ranking).backward()
            optimizer.step()
        self.state_ = {
            name: value.detach().cpu().numpy()
            for name, value in network.state_dict().items()
        }
        self.input_features_ = x.shape[1]
        return self

    def _network(self) -> torch.nn.Module:
        network = torch.nn.Sequential(
            torch.nn.Linear(self.input_features_, self.hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_units, 1),
        )
        network.load_state_dict(
            {
                name: torch.tensor(value)
                for name, value in self.state_.items()
            }
        )
        network.eval()
        return network

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        normalized = (
            np.asarray(x, dtype=np.float32) - self.mean_
        ) / self.scale_
        with torch.inference_mode():
            logits = self._network()(torch.tensor(normalized)).squeeze(1)
            positive = torch.sigmoid(logits).numpy()
        return np.column_stack([1.0 - positive, positive])

