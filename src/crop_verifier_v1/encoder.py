from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


SLOTS = (
    "first_stable_real_hit",
    "maximum_confidence_real_hit",
    "latest_real_hit",
)


def select_real_crop_rows(group: pd.DataFrame) -> pd.DataFrame:
    real = group[group["is_detector_hit"].astype(bool)].copy()
    if real.empty:
        raise RuntimeError(f"Track has no real detector hit: {group.name}")
    real = real.sort_values(
        ["frame_number", "detector_confidence", "image_path"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    stable = real.iloc[min(1, len(real) - 1)]
    maximum = real.sort_values(
        ["detector_confidence", "frame_number", "image_path"],
        ascending=[False, True, True],
    ).iloc[0]
    latest = real.iloc[-1]
    selected = pd.DataFrame([stable, maximum, latest]).reset_index(drop=True)
    selected.insert(0, "crop_slot", SLOTS)
    return selected


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
        canvas[dy1 : dy1 + sy2 - sy1, dx1 : dx1 + sx2 - sx1] = image[
            sy1:sy2, sx1:sx2
        ]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)


class FrozenYoloTrackEncoder:
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
        self.input_size = int(input_size)
        self.context_multiplier = float(context_multiplier)
        self.padding_value = int(padding_value)
        self.batch_size = int(batch_size)
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        wrapper = YOLO(str(checkpoint))
        self.layers = list(wrapper.model.model[: int(last_layer_index) + 1])
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

    def encode(
        self, observations: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        metadata: list[dict[str, Any]] = []
        crops: list[np.ndarray] = []
        for track_key, group in observations.groupby("track_key", sort=True):
            selected = select_real_crop_rows(group)
            unique_frames = int(selected["frame_number"].nunique())
            for row in selected.itertuples(index=False):
                image = cv2.imread(str(row.image_path))
                if image is None:
                    raise RuntimeError(f"Unreadable crop source: {row.image_path}")
                crops.append(
                    context_crop(
                        image,
                        [float(row.x1), float(row.y1), float(row.x2), float(row.y2)],
                        self.context_multiplier,
                        self.input_size,
                        self.padding_value,
                    )
                )
                metadata.append({
                    "track_key": str(track_key),
                    "grouped_scene_id": str(row.grouped_scene_id),
                    "crop_slot": str(row.crop_slot),
                    "frame_number": int(row.frame_number),
                    "image_path": str(row.image_path),
                    "x1": float(row.x1),
                    "y1": float(row.y1),
                    "x2": float(row.x2),
                    "y2": float(row.y2),
                    "unique_crop_frames": unique_frames,
                    "is_detector_hit": bool(row.is_detector_hit),
                    "interpolated": bool(row.interpolated),
                })
        if not crops:
            raise RuntimeError("No real detector crops were available")
        vectors: list[np.ndarray] = []
        for start in range(0, len(crops), self.batch_size):
            batch = np.stack(crops[start : start + self.batch_size])
            tensor = torch.from_numpy(
                batch[:, :, :, ::-1].copy().transpose(0, 3, 1, 2)
            ).float().div_(255.0).to(self.device)
            vectors.append(self._forward(tensor).numpy())
        encoded = np.concatenate(vectors, axis=0)
        crop_metadata = pd.DataFrame(metadata)
        track_rows: list[dict[str, Any]] = []
        track_vectors: list[np.ndarray] = []
        for track_key, group in crop_metadata.groupby("track_key", sort=True):
            indices = group.index.to_numpy()
            matrix = encoded[indices]
            if matrix.shape[0] != 3:
                raise RuntimeError("Each track must have exactly three crop slots")
            if group["interpolated"].astype(bool).any():
                raise RuntimeError("Interpolated crop reached visual encoder")
            track_rows.append({
                "track_key": str(track_key),
                "grouped_scene_id": str(group["grouped_scene_id"].iloc[0]),
                "unique_crop_frames": int(group["unique_crop_frames"].iloc[0]),
            })
            track_vectors.append(np.mean(matrix, axis=0))
        result = np.stack(track_vectors).astype(np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("Visual embeddings contain NaN/Inf")
        return pd.DataFrame(track_rows), crop_metadata, result

