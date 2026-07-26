from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config_loader import resolve_local_path, sha256_file


class FrozenPersonDetector:
    def __init__(
        self,
        settings: dict[str, Any],
        checkpoint_override: Path | None = None,
        device: str = "auto",
    ) -> None:
        from ultralytics import YOLO

        checkpoint = checkpoint_override or resolve_local_path(settings["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Local detector checkpoint is missing: {checkpoint}")
        expected = settings.get("expected_sha256")
        if expected and sha256_file(checkpoint) != expected:
            raise RuntimeError("Frozen detector checkpoint SHA-256 mismatch")
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = sha256_file(checkpoint)
        self.settings = settings
        self.device = None if device == "auto" else device
        self.model = YOLO(str(checkpoint))

    def candidates(self, frame: np.ndarray) -> list[dict[str, Any]]:
        result = self.model.predict(
            source=frame,
            conf=float(self.settings["candidate_floor"]),
            iou=float(self.settings["iou_threshold"]),
            imgsz=int(self.settings["image_size"]),
            classes=[0],
            device=self.device,
            verbose=False,
        )[0]
        rows: list[dict[str, Any]] = []
        if result.boxes is None:
            return rows
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidence = result.boxes.conf.detach().cpu().numpy()
        for box, score in zip(boxes, confidence):
            rows.append(
                {
                    "box": [float(value) for value in box],
                    "confidence": float(score),
                    "raw_confidence": float(score),
                    "class_id": 0,
                    "track_id": None,
                    "source": "detector",
                    "interpolated": False,
                    "confirmed": score >= float(self.settings["standard_threshold"]),
                }
            )
        return rows

    def standard(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        threshold = float(self.settings["standard_threshold"])
        return [dict(row) for row in candidates if float(row["confidence"]) >= threshold]
