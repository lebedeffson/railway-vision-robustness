from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .safety_banner import draw_research_warning


CSV_FIELDS = [
    "frame_number",
    "timestamp",
    "track_id",
    "event_type",
    "confidence",
    "bbox",
    "source",
    "confirmed",
    "interpolated",
]


def _dashed_rectangle(
    image: np.ndarray, box: list[float], color: tuple[int, int, int]
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    for x in range(x1, x2, 12):
        cv2.line(image, (x, y1), (min(x + 7, x2), y1), color, 2)
        cv2.line(image, (x, y2), (min(x + 7, x2), y2), color, 2)
    for y in range(y1, y2, 12):
        cv2.line(image, (x1, y), (x1, min(y + 7, y2)), color, 2)
        cv2.line(image, (x2, y), (x2, min(y + 7, y2)), color, 2)


def annotate(
    frame: np.ndarray, detections: list[dict[str, Any]], research: bool
) -> np.ndarray:
    output = draw_research_warning(frame) if research else frame.copy()
    for row in detections:
        source = str(row.get("source", "detector"))
        color = (0, 220, 0) if source == "detector" else (0, 165, 255)
        if row.get("interpolated"):
            color = (255, 0, 255)
            _dashed_rectangle(output, row["box"], color)
        else:
            x1, y1, x2, y2 = [int(round(value)) for value in row["box"]]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = f"{source} {float(row['confidence']):.2f}"
        if row.get("track_id") is not None:
            label = f"#{row['track_id']} {label}"
        cv2.putText(
            output,
            label,
            (int(row["box"][0]), max(18, int(row["box"][1]) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


class RunWriter:
    def __init__(
        self, root: Path, fps: float, size: tuple[int, int], config: dict[str, Any]
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.video = cv2.VideoWriter(
            str(root / "annotated_video.mp4"),
            cv2.VideoWriter_fourcc(*str(config["output"]["codec"])),
            fps,
            size,
        )
        if not self.video.isOpened():
            raise RuntimeError("Could not create annotated video")
        self.detections: list[dict[str, Any]] = []
        self.tracks: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

    def write(
        self,
        frame: np.ndarray,
        rows: list[dict[str, Any]],
        events: list[dict[str, Any]],
        frame_number: int,
        timestamp: float,
    ) -> None:
        self.video.write(frame)
        for row in rows:
            record = {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "track_id": row.get("track_id", ""),
                "event_type": "detection",
                "confidence": row["confidence"],
                "bbox": json.dumps(row["box"]),
                "source": row.get("source", "detector"),
                "confirmed": bool(row.get("confirmed", True)),
                "interpolated": bool(row.get("interpolated", False)),
            }
            self.detections.append(record)
            self.events.append(record)
            if row.get("track_id") is not None:
                self.tracks.append(record)
        for event in events:
            self.events.append(
                {
                    "frame_number": frame_number,
                    "timestamp": timestamp,
                    "track_id": event.get("track_id", ""),
                    "event_type": event.get("event", "tracker"),
                    "confidence": "",
                    "bbox": "",
                    "source": "tracker",
                    "confirmed": bool(event.get("confirmed", False)),
                    "interpolated": False,
                }
            )

    def _csv(self, name: str, rows: list[dict[str, Any]]) -> None:
        with (self.root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def close(
        self,
        runtime: dict[str, Any],
        config: dict[str, Any],
        provenance: dict[str, Any],
    ) -> None:
        self.video.release()
        self._csv("detections.csv", self.detections)
        self._csv("tracks.csv", self.tracks)
        self._csv("events.csv", self.events)
        (self.root / "runtime.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (self.root / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        (self.root / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.root / "README.txt").write_text(
            "RESEARCH DEMONSTRATOR ONLY\n"
            "Temporal mode has a high false-alarm rate and is not for safety deployment.\n",
            encoding="utf-8",
        )
