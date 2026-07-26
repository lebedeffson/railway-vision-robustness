from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2

from .models import ReviewEvent


def _annotate_event_frame(
    frame, event: ReviewEvent, frame_number: int
):  # numpy type deliberately optional
    output = frame.copy()
    rows = [item for item in event.detections if item.frame_number == frame_number]
    for item in rows:
        color = (0, 220, 0)
        if item.source == "TEMPORAL_ONLY":
            color = (0, 165, 255)
        if item.interpolated:
            color = (255, 0, 255)
        x1, y1, x2, y2 = [int(round(value)) for value in item.box]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{item.source} {item.confidence:.2f}",
            (x1, max(y1 - 5, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"{event.event_id} | HUMAN REVIEW REQUIRED",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 230),
        2,
        cv2.LINE_AA,
    )
    if event.source_label == "TEMPORAL_ONLY":
        cv2.putText(
            output,
            "TEMPORAL-ONLY: HIGH FALSE-POSITIVE RISK",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 100, 255),
            2,
            cv2.LINE_AA,
        )
    return output


class EventClipWriter:
    def __init__(self, pre_roll_seconds: float, post_roll_seconds: float) -> None:
        self.pre_roll_seconds = float(pre_roll_seconds)
        self.post_roll_seconds = float(post_roll_seconds)

    def write(
        self,
        video_path: Path,
        event: ReviewEvent,
        destination: Path,
    ) -> tuple[Path, Path]:
        destination.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open event source video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 1.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        first_frame = max(
            int((event.start_time - self.pre_roll_seconds) * fps), 0
        )
        last_frame = min(
            int((event.end_time + self.post_roll_seconds) * fps), total - 1
        )
        clip_path = destination / "clip.mp4"
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create event clip: {clip_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        thumbnail_path = destination / "thumbnail.jpg"
        thumbnail_frame = max(
            event.detections,
            key=lambda item: item.confidence,
        ).frame_number
        for frame_number in range(first_frame, last_frame + 1):
            ok, frame = capture.read()
            if not ok:
                break
            annotated = _annotate_event_frame(frame, event, frame_number)
            writer.write(annotated)
            if frame_number == thumbnail_frame:
                cv2.imwrite(str(thumbnail_path), annotated)
        writer.release()
        capture.release()
        if not thumbnail_path.is_file():
            fallback = cv2.VideoCapture(str(clip_path))
            ok, frame = fallback.read()
            fallback.release()
            if ok:
                cv2.imwrite(str(thumbnail_path), frame)
        event_json = event.to_record()
        event_json.update(
            {
                "clip_start_time": first_frame / fps,
                "clip_end_time": last_frame / fps,
                "review_status": "PENDING",
                "operator_comment": "",
            }
        )
        (destination / "event.json").write_text(
            json.dumps(event_json, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fields = [
            "frame_number",
            "timestamp",
            "track_id",
            "confidence",
            "box",
            "source",
            "confirmed",
            "interpolated",
            "motion",
            "candidate_id",
            "processing_status",
        ]
        with (destination / "detections.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer_csv = csv.DictWriter(handle, fieldnames=fields)
            writer_csv.writeheader()
            for row in event.detections_as_records():
                row["box"] = json.dumps(row["box"])
                writer_csv.writerow(row)
        return clip_path, thumbnail_path
