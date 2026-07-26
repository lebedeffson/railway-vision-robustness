from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.operator_assistant_evidence_v2.common import (
    OUTPUT,
    PROJECT,
    assert_test_sealed,
    atomic_csv,
    config,
    sha256,
)


def _stream_bbox(
    frame_object: dict[str, Any], stream: str
) -> tuple[list[float], str] | None:
    for bbox in frame_object.get("object_data", {}).get("bbox", []):
        if bbox.get("coordinate_system") == stream or str(
            bbox.get("name", "")
        ).startswith(f"{stream}__bbox__"):
            attributes = bbox.get("attributes", {}).get("text", [])
            occlusion = next(
                (
                    str(item.get("val", "undefined"))
                    for item in attributes
                    if item.get("name") == "occlusion"
                ),
                "undefined",
            )
            return [float(value) for value in bbox["val"]], occlusion
    return None


def main() -> None:
    assert_test_sealed()
    settings = config()
    benchmark = settings["benchmark"]
    index = pd.read_csv(PROJECT / benchmark["development_index"])
    selected = index[
        (index["grouped_scene_id"] == benchmark["grouped_scene_id"])
        & (index["source_sequence_id"] == benchmark["source_sequence_id"])
    ].sort_values("frame_number")
    if len(selected) != int(benchmark["frames"]):
        raise RuntimeError(f"Expected 100 development frames, got {len(selected)}")
    source_numbers = selected["frame_number"].astype(int).tolist()
    expected_numbers = list(
        range(source_numbers[0], source_numbers[0] + int(benchmark["frames"]))
    )
    if source_numbers != expected_numbers:
        raise RuntimeError("Development source frames are not contiguous")
    annotation_path = PROJECT / benchmark["annotation"]
    annotation_hash = sha256(annotation_path)
    openlabel = json.loads(annotation_path.read_text(encoding="utf-8"))["openlabel"]
    video_path = OUTPUT / "development_event_benchmark_v2.mp4"
    manifest_path = OUTPUT / "development_event_benchmark_v2_manifest.csv"
    boxes_path = OUTPUT / "GT_PERSON_BOXES.csv"
    episodes_path = OUTPUT / "GT_PERSON_EPISODES.csv"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(benchmark["fps"]),
        (int(benchmark["width"]), int(benchmark["height"])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create benchmark video: {video_path}")
    manifest_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    episode_frames: dict[str, list[int]] = {}
    try:
        for video_frame_id, source in enumerate(selected.itertuples(index=False)):
            image_path = Path(str(source.image_path))
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Missing development frame: {image_path.name}")
            source_height, source_width = frame.shape[:2]
            resized = cv2.resize(
                frame,
                (int(benchmark["width"]), int(benchmark["height"])),
                interpolation=cv2.INTER_AREA,
            )
            writer.write(resized)
            source_frame_id = int(source.frame_number)
            frame_payload = openlabel["frames"].get(str(source_frame_id))
            if frame_payload is None:
                raise RuntimeError(
                    f"Annotation frame missing: {source_frame_id}"
                )
            manifest_rows.append(
                {
                    "video_frame_id": video_frame_id,
                    "scene_id": benchmark["grouped_scene_id"],
                    "source_sequence": benchmark["source_sequence_id"],
                    "source_frame_id": source_frame_id,
                    "timestamp": float(source.timestamp),
                    "annotation_file": benchmark["annotation"],
                    "source_image": image_path.relative_to(PROJECT).as_posix(),
                    "source_image_sha256": sha256(image_path),
                    "annotation_sha256": annotation_hash,
                }
            )
            for object_id, frame_object in frame_payload.get("objects", {}).items():
                object_record = openlabel["objects"].get(object_id, {})
                if object_record.get("type") != "person":
                    continue
                result = _stream_bbox(frame_object, benchmark["stream"])
                if result is None:
                    continue
                center_box, occlusion = result
                cx, cy, width, height = center_box
                scale_x = float(benchmark["width"]) / source_width
                scale_y = float(benchmark["height"]) / source_height
                x1 = max((cx - width / 2) * scale_x, 0.0)
                y1 = max((cy - height / 2) * scale_y, 0.0)
                x2 = min((cx + width / 2) * scale_x, float(benchmark["width"]))
                y2 = min((cy + height / 2) * scale_y, float(benchmark["height"]))
                if x2 <= x1 or y2 <= y1:
                    raise RuntimeError(
                        f"Invalid person box {object_id} frame {source_frame_id}"
                    )
                episode_frames.setdefault(object_id, []).append(video_frame_id)
                box_rows.append(
                    {
                        "episode_id": object_id,
                        "scene_id": benchmark["grouped_scene_id"],
                        "person_id": object_id,
                        "video_frame_id": video_frame_id,
                        "source_frame_id": source_frame_id,
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "temporarily_occluded": occlusion not in {"0-25 %", "0"},
                        "occlusion": occlusion,
                        "annotation_sha256": annotation_hash,
                    }
                )
    finally:
        writer.release()
    episodes = []
    for object_id, frames in sorted(episode_frames.items()):
        if not frames:
            continue
        episodes.append(
            {
                "episode_id": object_id,
                "scene_id": benchmark["grouped_scene_id"],
                "person_id": object_id,
                "start_frame": min(frames),
                "end_frame": max(frames),
                "bbox_per_frame_reference": (
                    f"GT_PERSON_BOXES.csv#episode_id={object_id}"
                ),
                "temporarily_occluded": any(
                    row["temporarily_occluded"]
                    for row in box_rows
                    if row["episode_id"] == object_id
                ),
                "annotation_author": "OSDaR23_OPENLABEL",
                "verification_author": "SOURCE_OBJECT_UUID",
            }
        )
    if not episodes:
        raise RuntimeError("No source-ID person episodes in selected camera stream")
    atomic_csv(manifest_path, pd.DataFrame(manifest_rows))
    atomic_csv(boxes_path, pd.DataFrame(box_rows))
    atomic_csv(episodes_path, pd.DataFrame(episodes))
    shutil.copyfile(
        PROJECT
        / "protocol/operator_assistant_evidence_v2/GT_EPISODE_ANNOTATION_PROTOCOL.md",
        OUTPUT / "GT_EPISODE_ANNOTATION_PROTOCOL.md",
    )
    print(
        json.dumps(
            {
                "video_frames": len(manifest_rows),
                "person_episodes": len(episodes),
                "person_boxes": len(box_rows),
                "video_sha256": sha256(video_path),
                "manifest_sha256": sha256(manifest_path),
                "annotation_sha256": annotation_hash,
                "test_status": "SEALED",
                "test_access_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
