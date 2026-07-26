from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_demo import FinalDemoApp


def combine(left: Path, right: Path, output: Path) -> None:
    first = cv2.VideoCapture(str(left))
    second = cv2.VideoCapture(str(right))
    fps = float(first.get(cv2.CAP_PROP_FPS) or 10.0)
    width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create comparison video")
    while True:
        ok_left, left_frame = first.read()
        ok_right, right_frame = second.read()
        if not ok_left or not ok_right:
            break
        cv2.putText(
            left_frame,
            "FRAME BASELINE",
            (15, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            right_frame,
            "TEMPORAL RESEARCH",
            (15, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(cv2.hconcat([left_frame, right_frame]))
    first.release()
    second.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/final_demo.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    root = args.output.parent / f"{args.output.stem}_runs"
    frame_root = root / "frame_baseline"
    temporal_root = root / "temporal_research"
    FinalDemoApp(args.config, args.checkpoint, args.device).run_video(
        args.input, frame_root, "frame_baseline"
    )
    FinalDemoApp(args.config, args.checkpoint, args.device).run_video(
        args.input, temporal_root, "temporal_research"
    )
    combine(
        frame_root / "annotated_video.mp4",
        temporal_root / "annotated_video.mp4",
        args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
