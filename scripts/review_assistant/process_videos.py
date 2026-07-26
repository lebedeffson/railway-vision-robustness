from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.review_assistant.config import load_config
from src.review_assistant.processor_v2 import FailSafeReviewProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an operator-review event queue from local videos."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["conservative_review", "high_recall_review", "combined_queue"],
        default="combined_queue",
    )
    parser.add_argument("--camera-id", default="local-camera")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--verifier", type=Path)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    extensions = {str(value).lower() for value in config["input"]["extensions"]}
    if args.input.is_file():
        videos = [args.input]
    else:
        iterator = args.input.rglob("*") if args.recursive else args.input.glob("*")
        videos = sorted(path for path in iterator if path.suffix.lower() in extensions)
    if not videos:
        raise SystemExit("No supported local videos found.")
    results = []
    for video in videos:
        processor = FailSafeReviewProcessor(
            checkpoint=args.checkpoint,
            verifier=args.verifier,
            encoder=args.encoder,
            database=args.database,
            device=args.device,
        )

        def progress(row: dict) -> None:
            if not row.get("error"):
                print(
                    f"\r{video.name}: {row['processed_frames']}/"
                    f"{row['total_frames']} frames · {row['fps']:.2f} FPS · "
                    f"ETA {row['eta_seconds']:.0f}s · {row['events']} events",
                    end="",
                    flush=True,
                )

        result = processor.process(
            video, mode=args.mode, camera_id=args.camera_id, progress=progress
        )
        print()
        results.append(result)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
