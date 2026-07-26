from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_demo import FinalDemoApp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("frame_baseline", "temporal_research"),
        required=True,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--config", type=Path, default=Path("configs/final_demo.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    app = FinalDemoApp(args.config, args.checkpoint, args.device)
    runtime = app.run_image_folder(args.input, args.output, args.mode, args.fps)
    print(json.dumps({"run_root": str(args.output), "runtime": runtime}, indent=2))


if __name__ == "__main__":
    main()
