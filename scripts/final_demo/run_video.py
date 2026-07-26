from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_demo import FinalDemoApp


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--input", required=True, type=Path)
    value.add_argument(
        "--mode",
        choices=("frame_baseline", "temporal_research"),
        required=True,
    )
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--config", type=Path, default=Path("configs/final_demo.yaml"))
    value.add_argument("--checkpoint", type=Path)
    value.add_argument("--device", default=None)
    return value


def main() -> None:
    args = parser().parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (
        args.output.parent / f"{args.output.stem}_{run_id}"
        if args.output.suffix.lower() == ".mp4"
        else args.output
    )
    app = FinalDemoApp(args.config, args.checkpoint, args.device)
    runtime = app.run_video(args.input, root, args.mode)
    if args.output.suffix.lower() == ".mp4":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / "annotated_video.mp4", args.output)
    print(json.dumps({"run_root": str(root), "runtime": runtime}, indent=2))


if __name__ == "__main__":
    main()
