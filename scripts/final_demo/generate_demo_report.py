from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for runtime_path in sorted(args.runs.rglob("runtime.json")):
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        rows.append(
            f"| {runtime_path.parent.name} | {runtime['frames']} | "
            f"{runtime['mean_latency_ms']:.2f} | {runtime['p95_latency_ms']:.2f} | "
            f"{runtime['fps']:.2f} |"
        )
    content = [
        "# Final demonstrator runtime",
        "",
        "Research demonstrator only. Temporal output is not approved for safety use.",
        "",
        "| Run | Frames | Mean ms | p95 ms | FPS |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(content), encoding="utf-8")


if __name__ == "__main__":
    main()
