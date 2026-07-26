from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.review_assistant.config import load_config, resolve_path
from src.review_assistant.reporting import generate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config()
    metrics = generate_report(
        args.database or resolve_path(config["storage"]["database"]),
        args.run_id,
        args.output or resolve_path(config["storage"]["output_root"]),
        resolved_config={
            key: value for key, value in config.items() if not key.startswith("_")
        },
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
