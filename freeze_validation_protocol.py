from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
VAL_RAW = PROJECT_DIR / "outputs/final_practice/unified_diagnostics_val_raw.csv"
VAL_CONFIG = VAL_RAW.with_suffix(".json")
VAL_STATS = PROJECT_DIR / "outputs/final_practice/09_statistics_val/model_comparison_m0_m4.csv"
OUTPUT = PROJECT_DIR / "outputs/final_practice/validation_freeze.json"
DEADLINE_PREPARATION = (
    PROJECT_DIR / "outputs/final_practice/deadline/deadline_validation_preparation.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze validation-selected final protocol")
    parser.add_argument("--raw", type=Path, default=VAL_RAW)
    parser.add_argument("--config", type=Path, default=VAL_CONFIG)
    parser.add_argument("--statistics", type=Path, default=VAL_STATS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for path in (args.raw, args.config, args.statistics):
        if not path.is_file():
            raise FileNotFoundError(path)
    data = pd.read_csv(args.raw)
    required = {"sequence_id", "split", "attack", "adaptive", "epsilon", "seed", "defense"}
    missing = required - set(data)
    if missing or data.empty or data["sequence_id"].isna().any():
        raise RuntimeError(f"Validation freeze rejected; missing={sorted(missing)}, rows={len(data)}")
    if set(data["split"].astype(str)) != {"val"}:
        raise RuntimeError("Validation freeze accepts only split=val")
    subprocess.run(
        [str(PROJECT_DIR / ".venv/bin/python"), "prepare_deadline_validation.py"],
        cwd=PROJECT_DIR, check=True,
    )
    if not DEADLINE_PREPARATION.is_file():
        raise RuntimeError("Deadline validation gate was not frozen before test")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()
    payload = {
        "status": "FROZEN_ON_VALIDATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git or None,
        "statistical_unit": "sequence_id",
        "test_opened": False,
        "rows": len(data),
        "sequences": int(data["sequence_id"].nunique()),
        "protocol": config,
        "artifacts": {
            str(path.relative_to(PROJECT_DIR)): sha256(path)
            for path in (args.raw, args.config, args.statistics, DEADLINE_PREPARATION)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
