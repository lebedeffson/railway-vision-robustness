from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from download_osdar23_direct import SEQUENCES, sequence_is_complete


PROJECT_DIR = Path(__file__).resolve().parent
STATE_PATH = PROJECT_DIR / "outputs/final_practice/training_pipeline_state.json"


def write_state(stage: str, status: str, **extra: object) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"stage": stage, "status": status, **extra}, indent=2) + "\n",
        encoding="utf-8",
    )


def run(stage: str, script: str, *arguments: str) -> None:
    write_state(stage, "RUNNING", command=[sys.executable, script, *arguments])
    subprocess.run([sys.executable, script, *arguments], cwd=PROJECT_DIR, check=True)
    write_state(stage, "PASS")


def main() -> None:
    missing = [sequence for sequence in SEQUENCES if not sequence_is_complete(sequence)]
    if missing:
        write_state("download", "BLOCKED", missing_sequences=missing)
        raise SystemExit(
            f"Dataset download is incomplete: {len(missing)} sequences missing"
        )
    run("build_dataset", "build_yolo_dataset.py")
    run("validate_labels", "validate_yolo_dataset.py")
    run("sequence_audit", "audit_final_practice.py")
    run("train", "train_yolo_baseline.py")
    run("clean_test", "evaluate_baseline.py")
    run("delivery_zip", "build_final_delivery.py")
    write_state("complete", "PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_state("failed", "FAIL", error=f"{type(error).__name__}: {error}")
        raise
