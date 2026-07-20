from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from download_osdar23_direct import SEQUENCES, sequence_is_complete


PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / ".venv/bin/python"
STATE = PROJECT_DIR / "outputs/final_practice/wait_and_train_state.json"
POLL_SECONDS = 30


def update(status: str, **extra: object) -> None:
    payload = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)


def environment_ready() -> bool:
    if not VENV_PYTHON.is_file():
        return False
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", "import torch, ultralytics, cv2, pandas, sklearn, scipy"],
        cwd=PROJECT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    while True:
        missing = [name for name in SEQUENCES if not sequence_is_complete(name)]
        ready = environment_ready()
        if not missing and ready:
            break
        update(
            "WAITING",
            complete_sequences=len(SEQUENCES) - len(missing),
            total_sequences=len(SEQUENCES),
            missing_sequences=missing,
            environment_ready=ready,
        )
        time.sleep(POLL_SECONDS)
    update("STARTING_PIPELINE")
    subprocess.run(
        [str(VENV_PYTHON), "run_training_pipeline.py"],
        cwd=PROJECT_DIR,
        check=True,
    )
    update("COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        update("FAIL", error=f"{type(error).__name__}: {error}")
        raise
