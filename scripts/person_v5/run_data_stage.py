from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.person_v5.download_crowdhuman import acquire
from scripts.person_v5.prepare_crowdhuman import prepare


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "outputs/person_v5"
STATUS = OUTPUT / "pipeline_status.json"
DOWNLOADS = PROJECT / "data/crowdhuman_downloads"
DATASET = PROJECT / "data/crowdhuman"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def status(stage: str, state: str, **extra: Any) -> None:
    payload = (
        json.loads(STATUS.read_text(encoding="utf-8"))
        if STATUS.is_file()
        else {
            "protocol_id": "canonical-v5-person-data-first-v1",
            "test_opened": False,
            "training_started": False,
            "stages": {},
        }
    )
    payload["updated_at"] = now()
    payload["stages"][stage] = {"status": state, **extra}
    atomic_json(STATUS, payload)


def run() -> None:
    marker = PROJECT / "outputs/person_v3/test/TEST_OPENED.json"
    if marker.exists():
        raise RuntimeError("Railway test is not sealed")
    try:
        status("crowdhuman_download", "running", started_at=now())
        acquisition = acquire(DOWNLOADS, workers=3)
        status(
            "crowdhuman_download",
            "success",
            finished_at=now(),
            manifest=(
                "outputs/person_v5/protocol/"
                "data_acquisition_manifest.json"
            ),
            files=len(acquisition["files"]),
        )
        status("crowdhuman_conversion", "running", started_at=now())
        conversion = prepare(DOWNLOADS, DATASET)
        status(
            "crowdhuman_conversion",
            "success",
            finished_at=now(),
            audit=(
                "outputs/person_v5/protocol/"
                "crowdhuman_conversion_audit.json"
            ),
            train_images=conversion["splits"]["train"]["linked_images"],
            validation_images=conversion["splits"]["val"]["linked_images"],
        )
        status(
            "GPU_training",
            "blocked_pending_runtime_and_data_audit",
            test_opened=False,
        )
    except Exception as error:
        status(
            "data_stage",
            "failed",
            finished_at=now(),
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    run()

