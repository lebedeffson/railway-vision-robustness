from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.operator_assistant_evidence_v2.common import (
    CONFIG_PATH,
    OUTPUT,
    PROJECT,
    assert_test_sealed,
    atomic_json,
    config,
    sha256,
)


LOCK = OUTPUT / "EVENT_PROTOCOL_LOCK.json"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE_NOT_GIT"


def _record(relative: str) -> dict[str, Any]:
    path = PROJECT / relative
    if not path.is_file():
        raise RuntimeError(f"Frozen input missing: {relative}")
    return {
        "path": relative,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def payload() -> dict[str, Any]:
    settings = config()
    review = __import__("yaml").safe_load(
        (PROJECT / settings["frozen_pipeline"]["review_config"]).read_text(
            encoding="utf-8"
        )
    )
    sensitivity = settings["events"]["sensitivity"]
    combinations = (
        len(sensitivity["join_time_seconds"])
        * len(sensitivity["minimum_iou"])
        * len(sensitivity["maximum_center_distance_ratio"])
        * len(sensitivity["reopen_window_seconds"])
    )
    if combinations != int(sensitivity["expected_configurations"]):
        raise RuntimeError(f"Unexpected grid size: {combinations}")
    files = {
        "benchmark_video": _record(
            "outputs/operator_assistant_evidence_v2/"
            "development_event_benchmark_v2.mp4"
        ),
        "benchmark_manifest": _record(
            "outputs/operator_assistant_evidence_v2/"
            "development_event_benchmark_v2_manifest.csv"
        ),
        "gt_person_episodes": _record(
            "outputs/operator_assistant_evidence_v2/GT_PERSON_EPISODES.csv"
        ),
        "gt_person_boxes": _record(
            "outputs/operator_assistant_evidence_v2/GT_PERSON_BOXES.csv"
        ),
        "source_annotation": _record(settings["benchmark"]["annotation"]),
        "protocol_config": _record(
            "configs/operator_assistant_evidence_v2.yaml"
        ),
        "review_config": _record(settings["frozen_pipeline"]["review_config"]),
        "detector_checkpoint": _record(review["detector"]["checkpoint"]),
        "verifier_model": _record(review["verifier"]["model"]),
        "verifier_encoder": _record(
            review["verifier"]["encoder_checkpoint"]
        ),
        "event_aggregator_source": _record(
            "src/review_assistant/event_aggregator_v2.py"
        ),
        "event_model_source": _record("src/review_assistant/models.py"),
        "processor_source": _record("src/review_assistant/processor_v2.py"),
        "database_source": _record("src/review_assistant/database.py"),
        "temporal_pipeline_source": _record(
            "src/final_demo/temporal_pipeline.py"
        ),
        "tracker_source": _record("src/temporal_safety/tracker_base.py"),
        "evidence_runner_source": _record(
            "scripts/operator_assistant_evidence_v2/run_evidence.py"
        ),
        "episode_protocol": _record(
            "protocol/operator_assistant_evidence_v2/"
            "GT_EPISODE_ANNOTATION_PROTOCOL.md"
        ),
    }
    return {
        "protocol_id": settings["protocol_id"],
        "parent_protocol": settings["parent_protocol"],
        "git_commit_before_computation": _git_commit(),
        "inputs": files,
        "benchmark": settings["benchmark"],
        "baseline_event_parameters": settings["events"]["baseline"],
        "sensitivity_grid": {
            **sensitivity,
            "number_of_configurations": combinations,
        },
        "runtime": settings["runtime"],
        "test_status": "SEALED",
        "test_access_count": 0,
        "training": "FORBIDDEN",
        "multistream_benchmark": "DEFERRED",
        "immutable_after_first_pipeline_run": True,
    }


def main() -> None:
    assert_test_sealed()
    current = payload()
    if LOCK.exists():
        existing = json.loads(LOCK.read_text(encoding="utf-8"))
        stable_existing = {
            key: value
            for key, value in existing.items()
            if key != "git_commit_before_computation"
        }
        stable_current = {
            key: value
            for key, value in current.items()
            if key != "git_commit_before_computation"
        }
        if stable_existing != stable_current:
            raise RuntimeError("Frozen event protocol differs; use a new amendment")
        print("verified EVENT_PROTOCOL_LOCK.json")
        return
    atomic_json(LOCK, current)
    print("created EVENT_PROTOCOL_LOCK.json")


if __name__ == "__main__":
    main()
