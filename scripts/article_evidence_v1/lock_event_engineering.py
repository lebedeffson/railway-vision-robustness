from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.article_evidence_v1.common import atomic_json, sha256


CONFIG_PATH = PROJECT / "configs/event_engineering_evidence_v1.yaml"
LOCK_PATH = (
    PROJECT
    / "outputs/article_evidence_v1/operator_assistant/EVENT_ENGINEERING_LOCK.json"
)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE_NOT_GIT"


def _input_record(relative: str) -> dict[str, Any]:
    path = PROJECT / relative
    return {
        "path": relative,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def build_lock() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    sensitivity = config["sensitivity"]
    combinations = (
        len(sensitivity["join_time_seconds"])
        * len(sensitivity["minimum_iou"])
        * len(sensitivity["maximum_center_distance_ratio"])
        * len(sensitivity["reopen_window_seconds"])
    )
    if combinations != int(sensitivity["expected_configurations"]):
        raise RuntimeError(f"Unexpected event grid size: {combinations}")
    inputs = {
        key: _input_record(relative)
        for key, relative in config["inputs"].items()
    }
    return {
        "protocol_id": config["protocol_id"],
        "parent_protocol": config["parent_protocol"],
        "git_commit_before_computation": _git_commit(),
        "config_sha256": sha256(CONFIG_PATH),
        "inputs": inputs,
        "event_parameter_grid": {
            "join_time_seconds": sensitivity["join_time_seconds"],
            "minimum_iou": sensitivity["minimum_iou"],
            "maximum_center_distance_ratio": sensitivity[
                "maximum_center_distance_ratio"
            ],
            "reopen_window_seconds": sensitivity["reopen_window_seconds"],
            "number_of_configurations": combinations,
        },
        "clip_window_seconds": {
            "pre_roll": sensitivity["clip_pre_roll_seconds"],
            "post_roll": sensitivity["clip_post_roll_seconds"],
        },
        "scaling_streams": config["scaling"]["streams"],
        "scaling_measured_frames_per_stream": config["scaling"][
            "measured_frames_per_stream"
        ],
        "ground_truth_episode_missing_policy": config[
            "ground_truth_episode_metrics"
        ]["missing_policy"],
        "test_status": "SEALED",
        "test_access_count": 0,
        "training": "FORBIDDEN",
        "article_editing": "OUT_OF_SCOPE",
        "immutable_after_first_computation": True,
    }


def main() -> None:
    payload = build_lock()
    if LOCK_PATH.exists():
        existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        stable_existing = {
            key: value
            for key, value in existing.items()
            if key != "git_commit_before_computation"
        }
        stable_payload = {
            key: value
            for key, value in payload.items()
            if key != "git_commit_before_computation"
        }
        if stable_existing != stable_payload:
            raise RuntimeError(
                "EVENT_ENGINEERING_LOCK already exists and differs; "
                "create a new amendment instead of rewriting it."
            )
        print(f"verified immutable lock: {LOCK_PATH.relative_to(PROJECT)}")
        return
    atomic_json(LOCK_PATH, payload)
    print(f"created immutable lock: {LOCK_PATH.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
