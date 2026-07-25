from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from scripts.person_canonical_v5.common import (
    PROJECT,
    assert_test_sealed,
    atomic_json,
    now,
    sha256,
)


CONFIG = PROJECT / "configs/person_v5/execution_v5.yaml"
EXECUTION_LOCK = PROJECT / "protocol/v5/V5_EXECUTION_LOCK.json"
RUNTIME_LOCK = PROJECT / "protocol/v5/V5_RUNTIME_LOCK.json"
RUNTIME_ROOT = PROJECT / "runtime/v5"
RUN_STATE = RUNTIME_ROOT / "RUN_STATE.json"
HEARTBEAT = RUNTIME_ROOT / "heartbeat.json"
COMPLETED = RUNTIME_ROOT / "completed"
RESULTS = PROJECT / "results/v5"
REPORTS = PROJECT / "reports/v5"
RELEASE = PROJECT / "release/v5"
OUTPUT = PROJECT / "outputs/person_canonical_v5"


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def assert_execution_locked() -> dict[str, Any]:
    assert_test_sealed()
    if not EXECUTION_LOCK.is_file() or not RUNTIME_LOCK.is_file():
        raise RuntimeError("Person-v5 execution is not fully locked")
    execution = json.loads(EXECUTION_LOCK.read_text(encoding="utf-8"))
    runtime = json.loads(RUNTIME_LOCK.read_text(encoding="utf-8"))
    if execution["test_status"] != "SEALED":
        raise RuntimeError("Execution lock does not seal test")
    if execution["attacks_status"] != "BLOCKED":
        raise RuntimeError("Execution lock does not block attacks")
    if runtime["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Execution config changed after runtime lock")
    if runtime["execution_lock_sha256"] != sha256(EXECUTION_LOCK):
        raise RuntimeError("Execution lock changed after runtime lock")
    for relative, expected in runtime["implementation_sha256"].items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Runtime implementation changed: {relative}")
    return runtime


def update_state(
    candidate: str,
    fold: int | None,
    stage: str,
    status: str,
    **values: Any,
) -> None:
    previous = (
        json.loads(RUN_STATE.read_text(encoding="utf-8"))
        if RUN_STATE.is_file()
        else {}
    )
    payload = {
        **previous,
        "protocol_id": config()["protocol_id"],
        "candidate": candidate,
        "fold": fold,
        "stage": stage,
        "service_status": status,
        "test_access_count": 0,
        "attack_runs": 0,
        "updated_at": now(),
        **values,
    }
    atomic_json(RUN_STATE, payload)
    atomic_json(
        HEARTBEAT,
        {
            "protocol_id": config()["protocol_id"],
            "candidate": candidate,
            "fold": fold,
            "stage": stage,
            "status": status,
            "updated_at": payload["updated_at"],
        },
    )


def mark_complete(candidate: str, fold: int, payload: dict[str, Any]) -> Path:
    required = ("checkpoint_sha256", "evaluator_consistency", "lost_GT")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"Refusing incomplete marker; missing {missing}")
    if payload["evaluator_consistency"] != "PASS" or payload["lost_GT"] != 0:
        raise RuntimeError("Refusing completion marker for invalid evaluation")
    path = COMPLETED / f"{candidate}_fold{fold}.done"
    atomic_json(
        path,
        {
            "status": "RUNTIME_PASS",
            "candidate": candidate,
            "fold": fold,
            "completed_at": now(),
            "test_access_count": 0,
            "attack_runs": 0,
            **payload,
        },
    )
    return path


@contextmanager
def exclusive_execution() -> Iterator[None]:
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = RUNTIME_ROOT / "person_v5_execution.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another person-v5 execution is active") from error
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield
