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


CONFIG = PROJECT / "configs/person_v5/expedited_screening.yaml"
LOCK = PROJECT / "protocol/v5/V5_EXPEDITED_SCREENING_LOCK.json"
ROOT = PROJECT / "outputs/person_canonical_v5/screening"
STATUS = ROOT / "screening_status.json"
TRACE = ROOT / "decision_trace.json"
LEADERBOARD = ROOT / "candidate_leaderboard.csv"


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def assert_locked() -> dict[str, Any]:
    assert_test_sealed()
    if not LOCK.is_file():
        raise RuntimeError("Expedited screening protocol is not locked")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["protocol_id"] != "canonical-v5-expedited-screening-v1":
        raise RuntimeError("Unexpected expedited screening protocol")
    if lock["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Expedited screening config changed after lock")
    for relative, expected in lock["implementation_sha256"].items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Screening implementation changed: {relative}")
    if lock["article_evidence"] or not lock["candidate_screening_only"]:
        raise RuntimeError("Screening claim boundary is invalid")
    if lock["test_status"] != "SEALED":
        raise RuntimeError("Screening lock does not seal test")
    return lock


def write_status(**values: Any) -> None:
    previous = (
        json.loads(STATUS.read_text(encoding="utf-8"))
        if STATUS.is_file()
        else {}
    )
    atomic_json(
        STATUS,
        {
            **previous,
            "protocol_id": config()["protocol_id"],
            "article_evidence": False,
            "candidate_screening_only": True,
            "test_status": "SEALED",
            "attacks_status": "BLOCKED",
            "test_access_count": 0,
            "updated_at": now(),
            **values,
        },
    )


def append_decision(
    candidate: str,
    fidelity: int,
    decision: str,
    reason: list[str],
    metrics: dict[str, Any] | None = None,
) -> None:
    payload = (
        json.loads(TRACE.read_text(encoding="utf-8"))
        if TRACE.is_file()
        else {
            "protocol_id": config()["protocol_id"],
            "immutable_append_only": True,
            "article_evidence": False,
            "decisions": [],
        }
    )
    key = (candidate, fidelity)
    existing = {
        (row["candidate"], int(row["fidelity"]))
        for row in payload["decisions"]
    }
    if key in existing:
        return
    payload["decisions"].append(
        {
            "candidate": candidate,
            "fidelity": fidelity,
            "decision": decision,
            "reason": reason,
            "metrics": metrics or {},
            "recorded_at": now(),
            "test_used": False,
        }
    )
    atomic_json(TRACE, payload)


@contextmanager
def exclusive() -> Iterator[None]:
    ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = ROOT / "screening.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another expedited screening process is active") from error
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        yield

