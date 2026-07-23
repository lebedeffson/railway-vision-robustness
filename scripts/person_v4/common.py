from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_DIR / "configs/canonical_v4_person_dg_nwd.yaml"
OUTPUT_ROOT = PROJECT_DIR / "outputs/person_v4"
LOCK_PATH = OUTPUT_ROOT / "protocol/protocol_lock.json"
EXPEDITED_PROTOCOL_PATH = (
    PROJECT_DIR
    / "configs/canonical_v4_person_dg_nwd_expedited.yaml"
)
EXPEDITED_OUTPUT_ROOT = PROJECT_DIR / "outputs/person_v4_expedited"
EXPEDITED_LOCK_PATH = (
    EXPEDITED_OUTPUT_ROOT / "protocol/protocol_lock.json"
)
TEST_MARKER = PROJECT_DIR / "outputs/person_v3/test/TEST_OPENED.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT_DIR, text=True
    ).strip()


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def assert_test_sealed() -> None:
    if TEST_MARKER.exists():
        raise RuntimeError(f"Canonical v4 test is not sealed: {TEST_MARKER}")


def assert_locked() -> dict[str, Any]:
    assert_test_sealed()
    if EXPEDITED_LOCK_PATH.is_file():
        lock = json.loads(
            EXPEDITED_LOCK_PATH.read_text(encoding="utf-8")
        )
        if lock["protocol_sha256"] != sha256(EXPEDITED_PROTOCOL_PATH):
            raise RuntimeError(
                "Canonical v4 expedited protocol changed after lock"
            )
        if lock["parent_protocol_sha256"] != sha256(PROTOCOL_PATH):
            raise RuntimeError(
                "Canonical v4 parent protocol changed after amendment"
            )
        if lock["git_commit"] != git("rev-parse", "HEAD"):
            raise RuntimeError(
                "Canonical v4 expedited code differs from its lock"
            )
        return lock
    if not LOCK_PATH.is_file():
        raise RuntimeError("Canonical v4 protocol lock is missing")
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("Canonical v4 protocol changed after lock")
    if lock["git_commit"] != git("rev-parse", "HEAD"):
        raise RuntimeError(
            "Canonical v4 code commit differs from the frozen protocol lock"
        )
    return lock
