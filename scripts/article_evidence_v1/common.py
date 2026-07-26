from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT / "configs/article_evidence_v1.yaml"
OUTPUT = PROJECT / "outputs/article_evidence_v1"


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def ensure_development_path(path: Path) -> None:
    resolved = "/" + path.resolve().as_posix().lower().strip("/") + "/"
    for token in config()["test"]["forbidden_path_tokens"]:
        if token.lower() in resolved:
            raise RuntimeError(f"Sealed-test path prohibited: {path}")


def check_no_test_markers() -> None:
    markers = [
        PROJECT / "outputs/temporal_safety_v1/test/TEST_OPENED.json",
        PROJECT / "outputs/temporal_verifier_v1/test/TEST_OPENED.json",
        PROJECT / "outputs/crop_verifier_v1/test/TEST_OPENED.json",
        PROJECT / "outputs/person_v3/test/TEST_OPENED.json",
    ]
    present = [str(path.relative_to(PROJECT)) for path in markers if path.exists()]
    if present:
        raise RuntimeError(f"Test-open markers exist: {present}")
