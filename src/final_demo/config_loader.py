from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT / "configs/final_demo.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol_id"] != "railway-vision-final-closure-v1":
        raise RuntimeError("Final demo requires the frozen closure protocol")
    if config["test_access"]["status"] != "SEALED":
        raise RuntimeError("Railway test must remain sealed")
    return config


def resolve_local_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def assert_not_test_input(path: Path, config: dict[str, Any]) -> None:
    normalized = path.resolve().as_posix().lower()
    for token in config["test_access"]["prohibit_paths_containing"]:
        if str(token).lower() in normalized:
            raise RuntimeError(f"Sealed-test-like input is prohibited: {token}")
