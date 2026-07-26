from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/review_assistant_v1.yaml"
PRODUCT_ID = "railway-person-review-assistant-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(
    path: str | Path = DEFAULT_CONFIG,
    *,
    overrides: dict[str, str | Path | None] | None = None,
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("product_id") != PRODUCT_ID:
        raise RuntimeError(f"Expected product_id={PRODUCT_ID}")
    review = config["review"]
    if (
        review.get("human_confirmation_required") is not True
        or review.get("autonomous_alarm") is not False
        or review.get("safety_actuation") is not False
    ):
        raise RuntimeError("Operator review safety contract is invalid")
    if config["test_access"] != {
        "status": "SEALED",
        "access_count": 0,
        "prohibit_paths_containing": ["sealed_test", "/test/"],
    }:
        raise RuntimeError("Railway test seal is not intact")
    result = copy.deepcopy(config)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        section, field = key.split(".", 1)
        result[section][field] = str(value)
    result["_config_path"] = str(config_path)
    return result


def assert_safe_input(path: Path, config: dict[str, Any]) -> None:
    normalized = path.resolve().as_posix().lower()
    for token in config["test_access"]["prohibit_paths_containing"]:
        if token.lower() in normalized:
            raise RuntimeError(f"Sealed-test-like input is prohibited: {token}")


def verify_model_file(path_value: str | Path, expected_sha256: str | None) -> Path:
    path = resolve_path(path_value)
    if not path.is_file():
        suffix = f" Expected SHA-256: {expected_sha256}." if expected_sha256 else ""
        raise FileNotFoundError(f"Required local model is missing: {path}.{suffix}")
    actual = sha256_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise RuntimeError(
            f"Model SHA-256 mismatch for {path.name}: expected "
            f"{expected_sha256}, got {actual}"
        )
    return path
