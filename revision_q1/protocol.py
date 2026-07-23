from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "config/revision_q1_protocol.yaml"
SELECTION_ACTIONS = {
    "fit_normalization",
    "select_normalization",
    "fit_scene_thresholds",
    "select_features",
    "select_models",
    "tune_thresholds",
}


def load_protocol(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid revision protocol: {path}")
    if payload.get("statistical_unit") != "sequence_id":
        raise RuntimeError("Revision protocol must use sequence_id")
    if int(payload.get("bootstrap_iterations", 0)) < 5000:
        raise RuntimeError("Revision protocol requires at least 5000 bootstraps")
    return payload


def assert_split_action(action: str, split: str) -> None:
    if split == "test" and action in SELECTION_ACTIONS:
        raise RuntimeError(f"Test leakage guard: {action} is forbidden on test")


def output_root(protocol: dict[str, Any] | None = None) -> Path:
    protocol = protocol or load_protocol()
    return PROJECT_DIR / str(protocol["output_root"])
