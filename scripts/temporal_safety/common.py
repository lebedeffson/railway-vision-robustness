from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "configs/temporal_safety_v1.yaml"
LOCK = PROJECT / "outputs/temporal_safety_v1/protocol/protocol_lock.json"
OUTPUT = PROJECT / "outputs/temporal_safety_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def assert_test_sealed() -> None:
    protocol = config()
    for key in ("marker", "legacy_marker"):
        marker = PROJECT / protocol["test_access"][key]
        if marker.exists():
            raise RuntimeError(f"Railway test is not sealed: {marker}")


def assert_locked() -> dict[str, Any]:
    assert_test_sealed()
    if not LOCK.is_file():
        raise RuntimeError("Temporal safety protocol is not locked")
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    if payload["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Temporal safety config changed after lock")
    for relative, expected in payload["implementation_sha256"].items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Temporal safety implementation changed: {relative}")
    if payload["test_status"] != "SEALED" or payload["test_access_count"] != 0:
        raise RuntimeError("Temporal safety test isolation contract failed")
    return payload


def fold_scenes(fold: int) -> set[str]:
    folds = json.loads(
        (PROJECT / config()["data"]["folds"]).read_text(encoding="utf-8")
    )["folds"]
    return set(map(str, folds[str(fold)]))


def source_frames(fold: int, role: str) -> pd.DataFrame:
    manifest = pd.read_csv(
        PROJECT / config()["data"]["manifest"], dtype={"frame_id": str}
    )
    development = manifest[manifest["split"].isin(["train", "val"])].copy()
    heldout = fold_scenes(fold)
    if role == "train":
        excluded = set().union(
            *[
                fold_scenes(int(excluded_fold))
                for excluded_fold in config()["selection"]["fit_excluded_folds"]
            ]
        )
        source = development[
            ~development["grouped_scene_id"].astype(str).isin(excluded)
        ].copy()
    elif role == "heldout":
        source = development[
            development["grouped_scene_id"].astype(str).isin(heldout)
        ].copy()
    else:
        raise ValueError(f"Unsupported role: {role}")
    test_scenes = set(
        manifest.loc[manifest["split"].eq("test"), "grouped_scene_id"].astype(str)
    )
    observed = set(source["grouped_scene_id"].astype(str))
    if observed & test_scenes:
        raise RuntimeError("Temporal source intersects sealed test")
    if role == "heldout" and observed != heldout:
        raise RuntimeError("Held-out scenes do not match the frozen fold")
    if role == "train":
        prohibited = set().union(
            *[
                fold_scenes(int(excluded_fold))
                for excluded_fold in config()["selection"]["fit_excluded_folds"]
            ]
        )
        if observed & prohibited:
            raise RuntimeError("Tracker fit role overlaps screening/confirmation scenes")
    source["image_path"] = source["output_image"].map(
        lambda value: str(Path(value).resolve())
    )
    source["frame_number"] = pd.to_numeric(source["frame_id"], errors="raise")
    index_path = OUTPUT / "data/frame_sequence_index.csv"
    if index_path.is_file():
        index = pd.read_csv(index_path)
        dimensions = index.set_index("image_path")[["width", "height", "timestamp"]]
        source = source.join(dimensions, on="image_path")
    else:
        source["width"] = 2464
        source["height"] = 1600
        source["timestamp"] = source["frame_number"] / float(
            config()["data"]["expected_nominal_fps"]
        )
    source = source.sort_values(
        ["grouped_scene_id", "subsequence_id", "frame_number", "image_path"]
    ).reset_index(drop=True)
    if source.duplicated(["subsequence_id", "frame_number"]).any():
        raise RuntimeError("Duplicate temporal frames")
    return source


def prediction_path(fold: int, role: str) -> Path:
    detector = config()["detector"][f"fold_{fold}"]
    key = "tuning_predictions" if role == "train" else "heldout_predictions"
    if key not in detector:
        raise RuntimeError(f"No frozen predictions for fold={fold}, role={role}")
    return PROJECT / detector[key]
