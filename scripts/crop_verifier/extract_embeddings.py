from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.crop_verifier.common import (
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_csv,
    atomic_json,
    config,
    sha256,
)
from src.crop_verifier_v1 import FrozenYoloTrackEncoder


ROLES = ("support", "screening", "confirmation")


def observations_path(role: str) -> Path:
    return (
        PROJECT
        / config()["frozen_inputs"]["tracks_root"]
        / role
        / "track_observations.csv"
    )


def encode_role(
    role: str, encoder: FrozenYoloTrackEncoder
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = OUTPUT / "embeddings" / role
    metadata_path = root / "track_metadata.csv"
    crop_path = root / "crop_metadata.csv"
    features_path = root / "visual_embeddings.npz"
    if metadata_path.is_file() and crop_path.is_file() and features_path.is_file():
        return pd.read_csv(metadata_path), pd.read_csv(crop_path)
    observations = pd.read_csv(observations_path(role))
    metadata, crops, embeddings = encoder.encode(observations)
    atomic_csv(metadata_path, metadata)
    atomic_csv(crop_path, crops)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(features_path, embeddings=embeddings)
    audit = {
        "role": role,
        "tracks": len(metadata),
        "scenes_with_tracks": int(metadata["grouped_scene_id"].nunique()),
        "embedding_dimension": int(embeddings.shape[1]),
        "crop_slots_per_track": 3,
        "unique_crop_frames": {
            str(key): int(value)
            for key, value in metadata["unique_crop_frames"]
            .value_counts()
            .sort_index()
            .items()
        },
        "interpolated_crops": int(crops["interpolated"].astype(bool).sum()),
        "non_detector_crops": int((~crops["is_detector_hit"].astype(bool)).sum()),
        "image_paths_public": False,
        "test_used": False,
        "checkpoint_sha256": config()["encoder"]["checkpoint_sha256"],
        "frozen_backbone": True,
    }
    atomic_json(root / "EMBEDDING_AUDIT.json", audit)
    return metadata, crops


def main() -> None:
    assert_locked()
    settings = config()["encoder"]
    checkpoint = PROJECT / settings["checkpoint"]
    if sha256(checkpoint) != settings["checkpoint_sha256"]:
        raise RuntimeError("Frozen visual checkpoint hash changed")
    encoder = FrozenYoloTrackEncoder(
        checkpoint=checkpoint,
        last_layer_index=int(settings["backbone_last_layer_index"]),
        input_size=int(settings["input_size"]),
        context_multiplier=float(settings["crop_context_multiplier"]),
        padding_value=int(settings["padding_value"]),
        batch_size=int(settings["batch_size"]),
    )
    metadata: dict[str, pd.DataFrame] = {}
    crops: dict[str, pd.DataFrame] = {}
    for role in ROLES:
        metadata[role], crops[role] = encode_role(role, encoder)
    leakage: dict[str, object] = {}
    for left_index, left in enumerate(ROLES):
        for right in ROLES[left_index + 1 :]:
            left_scenes = set(metadata[left]["grouped_scene_id"].astype(str))
            right_scenes = set(metadata[right]["grouped_scene_id"].astype(str))
            left_images = set(crops[left]["image_path"].astype(str))
            right_images = set(crops[right]["image_path"].astype(str))
            leakage[f"{left}_vs_{right}"] = {
                "scene_overlap": sorted(left_scenes & right_scenes),
                "crop_source_overlap_count": len(left_images & right_images),
            }
    passed = all(
        not row["scene_overlap"] and row["crop_source_overlap_count"] == 0
        for row in leakage.values()
    )
    payload = {
        "status": "PASS" if passed else "FAIL",
        "roles": {
            role: {
                "tracks": len(metadata[role]),
                "scenes_with_tracks": int(
                    metadata[role]["grouped_scene_id"].nunique()
                ),
                "crop_sources": int(crops[role]["image_path"].nunique()),
            }
            for role in ROLES
        },
        "cross_role_checks": leakage,
        "interpolated_crops": int(
            sum(frame["interpolated"].astype(bool).sum() for frame in crops.values())
        ),
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    atomic_json(OUTPUT / "audit/CROP_LEAKAGE_AUDIT.json", payload)
    if not passed or payload["interpolated_crops"] != 0:
        raise RuntimeError("Crop leakage audit failed")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
