from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from canonical_m4_runtime import feature_sets, load_image, load_model
from extract_feature_consistency import FeatureHook
from person_v8b.common import (
    OUTPUT_ROOT,
    ROOT,
    assert_locked,
    assert_test_sealed,
    atomic_json,
    load_config,
    sha256_file,
)


def run_f0() -> dict[str, object]:
    lock = assert_locked()
    config = load_config()
    assert_test_sealed()
    manifest_path = ROOT / config["immutable_inputs"]["development_manifest"]["path"]
    manifest = pd.read_csv(manifest_path)
    required = {
        "output_image",
        "output_label",
        "grouped_scene_id",
        "person_v3_split",
    }
    missing_columns = sorted(required - set(manifest.columns))
    failures: list[str] = []
    if missing_columns:
        failures.append(f"MISSING_MANIFEST_COLUMNS:{missing_columns}")
    else:
        if len(manifest) != 1085:
            failures.append(f"FRAME_COUNT:{len(manifest)}")
        if manifest["grouped_scene_id"].astype(str).nunique() != 15:
            failures.append(
                f"SCENE_COUNT:{manifest['grouped_scene_id'].astype(str).nunique()}"
            )
        if set(manifest["person_v3_split"].astype(str)) != {"development"}:
            failures.append("NON_DEVELOPMENT_ROW")
        missing_images = sum(not Path(value).is_file() for value in manifest["output_image"])
        missing_labels = sum(not Path(value).is_file() for value in manifest["output_label"])
        if missing_images:
            failures.append(f"MISSING_IMAGES:{missing_images}")
        if missing_labels:
            failures.append(f"MISSING_LABELS:{missing_labels}")

    hook_shapes: dict[str, list[int]] = {}
    finite = False
    checkpoint = ROOT / config["immutable_inputs"]["detector_checkpoint"]["path"]
    if not failures:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_model(checkpoint, device)
        hook = FeatureHook(model)
        tiling = yaml.safe_load(
            (
                ROOT / config["immutable_inputs"]["tiling_protocol"]["path"]
            ).read_text(encoding="utf-8")
        )
        try:
            image = load_image(Path(manifest.iloc[0]["output_image"]), device)
            features = feature_sets(hook, model, image, tiling)
            hook_shapes = {
                layer: list(tensor.shape)
                for layer, tensor in zip(
                    config["feature_extraction"]["layers"], features, strict=True
                )
            }
            finite = all(bool(torch.isfinite(tensor).all()) for tensor in features)
            if not finite:
                failures.append("NONFINITE_FEATURE_HOOK")
            if any(shape[0] != 4 for shape in hook_shapes.values()):
                failures.append("TILE_BATCH_NOT_FOUR")
        finally:
            hook.close()

    payload: dict[str, object] = {
        "protocol_id": config["protocol_id"],
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "development_frames": len(manifest),
        "development_scenes": (
            manifest["grouped_scene_id"].astype(str).nunique()
            if "grouped_scene_id" in manifest
            else 0
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "feature_hook_shapes": hook_shapes,
        "features_finite": finite,
        "detector_training": "FORBIDDEN",
        "detector_development_overlap": "DISCLOSED_12_OF_15_SCENES",
        "risk_outer_validation": "LOSO_15_SCENES",
        "test_status": "SEALED",
        "test_access_count": 0,
        "attacks_status": "OUT_OF_SCOPE",
        "protocol_lock_implementation_commit": lock["implementation_commit"],
    }
    atomic_json(OUTPUT_ROOT / "audit/F0_AUDIT.json", payload)
    return payload


def main() -> int:
    payload = run_f0()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

