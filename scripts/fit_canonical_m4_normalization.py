from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch

from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    assert_role_allowed,
    atomic_json,
    load_protocol,
    sha256,
)
from canonical_m4_runtime import (
    FrameInput,
    feature_sets,
    load_image,
    load_model,
)
from extract_feature_consistency import FeatureHook
from revision_q1.normalization import (
    LAYERS,
    distribution_diagnostics,
    fit_channel_statistics,
    membership,
    saturation_warning,
)


SOURCE_MANIFEST = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"
INHERITED_PROTOCOL = PROJECT_DIR / "config/revision_q1_protocol.yaml"
DESTINATION = OUTPUT_ROOT / "normalization"


def path_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def fit(checkpoint: Path) -> dict:
    assert_role_allowed("normalization")
    protocol = load_protocol()
    gate = json.loads(
        (OUTPUT_ROOT / "validation/quality_gate.json").read_text(encoding="utf-8")
    )
    if gate["checkpoint_sha256"] != sha256(checkpoint):
        raise RuntimeError("Normalization checkpoint differs from frozen gate")
    inherited = __import__("yaml").safe_load(
        INHERITED_PROTOCOL.read_text(encoding="utf-8")
    )
    positions = int(inherited["normalization"]["sample_positions_per_image"])
    seed = int(inherited["random_seed"])
    source = pd.read_csv(SOURCE_MANIFEST)
    source = source[source["split"].eq("val")].sort_values(
        ["grouped_scene_id", "subsequence_id", "frame_id"]
    )
    if len(source) != int(protocol["dataset"]["split_frame_counts"]["val"]):
        raise RuntimeError("Normalization validation frame count mismatch")
    device = torch.device("cuda:0")
    model = load_model(checkpoint, device)
    hook = FeatureHook(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    samples: dict[str, list[torch.Tensor]] = {name: [] for name in LAYERS}
    image_paths = []
    try:
        for row in source.itertuples(index=False):
            frame = FrameInput(
                Path(row.output_image),
                Path(row.output_label),
                str(row.grouped_scene_id),
                str(row.subsequence_id),
            )
            image_paths.append(str(frame.image_path.resolve()))
            image = load_image(frame.image_path, device)
            features = feature_sets(hook, model, image, protocol)
            for layer, feature in zip(LAYERS, features, strict=True):
                cpu = feature.detach().float().cpu()
                batch, channels, height, width = cpu.shape
                flattened = cpu.view(batch, channels, height * width)
                count = min(positions, height * width)
                for tile_index in range(batch):
                    indices = torch.randperm(
                        height * width, generator=generator
                    )[:count]
                    samples[layer].append(flattened[tile_index, :, indices])
            del image, features
    finally:
        hook.close()
    combined = {
        layer: torch.cat(values, dim=1)
        for layer, values in samples.items()
    }
    statistics = fit_channel_statistics(combined)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    statistics_path = DESTINATION / "layer_channel_statistics.pt"
    torch.save({
        "statistics": statistics,
        "fit_split": "val",
        "fit_inputs": "clean_only",
        "per_layer": True,
        "per_channel": True,
        "image_count": len(image_paths),
        "tile_count": len(image_paths) * int(protocol["tiling"]["expected_tiles_per_frame"]),
        "image_path_hash": path_hash(image_paths),
        "protocol_id": protocol["protocol_id"],
        "checkpoint_sha256": sha256(checkpoint),
        "sample_positions_per_tile": positions,
        "sampling_seed": seed,
        "inherited_protocol": str(INHERITED_PROTOCOL.resolve()),
        "inherited_protocol_sha256": sha256(INHERITED_PROTOCOL),
    }, statistics_path)
    rows = []
    for layer in LAYERS:
        feature = combined[layer].unsqueeze(0).unsqueeze(2)
        for mode in ("N1_quantile", "N2_robust_sigmoid", "N3_zscore_sigmoid"):
            diagnostics = distribution_diagnostics(
                membership(feature, statistics[layer], mode)
            )
            rows.append({
                "split": "val",
                "input_state": "clean",
                "layer": layer,
                "normalization": mode,
                **diagnostics,
                "saturation_warning": saturation_warning(
                    diagnostics, float(protocol["normalization"]["saturation_max"])
                ),
            })
    audit = pd.DataFrame(rows)
    audit_path = DESTINATION / "normalization_audit.csv"
    temporary = audit_path.with_suffix(".csv.tmp")
    audit.to_csv(temporary, index=False)
    os.replace(temporary, audit_path)
    selected = None
    candidates = [
        str(protocol["normalization"]["primary"]),
        str(protocol["normalization"]["fallback"]),
    ]
    for candidate in candidates:
        subset = audit[audit["normalization"].eq(candidate)]
        if len(subset) == 3 and not bool(subset["saturation_warning"].any()):
            selected = candidate
            break
    if selected is None:
        raise RuntimeError("Canonical M4 N1 and frozen N2 normalization both saturate")
    payload = {
        "status": "PASS",
        "selected_normalization": selected,
        "selection_rule": "N1_if_all_layers_below_20_percent_else_frozen_N2",
        "fit_split": "val",
        "fit_inputs": "clean_only",
        "test_used": False,
        "checkpoint_sha256": sha256(checkpoint),
        "statistics_sha256": sha256(statistics_path),
        "audit_sha256": sha256(audit_path),
        "image_count": len(image_paths),
        "image_path_hash": path_hash(image_paths),
        "sample_positions_per_tile": positions,
        "sampling_seed": seed,
    }
    atomic_json(DESTINATION / "normalization_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
