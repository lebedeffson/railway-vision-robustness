from __future__ import annotations

import pandas as pd

from scripts.temporal_safety.common import (
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_json,
    config,
    prediction_path,
    sha256,
    source_frames,
)


ROLES = [(0, "train"), (0, "heldout"), (1, "heldout")]


def main() -> None:
    assert_locked()
    protocol = config()
    rows = []
    sources = {}
    for fold, role in ROLES:
        source = source_frames(fold, role)
        path = prediction_path(fold, role)
        table = pd.read_csv(path)
        allowed = set(source["image_path"])
        if role == "train":
            table = table[table["image_path"].astype(str).isin(allowed)].copy()
        if not set(table["image_path"].astype(str)) <= allowed:
            raise RuntimeError("Frozen prediction source leaks outside its role")
        predictions = table[table["kind"].eq("prediction")]
        if not predictions.empty and float(predictions["confidence"].min()) < float(
            protocol["detector"]["candidate_floor"]
        ) - 1e-12:
            raise RuntimeError("Prediction below frozen candidate floor")
        table.insert(0, "prediction_role", role)
        table.insert(0, "detector_fold", fold)
        rows.append(table)
        sources[f"fold_{fold}_{role}"] = {
            "path": str(path.relative_to(PROJECT)),
            "sha256": sha256(path),
            "frames": len(source),
            "scenes": sorted(set(source["grouped_scene_id"].astype(str))),
        }
    combined = pd.concat(rows, ignore_index=True)
    destination = OUTPUT / "baseline/raw_predictions.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.tmp")
    combined.to_parquet(temporary, index=False)
    temporary.replace(destination)
    atomic_json(
        OUTPUT / "baseline/RAW_PREDICTIONS_AUDIT.json",
        {
            "protocol_id": protocol["protocol_id"],
            "candidate_floor": protocol["detector"]["candidate_floor"],
            "sources": sources,
            "parquet_sha256": sha256(destination),
            "same_predictions_for_all_trackers": True,
            "test_used": False,
        },
    )


if __name__ == "__main__":
    main()
