from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_final_practice import label_path, load_manifest
from revision_q1.protocol import assert_split_action, load_protocol, output_root


PROJECT_DIR = Path(__file__).resolve().parents[1]


def frame_geometry(label: Path, small_threshold: float) -> dict[str, float | int]:
    areas: list[float] = []
    if label.is_file():
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            areas.append(float(fields[3]) * float(fields[4]))
    return {
        "object_count": len(areas),
        "median_bbox_area_ratio": float(np.median(areas)) if areas else 0.0,
        "small_object_fraction": (
            float(np.mean(np.asarray(areas) < small_threshold)) if areas else 0.0
        ),
    }


def tertiles(values: pd.Series) -> tuple[float, float]:
    return float(values.quantile(1 / 3)), float(values.quantile(2 / 3))


def labels(values: pd.Series, thresholds: tuple[float, float], prefix: str) -> pd.Series:
    low, high = thresholds
    return pd.Series(
        np.where(
            values <= low,
            f"low_{prefix}",
            np.where(values <= high, f"medium_{prefix}", f"high_{prefix}"),
        ),
        index=values.index,
    )


def main() -> None:
    protocol = load_protocol()
    parser = argparse.ArgumentParser(description="Freeze scene difficulty on validation tertiles")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data/yolo_osdar23/manifest.csv")
    parser.add_argument("--output", type=Path, default=output_root(protocol))
    args = parser.parse_args()
    assert_split_action("fit_scene_thresholds", "val")
    threshold = float(protocol["scene_difficulty"]["small_object_area_ratio"])
    rows: list[dict[str, object]] = []
    for row in load_manifest(args.manifest):
        image = Path(row["image_path"])
        rows.append({
            "sequence_id": row["sequence_id"],
            "image_path": row["image_path"],
            "split": row["split"],
            **frame_geometry(label_path(image), threshold),
        })
    frame = pd.DataFrame(rows)
    validation = frame[frame["split"] == "val"]
    object_thresholds = tertiles(validation["object_count"])
    small_thresholds = tertiles(validation["small_object_fraction"])
    frame["object_count_stratum"] = labels(
        frame["object_count"], object_thresholds, "object_count"
    )
    frame["small_object_stratum"] = labels(
        frame["small_object_fraction"], small_thresholds, "small_object_fraction"
    )
    raw = args.output / "raw"
    config = args.output / "config"
    tables = args.output / "tables"
    for directory in (raw, config, tables):
        directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(raw / "scene_difficulty.csv", index=False)
    payload = {
        "fit_split": "val",
        "object_count_tertiles": list(object_thresholds),
        "small_object_fraction_tertiles": list(small_thresholds),
        "small_object_area_ratio": threshold,
        "validation_frames": len(validation),
        "validation_sequences": int(validation["sequence_id"].nunique()),
    }
    (config / "scene_difficulty_thresholds.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    summary = frame.groupby(
        ["split", "object_count_stratum", "small_object_stratum"], as_index=False
    ).agg(frames=("image_path", "count"), sequences=("sequence_id", "nunique"))
    summary["status"] = np.where(
        summary["sequences"] < int(protocol["scene_difficulty"]["exploratory_min_sequences"]),
        "exploratory", "confirmatory",
    )
    summary.to_csv(tables / "09_scene_difficulty_analysis.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
