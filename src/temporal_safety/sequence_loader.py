from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


TIMESTAMP = re.compile(r"_([0-9]{10}\.[0-9]+)\.[^.]+$")


def timestamp_from_path(path: str) -> float | None:
    match = TIMESTAMP.search(Path(path).name)
    return float(match.group(1)) if match else None


def validate_sequence_index(frame: pd.DataFrame) -> dict[str, int | float]:
    required = {
        "grouped_scene_id",
        "source_sequence_id",
        "frame_id",
        "frame_number",
        "timestamp",
        "redacted_image_id",
        "width",
        "height",
        "person_gt_count",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Sequence index misses columns: {sorted(missing)}")
    duplicates = int(
        frame.duplicated(["source_sequence_id", "frame_number"]).sum()
    )
    non_monotonic = 0
    for _, sequence in frame.groupby("source_sequence_id", sort=False):
        numbers = sequence["frame_number"].to_numpy(dtype=float)
        times = sequence["timestamp"].dropna().to_numpy(dtype=float)
        non_monotonic += int((numbers[1:] <= numbers[:-1]).any())
        if len(times) > 1:
            non_monotonic += int((times[1:] <= times[:-1]).any())
    if duplicates or non_monotonic:
        raise ValueError("Frame order audit failed")
    return {
        "frames": int(len(frame)),
        "scenes": int(frame["grouped_scene_id"].nunique()),
        "subsequences": int(frame["source_sequence_id"].nunique()),
        "duplicate_frames": duplicates,
        "non_monotonic_sequences": non_monotonic,
    }
