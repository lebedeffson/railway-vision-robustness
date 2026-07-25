from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.temporal_safety.common import OUTPUT, PROJECT, atomic_csv, atomic_json, config
from src.temporal_safety.sequence_loader import timestamp_from_path, validate_sequence_index


def person_count(label_path: str) -> int:
    path = Path(label_path)
    if not path.is_file():
        raise RuntimeError(f"Missing label: {path}")
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and int(line.split()[0]) == 0
    )


def main() -> None:
    protocol = config()
    manifest = pd.read_csv(
        PROJECT / protocol["data"]["manifest"], dtype={"frame_id": str}
    )
    development = manifest[manifest["split"].isin(["train", "val"])].copy()
    rows = []
    gaps = []
    for record in development.itertuples(index=False):
        image_path = Path(record.output_image).resolve()
        with Image.open(image_path) as image:
            width, height = image.size
        timestamp = timestamp_from_path(str(record.source_image))
        relative_id = hashlib.sha256(
            f"{record.subsequence_id}/{record.frame_id}".encode("utf-8")
        ).hexdigest()[:20]
        rows.append(
            {
                "grouped_scene_id": str(record.grouped_scene_id),
                "source_sequence_id": str(record.subsequence_id),
                "subsequence_id": str(record.subsequence_id),
                "frame_id": str(record.frame_id),
                "frame_number": int(record.frame_id),
                "timestamp": timestamp,
                "redacted_image_id": relative_id,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "person_gt_count": person_count(str(record.output_label)),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["grouped_scene_id", "source_sequence_id", "frame_number"]
    ).reset_index(drop=True)
    audit = validate_sequence_index(frame)
    for subsequence, sequence in frame.groupby("source_sequence_id", sort=False):
        differences = sequence["frame_number"].diff().dropna()
        for index in differences[differences.gt(1)].index:
            gaps.append(
                {
                    "source_sequence_id": str(subsequence),
                    "previous_frame_number": int(
                        sequence.iloc[sequence.index.get_loc(index) - 1]["frame_number"]
                    )
                    if sequence.index.get_loc(index) > 0
                    else None,
                    "current_frame_number": int(sequence.loc[index, "frame_number"]),
                    "gap_frames": int(differences.loc[index] - 1),
                }
            )
    timestamp_diffs = []
    for _, sequence in frame.groupby("source_sequence_id", sort=False):
        timestamp_diffs.extend(
            sequence["timestamp"].diff().dropna().loc[lambda values: values > 0].tolist()
        )
    observed_fps = (
        1.0 / float(pd.Series(timestamp_diffs).median()) if timestamp_diffs else None
    )
    audit.update(
        {
            "test_rows_read": 0,
            "test_labels_read": False,
            "test_images_read": False,
            "observed_median_fps": observed_fps,
            "fps_source": protocol["data"]["timestamp_source"],
            "documented_gaps": len(gaps),
        }
    )
    destination = OUTPUT / "data"
    atomic_csv(destination / "frame_sequence_index.csv", frame)
    atomic_csv(destination / "sequence_gaps.csv", pd.DataFrame(gaps))
    atomic_json(destination / "SEQUENCE_INDEX_AUDIT.json", audit)


if __name__ == "__main__":
    main()
