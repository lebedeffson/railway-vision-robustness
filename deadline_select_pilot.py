from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from audit_final_practice import label_path, load_manifest
from revision_q1.scene_difficulty import frame_geometry


PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST = PROJECT_DIR / "data/yolo_osdar23/manifest.csv"
DATA = PROJECT_DIR / "data/yolo_osdar23/data.yaml"
ROOT = PROJECT_DIR / "outputs/final_practice/deadline"


def main() -> None:
    config = ROOT / "config"
    config.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, object]] = []
    by_sequence: dict[str, list[dict[str, object]]] = {}
    for row in load_manifest(MANIFEST):
        if row["split"] != "val":
            continue
        geometry = frame_geometry(label_path(Path(row["image_path"])), 0.001)
        by_sequence.setdefault(row["sequence_id"], []).append({**row, **geometry})
    if len(by_sequence) != 3:
        raise RuntimeError(
            f"Frozen validation split has {len(by_sequence)} sequences, expected 3"
        )
    for sequence_id, rows in sorted(by_sequence.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                int(row["object_count"]), float(row["small_object_fraction"]),
                str(row["image_path"]),
            ),
        )
        positions = np.linspace(0, len(ordered) - 1, 4).round().astype(int)
        for rank, position in enumerate(positions):
            selected_rows.append({
                "sequence_id": sequence_id,
                "image_path": ordered[int(position)]["image_path"],
                "object_count": ordered[int(position)]["object_count"],
                "small_object_fraction": ordered[int(position)]["small_object_fraction"],
                "difficulty_rank_within_sequence": rank,
                "selection_source": "clean_validation_geometry_only",
            })
    pilot_manifest = config / "pilot_manifest.csv"
    with pilot_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    image_list = config / "pilot_images.txt"
    image_list.write_text(
        "\n".join(str(row["image_path"]) for row in selected_rows) + "\n",
        encoding="utf-8",
    )
    base = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    pilot_data = config / "pilot_data.yaml"
    base["val"] = str(image_list)
    base["test"] = str(image_list)
    pilot_data.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    digest = hashlib.sha256(pilot_manifest.read_bytes()).hexdigest()
    payload = {
        "status": "FROZEN_BEFORE_PILOT_RESULTS",
        "requested_independent_sequences": 12,
        "available_validation_sequences": len(by_sequence),
        "selected_frames": len(selected_rows),
        "frames_per_sequence": 4,
        "independence_limitation": (
            "12 independent validation sequences do not exist; the pilot uses 12 frames "
            "across all 3 validation sequences and is smoke-test evidence only"
        ),
        "manifest_sha256": digest,
    }
    (config / "pilot_selection.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
