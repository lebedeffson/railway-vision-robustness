from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml

from audit_final_practice import load_manifest


PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST = PROJECT_DIR / "data/yolo_osdar23_v2/manifest.csv"
DATA = PROJECT_DIR / "data/yolo_osdar23_v2/data.yaml"
OUTPUT = PROJECT_DIR / "outputs/canonical_v2/pilot"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in load_manifest(MANIFEST):
        if row["split"] != "val":
            continue
        grouped.setdefault(row["sequence_id"], []).append(row)
    if len(grouped) != 5:
        raise RuntimeError(f"Canonical v2 pilot requires five validation scenes, got {len(grouped)}")
    selected: list[dict[str, object]] = []
    for grouped_scene_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: str(row["image_path"]))
        positions = (0, len(ordered) // 2, len(ordered) - 1)
        for rank, position in enumerate(positions):
            row = ordered[int(position)]
            selected.append({
                "grouped_scene_id": grouped_scene_id,
                "sequence_id": grouped_scene_id,
                "subsequence_id": row.get("source_sequence", grouped_scene_id),
                "image_path": row["image_path"],
                "frame_order_rank": rank,
                "selection_source": "deterministic_even_frame_order_no_model_or_difficulty",
            })
    manifest = OUTPUT / "pilot_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)
    (OUTPUT / "pilot_frames.csv").write_bytes(manifest.read_bytes())
    images = OUTPUT / "pilot_images.txt"
    images.write_text(
        "\n".join(str(row["image_path"]) for row in selected) + "\n",
        encoding="utf-8",
    )
    data = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    data["val"] = str(images.resolve())
    data["test"] = str(images.resolve())
    (OUTPUT / "pilot_data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    payload = {
        "status": "FROZEN_BEFORE_CANONICAL_ATTACK_RESULTS",
        "selection_split": "val_v2",
        "test_used": False,
        "grouped_scenes": 5,
        "frames": len(selected),
        "frames_per_scene": 3,
        "selection_method": "deterministic_even_frame_order_no_model_or_difficulty",
        "pilot_manifest_sha256": digest,
    }
    (OUTPUT / "pilot_selection.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
