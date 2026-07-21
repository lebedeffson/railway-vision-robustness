from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

from audit_final_practice import load_manifest
from revision_q1.protocol import load_protocol, output_root


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    protocol = load_protocol()
    root = output_root(protocol)
    parser = argparse.ArgumentParser(description="Stage 1/2 clean metrics by test sequence")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data/yolo_osdar23/manifest.csv")
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data/yolo_osdar23/data.yaml")
    parser.add_argument("--output", type=Path, default=root / "raw/checkpoint_sequence_metrics.csv")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()
    if args.output.is_file():
        print(f"Checkpoint sequence metrics already complete: {args.output}")
        return
    base = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    names = base["names"]
    sequences: dict[str, list[str]] = {}
    for row in load_manifest(args.manifest):
        if row["split"] == "test":
            sequences.setdefault(row["sequence_id"], []).append(row["image_path"])
    config_dir = root / "config/checkpoint_sequence_data"
    config_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "stage2_best": PROJECT_DIR / protocol["primary_checkpoint"],
        "stage1_best": PROJECT_DIR / protocol["sensitivity_checkpoint"],
    }
    rows: list[dict[str, object]] = []
    for checkpoint_name, checkpoint in checkpoints.items():
        model = YOLO(str(checkpoint))
        for sequence_id, images in sorted(sequences.items()):
            image_list = config_dir / f"{sequence_id}.txt"
            image_list.write_text("\n".join(images) + "\n", encoding="utf-8")
            data_config = config_dir / f"{sequence_id}.yaml"
            data_config.write_text(yaml.safe_dump({
                "path": str(PROJECT_DIR),
                "train": str(image_list),
                "val": str(image_list),
                "test": str(image_list),
                "names": names,
            }, sort_keys=False), encoding="utf-8")
            result = model.val(
                data=str(data_config), split="test", imgsz=args.imgsz,
                batch=1, workers=0, device=args.device, plots=False, verbose=False,
                project=str(root / "logs/checkpoint_sequence_eval"),
                name=f"{checkpoint_name}_{sequence_id}", exist_ok=True,
            )
            rows.append({
                "checkpoint": checkpoint_name,
                "sequence_id": sequence_id,
                "frames": len(images),
                "map50": float(result.box.map50),
                "map50_95": float(result.box.map),
                "precision": float(result.box.mp),
                "recall": float(result.box.mr),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "sequences": len(sequences)}, indent=2))


if __name__ == "__main__":
    main()
