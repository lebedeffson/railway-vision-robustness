from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL = PROJECT_DIR / "config/canonical_v2_protocol.yaml"
OUTPUT = PROJECT_DIR / "outputs/canonical_v2/training"
NAME = "yolo11m_canonical_v2"


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    config = protocol["model"]
    weights = OUTPUT / NAME / "weights"
    best = weights / "best.pt"; last = weights / "last.pt"
    marker = OUTPUT / NAME / "TRAINING_COMPLETE"
    if marker.is_file() and best.is_file():
        print(best); return
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical v2 training requires CUDA")
    model = YOLO(str(last) if last.is_file() else config["initialization"])
    if last.is_file():
        model.train(resume=True)
    else:
        model.train(
            data=str(PROJECT_DIR / protocol["dataset"]), task="detect",
            epochs=int(config["epochs"]), patience=int(config["patience"]),
            imgsz=int(config["imgsz"]), batch=int(config["batch"]), device=0, workers=2,
            optimizer="AdamW", lr0=0.0003, lrf=0.1, weight_decay=0.0005,
            pretrained=True, amp=True, cos_lr=True, freeze=0,
            hsv_h=.01, hsv_s=.35, hsv_v=.25, translate=.05, scale=.20,
            fliplr=.5, mosaic=.30, close_mosaic=5, cache=False,
            seed=int(config["seed"]), deterministic=True, val=True, plots=True,
            save=True, save_period=5, project=str(OUTPUT), name=NAME,
            exist_ok=False, verbose=True,
        )
    if not best.is_file():
        raise FileNotFoundError(best)
    marker.write_text(f"best={best}\n", encoding="utf-8")
    selection = {
        "selected_checkpoint": str(best.resolve()),
        "selection_split": "val_v2", "test_used": False,
        "configuration_A": config,
        "configuration_B": {
            "status": "rejected_before_training_by_resource_feasibility",
            "reason": config["rejected_configuration_B"],
        },
    }
    (OUTPUT / NAME / "checkpoint_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(best)


if __name__ == "__main__":
    main()
