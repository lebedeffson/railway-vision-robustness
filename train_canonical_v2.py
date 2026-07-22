from __future__ import annotations

import json
import hashlib
import csv
import os
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
PROTOCOL = PROJECT_DIR / "config/canonical_v2_protocol.yaml"
OUTPUT = PROJECT_DIR / "outputs/canonical_v2/training"
NAME = "yolo11m_canonical_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_selection(best: Path, config: dict) -> None:
    checkpoint_hash = sha256(best)
    marker = best.parent.parent / "TRAINING_COMPLETE"
    marker.write_text(
        f"best={best.resolve()}\nsha256={checkpoint_hash}\n",
        encoding="utf-8",
    )
    selection = {
        "selected_checkpoint": str(best.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "selection_split": "val_v2", "test_used": False,
        "configuration_A": config,
        "configuration_B": {
            "status": "rejected_before_training_by_resource_feasibility",
            "reason": config["rejected_configuration_B"],
        },
    }
    (best.parent.parent / "checkpoint_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "training_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    with (OUTPUT / "checkpoint_selection.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "selected_checkpoint", "checkpoint_sha256", "selection_split",
            "test_used", "selection_metric",
        ])
        writer.writeheader(); writer.writerow({
            "selected_checkpoint": str(best.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "selection_split": "val_v2", "test_used": False,
            "selection_metric": "validation_fitness_best.pt",
        })
    (OUTPUT / "checkpoint_provenance.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    aliases = OUTPUT / "weights"
    aliases.mkdir(parents=True, exist_ok=True)
    for source in (best, best.with_name("last.pt")):
        if not source.is_file():
            continue
        destination = aliases / source.name
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() == sha256(source):
                continue
            raise RuntimeError(f"Training alias already exists with another hash: {destination}")
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    run = best.parent.parent
    for source_name, destination_name in (
        ("results.csv", "results.csv"), ("results.png", "training_curves.png")
    ):
        source = run / source_name
        if source.is_file():
            shutil.copy2(source, OUTPUT / destination_name)


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    config = protocol["model"]
    weights = OUTPUT / NAME / "weights"
    best = weights / "best.pt"; last = weights / "last.pt"
    marker = OUTPUT / NAME / "TRAINING_COMPLETE"
    if marker.is_file() and best.is_file():
        record_selection(best, config)
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
    record_selection(best, config)
    print(best)


if __name__ == "__main__":
    main()
