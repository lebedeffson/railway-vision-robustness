from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

from scripts.canonical_m4_trainer import DifferentialLRDetectionTrainer
from scripts.person_v3.common import (
    DATASET_ROOT,
    FOLDS_PATH,
    OUTPUT_ROOT,
    PROJECT_DIR,
    assert_locked,
    atomic_json,
    atomic_text,
    load_protocol,
    now,
    sha256,
)
from scripts.run_micro_overfit import GradientLogger


INITIALIZATION = PROJECT_DIR / "yolo11m.pt"


def training_root(fold: int) -> Path:
    return OUTPUT_ROOT / f"scene_cv/fold_{fold}/seed_20260723"


def environment() -> dict[str, Any]:
    return {
        "created_at": now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True
        ).strip(),
    }


def train_stage(
    *,
    initialization: Path,
    data: Path,
    root: Path,
    stage_name: str,
    stage: dict[str, Any],
    protocol: dict[str, Any],
) -> Path:
    run = root / stage_name
    best = run / "weights/best.pt"
    last = run / "weights/last.pt"
    marker = run / "TRAINING_COMPLETE.json"
    if marker.is_file() and best.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload["best_sha256"] != sha256(best):
            raise RuntimeError(f"Person v3 completed checkpoint changed: {best}")
        return best
    DifferentialLRDetectionTrainer.backbone_lr = float(stage["backbone_lr"])
    DifferentialLRDetectionTrainer.head_lr = float(stage["head_lr"])
    logger = GradientLogger(root / f"{stage_name}_gradient_metrics.csv")
    model = YOLO(str(last if last.is_file() else initialization))
    model.add_callback("on_train_start", logger.train_start)
    model.add_callback("on_train_epoch_start", logger.epoch_start)
    model.add_callback("on_train_batch_end", logger.before_zero_grad)
    model.add_callback("on_train_epoch_end", logger.epoch_end)
    if last.is_file():
        model.train(resume=True, trainer=DifferentialLRDetectionTrainer)
    else:
        augmentations = dict(protocol["pipeline"]["augmentations"])
        model.train(
            trainer=DifferentialLRDetectionTrainer,
            data=str(data.resolve()),
            task="detect",
            imgsz=int(protocol["pipeline"]["input_size"]),
            epochs=int(stage["epochs"]),
            patience=int(stage["epochs"]),
            batch=int(protocol["pipeline"]["batch"]),
            device=0,
            workers=int(protocol["pipeline"]["workers"]),
            optimizer="AdamW",
            lr0=float(stage["head_lr"]),
            lrf=0.10,
            weight_decay=0.0005,
            pretrained=True,
            amp=bool(protocol["pipeline"]["AMP"]),
            cos_lr=True,
            freeze=int(stage["freeze_layers"]),
            seed=int(protocol["pipeline"]["seed"]),
            deterministic=True,
            cache=False,
            val=True,
            plots=True,
            save=True,
            save_period=5,
            close_mosaic=0,
            project=str(root.resolve()),
            name=stage_name,
            exist_ok=False,
            verbose=True,
            **augmentations,
        )
    if not best.is_file() or not last.is_file():
        raise RuntimeError(f"Person v3 {stage_name} checkpoints are missing")
    gradient = root / f"{stage_name}_gradient_metrics.csv"
    if not gradient.is_file():
        raise RuntimeError(f"Person v3 {stage_name} gradient audit is missing")
    atomic_json(marker, {
        "status": "PASS",
        "stage": stage_name,
        "finished_at": now(),
        "initialization": str(initialization.resolve()),
        "initialization_sha256": sha256(initialization),
        "data": str(data.resolve()),
        "data_sha256": sha256(data),
        "best": str(best.resolve()),
        "best_sha256": sha256(best),
        "last": str(last.resolve()),
        "last_sha256": sha256(last),
        "gradient_metrics": str(gradient.resolve()),
        "test_used": False,
    })
    return best


def train(fold: int) -> dict[str, Any]:
    lock = assert_locked()
    protocol = load_protocol()
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))["folds"]
    if str(fold) not in folds:
        raise RuntimeError(f"Person v3 fold is not frozen: {fold}")
    if not torch.cuda.is_available():
        raise RuntimeError("Person v3 training requires CUDA")
    data = DATASET_ROOT / f"folds/fold_{fold}/data.yaml"
    if not data.is_file():
        raise RuntimeError(f"Person v3 fold dataset is missing: {data}")
    root = training_root(fold)
    root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": lock["protocol_sha256"],
        "fold": fold,
        "heldout_scenes": folds[str(fold)],
        "seed": protocol["pipeline"]["seed"],
        "data": str(data.resolve()),
        "data_sha256": sha256(data),
        "target": protocol["target"],
        "pipeline": protocol["pipeline"],
        "test_usage": "forbidden",
    }
    atomic_text(
        root / "resolved_config.yaml",
        yaml.safe_dump(resolved, sort_keys=False),
    )
    atomic_json(root / "environment.json", environment())
    current = INITIALIZATION
    stages = {}
    for stage_name in ("stage1", "stage2"):
        current = train_stage(
            initialization=current,
            data=data,
            root=root,
            stage_name=stage_name,
            stage=protocol["pipeline"]["staged_training"][stage_name],
            protocol=protocol,
        )
        stages[stage_name] = {
            "checkpoint": str(current.resolve()),
            "sha256": sha256(current),
        }
    result = {
        "status": "PASS",
        "finished_at": now(),
        "protocol_id": protocol["protocol_id"],
        "fold": fold,
        "seed": protocol["pipeline"]["seed"],
        "heldout_scenes": folds[str(fold)],
        "final_checkpoint": str(current.resolve()),
        "final_checkpoint_sha256": sha256(current),
        "stages": stages,
        "test_used": False,
    }
    atomic_json(root / "TRAINING_COMPLETE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.fold), indent=2))


if __name__ == "__main__":
    main()

