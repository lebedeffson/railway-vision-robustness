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
from scripts.person_v5.common import (
    OUTPUT,
    PROJECT,
    assert_runtime_locked,
    atomic_json,
    atomic_text,
    load_protocol,
    now,
    sha256,
)
from scripts.person_v5.hard_mining import mine
from scripts.run_micro_overfit import GradientLogger


RUNTIME_PATH = PROJECT / "configs/canonical_v5_person_data_first_runtime.yaml"


def runtime() -> dict[str, Any]:
    return yaml.safe_load(RUNTIME_PATH.read_text(encoding="utf-8"))


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
            ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
        ).strip(),
    }


def ensure_crowdhuman_yaml() -> Path:
    root = PROJECT / "data/crowdhuman/yolo_visible_person"
    path = root / "data.yaml"
    payload = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": {0: "person"},
    }
    atomic_text(path, yaml.safe_dump(payload, sort_keys=False))
    return path


def external_pretrain() -> Path:
    assert_runtime_locked()
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical v5 external pretraining requires CUDA")
    protocol = load_protocol()
    config = protocol["external_pretraining"]
    root = PROJECT / runtime()["external_pretraining"]["output"]
    best = root / "weights/best.pt"
    last = root / "weights/last.pt"
    marker = root / "TRAINING_COMPLETE.json"
    if marker.is_file() and best.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload["best_sha256"] != sha256(best):
            raise RuntimeError("CrowdHuman best checkpoint changed")
        return best
    data = ensure_crowdhuman_yaml()
    model = YOLO(
        str(last if last.is_file() else PROJECT / "yolo11m.pt")
    )
    if last.is_file():
        model.train(resume=True)
    else:
        augmentation = config["augmentation"]
        model.train(
            data=str(data.resolve()),
            task="detect",
            imgsz=int(config["imgsz"]),
            epochs=int(config["epochs"]),
            patience=int(config["patience"]),
            batch=int(config["batch"]),
            nbs=int(config["nominal_batch_size"]),
            device=0,
            workers=int(config["workers"]),
            optimizer=str(config["optimizer"]),
            pretrained=True,
            amp=bool(config["amp"]),
            seed=int(config["seed"]),
            deterministic=bool(config["deterministic"]),
            cache=False,
            val=True,
            plots=True,
            save=True,
            save_period=int(
                runtime()["external_pretraining"]["save_period"]
            ),
            close_mosaic=0,
            project=str(root.parent.resolve()),
            name=root.name,
            exist_ok=False,
            verbose=True,
            **augmentation,
        )
    if not best.is_file() or not last.is_file():
        raise RuntimeError("CrowdHuman checkpoints are missing")
    atomic_json(
        marker,
        {
            "status": "PASS",
            "finished_at": now(),
            "data": str(data.resolve()),
            "data_sha256": sha256(data),
            "best": str(best.resolve()),
            "best_sha256": sha256(best),
            "last": str(last.resolve()),
            "last_sha256": sha256(last),
            "selection": "maximum_external_validation_mAP50",
            "crowdhuman_test_used": False,
            "railway_test_used": False,
        },
    )
    return best


def train_stage(
    *,
    fold: int,
    name: str,
    initialization: Path,
    data: Path,
    stage: dict[str, Any],
) -> Path:
    protocol = load_protocol()
    root = OUTPUT / f"railway/fold_{fold}"
    run = root / name
    best = run / "weights/best.pt"
    last = run / "weights/last.pt"
    marker = run / "TRAINING_COMPLETE.json"
    if marker.is_file() and best.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload["best_sha256"] != sha256(best):
            raise RuntimeError(f"V5 fold {fold} checkpoint changed")
        return best
    DifferentialLRDetectionTrainer.backbone_lr = float(
        stage["backbone_lr"]
    )
    DifferentialLRDetectionTrainer.head_lr = float(stage["head_lr"])
    logger = GradientLogger(root / f"{name}_gradient_metrics.csv")
    model = YOLO(str(last if last.is_file() else initialization))
    model.add_callback("on_train_start", logger.train_start)
    model.add_callback("on_train_epoch_start", logger.epoch_start)
    model.add_callback("on_train_batch_end", logger.before_zero_grad)
    model.add_callback("on_train_epoch_end", logger.epoch_end)
    if last.is_file():
        model.train(resume=True, trainer=DifferentialLRDetectionTrainer)
    else:
        model.train(
            trainer=DifferentialLRDetectionTrainer,
            data=str(data.resolve()),
            task="detect",
            imgsz=int(protocol["railway_finetuning"]["imgsz"]),
            epochs=int(stage["epochs"]),
            patience=int(stage["epochs"]),
            batch=int(protocol["railway_finetuning"]["batch"]),
            nbs=int(protocol["railway_finetuning"]["nominal_batch_size"]),
            device=0,
            workers=int(protocol["railway_finetuning"]["workers"]),
            optimizer="AdamW",
            lr0=float(stage["head_lr"]),
            lrf=0.10,
            weight_decay=0.0005,
            pretrained=True,
            amp=bool(protocol["railway_finetuning"]["amp"]),
            cos_lr=True,
            freeze=int(stage["freeze_layers"]),
            seed=int(protocol["railway_finetuning"]["seed"]),
            deterministic=True,
            cache=False,
            val=True,
            plots=True,
            save=True,
            save_period=int(runtime()["railway"]["save_period"]),
            close_mosaic=0,
            project=str(root.resolve()),
            name=name,
            exist_ok=False,
            verbose=True,
            **runtime()["railway"]["augmentation"],
        )
    if not best.is_file() or not last.is_file():
        raise RuntimeError(f"V5 {name} checkpoints are missing")
    atomic_json(
        marker,
        {
            "status": "PASS",
            "fold": fold,
            "stage": name,
            "finished_at": now(),
            "initialization": str(initialization.resolve()),
            "initialization_sha256": sha256(initialization),
            "data": str(data.resolve()),
            "data_sha256": sha256(data),
            "best": str(best.resolve()),
            "best_sha256": sha256(best),
            "last": str(last.resolve()),
            "last_sha256": sha256(last),
            "test_used": False,
        },
    )
    return best


def train_fold(fold: int, external_checkpoint: Path) -> dict[str, Any]:
    assert_runtime_locked()
    if fold not in (0, 1):
        raise RuntimeError("V5 triage runtime only permits folds 0 and 1")
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical v5 railway training requires CUDA")
    root = OUTPUT / f"railway/fold_{fold}"
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "environment.json", environment())
    original = (
        PROJECT / runtime()["railway"]["source_data_pattern"].format(fold=fold)
    )
    stages = runtime()["railway"]["stages"]
    warmup = train_stage(
        fold=fold,
        name="warmup",
        initialization=external_checkpoint,
        data=original,
        stage=stages["warmup"],
    )
    mined = mine(fold, warmup)
    final = train_stage(
        fold=fold,
        name="full",
        initialization=warmup,
        data=mined,
        stage=stages["full"],
    )
    result = {
        "status": "PASS",
        "fold": fold,
        "finished_at": now(),
        "external_checkpoint": str(external_checkpoint.resolve()),
        "external_checkpoint_sha256": sha256(external_checkpoint),
        "warmup_checkpoint_sha256": sha256(warmup),
        "hard_mining_data": str(mined.resolve()),
        "final_checkpoint": str(final.resolve()),
        "final_checkpoint_sha256": sha256(final),
        "test_used": False,
    }
    atomic_json(root / "TRAINING_COMPLETE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("external", "fold"), required=True
    )
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    checkpoint = external_pretrain()
    if args.stage == "external":
        print(json.dumps({"checkpoint": str(checkpoint)}, indent=2))
        return
    if args.fold is None:
        raise SystemExit("--fold is required for railway fold training")
    print(json.dumps(train_fold(args.fold, checkpoint), indent=2))


if __name__ == "__main__":
    main()

