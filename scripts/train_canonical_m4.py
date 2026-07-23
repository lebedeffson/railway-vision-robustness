from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    assert_role_allowed,
    atomic_json,
    load_protocol,
    now,
    sha256,
)
from canonical_m4_trainer import DifferentialLRDetectionTrainer
from run_micro_overfit import GradientLogger


TILING_ROOT = OUTPUT_ROOT / "tiling_audit"
INITIALIZATION = PROJECT_DIR / "yolo11m.pt"


def environment_snapshot() -> dict[str, Any]:
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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def training_root(mode: str, seed: int, fold: int | None) -> Path:
    if mode == "cv":
        if fold is None:
            raise ValueError("CV training requires a fold")
        return OUTPUT_ROOT / "scene_cv" / f"fold_{fold}" / f"seed_{seed}"
    return OUTPUT_ROOT / "full_training" / f"seed_{seed}"


def dataset_path(mode: str, fold: int | None) -> Path:
    if mode == "cv":
        if fold is None:
            raise ValueError("CV training requires a fold")
        return TILING_ROOT / "scene_cv_folds" / f"fold_{fold}" / "data.yaml"
    return TILING_ROOT / "data.yaml"


def stage_train(
    *,
    initialization: Path,
    data: Path,
    run_root: Path,
    stage_name: str,
    config: dict[str, Any],
    protocol: dict[str, Any],
    seed: int,
) -> Path:
    run = run_root / stage_name
    best = run / "weights" / "best.pt"
    last = run / "weights" / "last.pt"
    completion = run / "TRAINING_COMPLETE.json"
    if completion.is_file() and best.is_file():
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if payload.get("best_sha256") != sha256(best):
            raise RuntimeError(f"Completed stage checkpoint changed: {best}")
        return best

    DifferentialLRDetectionTrainer.backbone_lr = float(config["backbone_lr"])
    DifferentialLRDetectionTrainer.head_lr = float(config["head_lr"])
    logger = GradientLogger(run_root / f"{stage_name}_gradient_metrics.csv")
    model = YOLO(str(last if last.is_file() else initialization))
    model.add_callback("on_train_start", logger.train_start)
    model.add_callback("on_train_epoch_start", logger.epoch_start)
    model.add_callback("on_train_batch_end", logger.before_zero_grad)
    model.add_callback("on_train_epoch_end", logger.epoch_end)
    if last.is_file():
        model.train(resume=True, trainer=DifferentialLRDetectionTrainer)
    else:
        augmentations = dict(protocol["model"]["augmentations"])
        augmentations.pop("random_crop", None)
        model.train(
            trainer=DifferentialLRDetectionTrainer,
            data=str(data.resolve()),
            task="detect",
            imgsz=int(protocol["model"]["input_size"]),
            epochs=int(config["epochs"]),
            patience=int(protocol["model"]["patience"]),
            batch=int(protocol["model"]["batch"]),
            device=0,
            workers=int(protocol["model"]["workers"]),
            optimizer=str(protocol["model"]["optimizer"]),
            lr0=float(config["head_lr"]),
            lrf=float(protocol["model"]["lrf"]),
            weight_decay=float(protocol["model"]["weight_decay"]),
            pretrained=True,
            amp=bool(protocol["model"]["amp"]),
            cos_lr=True,
            freeze=int(config["freeze_layers"]),
            seed=seed,
            deterministic=bool(protocol["model"]["deterministic"]),
            cache=bool(protocol["model"]["cache"]),
            val=True,
            plots=True,
            save=True,
            save_period=int(protocol["model"]["save_period"]),
            close_mosaic=0,
            project=str(run_root.resolve()),
            name=stage_name,
            exist_ok=False,
            verbose=True,
            **augmentations,
        )
    if not best.is_file() or not last.is_file():
        raise FileNotFoundError(f"Stage {stage_name} did not produce best.pt and last.pt")
    gradient_path = run_root / f"{stage_name}_gradient_metrics.csv"
    if not gradient_path.is_file():
        raise RuntimeError(f"Gradient audit missing for {stage_name}")
    atomic_json(
        completion,
        {
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
            "gradient_metrics": str(gradient_path.resolve()),
            "test_used": False,
        },
    )
    return best


def train(mode: str, seed: int, fold: int | None) -> dict[str, Any]:
    assert_role_allowed("scene_cv" if mode == "cv" else "full_training")
    protocol = load_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical M4 training requires CUDA")
    if mode == "cv":
        if seed != int(protocol["scene_cv"]["seed"]):
            raise RuntimeError("Scene-CV seed differs from frozen protocol")
        if fold not in range(int(protocol["scene_cv"]["folds"])):
            raise RuntimeError("Scene-CV fold differs from frozen protocol")
    elif seed not in [int(value) for value in protocol["full_training"]["seeds"]]:
        raise RuntimeError("Full-training seed differs from frozen protocol")
    data = dataset_path(mode, fold)
    audit = TILING_ROOT / "tiling_audit.json"
    if not data.is_file() or not audit.is_file():
        raise RuntimeError("Canonical M4 tiling audit is incomplete")
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    if audit_payload.get("status") != "PASS" or audit_payload.get("test_used"):
        raise RuntimeError("Canonical M4 tiling audit did not pass")
    root = training_root(mode, seed, fold)
    root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "protocol_id": protocol["protocol_id"],
        "mode": mode,
        "fold": fold,
        "seed": seed,
        "data": str(data.resolve()),
        "data_sha256": sha256(data),
        "initialization": str(INITIALIZATION.resolve()),
        "initialization_sha256": sha256(INITIALIZATION),
        "model": protocol["model"],
        "test_usage": "forbidden",
    }
    atomic_text(root / "resolved_config.yaml", yaml.safe_dump(resolved, sort_keys=False))
    atomic_json(root / "environment.json", environment_snapshot())
    current = INITIALIZATION
    stages: dict[str, Any] = {}
    for stage_name in ("stage1", "stage2", "stage3"):
        current = stage_train(
            initialization=current,
            data=data,
            run_root=root,
            stage_name=stage_name,
            config=dict(protocol["model"][stage_name]),
            protocol=protocol,
            seed=seed,
        )
        stages[stage_name] = {
            "checkpoint": str(current.resolve()),
            "sha256": sha256(current),
        }
    completion = {
        "status": "PASS",
        "finished_at": now(),
        "protocol_id": protocol["protocol_id"],
        "mode": mode,
        "fold": fold,
        "seed": seed,
        "final_checkpoint": str(current.resolve()),
        "final_checkpoint_sha256": sha256(current),
        "stages": stages,
        "test_evaluated": False,
    }
    atomic_json(root / "TRAINING_COMPLETE.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cv", "official"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    print(json.dumps(train(args.mode, args.seed, args.fold), indent=2))


if __name__ == "__main__":
    main()
