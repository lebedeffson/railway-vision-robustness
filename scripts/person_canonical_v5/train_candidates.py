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

from scripts.canonical_m4_trainer import DifferentialLRDetectionTrainer
from scripts.person_canonical_v5.common import (
    OUTPUT,
    PROJECT,
    PROTOCOL_ROOT,
    assert_test_sealed,
    atomic_json,
    now,
    sha256,
)
from scripts.run_micro_overfit import GradientLogger
from src.models.coordinate_attention import register_ultralytics_modules
from src.models.p2_head import build_p2_model


CONFIG = PROJECT / "configs/person_v5/candidate_runtime.yaml"
LOCK = PROTOCOL_ROOT / "candidate_runtime_lock.json"
CANDIDATE_ROOT = OUTPUT / "candidates"


def runtime() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def assert_runtime_locked() -> dict[str, Any]:
    assert_test_sealed()
    if not LOCK.is_file():
        raise RuntimeError("Person-v5 candidate runtime is not locked")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["config_sha256"] != sha256(CONFIG):
        raise RuntimeError("Candidate runtime config changed after lock")
    if not lock["training_allowed"]:
        raise RuntimeError("Candidate training is blocked")
    return lock


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


def candidate_base(candidate: str) -> str:
    return candidate.split("-fraction", 1)[0]


def model_config(candidate: str) -> Path:
    base = candidate_base(candidate)
    if base in {"V5-A", "V5-D"}:
        return PROJECT / "configs/person_v5/models/yolo11m-p2.yaml"
    if base == "V5-B":
        return PROJECT / "configs/person_v5/models/yolo11m-p2-ca.yaml"
    if base == "V5-C":
        return PROJECT / "yolo11m.pt"
    raise ValueError(f"Unknown person-v5 candidate: {candidate}")


def data_config(candidate: str, fold: int) -> Path:
    base = candidate_base(candidate)
    if base in {"V5-A", "V5-B"}:
        return PROJECT / f"outputs/person_v3/dataset/folds/fold_{fold}/data.yaml"
    if "-fraction" in candidate:
        fraction = int(candidate.rsplit("-fraction", 1)[1])
    else:
        selection = (
            PROTOCOL_ROOT / "pasting_fraction_lock.json"
        )
        if not selection.is_file():
            raise RuntimeError("Pasting fraction is not frozen")
        fraction = int(
            json.loads(selection.read_text(encoding="utf-8"))[
                "selected_fraction_percent"
            ]
        )
    return (
        OUTPUT
        / f"instance_pasting/fold_{fold}/fraction_{fraction}/dataset/data.yaml"
    )


def initial_model(candidate: str) -> YOLO:
    config = model_config(candidate)
    if config.suffix == ".yaml":
        return build_p2_model(config, pretrained=PROJECT / "yolo11m.pt")
    return YOLO(str(config))


def train_stage(
    *,
    candidate: str,
    fold: int,
    stage_name: str,
    initialization: Path | None,
    data: Path,
) -> Path:
    config = runtime()
    stage = config["training"]["stages"][stage_name]
    root = CANDIDATE_ROOT / candidate / f"fold_{fold}"
    run = root / stage_name
    best = run / "weights/best.pt"
    last = run / "weights/last.pt"
    marker = run / "TRAINING_COMPLETE.json"
    if marker.is_file() and best.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload["best_sha256"] != sha256(best):
            raise RuntimeError(f"Completed checkpoint changed: {best}")
        return best
    register_ultralytics_modules()
    DifferentialLRDetectionTrainer.backbone_lr = float(
        stage["backbone_lr"]
    )
    DifferentialLRDetectionTrainer.head_lr = float(stage["head_lr"])
    logger = GradientLogger(root / f"{stage_name}_gradient_metrics.csv")
    if last.is_file():
        model = YOLO(str(last))
    elif initialization is not None:
        model = YOLO(str(initialization))
    else:
        model = initial_model(candidate)
    model.add_callback("on_train_start", logger.train_start)
    model.add_callback("on_train_epoch_start", logger.epoch_start)
    model.add_callback("on_train_batch_end", logger.before_zero_grad)
    model.add_callback("on_train_epoch_end", logger.epoch_end)
    if last.is_file():
        model.train(resume=True, trainer=DifferentialLRDetectionTrainer)
    else:
        training = config["training"]
        model.train(
            trainer=DifferentialLRDetectionTrainer,
            data=str(data.resolve()),
            task="detect",
            imgsz=int(training["imgsz"]),
            epochs=int(stage["epochs"]),
            patience=int(stage["epochs"]),
            batch=int(training["batch"]),
            nbs=int(training["nominal_batch_size"]),
            device=0,
            workers=int(training["workers"]),
            optimizer=str(training["optimizer"]),
            lr0=float(stage["head_lr"]),
            lrf=0.10,
            weight_decay=float(training["weight_decay"]),
            pretrained=True,
            amp=bool(training["amp"]),
            cos_lr=True,
            freeze=int(stage["freeze_layers"]),
            seed=int(training["seed"]),
            deterministic=bool(training["deterministic"]),
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
            **training["augmentations"],
        )
    if not best.is_file() or not last.is_file():
        raise RuntimeError(f"Candidate checkpoints missing: {candidate} fold {fold}")
    atomic_json(
        marker,
        {
            "status": "PASS",
            "candidate": candidate,
            "fold": fold,
            "stage": stage_name,
            "finished_at": now(),
            "data": str(data.resolve()),
            "data_sha256": sha256(data),
            "initialization": str(initialization.resolve())
            if initialization is not None
            else str(model_config(candidate).resolve()),
            "best": str(best.resolve()),
            "best_sha256": sha256(best),
            "last": str(last.resolve()),
            "last_sha256": sha256(last),
            "test_used": False,
        },
    )
    return best


def train(candidate: str, fold: int) -> dict[str, Any]:
    lock = assert_runtime_locked()
    if candidate_base(candidate) not in lock["allowed_candidates"]:
        raise RuntimeError(f"Candidate is not allowed: {candidate}")
    if fold not in (0, 1):
        raise RuntimeError("Candidate triage only permits folds 0 and 1")
    if not torch.cuda.is_available():
        raise RuntimeError("Person-v5 candidate training requires CUDA")
    data = data_config(candidate, fold)
    if not data.is_file():
        raise RuntimeError(f"Candidate dataset is not ready: {data}")
    root = CANDIDATE_ROOT / candidate / f"fold_{fold}"
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "environment.json", environment())
    atomic_json(
        root / "resolved_config.json",
        {
            "candidate": candidate,
            "fold": fold,
            "runtime_config_sha256": sha256(CONFIG),
            "data": str(data.resolve()),
            "data_sha256": sha256(data),
            "model": str(model_config(candidate).resolve()),
            "model_sha256": sha256(model_config(candidate)),
            "seed": runtime()["training"]["seed"],
            "test_used": False,
        },
    )
    current: Path | None = None
    for stage_name in ("stage1", "stage2"):
        current = train_stage(
            candidate=candidate,
            fold=fold,
            stage_name=stage_name,
            initialization=current,
            data=data,
        )
    assert current is not None
    payload = {
        "status": "PASS",
        "candidate": candidate,
        "fold": fold,
        "finished_at": now(),
        "checkpoint": str(current.resolve()),
        "checkpoint_sha256": sha256(current),
        "test_used": False,
    }
    atomic_json(root / "TRAINING_COMPLETE.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.candidate, args.fold), indent=2))

