from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from scripts.person_v4.common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    assert_locked,
    atomic_json,
    load_protocol,
    now,
    sha256,
)
from scripts.person_v4.trainer import PersonDGTrainer
from scripts.run_micro_overfit import GradientLogger


PERSON_ROOT = PROJECT_DIR / "outputs/person_v3"
FOLDS_PATH = PERSON_ROOT / "protocol/folds.json"
TILE_MANIFEST = (
    PROJECT_DIR / "outputs/canonical_m4/tiling_audit/tile_manifest.csv"
)
INITIALIZATION = PROJECT_DIR / "yolo11m.pt"


def training_root(variant: str, fold: int) -> Path:
    return OUTPUT_ROOT / f"variants/{variant}/fold_{fold}/seed_20260723"


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


def tile_scene_map() -> dict[str, str]:
    frame = pd.read_csv(TILE_MANIFEST)
    return {
        Path(row.tile_image).stem: str(row.grouped_scene_id)
        for row in frame.itertuples(index=False)
    }


def train_images(fold: int) -> list[Path]:
    path = PERSON_ROOT / f"dataset/folds/fold_{fold}/train.txt"
    return [
        Path(value.strip()).absolute()
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]


def nwd_constant(fold: int, input_size: int) -> tuple[float, dict[str, Any]]:
    values = []
    source_width = 2350.0
    source_height = 1431.0
    gain = min(input_size / source_width, input_size / source_height)
    for image in train_images(fold):
        label = (
            PERSON_ROOT
            / "dataset/labels/development"
            / f"{image.stem}.txt"
        )
        if not label.is_file():
            raise RuntimeError(f"Person label is missing: {label}")
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, _, _, width, height = map(float, line.split())
            if int(class_id) != 0:
                raise RuntimeError("Non-person class in the person-only view")
            width_px = width * source_width * gain
            height_px = height * source_height * gain
            values.append(math.sqrt(width_px * height_px))
    if not values:
        raise RuntimeError("No person boxes in fold training portion")
    constant = max(float(np.median(values)), 1.0)
    return constant, {
        "rule": "median_sqrt_area_after_letterbox_fold_train_only",
        "fold": fold,
        "count": len(values),
        "median_px": constant,
        "mean_px": float(np.mean(values)),
        "q01_px": float(np.quantile(values, 0.01)),
        "q99_px": float(np.quantile(values, 0.99)),
        "input_size": input_size,
        "source_tile_size": [int(source_width), int(source_height)],
        "letterbox_gain": gain,
    }


def experiment_config(
    variant: str,
    fold: int,
    root: Path,
) -> tuple[dict[str, Any], dict[str, str], list[str], dict[str, Any]]:
    protocol = load_protocol()
    definition = protocol["matrix"][variant]
    constant, audit = nwd_constant(
        fold, int(protocol["training"]["input_size"])
    )
    mapping = tile_scene_map()
    stems = {image.stem for image in train_images(fold)}
    scene_names = sorted({mapping[stem] for stem in stems})
    if len(scene_names) < 2:
        raise RuntimeError("Fold train portion has fewer than two scenes")
    nwd = protocol["nwd"]
    config = {
        "variant": variant,
        "fold": fold,
        "seed": int(protocol["training"]["seed"]),
        "scene_sampler": bool(definition["group_dro"]),
        "nwd_constant": constant,
        "nwd": {
            "reference_area": float(
                nwd["area_alpha"]["reference_area_px2"]
            ),
            "area_temperature": float(
                nwd["area_alpha"]["temperature"]
            ),
            "candidate_min_similarity": float(
                nwd["assignment_candidate_min_similarity"]
            ),
        },
        "qfl": {
            "beta": float(protocol["quality_focal_loss"]["beta"]),
            "iou_weight": float(
                protocol["quality_focal_loss"]["quality_target"][
                    "IoU_weight"
                ]
            ),
            "nwd_weight": float(
                protocol["quality_focal_loss"]["quality_target"][
                    "NWD_weight"
                ]
            ),
        },
        "group_dro": {
            "enabled": bool(definition["group_dro"]),
            "eta": float(protocol["group_dro"]["eta"]),
        },
        "mixstyle": {
            "enabled": bool(definition["mixstyle"]),
            "probability": float(protocol["mixstyle"]["probability"]),
            "beta_alpha": float(protocol["mixstyle"]["beta_alpha"]),
            "epsilon": float(protocol["mixstyle"]["epsilon"]),
            "layer_indices": list(
                protocol["mixstyle"]["backbone_layer_indices"]
            ),
        },
        "scale_aware": {
            "enabled": bool(definition["scale_aware_augmentation"]),
            "probability": float(
                protocol["scale_aware_augmentation"]["probability"]
            ),
            "zoom_min": float(
                protocol["scale_aware_augmentation"]["zoom_min"]
            ),
            "zoom_max": float(
                protocol["scale_aware_augmentation"]["zoom_max"]
            ),
            "minimum_box_px": float(
                protocol["scale_aware_augmentation"][
                    "minimum_box_after_transform_px"
                ]
            ),
        },
        "swad": {"enabled": bool(definition["swad"])},
        "root": str(root.resolve()),
    }
    return config, mapping, scene_names, audit


class EpochTrajectory:
    """Save the stage-2 EMA state after every validation epoch."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.destination.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []

    def __call__(self, trainer: Any) -> None:
        epoch = int(trainer.epoch) + 1
        ema_model = trainer.ema.ema if trainer.ema else trainer.model
        state = {
            key: value.detach().cpu()
            for key, value in ema_model.state_dict().items()
        }
        checkpoint = self.destination / f"epoch_{epoch:03d}.pt"
        torch.save(state, checkpoint)
        metrics = {
            key: float(value)
            for key, value in trainer.metrics.items()
            if isinstance(value, (int, float))
        }
        row = {
            "epoch": epoch,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            **metrics,
        }
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(
            self.destination / "trajectory.csv", index=False
        )


def select_swad_epochs(results: pd.DataFrame, config: dict[str, Any]) -> list[int]:
    columns = ["val/box_loss", "val/cls_loss", "val/dfl_loss"]
    losses = results[columns].sum(axis=1).to_numpy(dtype=float)
    minimum = float(losses.min())
    start_limit = minimum * (
        1 + float(config["start_tolerance_relative"])
    )
    start = int(np.flatnonzero(losses <= start_limit)[0])
    end = len(losses) - 1
    patience = int(config["end_patience_epochs"])
    degradation = float(config["end_degradation_relative"])
    for index in range(start + patience - 1, len(losses)):
        window = losses[index - patience + 1 : index + 1]
        historical = float(losses[: index - patience + 2].min())
        if bool(np.all(window > historical * (1 + degradation))):
            end = index - patience
            break
    selected = list(range(start + 1, end + 2))
    minimum_count = int(config["minimum_averaged_checkpoints"])
    if len(selected) < minimum_count:
        selected = sorted(
            (np.argsort(losses)[:minimum_count] + 1).tolist()
        )
    return selected


def build_swad(
    run: Path,
    stage_best: Path,
    protocol: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    trajectory = run / "trajectory"
    frame = pd.read_csv(run / "results.csv")
    selected = select_swad_epochs(frame, protocol["swad"])
    states = [
        torch.load(
            trajectory / f"epoch_{epoch:03d}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for epoch in selected
    ]
    averaged: dict[str, torch.Tensor] = {}
    for key in states[0]:
        value = states[0][key]
        if not torch.is_floating_point(value):
            averaged[key] = value
            continue
        total = torch.zeros_like(value, dtype=torch.float32)
        for state in states:
            total += state[key].float()
        averaged[key] = (total / len(states)).to(dtype=value.dtype)
    checkpoint = torch.load(
        stage_best, map_location="cpu", weights_only=False
    )
    model = checkpoint.get("ema") or checkpoint["model"]
    model.load_state_dict(averaged, strict=True)
    checkpoint["model"] = copy.deepcopy(model).half()
    checkpoint["ema"] = copy.deepcopy(model).half()
    checkpoint["optimizer"] = None
    checkpoint["updates"] = None
    destination = run / "weights/swad.pt"
    torch.save(checkpoint, destination)
    audit = {
        "status": "PASS",
        "selected_epochs": selected,
        "checkpoint_count": len(selected),
        "source": "stage2_EMA_epoch_trajectory",
        "selection_split": "development_fold_validation_only",
        "test_used": False,
        "output": str(destination.resolve()),
        "output_sha256": sha256(destination),
    }
    atomic_json(run / "SWAD_SELECTION.json", audit)
    return destination, audit


def train_stage(
    *,
    variant: str,
    fold: int,
    initialization: Path,
    data: Path,
    root: Path,
    stage_name: str,
    stage: dict[str, Any],
    protocol: dict[str, Any],
    config: dict[str, Any],
    mapping: dict[str, str],
    scene_names: list[str],
) -> Path:
    run = root / stage_name
    best = run / "weights/best.pt"
    last = run / "weights/last.pt"
    marker = run / "TRAINING_COMPLETE.json"
    if marker.is_file() and best.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload["best_sha256"] != sha256(best):
            raise RuntimeError("Completed person v4 checkpoint changed")
        if variant == "A3" and stage_name == "stage2":
            swad = run / "weights/swad.pt"
            if not swad.is_file():
                raise RuntimeError("Completed A3 stage2 misses SWAD")
            return swad
        return best
    PersonDGTrainer.backbone_lr = float(stage["backbone_lr"])
    PersonDGTrainer.head_lr = float(stage["head_lr"])
    PersonDGTrainer.experiment_config = dict(config)
    PersonDGTrainer.scene_by_stem = dict(mapping)
    PersonDGTrainer.scene_names = list(scene_names)
    PersonDGTrainer.dro_log_path = (
        root / f"{stage_name}_group_dro.csv"
        if config["group_dro"]["enabled"] else None
    )
    PersonDGTrainer.augmentation_log = []
    logger = GradientLogger(root / f"{stage_name}_gradient_metrics.csv")
    model = YOLO(str(last if last.is_file() else initialization))
    model.add_callback("on_train_start", logger.train_start)
    model.add_callback("on_train_epoch_start", logger.epoch_start)
    model.add_callback("on_train_batch_end", logger.before_zero_grad)
    model.add_callback("on_train_epoch_end", logger.epoch_end)
    recorder = None
    if variant == "A3" and stage_name == "stage2":
        recorder = EpochTrajectory(run / "trajectory")
        model.add_callback("on_fit_epoch_end", recorder)
    if last.is_file():
        model.train(resume=True, trainer=PersonDGTrainer)
    else:
        augmentations = dict(
            protocol["training"]["common_augmentations"]
        )
        if variant == "A3":
            photometric = protocol["scale_aware_augmentation"][
                "photometric"
            ]
            augmentations.update(photometric)
        model.train(
            trainer=PersonDGTrainer,
            data=str(data.resolve()),
            task="detect",
            imgsz=int(protocol["training"]["input_size"]),
            epochs=int(stage["epochs"]),
            patience=int(stage["epochs"]),
            batch=int(protocol["training"]["batch"]),
            nbs=int(protocol["training"]["nominal_batch_size"]),
            device=0,
            workers=int(protocol["training"]["workers"]),
            optimizer="AdamW",
            lr0=float(stage["head_lr"]),
            lrf=0.10,
            weight_decay=float(protocol["training"]["weight_decay"]),
            pretrained=True,
            amp=bool(protocol["training"]["amp"]),
            cos_lr=True,
            freeze=int(stage["freeze_layers"]),
            seed=int(protocol["training"]["seed"]),
            deterministic=bool(protocol["training"]["deterministic"]),
            cache=False,
            val=True,
            plots=True,
            save=True,
            save_period=1 if recorder else 5,
            close_mosaic=0,
            project=str(root.resolve()),
            name=stage_name,
            exist_ok=False,
            verbose=True,
            **augmentations,
        )
    if not best.is_file() or not last.is_file():
        raise RuntimeError(f"Person v4 {stage_name} checkpoints missing")
    selected = best
    swad_audit = None
    if recorder is not None:
        selected, swad_audit = build_swad(
            run, best, protocol["swad"]
        )
    if PersonDGTrainer.augmentation_log:
        pd.DataFrame(PersonDGTrainer.augmentation_log).to_csv(
            root / f"{stage_name}_scale_augmentation.csv", index=False
        )
    payload = {
        "status": "PASS",
        "variant": variant,
        "fold": fold,
        "stage": stage_name,
        "finished_at": now(),
        "initialization": str(initialization.resolve()),
        "initialization_sha256": sha256(initialization),
        "best": str(best.resolve()),
        "best_sha256": sha256(best),
        "last": str(last.resolve()),
        "last_sha256": sha256(last),
        "selected": str(selected.resolve()),
        "selected_sha256": sha256(selected),
        "swad": swad_audit,
        "test_used": False,
    }
    atomic_json(marker, payload)
    return selected


def train(variant: str, fold: int) -> dict[str, Any]:
    assert_locked()
    protocol = load_protocol()
    if variant not in {"A1", "A2", "A3"}:
        raise ValueError("Only A1-A3 require training")
    if fold not in range(5):
        raise ValueError("Fold is not frozen")
    if not torch.cuda.is_available():
        raise RuntimeError("Person v4 training requires CUDA")
    data = PERSON_ROOT / f"dataset/folds/fold_{fold}/data.yaml"
    root = training_root(variant, fold)
    root.mkdir(parents=True, exist_ok=True)
    config, mapping, scene_names, constant_audit = experiment_config(
        variant, fold, root
    )
    atomic_json(root / "nwd_constant.json", constant_audit)
    resolved = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(
            PROJECT_DIR / "configs/canonical_v4_person_dg_nwd.yaml"
        ),
        "variant": variant,
        "fold": fold,
        "experiment": config,
        "training": protocol["training"],
        "test_usage": "forbidden",
    }
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    atomic_json(root / "environment.json", environment())
    current = INITIALIZATION
    stages = {}
    for stage_name in ("stage1", "stage2"):
        current = train_stage(
            variant=variant,
            fold=fold,
            initialization=current,
            data=data,
            root=root,
            stage_name=stage_name,
            stage=protocol["training"]["stages"][stage_name],
            protocol=protocol,
            config=config,
            mapping=mapping,
            scene_names=scene_names,
        )
        stages[stage_name] = {
            "selected_checkpoint": str(current.resolve()),
            "selected_sha256": sha256(current),
        }
    result = {
        "status": "PASS",
        "finished_at": now(),
        "protocol_id": protocol["protocol_id"],
        "variant": variant,
        "fold": fold,
        "seed": protocol["training"]["seed"],
        "final_checkpoint": str(current.resolve()),
        "final_checkpoint_sha256": sha256(current),
        "nwd_constant": constant_audit,
        "stages": stages,
        "test_used": False,
    }
    atomic_json(root / "TRAINING_COMPLETE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A1", "A2", "A3"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.variant, args.fold), indent=2))


if __name__ == "__main__":
    main()
