from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from rescue_common import (
    PROJECT_DIR,
    OUTPUT_ROOT,
    assert_test_sealed,
    atomic_json,
    completed_marker,
    ensure_branch,
    environment_snapshot,
    load_protocol,
    now,
    sha256,
)


OUTPUT = OUTPUT_ROOT / "micro_overfit"
DATASET = OUTPUT / "dataset"
RUN = OUTPUT / "training"


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise RuntimeError(f"Micro-overfit target already differs: {destination}")
    os.link(source, destination)


def label_classes(path: Path) -> set[int]:
    return {
        int(line.split()[0])
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    }


def prepare_subset(protocol: dict[str, Any]) -> tuple[Path, Path]:
    repaired = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"
    manifest_path = repaired if repaired.is_file() else PROJECT_DIR / protocol["split_manifest"]
    manifest = pd.read_csv(manifest_path)
    train = manifest[manifest["split"] == "train"].copy()
    scene_classes: dict[str, set[int]] = {}
    for scene, rows in train.groupby("grouped_scene_id"):
        classes: set[int] = set()
        for path in rows["output_label"]:
            classes.update(label_classes(Path(path)))
        scene_classes[str(scene)] = classes
    pairs = []
    scenes = sorted(scene_classes)
    for left_index, left in enumerate(scenes):
        for right in scenes[left_index:]:
            coverage = scene_classes[left] | scene_classes[right]
            pairs.append((-len(coverage), left, right))
    _, first, second = min(pairs)
    chosen_scenes = [first] if first == second else [first, second]
    scope = train[train["grouped_scene_id"].astype(str).isin(chosen_scenes)].copy()
    scope = scope.sort_values(["grouped_scene_id", "subsequence_id", "frame_id", "output_image"])
    frame_count = int(protocol["micro_overfit"]["frames"])
    selected_indices: list[int] = []
    covered: set[int] = set()
    remaining = list(scope.index)
    while remaining and len(selected_indices) < frame_count:
        best = max(
            remaining,
            key=lambda index: (
                len(label_classes(Path(scope.loc[index, "output_label"])) - covered),
                -len(selected_indices),
                str(scope.loc[index, "output_image"]),
            ),
        )
        selected_indices.append(best)
        covered.update(label_classes(Path(scope.loc[best, "output_label"])))
        remaining.remove(best)
        if len(covered) >= int(protocol["micro_overfit"]["minimum_classes"]):
            break
    for index in remaining:
        if len(selected_indices) >= frame_count:
            break
        selected_indices.append(index)
    selected = scope.loc[selected_indices].sort_values(
        ["grouped_scene_id", "subsequence_id", "frame_id", "output_image"]
    )
    if len(selected) != frame_count:
        raise RuntimeError(f"Could select only {len(selected)} micro-overfit frames")
    if len(covered) < int(protocol["micro_overfit"]["minimum_classes"]):
        raise RuntimeError(f"Micro-overfit subset covers only classes {sorted(covered)}")
    rows = []
    for row in selected.itertuples(index=False):
        source_image = Path(row.output_image)
        source_label = Path(row.output_label)
        for split in ("train", "val"):
            image = DATASET / "images" / split / source_image.name
            label = DATASET / "labels" / split / source_label.name
            hardlink(source_image, image)
            hardlink(source_label, label)
        rows.append({
            **row._asdict(),
            "split": "train",
            "micro_train_image": str((DATASET / "images/train" / source_image.name).resolve()),
            "micro_val_image": str((DATASET / "images/val" / source_image.name).resolve()),
        })
    subset_manifest = OUTPUT / "subset_manifest.csv"
    pd.DataFrame(rows).to_csv(subset_manifest, index=False)
    data = {
        "path": str(DATASET.resolve()),
        "train": "images/train", "val": "images/val",
        "nc": 6, "names": protocol["class_names"],
    }
    data_yaml = DATASET / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    atomic_json(OUTPUT / "subset_selection.json", {
        "status": "PASS", "frames": len(selected), "scenes": chosen_scenes,
        "classes": sorted(covered), "selection_uses_model_results": False,
        "selection_split": "train_only", "seed": protocol["micro_overfit"]["seed"],
    })
    return data_yaml, subset_manifest


class GradientLogger:
    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.rows: list[dict[str, Any]] = []
        self.batch_rows: list[dict[str, float]] = []
        self.initial_weight_norm: float | None = None

    @staticmethod
    def weight_norm(model: torch.nn.Module) -> float:
        return math.sqrt(sum(float(parameter.detach().float().pow(2).sum()) for parameter in model.parameters()))

    def train_start(self, trainer: Any) -> None:
        self.initial_weight_norm = self.weight_norm(trainer.model)

    def epoch_start(self, trainer: Any) -> None:
        self.batch_rows = []
        torch.cuda.reset_peak_memory_stats()

    def before_zero_grad(self, trainer: Any) -> None:
        head_ids = {id(parameter) for parameter in trainer.model.model[-1].parameters()}
        backbone_sq = head_sq = 0.0
        trainable = zero_or_missing = 0
        for parameter in trainer.model.parameters():
            if not parameter.requires_grad:
                continue
            trainable += 1
            if parameter.grad is None:
                zero_or_missing += 1
                continue
            norm_sq = float(parameter.grad.detach().float().pow(2).sum())
            if norm_sq == 0:
                zero_or_missing += 1
            if id(parameter) in head_ids:
                head_sq += norm_sq
            else:
                backbone_sq += norm_sq
        self.batch_rows.append({
            "backbone_gradient_norm": math.sqrt(backbone_sq),
            "head_gradient_norm": math.sqrt(head_sq),
            "zero_or_missing_gradient_fraction": zero_or_missing / max(trainable, 1),
        })

    def epoch_end(self, trainer: Any) -> None:
        losses = trainer.tloss.detach().float().cpu().tolist() if trainer.tloss is not None else []
        if not isinstance(losses, list):
            losses = [float(losses)]
        averages = {
            key: sum(row[key] for row in self.batch_rows) / max(len(self.batch_rows), 1)
            for key in (
                "backbone_gradient_norm", "head_gradient_norm",
                "zero_or_missing_gradient_fraction",
            )
        }
        row = {
            "epoch": int(trainer.epoch) + 1,
            "learning_rate": float(next(iter(trainer.optimizer.param_groups))["lr"]),
            "box_loss": float(losses[0]) if len(losses) > 0 else math.nan,
            "cls_loss": float(losses[1]) if len(losses) > 1 else math.nan,
            "dfl_loss": float(losses[2]) if len(losses) > 2 else math.nan,
            **averages,
            "weight_norm": self.weight_norm(trainer.model),
            "initial_weight_norm": self.initial_weight_norm,
            "gpu_memory_peak_bytes": int(torch.cuda.max_memory_allocated()),
            "nan_or_inf": any(
                not math.isfinite(float(value))
                for value in [*losses, *averages.values()]
            ),
        }
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.destination, index=False)


def train(data_yaml: Path, protocol: dict[str, Any]) -> Path:
    weights = RUN / "micro_overfit" / "weights"
    best = weights / "best.pt"
    marker = RUN / "micro_overfit" / "TRAINING_COMPLETE"
    if marker.is_file() and best.is_file():
        return best
    last = weights / "last.pt"
    model = YOLO(str(last) if last.is_file() else str(PROJECT_DIR / "yolo11m.pt"))
    gradient_logger = GradientLogger(OUTPUT / "gradient_metrics.csv")
    model.add_callback("on_train_start", gradient_logger.train_start)
    model.add_callback("on_train_epoch_start", gradient_logger.epoch_start)
    model.add_callback("on_before_zero_grad", gradient_logger.before_zero_grad)
    model.add_callback("on_train_epoch_end", gradient_logger.epoch_end)
    settings = protocol["micro_overfit"]
    if last.is_file():
        model.train(resume=True)
    else:
        model.train(
            data=str(data_yaml), task="detect", imgsz=int(settings["imgsz"]),
            epochs=int(settings["epochs"]), patience=int(settings["epochs"]),
            batch=1, device=0, workers=2, optimizer="AdamW", lr0=3e-4,
            lrf=0.1, weight_decay=5e-4, pretrained=True, amp=True,
            cos_lr=True, freeze=0, seed=int(settings["seed"]), deterministic=True,
            mosaic=float(settings["mosaic"]), mixup=float(settings["mixup"]),
            translate=float(settings["translate"]), scale=float(settings["scale"]),
            perspective=float(settings["perspective"]), fliplr=float(settings["fliplr"]),
            hsv_h=float(settings["hsv_h"]), hsv_s=float(settings["hsv_s"]),
            hsv_v=float(settings["hsv_v"]), cache=False, val=True, plots=True,
            save=True, save_period=10, project=str(RUN), name="micro_overfit",
            exist_ok=False, verbose=True,
        )
    if not best.is_file():
        raise FileNotFoundError(best)
    marker.write_text(
        f"best={best.resolve()}\nsha256={sha256(best)}\n", encoding="utf-8"
    )
    return best


def main() -> None:
    ensure_branch()
    assert_test_sealed()
    protocol = load_protocol()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUTPUT / "environment.json", environment_snapshot())
    data_yaml, subset_manifest = prepare_subset(protocol)
    if not torch.cuda.is_available():
        block = {
            "status": "BLOCKED_INFRASTRUCTURE",
            "stage": "micro_overfit",
            "reason": "cuda_device_not_visible_to_execution_environment",
            "torch_cuda_version": torch.version.cuda,
            "torch_cuda_available": False,
            "scientific_gate_evaluated": False,
            "test_evaluated": False,
            "resume_action": "rerun the same committed pipeline in the host user service",
        }
        atomic_json(OUTPUT / "infrastructure_block.json", block)
        raise SystemExit(75)
    best = train(data_yaml, protocol)
    evaluation = OUTPUT / "evaluation"
    summary = evaluation / "baseline_rescue_summary.json"
    if not summary.is_file():
        subprocess.run([
            str(PROJECT_DIR / ".venv/bin/python"), "baseline_rescue_audit.py",
            "--model", str(best), "--data", str(data_yaml),
            "--manifest", str(subset_manifest), "--output", str(evaluation),
            "--splits", "train,val", "--imgsz", str(protocol["micro_overfit"]["imgsz"]),
        ], cwd=PROJECT_DIR, check=True)
    metrics = pd.read_csv(evaluation / "clean_metrics_train_val_test.csv")
    val = metrics[
        (metrics["split"] == "val") & (metrics["operating_point"] == "safety")
    ].iloc[0]
    history = pd.read_csv(RUN / "micro_overfit" / "results.csv")
    history.columns = [column.strip() for column in history.columns]
    loss_columns = [column for column in history if column.startswith("train/")]
    initial_loss = float(history.iloc[0][loss_columns].sum())
    final_loss = float(history.iloc[-1][loss_columns].sum())
    gradient = pd.read_csv(OUTPUT / "gradient_metrics.csv")
    result = {
        "status": "PASS" if (
            float(val["mAP50"]) >= float(protocol["micro_overfit"]["map50_min"])
            and float(val["recall"]) >= float(protocol["micro_overfit"]["recall_min"])
            and final_loss < initial_loss
            and not gradient["nan_or_inf"].astype(bool).any()
        ) else "FAIL",
        "checkpoint": str(best.resolve()), "checkpoint_sha256": sha256(best),
        "frames": int(protocol["micro_overfit"]["frames"]),
        "mAP50": float(val["mAP50"]), "recall": float(val["recall"]),
        "f1": float(val["f1"]), "initial_train_loss_sum": initial_loss,
        "final_train_loss_sum": final_loss,
        "loss_decreased": final_loss < initial_loss,
        "gradient_nan_or_inf": bool(gradient["nan_or_inf"].astype(bool).any()),
        "test_evaluated": False,
    }
    atomic_json(OUTPUT / "result.json", result)
    shutil.copy2(RUN / "micro_overfit" / "results.csv", OUTPUT / "metrics.csv")
    if (RUN / "micro_overfit" / "results.png").is_file():
        shutil.copy2(RUN / "micro_overfit" / "results.png", OUTPUT / "loss_curves.png")
    required = [
        OUTPUT / "result.json", OUTPUT / "metrics.csv",
        OUTPUT / "gradient_metrics.csv", evaluation / "threshold_selection.json",
    ]
    completed_marker(
        OUTPUT, inputs=[data_yaml, subset_manifest], outputs=required,
        extra={"stage": "micro_overfit", "checkpoint_sha256": sha256(best)},
    )
    if result["status"] != "PASS":
        print(
            "Micro-overfit scientific gate failed; complete diagnostics were "
            "saved and the parent pipeline must block the candidate matrix.",
            flush=True,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
