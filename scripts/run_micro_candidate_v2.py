from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from rescue_common import atomic_json, completed_marker, environment_snapshot
from rescue_v2_common import (
    PROJECT_DIR,
    assert_role_allowed,
    load_protocol,
    sha256,
)
from run_micro_overfit import GradientLogger


OUTPUT_ROOT = PROJECT_DIR / "outputs/rescue_v2/micro"
PYTHON = PROJECT_DIR / ".venv/bin/python"


def source_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    manifest = PROJECT_DIR / protocol["micro_dataset"]["source_manifest"]
    selection = PROJECT_DIR / protocol["micro_dataset"]["source_selection"]
    return {
        "manifest_sha256": sha256(manifest),
        "selection_sha256": sha256(selection),
    }


def register_m0(protocol: dict[str, Any]) -> dict[str, Any]:
    root = OUTPUT_ROOT / "M0"
    root.mkdir(parents=True, exist_ok=True)
    frozen = protocol["micro_candidates"]["M0"]
    result = {
        "protocol_id": protocol["protocol_id"],
        "candidate": "M0",
        "status": "REFERENCE_FAIL",
        "mode": frozen["mode"],
        "checkpoint": str((PROJECT_DIR / frozen["checkpoint"]).resolve()),
        "checkpoint_sha256": protocol["parent_checkpoint_sha256"],
        "imgsz": frozen["imgsz"],
        "mAP50": frozen["mAP50"],
        "recall": frozen["recall"],
        "f1": frozen["f1"],
        "small_recall": frozen["small_recall"],
        "medium_recall": 0.9933333333333333,
        "large_recall": 1.0,
        "micro_gate_passed": False,
        "test_evaluated": False,
        **source_hashes(protocol),
    }
    atomic_json(root / "result.json", result)
    completed_marker(
        root,
        inputs=[
            PROJECT_DIR / protocol["parent_checkpoint"],
            PROJECT_DIR / protocol["micro_dataset"]["source_manifest"],
        ],
        outputs=[root / "result.json"],
        extra={"stage": "micro_reference", "candidate": "M0"},
    )
    return result


def training_arguments(
    candidate: str, config: dict[str, Any], dataset: Path, root: Path
) -> dict[str, Any]:
    return {
        "data": str(dataset),
        "task": "detect",
        "imgsz": int(config["imgsz"]),
        "epochs": int(config["epochs"]),
        "patience": int(config["epochs"]),
        "batch": int(config["batch"]),
        "device": 0,
        "workers": 2,
        "optimizer": "AdamW",
        "lr0": 3e-4,
        "lrf": 0.1,
        "weight_decay": 5e-4,
        "pretrained": True,
        "amp": bool(config["amp"]),
        "cos_lr": True,
        "freeze": 0,
        "seed": int(config["seed"]),
        "deterministic": True,
        "mosaic": float(config["mosaic"]),
        "mixup": float(config["mixup"]),
        "translate": 0.0,
        "scale": 0.0,
        "perspective": 0.0,
        "degrees": 0.0,
        "shear": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "erasing": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "cache": False,
        "val": True,
        "plots": True,
        "save": True,
        "save_period": 10,
        "project": str(root / "training"),
        "name": candidate,
        "exist_ok": False,
        "verbose": True,
    }


def train(candidate: str, protocol: dict[str, Any], root: Path) -> Path:
    config = protocol["micro_candidates"][candidate]
    run = root / "training" / candidate
    marker = run / "TRAINING_COMPLETE"
    best = run / "weights/best.pt"
    if marker.is_file() and best.is_file():
        return best
    last = run / "weights/last.pt"
    model = YOLO(
        str(last) if last.is_file()
        else str((PROJECT_DIR / config["model"]).resolve())
    )
    logger = GradientLogger(root / "gradient_metrics.csv")
    model.add_callback("on_train_start", logger.train_start)
    model.add_callback("on_train_epoch_start", logger.epoch_start)
    model.add_callback("on_train_batch_end", logger.before_zero_grad)
    model.add_callback("on_train_epoch_end", logger.epoch_end)
    if last.is_file():
        model.train(resume=True)
    else:
        arguments = training_arguments(
            candidate,
            config,
            PROJECT_DIR / "outputs/rescue/micro_overfit/dataset/data.yaml",
            root,
        )
        (root / "resolved_config.yaml").write_text(
            yaml.safe_dump(arguments, sort_keys=False), encoding="utf-8"
        )
        model.train(**arguments)
    if not best.is_file():
        raise FileNotFoundError(best)
    marker.write_text(
        f"checkpoint={best.resolve()}\nsha256={sha256(best)}\n",
        encoding="utf-8",
    )
    return best


def evaluate(
    candidate: str, protocol: dict[str, Any], root: Path, checkpoint: Path
) -> Path:
    output = root / "evaluation"
    summary = output / "baseline_rescue_summary.json"
    if not summary.is_file():
        subprocess.run([
            str(PYTHON),
            "baseline_rescue_audit.py",
            "--model", str(checkpoint),
            "--data", str(
                PROJECT_DIR / "outputs/rescue/micro_overfit/dataset/data.yaml"
            ),
            "--manifest", str(
                PROJECT_DIR / protocol["micro_dataset"]["source_manifest"]
            ),
            "--output", str(output),
            "--splits", "train,val",
            "--imgsz", str(protocol["micro_candidates"][candidate]["imgsz"]),
        ], cwd=PROJECT_DIR, check=True)
    return summary


def result_from_outputs(
    candidate: str, protocol: dict[str, Any], root: Path, checkpoint: Path
) -> dict[str, Any]:
    metrics = pd.read_csv(root / "evaluation/clean_metrics_train_val_test.csv")
    detail = pd.read_csv(root / "evaluation/clean_metrics_by_class_and_size.csv")
    val = metrics[
        (metrics["split"] == "val")
        & (metrics["operating_point"] == "safety")
    ].iloc[0]
    size = detail[
        (detail["split"] == "val")
        & (detail["operating_point"] == "safety")
        & (detail["scope"] == "size")
    ].set_index("name")
    history = pd.read_csv(root / "training" / candidate / "results.csv")
    history.columns = [column.strip() for column in history.columns]
    loss_columns = [column for column in history if column.startswith("train/")]
    initial_loss = float(history.iloc[0][loss_columns].sum())
    final_loss = float(history.iloc[-1][loss_columns].sum())
    gradient = pd.read_csv(root / "gradient_metrics.csv")
    nonzero_gradient = bool(
        (gradient["backbone_gradient_norm"].astype(float) > 0).any()
        and (gradient["head_gradient_norm"].astype(float) > 0).any()
    )
    finite = bool(
        math.isfinite(initial_loss)
        and math.isfinite(final_loss)
        and not gradient["nan_or_inf"].astype(bool).any()
    )
    gate = protocol["micro_gate"]
    checks = {
        "mAP50": float(val["mAP50"]) >= float(gate["map50_min"]),
        "recall": float(val["recall"]) >= float(gate["recall_min"]),
        "small_recall": float(size.loc["small", "recall"])
        >= float(gate["small_recall_min"]),
        "medium_recall": float(size.loc["medium", "recall"])
        >= float(gate["medium_recall_min"]),
        "large_recall": float(size.loc["large", "recall"])
        >= float(gate["large_recall_min"]),
        "loss_decreased": final_loss < initial_loss,
        "finite": finite,
        "nonzero_gradient": nonzero_gradient,
    }
    return {
        "protocol_id": protocol["protocol_id"],
        "candidate": candidate,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "micro_gate_passed": all(checks.values()),
        "gate_checks": checks,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "imgsz": int(protocol["micro_candidates"][candidate]["imgsz"]),
        "seed": int(protocol["micro_candidates"][candidate]["seed"]),
        "mAP50": float(val["mAP50"]),
        "mAP50_95": float(val["mAP50-95"]),
        "precision": float(val["precision"]),
        "recall": float(val["recall"]),
        "f1": float(val["f1"]),
        "small_recall": float(size.loc["small", "recall"]),
        "medium_recall": float(size.loc["medium", "recall"]),
        "large_recall": float(size.loc["large", "recall"]),
        "initial_train_loss_sum": initial_loss,
        "final_train_loss_sum": final_loss,
        "nonzero_gradient_observed": nonzero_gradient,
        "gradient_nan_or_inf": bool(
            gradient["nan_or_inf"].astype(bool).any()
        ),
        "test_evaluated": False,
        **source_hashes(protocol),
    }


def run(candidate: str) -> dict[str, Any]:
    assert_role_allowed("micro")
    protocol = load_protocol()
    if candidate == "M0":
        return register_m0(protocol)
    if candidate not in {"M1", "M2"}:
        raise RuntimeError(
            f"{candidate} requires the crop/tiling runner, not this full-frame runner"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rescue-v2 micro training")
    root = OUTPUT_ROOT / candidate
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "environment.json", environment_snapshot())
    checkpoint = train(candidate, protocol, root)
    evaluate(candidate, protocol, root, checkpoint)
    result = result_from_outputs(candidate, protocol, root, checkpoint)
    atomic_json(root / "result.json", result)
    if (root / "training" / candidate / "results.png").is_file():
        shutil.copy2(
            root / "training" / candidate / "results.png",
            root / "training_curves.png",
        )
    completed_marker(
        root,
        inputs=[
            PROJECT_DIR / protocol["micro_dataset"]["source_manifest"],
            PROJECT_DIR / protocol["micro_dataset"]["source_selection"],
            PROJECT_DIR
            / "configs/rescue_v2/canonical_v2_small_signal_rescue_v2.yaml",
        ],
        outputs=[
            root / "result.json",
            root / "gradient_metrics.csv",
            root / "evaluation/threshold_selection.json",
            checkpoint,
        ],
        extra={
            "stage": "micro_candidate",
            "candidate": candidate,
            "scientific_status": result["status"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        },
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=["M0", "M1", "M2"])
    arguments = parser.parse_args()
    run(arguments.candidate)


if __name__ == "__main__":
    main()
