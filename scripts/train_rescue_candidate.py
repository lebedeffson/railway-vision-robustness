from __future__ import annotations

import argparse
import json
import os
import shutil
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
    sha256,
)
from run_micro_overfit import GradientLogger


RUNS = OUTPUT_ROOT / "runs"
RESCUE_DATA = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/data.yaml"
RESCUE_MANIFEST = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"
EXECUTION_POLICY = PROJECT_DIR / "configs/rescue/candidate_execution_policy.yaml"


def candidate_augmentations(name: str, mosaic: float) -> dict[str, float]:
    if name == "minimal":
        return {
            "hsv_h": 0.005, "hsv_s": 0.20, "hsv_v": 0.15,
            "translate": 0.02, "scale": 0.10, "perspective": 0.0,
            "fliplr": 0.5, "mosaic": mosaic, "mixup": 0.0,
        }
    return {
        "hsv_h": 0.01, "hsv_s": 0.30, "hsv_v": 0.20,
        "translate": 0.04, "scale": 0.15, "perspective": 0.0,
        "fliplr": 0.5, "mosaic": mosaic, "mixup": 0.0,
    }


def prepare_balanced_data(run_root: Path, protocol: dict[str, Any]) -> Path:
    frame = pd.read_csv(RESCUE_MANIFEST)
    train = frame[frame["split"] == "train"].copy()
    class_counts: Counter[int] = Counter()
    scene_counts = Counter(train["grouped_scene_id"].astype(str))
    frame_classes: dict[str, set[int]] = {}
    frame_small: dict[str, bool] = {}
    small_limit = float(protocol["object_sizes"]["small_area_ratio_max"])
    for row in train.itertuples(index=False):
        labels = [
            line.split() for line in Path(row.output_label).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        classes = {int(fields[0]) for fields in labels}
        for fields in labels:
            class_counts[int(fields[0])] += 1
        frame_classes[str(row.output_image)] = classes
        frame_small[str(row.output_image)] = any(
            float(fields[3]) * float(fields[4]) < small_limit for fields in labels
        )
    maximum = max(class_counts.values())
    scene_median = float(pd.Series(list(scene_counts.values())).median())
    rows = []
    image_lines = []
    maximum_weight = float(protocol["candidates"]["R4"]["sampler_max_frame_weight"])
    for row in train.itertuples(index=False):
        path = str(row.output_image)
        rare = max(
            (maximum / max(class_counts[class_id], 1)) ** 0.5
            for class_id in frame_classes[path]
        ) if frame_classes[path] else 1.0
        scene = (scene_median / max(scene_counts[str(row.grouped_scene_id)], 1)) ** 0.5
        small = 1.25 if frame_small[path] else 1.0
        weight = min(maximum_weight, max(1.0, rare * scene * small))
        repeats = max(1, int(round(weight)))
        image_lines.extend([path] * repeats)
        rows.append({
            "image_path": path, "grouped_scene_id": row.grouped_scene_id,
            "classes": "|".join(map(str, sorted(frame_classes[path]))),
            "contains_small_object": frame_small[path],
            "weight": weight, "repetitions": repeats,
        })
    weights = run_root / "sampler_weights.csv"
    pd.DataFrame(rows).to_csv(weights, index=False)
    train_list = run_root / "balanced_train.txt"
    train_list.write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    source = yaml.safe_load(RESCUE_DATA.read_text(encoding="utf-8"))
    source["train"] = str(train_list.resolve())
    balanced = run_root / "balanced_data.yaml"
    balanced.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return balanced


def train_stage(
    *, initialization: Path, data: Path, root: Path, stage: str,
    epochs: int, imgsz: int, seed: int, freeze: int, lr0: float,
    augmentations: dict[str, float], save_period: int, patience: int,
) -> Path:
    run = root / stage
    weights = run / "weights"
    best = weights / "best.pt"
    last = weights / "last.pt"
    marker = run / "TRAINING_COMPLETE"
    if marker.is_file() and best.is_file():
        return best
    model = YOLO(str(last) if last.is_file() else str(initialization))
    logger = GradientLogger(root / f"{stage}_gradient_metrics.csv")
    model.add_callback("on_train_start", logger.train_start)
    model.add_callback("on_train_epoch_start", logger.epoch_start)
    model.add_callback("on_train_batch_end", logger.before_zero_grad)
    model.add_callback("on_train_epoch_end", logger.epoch_end)
    if last.is_file():
        model.train(resume=True)
    else:
        model.train(
            data=str(data), task="detect", imgsz=imgsz, epochs=epochs,
            patience=max(patience, 15), batch=1, device=0, workers=2,
            optimizer="AdamW", lr0=lr0, lrf=0.1, weight_decay=5e-4,
            pretrained=True, amp=True, cos_lr=True, freeze=freeze,
            seed=seed, deterministic=True, cache=False, val=True, plots=True,
            save=True, save_period=save_period, close_mosaic=5,
            project=str(root), name=stage, exist_ok=False, verbose=True,
            **augmentations,
        )
    if not best.is_file():
        raise FileNotFoundError(best)
    marker.write_text(
        f"best={best.resolve()}\nsha256={sha256(best)}\n", encoding="utf-8"
    )
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("R1", "R2", "R3", "R4"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-candidate", choices=("R1", "R2", "R3"))
    args = parser.parse_args()
    ensure_branch()
    assert_test_sealed()
    protocol = load_protocol()
    execution = yaml.safe_load(EXECUTION_POLICY.read_text(encoding="utf-8"))
    candidate_name = args.candidate
    if candidate_name == "R4":
        if not args.base_candidate:
            parser.error("R4 requires --base-candidate")
        candidate = dict(protocol["candidates"][args.base_candidate])
        candidate["augmentations"] = protocol["candidates"][args.base_candidate]["augmentations"]
    else:
        candidate = protocol["candidates"][candidate_name]
    run_root = RUNS / candidate_name / f"seed_{args.seed}"
    run_root.mkdir(parents=True, exist_ok=True)
    completion = run_root / "COMPLETED.json"
    if completion.is_file():
        print(completion)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Rescue candidate training requires CUDA")
    data = prepare_balanced_data(run_root, protocol) if candidate_name == "R4" else RESCUE_DATA
    augmentations = candidate_augmentations(
        str(candidate["augmentations"]), float(candidate["mosaic"])
    )
    resolved = {
        "protocol_id": protocol["protocol_id"],
        "candidate": candidate_name, "base_candidate": args.base_candidate,
        "seed": args.seed, "imgsz": int(candidate["imgsz"]), "data": str(data.resolve()),
        "stage1": {
            "epochs": int(candidate["head_epochs"]), "freeze": int(candidate["freeze_layers"]),
            "lr0": float(candidate["lr0_head"]),
        },
        "stage2": {
            "epochs": int(candidate["finetune_epochs"]), "freeze": 0,
            "lr0": float(candidate["lr0_finetune"]),
        },
        "augmentations": augmentations,
        "checkpoint_interval_epochs": int(execution["checkpoint_interval_epochs"]),
        "test_usage": "forbidden",
    }
    (run_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    atomic_json(run_root / "environment.json", environment_snapshot())
    stage1 = train_stage(
        initialization=PROJECT_DIR / protocol["training"]["initialization"],
        data=data, root=run_root, stage="stage1",
        epochs=resolved["stage1"]["epochs"], imgsz=resolved["imgsz"], seed=args.seed,
        freeze=resolved["stage1"]["freeze"], lr0=resolved["stage1"]["lr0"],
        augmentations=augmentations,
        save_period=resolved["checkpoint_interval_epochs"],
        patience=resolved["stage1"]["epochs"],
    )
    stage2 = train_stage(
        initialization=stage1, data=data, root=run_root, stage="stage2",
        epochs=resolved["stage2"]["epochs"], imgsz=resolved["imgsz"], seed=args.seed,
        freeze=0, lr0=resolved["stage2"]["lr0"], augmentations=augmentations,
        save_period=resolved["checkpoint_interval_epochs"],
        patience=int(protocol["training"]["patience"]),
    )
    evaluation = run_root / "evaluation"
    command = [
        str(PROJECT_DIR / ".venv/bin/python"), "scripts/evaluate_rescue_candidate.py",
        "--candidate", candidate_name, "--seed", str(args.seed),
        "--checkpoint", str(stage2), "--data", str(data),
        "--manifest", str(RESCUE_MANIFEST), "--imgsz", str(resolved["imgsz"]),
        "--output", str(evaluation),
    ]
    completed = __import__("subprocess").run(command, cwd=PROJECT_DIR, check=True)
    result = json.loads((evaluation / "candidate_result.json").read_text(encoding="utf-8"))
    provenance = {
        "status": "PASS", "candidate": candidate_name, "seed": args.seed,
        "stage1_checkpoint": str(stage1.resolve()), "stage1_sha256": sha256(stage1),
        "stage2_checkpoint": str(stage2.resolve()), "stage2_sha256": sha256(stage2),
        "selection_split": "validation", "test_used": False,
        "candidate_result": result,
    }
    atomic_json(run_root / "checkpoint_provenance.json", provenance)
    completed_marker(
        run_root,
        inputs=[PROJECT_DIR / "configs/rescue/canonical_v2_rescue_v1.yaml", EXECUTION_POLICY, data],
        outputs=[
            run_root / "resolved_config.yaml", run_root / "environment.json",
            run_root / "checkpoint_provenance.json",
            evaluation / "candidate_result.json", stage1, stage2,
        ],
        extra={
            "stage": "rescue_candidate", "candidate": candidate_name,
            "seed": args.seed, "test_evaluated": False,
        },
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
