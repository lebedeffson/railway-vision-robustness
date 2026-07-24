from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from scripts.audit_evaluator import average_precision, match_dataset
from scripts.canonical_m4_trainer import DifferentialLRDetectionTrainer
from scripts.person_v3.evaluate import (
    atomic_csv,
    detections_frame,
    infer_all,
    person_gt,
    reparse,
)
from scripts.person_v3.run_pipeline import size_recall
from scripts.person_v4.common import atomic_json, now, sha256
from scripts.person_v4.proxy import classify_proxy_gate
from scripts.person_v4.swad import average_state_dicts, state_dict_sha256
from scripts.person_v4.train import EpochTrajectory
from scripts.person_v4.trainer import PersonDGTrainer
from scripts.run_micro_overfit import GradientLogger


PROJECT = Path(__file__).resolve().parents[2]
PROXY_CONFIG = PROJECT / "configs/person_v4_train_only_proxy.yaml"
RUNTIME_CONFIG = (
    PROJECT / "configs/person_v4_train_only_proxy_runtime.yaml"
)
PROTOCOL_ROOT = PROJECT / "protocols/person_v4_train_only_proxy_v1"
LOCK = PROTOCOL_ROOT / "protocol_lock.json"
FRAME_MANIFEST = PROTOCOL_ROOT / "proxy_frame_manifest.csv"
TILE_MANIFEST = PROTOCOL_ROOT / "proxy_tile_manifest.csv"
DEVELOPMENT_MANIFEST = (
    PROJECT / "outputs/person_v3/protocol/development_manifest.csv"
)
IMAGE_ROOT = (
    PROJECT / "outputs/person_v3/dataset/images/development"
)
LABEL_ROOT = (
    PROJECT / "outputs/person_v3/dataset/labels/development"
)
INITIALIZATION = PROJECT / "yolo11m.pt"
OUTPUT = PROJECT / "outputs/person_v4_proxy"
TEST_MARKER = PROJECT / "outputs/person_v3/test/TEST_OPENED.json"
EXPEDITED_SELECTION = (
    PROJECT / "outputs/person_v4_expedited/expedited_selection.json"
)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def assert_runtime_isolation() -> dict[str, Any]:
    proxy = load_yaml(PROXY_CONFIG)
    runtime = load_yaml(RUNTIME_CONFIG)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if TEST_MARKER.exists():
        raise RuntimeError("Proxy cannot run after test has been opened")
    selection = json.loads(EXPEDITED_SELECTION.read_text(encoding="utf-8"))
    if selection["status"] != "NO_EXPEDITED_WINNER":
        raise RuntimeError("Proxy is allowed only after external A3 failure")
    if lock["protocol_sha256"] != sha256(PROXY_CONFIG):
        raise RuntimeError("Proxy protocol differs from runtime lock")
    for relative, expected in lock["code_sha256"].items():
        if sha256(PROJECT / relative) != expected:
            raise RuntimeError(f"Proxy locked implementation changed: {relative}")
    forbidden = set(
        runtime["isolation"]["forbidden_outer_fold0_scenes"]
    )
    frames = pd.read_csv(FRAME_MANIFEST)
    tiles = pd.read_csv(TILE_MANIFEST)
    if set(frames["grouped_scene_id"].astype(str)) & forbidden:
        raise RuntimeError("Proxy frame manifest includes external fold0")
    if set(tiles["grouped_scene_id"].astype(str)) & forbidden:
        raise RuntimeError("Proxy tile manifest includes external fold0")
    if frames["proxy_split"].nunique() != 3:
        raise RuntimeError("Proxy split count differs from frozen protocol")
    if len(frames) != 2044 or len(tiles) != 8176:
        raise RuntimeError("Proxy frozen manifest cardinality changed")
    return {
        "proxy": proxy,
        "runtime": runtime,
        "lock": lock,
        "frames": frames,
        "tiles": tiles,
    }


def prepare_datasets(contract: dict[str, Any]) -> dict[str, Path]:
    datasets: dict[str, Path] = {}
    tiles = contract["tiles"]
    destination = OUTPUT / "datasets"
    destination.mkdir(parents=True, exist_ok=True)
    for split in contract["runtime"]["execution"]["splits"]:
        root = destination / split
        root.mkdir(parents=True, exist_ok=True)
        frame = tiles[tiles["proxy_split"].eq(split)]
        roles: dict[str, list[str]] = {}
        for role in ("source", "heldout"):
            paths = []
            for stem in frame.loc[
                frame["role"].eq(role), "tile_stem"
            ].astype(str):
                image = IMAGE_ROOT / f"{stem}.jpg"
                label = LABEL_ROOT / f"{stem}.txt"
                if not image.is_file() or not label.is_file():
                    raise RuntimeError(f"Proxy tile pair is missing: {stem}")
                paths.append(str(image.resolve()))
            if not paths:
                raise RuntimeError(f"Proxy {split} has empty {role} role")
            roles[role] = sorted(paths)
            (root / f"{role}.txt").write_text(
                "\n".join(roles[role]) + "\n", encoding="utf-8"
            )
        data = {
            "path": str(
                (PROJECT / "outputs/person_v3/dataset").resolve()
            ),
            "train": str((root / "source.txt").resolve()),
            "val": str((root / "heldout.txt").resolve()),
            "nc": 1,
            "names": {0: "person"},
        }
        data_path = root / "data.yaml"
        data_path.write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
        datasets[split] = data_path
    return datasets


def scene_mapping(
    split: str, contract: dict[str, Any]
) -> tuple[dict[str, str], list[str]]:
    frame = contract["tiles"]
    frame = frame[
        frame["proxy_split"].eq(split) & frame["role"].eq("source")
    ]
    mapping = dict(
        zip(
            frame["tile_stem"].astype(str),
            frame["grouped_scene_id"].astype(str),
        )
    )
    scenes = sorted(set(mapping.values()))
    if len(scenes) != 4:
        raise RuntimeError(f"Proxy {split} does not have four source scenes")
    return mapping, scenes


def proxy_nwd_constant(
    split: str, contract: dict[str, Any], input_size: int
) -> tuple[float, dict[str, Any]]:
    frame = contract["tiles"]
    stems = frame.loc[
        frame["proxy_split"].eq(split) & frame["role"].eq("source"),
        "tile_stem",
    ].astype(str)
    gain = min(input_size / 2350.0, input_size / 1431.0)
    values = []
    for stem in sorted(set(stems)):
        for line in (LABEL_ROOT / f"{stem}.txt").read_text().splitlines():
            if not line.strip():
                continue
            class_id, _, _, width, height = map(float, line.split())
            if int(class_id) != 0:
                raise RuntimeError("Proxy person label contains another class")
            values.append(
                math.sqrt(
                    width * 2350.0 * gain * height * 1431.0 * gain
                )
            )
    if not values:
        raise RuntimeError(f"Proxy {split} source has no person boxes")
    constant = max(float(np.median(values)), 1.0)
    return constant, {
        "rule": "median_sqrt_area_after_letterbox_proxy_source_only",
        "split": split,
        "person_box_count": len(values),
        "constant_px": constant,
        "test_used": False,
    }


def a3_config(
    split: str, run_root: Path, contract: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str], list[str], dict[str, Any]]:
    parent = load_yaml(
        PROJECT / "configs/canonical_v4_person_dg_nwd.yaml"
    )
    runtime = contract["runtime"]
    mapping, scenes = scene_mapping(split, contract)
    constant, audit = proxy_nwd_constant(
        split, contract, int(runtime["training"]["input_size"])
    )
    nwd = parent["nwd"]
    config = {
        "variant": "A3",
        "split": split,
        "seed": int(runtime["training"]["seed"]),
        "scene_sampler": True,
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
            "beta": float(parent["quality_focal_loss"]["beta"]),
            "iou_weight": float(
                parent["quality_focal_loss"]["quality_target"][
                    "IoU_weight"
                ]
            ),
            "nwd_weight": float(
                parent["quality_focal_loss"]["quality_target"][
                    "NWD_weight"
                ]
            ),
        },
        "group_dro": {
            "enabled": True,
            "eta": float(parent["group_dro"]["eta"]),
        },
        "mixstyle": {
            "enabled": True,
            "probability": float(parent["mixstyle"]["probability"]),
            "beta_alpha": float(parent["mixstyle"]["beta_alpha"]),
            "epsilon": float(parent["mixstyle"]["epsilon"]),
            "layer_indices": list(
                parent["mixstyle"]["backbone_layer_indices"]
            ),
        },
        "scale_aware": {
            "enabled": True,
            "probability": float(
                parent["scale_aware_augmentation"]["probability"]
            ),
            "zoom_min": float(
                parent["scale_aware_augmentation"]["zoom_min"]
            ),
            "zoom_max": float(
                parent["scale_aware_augmentation"]["zoom_max"]
            ),
            "minimum_box_px": float(
                parent["scale_aware_augmentation"][
                    "minimum_box_after_transform_px"
                ]
            ),
        },
        "swad": {"enabled": True},
        "root": str(run_root.resolve()),
    }
    return config, mapping, scenes, audit


def stage_checkpoint(
    split: str,
    variant: str,
    data: Path,
    root: Path,
    stage_name: str,
    initialization: Path,
    stage: dict[str, Any],
    contract: dict[str, Any],
    a3: tuple[dict, dict, list, dict] | None,
) -> Path:
    run = root / stage_name
    marker = run / "TRAINING_COMPLETE.json"
    best = run / "weights/best.pt"
    last = run / "weights/last.pt"
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        selected = Path(payload["selected"])
        if selected.is_file() and sha256(selected) == payload["selected_sha256"]:
            return selected
        raise RuntimeError(f"Proxy stage marker is stale: {run}")
    runtime = contract["runtime"]["training"]
    trainer: type = DifferentialLRDetectionTrainer
    augmentations = {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    }
    recorder = None
    if variant == "A3":
        assert a3 is not None
        config, mapping, scenes, _ = a3
        trainer = PersonDGTrainer
        PersonDGTrainer.experiment_config = dict(config)
        PersonDGTrainer.scene_by_stem = dict(mapping)
        PersonDGTrainer.scene_names = list(scenes)
        PersonDGTrainer.dro_log_path = root / f"{stage_name}_group_dro.csv"
        PersonDGTrainer.augmentation_log = []
        parent = load_yaml(
            PROJECT / "configs/canonical_v4_person_dg_nwd.yaml"
        )
        augmentations.update(
            parent["scale_aware_augmentation"]["photometric"]
        )
        if stage_name == "stage3":
            recorder = EpochTrajectory(run / "trajectory")
    trainer.backbone_lr = float(stage["backbone_lr"])
    trainer.head_lr = float(stage["head_lr"])
    logger = GradientLogger(root / f"{stage_name}_gradient_metrics.csv")
    model = YOLO(str(last if last.is_file() else initialization))
    model.add_callback("on_train_start", logger.train_start)
    model.add_callback("on_train_epoch_start", logger.epoch_start)
    model.add_callback("on_train_batch_end", logger.before_zero_grad)
    model.add_callback("on_train_epoch_end", logger.epoch_end)
    if recorder is not None:
        model.add_callback("on_fit_epoch_end", recorder)
    if last.is_file():
        model.train(resume=True, trainer=trainer)
    else:
        model.train(
            trainer=trainer,
            data=str(data.resolve()),
            task="detect",
            imgsz=int(runtime["input_size"]),
            epochs=int(stage["epochs"]),
            patience=int(stage["epochs"]),
            batch=int(runtime["batch"]),
            nbs=int(runtime["nominal_batch_size"]),
            device=0,
            workers=int(runtime["workers"]),
            optimizer=str(runtime["optimizer"]),
            lr0=float(stage["head_lr"]),
            lrf=0.10,
            weight_decay=float(runtime["weight_decay"]),
            pretrained=True,
            amp=bool(runtime["amp"]),
            cos_lr=True,
            freeze=int(stage["freeze_layers"]),
            seed=int(runtime["seed"]),
            deterministic=bool(runtime["deterministic"]),
            cache=False,
            val=True,
            plots=True,
            save=True,
            save_period=1,
            close_mosaic=0,
            project=str(root.resolve()),
            name=stage_name,
            exist_ok=True,
            verbose=True,
            **augmentations,
        )
    if not best.is_file() or not last.is_file():
        raise RuntimeError(f"Proxy stage checkpoints are missing: {run}")
    selected = best
    if variant == "A3" and stage_name == "stage3":
        stage2 = root / "stage2/weights/best.pt"
        trajectory = run / "trajectory"
        states = []
        for checkpoint in (
            stage2,
            trajectory / "epoch_001.pt",
            trajectory / "epoch_002.pt",
        ):
            if checkpoint == stage2:
                payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
                model_state = payload.get("ema") or payload["model"]
                states.append(
                    {
                        key: value.detach().cpu()
                        for key, value in model_state.float().state_dict().items()
                    }
                )
            else:
                states.append(
                    torch.load(
                        checkpoint, map_location="cpu", weights_only=True
                    )
                )
        averaged = average_state_dicts(states)
        checkpoint = torch.load(
            best, map_location="cpu", weights_only=False
        )
        target = checkpoint.get("ema") or checkpoint["model"]
        target.load_state_dict(averaged, strict=True)
        checkpoint["model"] = copy.deepcopy(target).half()
        checkpoint["ema"] = copy.deepcopy(target).half()
        checkpoint["optimizer"] = None
        selected = run / "weights/swad.pt"
        torch.save(checkpoint, selected)
        atomic_json(
            run / "SWAD_SELECTION.json",
            {
                "status": "PASS",
                "states": [
                    "stage2_best",
                    "stage3_epoch_1_EMA",
                    "stage3_epoch_2_EMA",
                ],
                "state_sha256": state_dict_sha256(averaged),
                "output_sha256": sha256(selected),
                "test_used": False,
            },
        )
    payload = {
        "status": "PASS",
        "protocol_id": contract["runtime"]["protocol_id"],
        "split": split,
        "variant": variant,
        "stage": stage_name,
        "finished_at": now(),
        "best": str(best.resolve()),
        "best_sha256": sha256(best),
        "last": str(last.resolve()),
        "last_sha256": sha256(last),
        "selected": str(selected.resolve()),
        "selected_sha256": sha256(selected),
        "test_used": False,
    }
    atomic_json(marker, payload)
    return selected


def train_variant(
    split: str,
    variant: str,
    data: Path,
    contract: dict[str, Any],
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Proxy training requires CUDA")
    root = OUTPUT / f"runs/{split}/{variant}"
    root.mkdir(parents=True, exist_ok=True)
    a3 = a3_config(split, root, contract) if variant == "A3" else None
    if a3 is not None:
        atomic_json(root / "nwd_constant.json", a3[3])
    current = INITIALIZATION
    stages = contract["runtime"]["training"]["stages"]
    result = {}
    for stage_name in ("stage1", "stage2", "stage3"):
        current = stage_checkpoint(
            split,
            variant,
            data,
            root,
            stage_name,
            current,
            stages[stage_name],
            contract,
            a3,
        )
        result[stage_name] = {
            "selected": str(current.resolve()),
            "selected_sha256": sha256(current),
        }
    atomic_json(
        root / "TRAINING_COMPLETE.json",
        {
            "status": "PASS",
            "protocol_id": contract["runtime"]["protocol_id"],
            "split": split,
            "variant": variant,
            "final_checkpoint": str(current.resolve()),
            "final_checkpoint_sha256": sha256(current),
            "stages": result,
            "test_used": False,
        },
    )
    return current


def heldout_source(
    split: str, contract: dict[str, Any]
) -> pd.DataFrame:
    frozen = contract["frames"]
    frozen = frozen[
        frozen["proxy_split"].eq(split)
        & frozen["role"].eq("heldout")
    ]
    stems = set(frozen["frame_stem"].astype(str))
    allowed_scenes = set(frozen["grouped_scene_id"].astype(str))
    source = pd.read_csv(DEVELOPMENT_MANIFEST)
    source["_stem"] = source["output_image"].map(
        lambda value: Path(value).stem
    )
    source = source[source["_stem"].isin(stems)].drop(columns=["_stem"])
    if len(source) != len(stems):
        raise RuntimeError(f"Proxy {split} heldout frame mapping is incomplete")
    if set(source["grouped_scene_id"].astype(str)) != allowed_scenes:
        raise RuntimeError(f"Proxy {split} heldout scene mapping changed")
    return source.sort_values(
        ["grouped_scene_id", "subsequence_id", "frame_id"]
    ).reset_index(drop=True)


def evaluate_variant(
    split: str,
    variant: str,
    checkpoint: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    root = OUTPUT / f"runs/{split}/{variant}/evaluation"
    result_path = root / "evaluation_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["checkpoint_sha256"] != sha256(checkpoint):
            raise RuntimeError("Proxy evaluation checkpoint changed")
        return result
    root.mkdir(parents=True, exist_ok=True)
    source = heldout_source(split, contract)
    gt = person_gt(source)
    split_index = int(split.rsplit("_", 1)[1])
    predictions, runtime_ms = infer_all(
        100 + split_index, checkpoint, source, root
    )
    detections = detections_frame(source, gt, predictions)
    atomic_csv(detections, root / "predictions_and_ground_truth.csv")
    map50 = average_precision(gt, predictions, 0, 0.50)
    map5095 = float(
        np.mean(
            [
                average_precision(gt, predictions, 0, threshold)
                for threshold in np.arange(0.50, 0.96, 0.05)
            ]
        )
    )
    reparsed = reparse(
        pd.read_csv(root / "predictions_and_ground_truth.csv")
    )
    repeated = average_precision(*reparsed, 0, 0.50)
    difference = abs(map50 - repeated)
    tolerance = float(
        contract["runtime"]["evaluation"][
            "independent_reparse_tolerance"
        ]
    )
    result = {
        "status": "PASS" if difference <= tolerance else "FAIL",
        "protocol_id": contract["runtime"]["protocol_id"],
        "split": split,
        "variant": variant,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "frames": len(source),
        "scenes": int(source["grouped_scene_id"].nunique()),
        "person_GT": int(sum(len(values) for values in gt.values())),
        "mAP50": float(map50),
        "mAP50_95": map5095,
        "mean_runtime_ms": runtime_ms / max(len(source), 1),
        "evaluator_consistency": (
            "PASS" if difference <= tolerance else "FAIL"
        ),
        "lost_GT": int(
            sum(len(values) for values in gt.values())
            - detections["kind"].eq("ground_truth").sum()
        ),
        "missing_scenes": 0,
        "no_nan_inf": bool(np.isfinite([map50, map5095]).all()),
        "test_used": False,
        "article_evidence": False,
    }
    atomic_json(result_path, result)
    if result["status"] != "PASS":
        raise RuntimeError(f"Proxy evaluator failed: {split}/{variant}")
    return result


def threshold_sweep(
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any],
) -> pd.DataFrame:
    start, stop, step = contract["runtime"]["evaluation"][
        "confidence_sweep"
    ]
    return pd.DataFrame(
        [
            {
                "threshold": round(float(value), 6),
                **match_dataset(gt, predictions, float(value), 0.50),
            }
            for value in np.arange(
                float(start), float(stop) + float(step) / 2, float(step)
            )
        ]
    )


def finalize(contract: dict[str, Any]) -> dict[str, Any]:
    destination = OUTPUT / "results"
    destination.mkdir(parents=True, exist_ok=True)
    variant_frames: dict[str, list[pd.DataFrame]] = {"A0": [], "A3": []}
    evaluation: dict[tuple[str, str], dict[str, Any]] = {}
    for split, variant in contract["runtime"]["execution"]["order"]:
        path = OUTPUT / f"runs/{split}/{variant}/evaluation"
        variant_frames[variant].append(
            pd.read_csv(path / "predictions_and_ground_truth.csv")
        )
        evaluation[(split, variant)] = json.loads(
            (path / "evaluation_result.json").read_text(encoding="utf-8")
        )
    selected_thresholds = {}
    per_variant_split: dict[tuple[str, str], dict[str, Any]] = {}
    for variant, frames in variant_frames.items():
        merged = pd.concat(frames, ignore_index=True)
        gt, predictions = reparse(merged)
        sweep = threshold_sweep(gt, predictions, contract)
        selected = sweep.sort_values(
            ["f1", "recall", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
        threshold = float(selected["threshold"])
        selected_thresholds[variant] = threshold
        sweep.to_csv(
            destination / f"{variant}_threshold_sweep.csv", index=False
        )
        for split in contract["runtime"]["execution"]["splits"]:
            frame = pd.read_csv(
                OUTPUT
                / f"runs/{split}/{variant}/evaluation/"
                "predictions_and_ground_truth.csv"
            )
            split_gt, split_predictions = reparse(frame)
            metrics = match_dataset(
                split_gt, split_predictions, threshold, 0.50
            )
            metrics.update(
                size_recall(split_gt, split_predictions, threshold)
            )
            per_variant_split[(split, variant)] = {
                "proxy_split": split,
                "variant": variant,
                "threshold": threshold,
                "mAP50": evaluation[(split, variant)]["mAP50"],
                **metrics,
                "evaluator_consistency": evaluation[
                    (split, variant)
                ]["evaluator_consistency"],
                "lost_GT": evaluation[(split, variant)]["lost_GT"],
                "NaN_Inf": int(
                    not evaluation[(split, variant)]["no_nan_inf"]
                ),
            }
    rows = []
    paired = []
    for split in contract["runtime"]["execution"]["splits"]:
        baseline = per_variant_split[(split, "A0")]
        candidate = per_variant_split[(split, "A3")]
        rows.extend((baseline, candidate))
        paired.append(
            {
                "proxy_split": split,
                "delta_mAP50": candidate["mAP50"] - baseline["mAP50"],
                "delta_recall": candidate["recall"] - baseline["recall"],
                "delta_small_recall": (
                    candidate["small_recall"]
                    - baseline["small_recall"]
                ),
                "NaN_Inf": max(
                    baseline["NaN_Inf"], candidate["NaN_Inf"]
                ),
                "lost_GT": max(
                    baseline["lost_GT"], candidate["lost_GT"]
                ),
                "evaluator_consistency": (
                    "PASS"
                    if baseline["evaluator_consistency"] == "PASS"
                    and candidate["evaluator_consistency"] == "PASS"
                    else "FAIL"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        destination / "per_split_metrics.csv", index=False
    )
    pd.DataFrame(paired).to_csv(
        destination / "paired_proxy_metrics.csv", index=False
    )
    gate = classify_proxy_gate(
        paired, contract["proxy"]["proxy_gate"]["require_all"]
    )
    gate.update(
        {
            "protocol_id": contract["runtime"]["protocol_id"],
            "thresholds": selected_thresholds,
            "external_A3_fold0_result": "FAIL",
            "interpretation": (
                "proxy_can_reject_this_failed_idea"
                if gate["status"] == "FAIL"
                else "proxy_is_too_easy_and_must_not_gate_future_runs"
            ),
            "test_opened": False,
        }
    )
    atomic_json(destination / "proxy_gate.json", gate)
    atomic_json(
        PROTOCOL_ROOT / "execution_status.json",
        {
            "status": "COMPLETE",
            "GPU_proxy_started": True,
            "active_A3_interrupted": False,
            "external_A3_fold0_result": "FAIL",
            "proxy_result": gate["status"],
            "test_opened": False,
            "article_evidence": False,
        },
    )
    return gate


def run() -> dict[str, Any]:
    contract = assert_runtime_isolation()
    datasets = prepare_datasets(contract)
    atomic_json(
        OUTPUT / "runtime_contract.json",
        {
            "status": "PASS",
            "protocol_id": contract["runtime"]["protocol_id"],
            "proxy_protocol_sha256": sha256(PROXY_CONFIG),
            "runtime_protocol_sha256": sha256(RUNTIME_CONFIG),
            "runtime_lock_sha256": sha256(LOCK),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT,
                text=True,
            ).strip(),
            "test_opened": False,
            "article_evidence": False,
        },
    )
    for split, variant in contract["runtime"]["execution"]["order"]:
        checkpoint = train_variant(
            split, variant, datasets[split], contract
        )
        evaluate_variant(split, variant, checkpoint, contract)
    return finalize(contract)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only", action="store_true", help="verify isolation only"
    )
    args = parser.parse_args()
    contract = assert_runtime_isolation()
    if args.check_only:
        datasets = prepare_datasets(contract)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "datasets": {
                        key: str(value) for key, value in datasets.items()
                    },
                    "test_opened": False,
                },
                indent=2,
            )
        )
        return
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
