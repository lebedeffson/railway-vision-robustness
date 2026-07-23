from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from audit_evaluator import average_precision, class_aware_nms, match_dataset
from rescue_common import atomic_json, completed_marker, environment_snapshot
from rescue_v2_common import PROJECT_DIR, assert_role_allowed, load_protocol, sha256
from run_micro_candidate_v2 import OUTPUT_ROOT, source_hashes, training_arguments
from run_micro_overfit import GradientLogger


VIEW_ROOT = PROJECT_DIR / "outputs/rescue_v2/view_datasets"
SIZE_THRESHOLDS = {"small": 0.001, "medium": 0.01}


def read_yolo_labels(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, box_width, box_height = map(float, line.split())
        rows.append({
            "class_id": int(class_id),
            "box": [
                (x - box_width / 2) * width,
                (y - box_height / 2) * height,
                (x + box_width / 2) * width,
                (y + box_height / 2) * height,
            ],
        })
    return rows


def clip_label(
    label: dict[str, Any], window: tuple[int, int, int, int], minimum_visible: float
) -> dict[str, Any] | None:
    left, top, right, bottom = window
    x1, y1, x2, y2 = label["box"]
    intersection = (
        max(0.0, min(x2, right) - max(x1, left))
        * max(0.0, min(y2, bottom) - max(y1, top))
    )
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0 or intersection / area < minimum_visible:
        return None
    return {
        "class_id": label["class_id"],
        "box": [
            max(x1, left) - left,
            max(y1, top) - top,
            min(x2, right) - left,
            min(y2, bottom) - top,
        ],
    }


def write_labels(path: Path, labels: list[dict[str, Any]], width: int, height: int) -> None:
    lines = []
    for label in labels:
        x1, y1, x2, y2 = label["box"]
        lines.append(
            f"{label['class_id']} "
            f"{(x1 + x2) / (2 * width):.10f} "
            f"{(y1 + y2) / (2 * height):.10f} "
            f"{(x2 - x1) / width:.10f} "
            f"{(y2 - y1) / height:.10f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def shift_window(center: float, side: int, maximum: int) -> tuple[int, int]:
    start = int(round(center - side / 2))
    start = min(max(start, 0), max(maximum - side, 0))
    return start, min(start + side, maximum)


def crop_windows(
    labels: list[dict[str, Any]],
    width: int,
    height: int,
    config: dict[str, Any],
    target_class_id: int,
) -> list[tuple[int, int, int, int, str]]:
    windows: list[tuple[int, int, int, int, str]] = []
    if config["include_full_frame_view"]:
        windows.append((0, 0, width, height, "full"))
    for index, label in enumerate(labels):
        x1, y1, x2, y2 = label["box"]
        relative_area = (x2 - x1) * (y2 - y1) / (width * height)
        if (
            label["class_id"] != target_class_id
            or relative_area > float(config["target_small_area_ratio_max"])
        ):
            continue
        margin = float(config["context_margin_bbox_multiples"])
        side = int(math.ceil(max(
            int(config["minimum_crop_side_px"]),
            (x2 - x1) * (1 + 2 * margin),
            (y2 - y1) * (1 + 2 * margin),
        )))
        side = min(side, width, height)
        left, right = shift_window((x1 + x2) / 2, side, width)
        top, bottom = shift_window((y1 + y2) / 2, side, height)
        windows.append((left, top, right, bottom, f"signal_{index:03d}"))
    return windows


def tiled_axis(length: int, overlap: float) -> list[tuple[int, int]]:
    side = int(math.ceil(length / (2 - overlap)))
    return [(0, side), (length - side, length)]


def tile_windows(
    width: int, height: int, config: dict[str, Any]
) -> list[tuple[int, int, int, int, str]]:
    overlap = float(config["overlap_fraction"])
    return [
        (left, top, right, bottom, f"tile_{row}_{column}")
        for row, (top, bottom) in enumerate(tiled_axis(height, overlap))
        for column, (left, right) in enumerate(tiled_axis(width, overlap))
    ]


def prepare_views(candidate: str, protocol: dict[str, Any]) -> tuple[Path, Path]:
    root = VIEW_ROOT / candidate
    manifest_path = root / "view_manifest.csv"
    data_path = root / "data.yaml"
    if manifest_path.is_file() and data_path.is_file():
        return data_path, manifest_path
    source = pd.read_csv(PROJECT_DIR / protocol["micro_dataset"]["source_manifest"])
    config = protocol["micro_candidates"][candidate]
    class_names = {int(key): str(value) for key, value in protocol["class_names"].items()}
    target_class_id = next(
        class_id
        for class_id, name in class_names.items()
        if name == config.get("target_class")
    ) if candidate == "M3" else -1
    records: list[dict[str, Any]] = []
    for source_row in source.itertuples(index=False):
        image_path = Path(source_row.output_image)
        label_path = Path(source_row.output_label)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        labels = read_yolo_labels(label_path, width, height)
        windows = (
            crop_windows(labels, width, height, config, target_class_id)
            if candidate == "M3"
            else tile_windows(width, height, config)
        )
        for left, top, right, bottom, view_id in windows:
            visible = [
                clipped
                for label in labels
                if (clipped := clip_label(
                    label,
                    (left, top, right, bottom),
                    float(config["minimum_visible_fraction"]),
                )) is not None
            ]
            if not visible:
                continue
            stem = f"{image_path.stem}__{view_id}"
            for split in ("train", "val"):
                destination = root / "images" / split / f"{stem}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.crop((left, top, right, bottom)).save(destination)
                write_labels(
                    root / "labels" / split / f"{stem}.txt",
                    visible,
                    right - left,
                    bottom - top,
                )
            records.append({
                "candidate": candidate,
                "view_id": view_id,
                "view_image": str((root / "images/val" / f"{stem}.png").resolve()),
                "original_image": str(image_path.resolve()),
                "original_label": str(label_path.resolve()),
                "grouped_scene_id": source_row.grouped_scene_id,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "view_width": right - left,
                "view_height": bottom - top,
                "visible_labels": len(visible),
            })
    manifest = pd.DataFrame(records).sort_values(["original_image", "view_id"])
    if manifest.empty or manifest["original_image"].nunique() != len(source):
        raise RuntimeError(f"{candidate} view generation lost source frames")
    root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    data_path.write_text(yaml.safe_dump({
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 6,
        "names": protocol["class_names"],
    }, sort_keys=False), encoding="utf-8")
    atomic_json(root / "generation_summary.json", {
        "status": "PASS",
        "candidate": candidate,
        "source_frames": int(manifest["original_image"].nunique()),
        "views": len(manifest),
        "uses_test": False,
        "gt_derived_windows": candidate == "M3",
        "eligible_for_full_training_without_deployable_roi": (
            bool(config.get("eligible_for_full_training_without_deployable_roi", True))
        ),
    })
    return data_path, manifest_path


def train(
    candidate: str, protocol: dict[str, Any], root: Path, data_path: Path
) -> Path:
    config = protocol["micro_candidates"][candidate]
    run = root / "training" / candidate
    best = run / "weights/best.pt"
    marker = run / "TRAINING_COMPLETE"
    if marker.is_file() and best.is_file():
        return best
    last = run / "weights/last.pt"
    model = YOLO(str(last) if last.is_file() else str(PROJECT_DIR / config["model"]))
    logger = GradientLogger(root / "gradient_metrics.csv")
    model.add_callback("on_train_start", logger.train_start)
    model.add_callback("on_train_epoch_start", logger.epoch_start)
    model.add_callback("on_train_batch_end", logger.before_zero_grad)
    model.add_callback("on_train_epoch_end", logger.epoch_end)
    if last.is_file():
        model.train(resume=True)
    else:
        arguments = training_arguments(candidate, config, data_path, root)
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


def original_ground_truth(manifest: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in manifest.drop_duplicates("original_image").itertuples(index=False):
        image = Image.open(row.original_image)
        result[row.original_image] = read_yolo_labels(
            Path(row.original_label), image.width, image.height
        )
    return result


def merged_predictions(
    checkpoint: Path, manifest: pd.DataFrame, imgsz: int, nms_iou: float
) -> dict[str, list[dict[str, Any]]]:
    model = YOLO(str(checkpoint))
    merged: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest.itertuples(index=False):
        result = model.predict(
            source=row.view_image,
            imgsz=imgsz,
            conf=0.001,
            iou=0.7,
            max_det=300,
            device=0,
            verbose=False,
        )[0]
        if result.boxes is None:
            continue
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            merged[row.original_image].append({
                "class_id": int(class_id),
                "confidence": float(confidence),
                "box": [
                    float(box[0] + row.left),
                    float(box[1] + row.top),
                    float(box[2] + row.left),
                    float(box[3] + row.top),
                ],
            })
    return {
        image_id: class_aware_nms(rows, nms_iou)
        for image_id, rows in merged.items()
    }


def matched_indices(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    confidence: float,
) -> set[int]:
    active = sorted(
        [row for row in predictions if row["confidence"] >= confidence],
        key=lambda row: -row["confidence"],
    )
    used: set[int] = set()
    from audit_evaluator import box_iou
    for prediction in active:
        available = [
            (index, box_iou(prediction["box"], target["box"]))
            for index, target in enumerate(ground_truth)
            if index not in used and prediction["class_id"] == target["class_id"]
        ]
        if available:
            index, overlap = max(available, key=lambda item: item[1])
            if overlap >= 0.5:
                used.add(index)
    return used


def evaluate(
    candidate: str,
    protocol: dict[str, Any],
    root: Path,
    checkpoint: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    ground_truth = original_ground_truth(manifest)
    config = protocol["micro_candidates"][candidate]
    nms_iou = float(config["inter_crop_nms_iou"] if candidate == "M3" else config["inter_tile_nms_iou"])
    predictions = merged_predictions(
        checkpoint, manifest, int(config["imgsz"]), nms_iou
    )
    thresholds = np.round(np.arange(0.001, 0.501, 0.001), 3)
    sweep = pd.DataFrame([
        {"confidence": float(value), **match_dataset(ground_truth, predictions, float(value))}
        for value in thresholds
    ])
    sweep["f2"] = (
        5 * sweep["precision"] * sweep["recall"]
        / np.maximum(4 * sweep["precision"] + sweep["recall"], 1e-12)
    )
    standard = sweep.sort_values(
        ["f1", "recall", "confidence"], ascending=[False, False, True]
    ).iloc[0]
    safety = sweep.sort_values(
        ["f2", "recall", "confidence"], ascending=[False, False, True]
    ).iloc[0]
    classes = sorted({
        target["class_id"] for rows in ground_truth.values() for target in rows
    })
    ap50 = [average_precision(ground_truth, predictions, class_id, 0.5) for class_id in classes]
    map50 = float(np.nanmean(ap50))
    map5095 = float(np.nanmean([
        average_precision(ground_truth, predictions, class_id, threshold)
        for class_id in classes
        for threshold in np.arange(0.5, 0.96, 0.05)
    ]))
    matched = {
        image_id: matched_indices(
            ground_truth[image_id],
            predictions.get(image_id, []),
            float(safety["confidence"]),
        )
        for image_id in ground_truth
    }
    size_counts = {
        name: {"gt": 0, "tp": 0} for name in ("small", "medium", "large")
    }
    for image_id, labels in ground_truth.items():
        image = Image.open(image_id)
        for index, label in enumerate(labels):
            x1, y1, x2, y2 = label["box"]
            area = (x2 - x1) * (y2 - y1) / (image.width * image.height)
            name = "small" if area < SIZE_THRESHOLDS["small"] else (
                "medium" if area < SIZE_THRESHOLDS["medium"] else "large"
            )
            size_counts[name]["gt"] += 1
            size_counts[name]["tp"] += int(index in matched[image_id])
    detection_rows = [
        {
            "image_path": image_id,
            "kind": "prediction",
            "class_id": row["class_id"],
            "confidence": row["confidence"],
            "x1": row["box"][0],
            "y1": row["box"][1],
            "x2": row["box"][2],
            "y2": row["box"][3],
        }
        for image_id, rows in predictions.items() for row in rows
    ]
    pd.DataFrame(detection_rows).to_csv(output / "merged_predictions.csv", index=False)
    sweep.to_csv(output / "threshold_sweep.csv", index=False)
    pd.DataFrame([
        {"name": name, **values, "recall": values["tp"] / max(values["gt"], 1)}
        for name, values in size_counts.items()
    ]).to_csv(output / "metrics_by_size.csv", index=False)
    selection = {
        "selection_split": "micro_val",
        "test_used_for_selection": False,
        "standard": standard.to_dict(),
        "safety": safety.to_dict(),
        "checkpoint_sha256": sha256(checkpoint),
    }
    atomic_json(output / "threshold_selection.json", selection)
    return {
        "mAP50": map50,
        "mAP50_95": map5095,
        "precision": float(safety["precision"]),
        "recall": float(safety["recall"]),
        "f1": float(safety["f1"]),
        "small_recall": size_counts["small"]["tp"] / max(size_counts["small"]["gt"], 1),
        "medium_recall": size_counts["medium"]["tp"] / max(size_counts["medium"]["gt"], 1),
        "large_recall": size_counts["large"]["tp"] / max(size_counts["large"]["gt"], 1),
    }


def result_from_outputs(
    candidate: str,
    protocol: dict[str, Any],
    root: Path,
    checkpoint: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
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
        all(math.isfinite(float(value)) for value in metrics.values())
        and math.isfinite(initial_loss)
        and math.isfinite(final_loss)
        and not gradient["nan_or_inf"].astype(bool).any()
    )
    gate = protocol["micro_gate"]
    checks = {
        "mAP50": metrics["mAP50"] >= float(gate["map50_min"]),
        "recall": metrics["recall"] >= float(gate["recall_min"]),
        "small_recall": metrics["small_recall"] >= float(gate["small_recall_min"]),
        "medium_recall": metrics["medium_recall"] >= float(gate["medium_recall_min"]),
        "large_recall": metrics["large_recall"] >= float(gate["large_recall_min"]),
        "loss_decreased": final_loss < initial_loss,
        "finite": finite,
        "nonzero_gradient": nonzero_gradient,
    }
    config = protocol["micro_candidates"][candidate]
    return {
        "protocol_id": protocol["protocol_id"],
        "candidate": candidate,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "micro_gate_passed": all(checks.values()),
        "gate_checks": checks,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "imgsz": int(config["imgsz"]),
        "seed": int(config["seed"]),
        **metrics,
        "initial_train_loss_sum": initial_loss,
        "final_train_loss_sum": final_loss,
        "nonzero_gradient_observed": nonzero_gradient,
        "gradient_nan_or_inf": bool(gradient["nan_or_inf"].astype(bool).any()),
        "test_evaluated": False,
        "eligible_for_full_training_without_deployable_roi": bool(
            config.get("eligible_for_full_training_without_deployable_roi", True)
        ),
        **source_hashes(protocol),
    }


def run(candidate: str) -> dict[str, Any]:
    assert_role_allowed("micro")
    if candidate not in {"M3", "M4"}:
        raise ValueError(candidate)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for rescue-v2 micro training")
    protocol = load_protocol()
    root = OUTPUT_ROOT / candidate
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "environment.json", environment_snapshot())
    data_path, manifest_path = prepare_views(candidate, protocol)
    checkpoint = train(candidate, protocol, root, data_path)
    metrics_path = root / "evaluation/metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = evaluate(candidate, protocol, root, checkpoint, manifest_path)
        atomic_json(metrics_path, metrics)
    result = result_from_outputs(candidate, protocol, root, checkpoint, metrics)
    atomic_json(root / "result.json", result)
    curves = root / "training" / candidate / "results.png"
    if curves.is_file():
        shutil.copy2(curves, root / "training_curves.png")
    completed_marker(
        root,
        inputs=[
            PROJECT_DIR / protocol["micro_dataset"]["source_manifest"],
            PROJECT_DIR / protocol["micro_dataset"]["source_selection"],
            PROJECT_DIR / "configs/rescue_v2/canonical_v2_small_signal_rescue_v2.yaml",
            manifest_path,
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
    parser.add_argument("--candidate", required=True, choices=["M3", "M4"])
    args = parser.parse_args()
    run(args.candidate)


if __name__ == "__main__":
    main()
