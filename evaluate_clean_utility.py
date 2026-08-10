from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import Tensor
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.metrics import ap_per_class, box_iou

from audit_final_practice import canonical_path
from checkpoint_selection import configured_checkpoint
from evaluate_image_level_detection import (
    get_ground_truth,
    detection_metrics,
    predict_batch,
)
from extract_attack_consistency import sequence_lookup
from extract_feature_consistency import defend, loader, to_device


PROJECT_DIR = Path(__file__).resolve().parent
MODEL = configured_checkpoint()
DATA = PROJECT_DIR / "data/yolo_osdar23/data.yaml"
MANIFEST = PROJECT_DIR / "data/yolo_osdar23/manifest.csv"
OUTPUT = PROJECT_DIR / "outputs/final_practice/03_clean_utility"
DEFENSES = ["none", "tnorm", "bilateral", "gaussian", "jpeg", "median"]
IOU_LEVELS = torch.linspace(0.50, 0.95, 10)


def structural_similarity(reference: Tensor, candidate: Tensor) -> float:
    kernel = 11
    padding = kernel // 2
    mu_x = functional.avg_pool2d(reference, kernel, stride=1, padding=padding)
    mu_y = functional.avg_pool2d(candidate, kernel, stride=1, padding=padding)
    sigma_x = functional.avg_pool2d(reference.square(), kernel, 1, padding) - mu_x.square()
    sigma_y = functional.avg_pool2d(candidate.square(), kernel, 1, padding) - mu_y.square()
    sigma_xy = functional.avg_pool2d(reference * candidate, kernel, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return float(score.mean().clamp(-1, 1))


def correct_matrix(prediction: Tensor, boxes: Tensor, classes: Tensor) -> np.ndarray:
    correct = np.zeros((len(prediction), len(IOU_LEVELS)), dtype=bool)
    if not len(prediction) or not len(boxes):
        return correct
    ious = box_iou(boxes, prediction[:, :4])
    class_match = classes[:, None] == prediction[:, 5].long()[None, :]
    for threshold_index, threshold in enumerate(IOU_LEVELS.to(ious.device)):
        gt_index, prediction_index = torch.where((ious >= threshold) & class_match)
        if not len(gt_index):
            continue
        matches = np.column_stack((
            gt_index.cpu().numpy(),
            prediction_index.cpu().numpy(),
            ious[gt_index, prediction_index].detach().cpu().numpy(),
        ))
        matches = matches[np.argsort(-matches[:, 2])]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.argsort(-matches[:, 2])]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), threshold_index] = True
    return correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean filter utility with AP and SSIM")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    yolo = YOLO(str(args.model))
    model = yolo.model.to(device).float().eval()
    overrides = model.args if isinstance(model.args, dict) else vars(model.args)
    model.args = get_cfg(overrides=overrides)
    model.criterion = None
    names_source = yolo.names
    names = (
        {int(key): value for key, value in names_source.items()}
        if isinstance(names_source, dict)
        else {index: value for index, value in enumerate(names_source)}
    )
    exact_sequences, named_sequences = sequence_lookup(args.manifest)
    metric_parts = {
        name: {"tp": [], "confidence": [], "predicted": [], "target": []}
        for name in DEFENSES
    }
    rows: list[dict[str, object]] = []
    data_loader = loader(args.data, args.split, args.imgsz, 1, 0, device.type == "cuda")
    for raw_batch in tqdm(data_loader, desc="clean utility"):
        batch = to_device(raw_batch, device)
        clean = batch["img"].detach()
        path = str(batch["im_file"][0])
        sequence_id = exact_sequences.get(
            canonical_path(path), named_sequences.get(Path(path).name)
        )
        if sequence_id is None:
            raise RuntimeError(f"No sequence_id for {path}")
        boxes, classes = get_ground_truth(batch, 0)
        for defense_name in DEFENSES:
            images = clean if defense_name == "none" else defend(clean, defense_name)
            prediction = predict_batch(model, images)[0]
            fixed = detection_metrics(prediction, boxes, classes, args.confidence)
            rows.append({
                "sequence_id": sequence_id,
                "image_path": path,
                "split": args.split,
                "defense": defense_name,
                "precision": fixed["precision"],
                "recall": fixed["recall"],
                "f1": fixed["f1"],
                "tp": fixed["tp"],
                "fp": fixed["fp"],
                "fn": fixed["fn"],
                "ssim": 1.0 if defense_name == "none" else structural_similarity(clean, images),
            })
            parts = metric_parts[defense_name]
            parts["tp"].append(correct_matrix(prediction, boxes, classes))
            parts["confidence"].append(prediction[:, 4].detach().cpu().numpy())
            parts["predicted"].append(prediction[:, 5].detach().cpu().numpy())
            parts["target"].append(classes.detach().cpu().numpy())

    args.output.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output / "clean_utility_raw.csv", index=False)
    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for defense_name in DEFENSES:
        parts = metric_parts[defense_name]
        tp = np.concatenate(parts["tp"], axis=0)
        confidence = np.concatenate(parts["confidence"])
        predicted = np.concatenate(parts["predicted"])
        target = np.concatenate(parts["target"])
        result = ap_per_class(tp, confidence, predicted, target, names=names)
        precision, recall, f1, ap, unique_classes = result[2:7]
        scope = raw[raw["defense"] == defense_name]
        total_tp, total_fp, total_fn = scope[["tp", "fp", "fn"]].sum()
        fixed_precision = total_tp / max(total_tp + total_fp, 1)
        fixed_recall = total_tp / max(total_tp + total_fn, 1)
        fixed_f1 = 2 * fixed_precision * fixed_recall / max(fixed_precision + fixed_recall, 1e-12)
        summary_rows.append({
            "defense": defense_name,
            "mAP50": float(ap[:, 0].mean()),
            "mAP50-95": float(ap.mean()),
            "precision": float(fixed_precision),
            "recall": float(fixed_recall),
            "f1": float(fixed_f1),
            "false_negatives_per_frame": float(total_fn / len(scope)),
            "ssim": float(scope["ssim"].mean()),
            "images": len(scope),
            "sequences": scope["sequence_id"].nunique(),
        })
        for index, class_id in enumerate(unique_classes.astype(int)):
            class_rows.append({
                "defense": defense_name,
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "precision_optimal": float(precision[index]),
                "recall_optimal": float(recall[index]),
                "f1_optimal": float(f1[index]),
                "AP50": float(ap[index, 0]),
                "AP50-95": float(ap[index].mean()),
            })
    pd.DataFrame(summary_rows).to_csv(args.output / "clean_utility_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(args.output / "clean_utility_by_class.csv", index=False)
    (args.output / "config.json").write_text(json.dumps({
        "model": str(args.model), "data": str(args.data), "split": args.split,
        "imgsz": args.imgsz, "confidence": args.confidence,
        "iou_levels": IOU_LEVELS.tolist(), "defenses": DEFENSES,
        "statistical_unit": "sequence_id",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
