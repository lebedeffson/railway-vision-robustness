from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from PIL import Image, ImageDraw
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.metrics import box_iou

from audit_final_practice import canonical_path, load_manifest
from evaluate_image_level_detection import get_ground_truth, predict_batch
from extract_feature_consistency import loader, to_device


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/weights/best.pt"
DEFAULT_DATA = PROJECT_DIR / "data/yolo_osdar23/data.yaml"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/canonical_v2/baseline_rescue_current"
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(splits) - set(SPLITS)
    if not splits or unknown:
        raise argparse.ArgumentTypeError(f"Invalid splits: {sorted(unknown)}")
    return splits


def match_counts(
    prediction: Tensor, gt_boxes: Tensor, gt_classes: Tensor, confidence: float
) -> dict[str, Any]:
    prediction = prediction[prediction[:, 4] >= confidence]
    prediction = prediction[prediction[:, 4].argsort(descending=True)]
    matched: set[int] = set()
    tp_by_class: defaultdict[int, int] = defaultdict(int)
    fp_by_class: defaultdict[int, int] = defaultdict(int)
    matched_gt: set[int] = set()
    for row in prediction:
        class_id = int(row[5])
        candidates = [
            index for index in range(len(gt_boxes))
            if index not in matched and int(gt_classes[index]) == class_id
        ]
        if candidates:
            candidate_tensor = torch.tensor(candidates, dtype=torch.long)
            ious = box_iou(row[:4].view(1, 4), gt_boxes[candidate_tensor]).view(-1)
            best = int(torch.argmax(ious))
            if float(ious[best]) >= 0.5:
                gt_index = candidates[best]
                matched.add(gt_index)
                matched_gt.add(gt_index)
                tp_by_class[class_id] += 1
                continue
        fp_by_class[class_id] += 1
    gt_by_class: defaultdict[int, int] = defaultdict(int)
    for class_id in gt_classes.tolist():
        gt_by_class[int(class_id)] += 1
    return {
        "tp": len(matched), "fp": len(prediction) - len(matched),
        "fn": len(gt_boxes) - len(matched), "tp_by_class": tp_by_class,
        "fp_by_class": fp_by_class, "gt_by_class": gt_by_class,
        "matched_gt": matched_gt,
    }


def collect_predictions(model: torch.nn.Module, data: Path, split: str, imgsz: int) -> list[dict[str, Any]]:
    device = torch.device("cuda:0")
    rows: list[dict[str, Any]] = []
    for raw in loader(data, split, imgsz, 1, 2, True):
        batch = to_device(raw, device)
        predictions = predict_batch(model, batch["img"], max_time_img=10.0)
        boxes, classes = get_ground_truth(batch, 0)
        rows.append({
            "image_path": str(batch["im_file"][0]),
            "prediction": predictions[0].detach().cpu(),
            "gt_boxes": boxes.detach().cpu(), "gt_classes": classes.detach().cpu(),
            "width": int(batch["img"].shape[-1]), "height": int(batch["img"].shape[-2]),
        })
    return rows


def aggregate(samples: list[dict[str, Any]], confidence: float) -> dict[str, float]:
    tp = fp = fn = 0
    image_f1: list[float] = []
    for sample in samples:
        counts = match_counts(
            sample["prediction"], sample["gt_boxes"], sample["gt_classes"], confidence
        )
        tp += counts["tp"]; fp += counts["fp"]; fn += counts["fn"]
        p = counts["tp"] / max(counts["tp"] + counts["fp"], 1)
        r = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        image_f1.append(2 * p * r / max(p + r, 1e-12))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    f2 = 5 * precision * recall / max(4 * precision + recall, 1e-12)
    return {
        "confidence": confidence, "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "f2": f2,
        "mean_image_f1": float(np.mean(image_f1)),
        "fn_per_frame": fn / max(len(samples), 1),
    }


def detail_rows(
    samples: list[dict[str, Any]], split: str, confidence: float,
    operating_point: str, names: dict[int, str], small: float, medium: float,
) -> list[dict[str, Any]]:
    class_counts: defaultdict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "gt": 0})
    size_counts = {name: {"tp": 0, "gt": 0} for name in ("small", "medium", "large")}
    for sample in samples:
        counts = match_counts(sample["prediction"], sample["gt_boxes"], sample["gt_classes"], confidence)
        for class_id, value in counts["tp_by_class"].items(): class_counts[class_id]["tp"] += value
        for class_id, value in counts["fp_by_class"].items(): class_counts[class_id]["fp"] += value
        for class_id, value in counts["gt_by_class"].items(): class_counts[class_id]["gt"] += value
        area = (
            (sample["gt_boxes"][:, 2] - sample["gt_boxes"][:, 0])
            * (sample["gt_boxes"][:, 3] - sample["gt_boxes"][:, 1])
            / (sample["width"] * sample["height"])
        )
        for index, value in enumerate(area.tolist()):
            size = "small" if value < small else "medium" if value < medium else "large"
            size_counts[size]["gt"] += 1
            if index in counts["matched_gt"]: size_counts[size]["tp"] += 1
    rows: list[dict[str, Any]] = []
    for class_id in sorted(names):
        values = class_counts[class_id]
        rows.append({
            "split": split, "operating_point": operating_point, "scope": "class",
            "name": names[class_id], "confidence": confidence, **values,
            "precision": values["tp"] / max(values["tp"] + values["fp"], 1),
            "recall": values["tp"] / max(values["gt"], 1),
        })
    for name, values in size_counts.items():
        rows.append({
            "split": split, "operating_point": operating_point, "scope": "size",
            "name": name, "confidence": confidence, **values, "fp": math.nan,
            "precision": math.nan, "recall": values["tp"] / max(values["gt"], 1),
        })
    return rows


def save_false_negative_examples(
    samples: list[dict[str, Any]], confidence: float, output: Path, limit: int = 12
) -> None:
    ranked = []
    for sample in samples:
        counts = match_counts(
            sample["prediction"], sample["gt_boxes"], sample["gt_classes"], confidence
        )
        ranked.append((counts["fn"], sample, counts))
    examples = output / "false_negative_examples"
    examples.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, (false_negatives, sample, counts) in enumerate(
        sorted(ranked, key=lambda value: (-value[0], value[1]["image_path"]))[:limit], 1
    ):
        image = Image.open(sample["image_path"]).convert("RGB").resize(
            (sample["width"], sample["height"])
        )
        draw = ImageDraw.Draw(image)
        for box in sample["gt_boxes"].tolist():
            draw.rectangle(box, outline="red", width=3)
        predictions = sample["prediction"]
        predictions = predictions[predictions[:, 4] >= confidence]
        for box in predictions[:, :4].tolist():
            draw.rectangle(box, outline="cyan", width=2)
        destination = examples / f"{rank:02d}_{Path(sample['image_path']).name}"
        image.save(destination)
        manifest.append({
            "rank": rank, "image_path": sample["image_path"],
            "rendered_path": str(destination), "false_negatives": false_negatives,
            "tp": counts["tp"], "fp": counts["fp"],
            "confidence": confidence,
        })
    pd.DataFrame(manifest).to_csv(output / "false_negative_examples.csv", index=False)


def per_scene_rows(
    samples: list[dict[str, Any]], split: str, confidence: float,
    operating_point: str, manifest_path: Path,
) -> list[dict[str, Any]]:
    lookup = {
        canonical_path(row["image_path"]): row["sequence_id"]
        for row in load_manifest(manifest_path) if row["split"] == split
    }
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        path = canonical_path(sample["image_path"])
        if path not in lookup:
            raise RuntimeError(f"Clean evaluation frame missing from manifest: {path}")
        grouped[lookup[path]].append(sample)
    return [
        {
            "split": split, "grouped_scene_id": scene,
            "sequence_id": scene, "operating_point": operating_point,
            "frames": len(values), **aggregate(values, confidence),
        }
        for scene, values in sorted(grouped.items())
    ]


def ultralytics_metrics(
    checkpoint: Path, data: Path, split: str, imgsz: int, output: Path
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    runner = YOLO(str(checkpoint))
    metrics = runner.val(
        data=str(data), split=split, imgsz=imgsz, batch=1, device=0, workers=2,
        conf=0.001, iou=0.7, max_det=300, plots=False, verbose=False,
        project=str(output / "ultralytics"), name=split, exist_ok=True,
    )
    overall = {
        "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
        "precision_ap_evaluator": float(metrics.box.mp),
        "recall_ap_evaluator": float(metrics.box.mr),
    }
    classes = []
    ap_indices = np.asarray(metrics.box.ap_class_index, dtype=int)
    for position, class_id in enumerate(ap_indices):
        classes.append({
            "split": split, "class_id": int(class_id), "class_name": metrics.names[int(class_id)],
            "AP50": float(metrics.box.ap50[position]), "AP50-95": float(metrics.box.ap[position]),
            "recall_ap_evaluator": float(metrics.box.r[position]),
            "precision_ap_evaluator": float(metrics.box.p[position]),
        })
    del metrics, runner
    torch.cuda.empty_cache()
    return overall, classes


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline rescue, threshold calibration and clean audit")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--small-area", type=float, default=0.001)
    parser.add_argument("--medium-area", type=float, default=0.01)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--splits", type=parse_splits, default=SPLITS)
    parser.add_argument(
        "--frozen-thresholds", type=Path,
        help="Use an already frozen validation selection; do not fit thresholds.",
    )
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = args.data.parent / "manifest.csv"
    if args.frozen_thresholds is None and "val" not in args.splits:
        parser.error("Threshold calibration requires validation in --splits")
    if args.frozen_thresholds is not None and "val" in args.splits:
        parser.error("Frozen-threshold evaluation must not reopen validation")
    args.output.mkdir(parents=True, exist_ok=True)
    model_hash = sha256(args.model)
    yolo = YOLO(str(args.model)); model = yolo.model.cuda().float().eval()
    overrides = model.args if isinstance(model.args, dict) else vars(model.args)
    model.args = get_cfg(overrides=overrides)
    predictions = {
        split: collect_predictions(model, args.data, split, args.imgsz)
        for split in args.splits
    }
    names = {int(key): str(value) for key, value in yolo.names.items()}
    del model, yolo
    torch.cuda.empty_cache()
    if args.frozen_thresholds is None:
        sweep = pd.DataFrame([
            aggregate(predictions["val"], float(threshold))
            for threshold in np.round(np.arange(0.001, 0.501, 0.001), 3)
        ])
        standard = sweep.sort_values(
            ["f1", "recall", "confidence"], ascending=[False, False, True]
        ).iloc[0]
        safety = sweep.sort_values(
            ["f2", "recall", "confidence"], ascending=[False, False, True]
        ).iloc[0]
        sweep.to_csv(args.output / "threshold_sweep.csv", index=False)
        selection = {
            "selection_split": "val", "test_used_for_selection": False,
            "checkpoint_path": str(args.model.resolve()),
            "checkpoint_sha256": model_hash,
            "standard": {"criterion": "maximum_aggregate_F1", **standard.to_dict()},
            "safety": {"criterion": "maximum_aggregate_F2", **safety.to_dict()},
        }
        (args.output / "threshold_selection.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.plot(sweep["recall"], sweep["precision"])
        axis.scatter(
            [standard["recall"], safety["recall"]],
            [standard["precision"], safety["precision"]],
        )
        axis.set(
            xlabel="Recall", ylabel="Precision",
            title="Validation precision-recall threshold sweep",
        )
        figure.tight_layout()
        figure.savefig(args.output / "precision_recall_curve.png", dpi=180)
        plt.close(figure)
        save_false_negative_examples(
            predictions["val"], float(standard["confidence"]), args.output
        )
    else:
        selection = json.loads(args.frozen_thresholds.read_text(encoding="utf-8"))
        if selection.get("selection_split") != "val" or selection.get("test_used_for_selection"):
            raise RuntimeError("Thresholds were not frozen exclusively on validation")
        if selection.get("checkpoint_sha256") != model_hash:
            raise RuntimeError("Frozen threshold checkpoint hash does not match evaluated model")
        standard = pd.Series(selection["standard"])
        safety = pd.Series(selection["safety"])
    clean_rows: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    ap_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    ap_by_split: dict[str, dict[str, float]] = {}
    for split in args.splits:
        ap, classes = ultralytics_metrics(args.model, args.data, split, args.imgsz, args.output)
        ap_by_split[split] = ap
        ap_rows.extend(classes)
        for point, threshold in (("standard", float(standard["confidence"])), ("safety", float(safety["confidence"]))):
            clean_rows.append({"split": split, "operating_point": point, **aggregate(predictions[split], threshold), **ap})
            detail.extend(detail_rows(
                predictions[split], split, threshold, point, names,
                args.small_area, args.medium_area,
            ))
            scene_rows.extend(per_scene_rows(
                predictions[split], split, threshold, point, args.manifest,
            ))
    pd.DataFrame(clean_rows).to_csv(args.output / "clean_metrics_train_val_test.csv", index=False)
    pd.DataFrame(detail).to_csv(args.output / "clean_metrics_by_class_and_size.csv", index=False)
    pd.DataFrame(ap_rows).to_csv(args.output / "ap_by_class.csv", index=False)
    pd.DataFrame(scene_rows).to_csv(args.output / "clean_metrics_per_scene.csv", index=False)
    if args.frozen_thresholds is None:
        validation_ap = ap_by_split["val"]
        sweep["mAP50"] = validation_ap["mAP50"]
        sweep["mAP50-95"] = validation_ap["mAP50-95"]
        sweep.to_csv(args.output / "threshold_sweep.csv", index=False)
    val_standard = next(
        (row for row in clean_rows if row["split"] == "val" and row["operating_point"] == "standard"),
        None,
    )
    payload = {
        "status": "PASS", "model": str(args.model.resolve()), "data": str(args.data.resolve()),
        "checkpoint_sha256": model_hash,
        "imgsz": args.imgsz, "evaluated_splits": list(args.splits),
        "thresholds": selection,
        "threshold_mode": (
            "fit_on_validation" if args.frozen_thresholds is None else "frozen_validation_thresholds"
        ),
    }
    if val_standard is not None:
        payload["quality_gate"] = {
            "recall_min": 0.35, "mAP50_min": 0.25,
            "observed_validation_recall": val_standard["recall"],
            "observed_validation_mAP50": val_standard["mAP50"],
            "passed": val_standard["recall"] >= .35 and val_standard["mAP50"] >= .25,
            "selection_uses_validation_only": True,
            "test_evaluated_before_gate": False,
        }
    (args.output / "baseline_rescue_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
