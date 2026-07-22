from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy

try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:
    from ultralytics.utils.ops import non_max_suppression

from extract_feature_consistency import (
    MODEL,
    DATA,
    ATTACKS,
    DEFENSES,
    EPS_LIST,
    loader,
    to_device,
    attack,
    defend,
)


OUTPUT = "outputs/diagnostics/image_detection"

IOU_THRESHOLD = 0.5
NMS_IOU = 0.7
NMS_CONF = 0.001
MAX_DETECTIONS = 300
DEFAULT_CONFIDENCE = 0.35
SEED = 42

COLUMNS = [
    "image",
    "image_path",
    "split",
    "attack",
    "epsilon_px",
    "epsilon",
    "defense",
    "confidence_threshold",
    "iou_threshold",
    "num_gt",
    "num_predictions",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
]


def predict_batch(
    model: nn.Module,
    images: Tensor,
    *,
    max_time_img: float = 0.05,
    return_diagnostics: bool = False,
) -> list[Tensor] | tuple[list[Tensor], list[dict[str, float | int | bool]]]:
    with torch.no_grad():
        output = model(images)

    if isinstance(output, tuple):
        output = output[0]

    class_count = max(1, int(getattr(model, "nc", output.shape[1] - 4)))
    candidates = (
        output[:, 4:4 + class_count].amax(1) > NMS_CONF
    ).sum(1).detach().cpu().tolist()
    started = time.perf_counter()
    predictions = non_max_suppression(
        output,
        conf_thres=NMS_CONF,
        iou_thres=NMS_IOU,
        max_det=MAX_DETECTIONS,
        max_time_img=max_time_img,
    )
    runtime_ms = (time.perf_counter() - started) * 1000.0
    time_limit_ms = (2.0 + max_time_img * len(images)) * 1000.0
    timed_out = runtime_ms > time_limit_ms
    diagnostics = [
        {
            "nms_timeout": timed_out,
            "nms_runtime_ms": runtime_ms,
            "nms_candidates_before": int(candidates[index]),
            "predictions_after_nms": int(len(prediction)),
            # Ultralytics checks its limit after assigning the current output.
            # Therefore a batch of one is complete even when it emits a warning.
            "nms_output_complete": len(images) == 1,
        }
        for index, prediction in enumerate(predictions)
    ]
    return (predictions, diagnostics) if return_diagnostics else predictions


def get_ground_truth(
    batch: dict[str, Any],
    image_index: int,
) -> tuple[Tensor, Tensor]:
    mask = batch["batch_idx"] == image_index
    classes = batch["cls"][mask].view(-1).long()
    boxes = batch["bboxes"][mask].clone()

    if boxes.numel() == 0:
        device = batch["img"].device

        return (
            torch.empty((0, 4), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    height, width = batch["img"].shape[2:]

    boxes[:, 0] *= width
    boxes[:, 1] *= height
    boxes[:, 2] *= width
    boxes[:, 3] *= height

    return xywh2xyxy(boxes), classes


def detection_metrics(
    prediction: Tensor | None,
    gt_boxes: Tensor,
    gt_classes: Tensor,
    confidence: float,
) -> dict[str, float | int]:
    if prediction is None or prediction.numel() == 0:
        prediction = torch.empty(
            (0, 6),
            device=gt_boxes.device,
        )
    else:
        prediction = prediction[
            prediction[:, 4] >= confidence
        ]

        prediction = prediction[
            prediction[:, 4].argsort(descending=True)
        ]

    num_predictions = int(len(prediction))
    num_gt = int(len(gt_boxes))

    if num_predictions == 0 and num_gt == 0:
        return {
            "num_gt": 0,
            "num_predictions": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        }

    matched_gt: set[int] = set()
    tp = 0

    for prediction_row in prediction:
        predicted_box = prediction_row[:4].view(1, 4)
        predicted_class = int(prediction_row[5].item())

        candidates = [
            index
            for index in range(num_gt)
            if index not in matched_gt
            and int(gt_classes[index].item()) == predicted_class
        ]

        if not candidates:
            continue

        candidate_indices = torch.tensor(
            candidates,
            dtype=torch.long,
            device=gt_boxes.device,
        )

        ious = box_iou(
            predicted_box,
            gt_boxes[candidate_indices],
        ).view(-1)

        best_position = int(torch.argmax(ious).item())
        best_iou = float(ious[best_position].item())

        if best_iou >= IOU_THRESHOLD:
            matched_gt.add(candidates[best_position])
            tp += 1

    fp = num_predictions - tp
    fn = num_gt - tp

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "num_gt": num_gt,
        "num_predictions": num_predictions,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def collect_clean_validation(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> list[tuple[Tensor, Tensor, Tensor]]:
    samples = []

    for raw_batch in tqdm(
        data_loader,
        desc="calibrate confidence",
        unit="batch",
    ):
        batch = to_device(raw_batch, device)
        predictions = predict_batch(model, batch["img"])

        for image_index, prediction in enumerate(predictions):
            gt_boxes, gt_classes = get_ground_truth(
                batch,
                image_index,
            )

            samples.append(
                (
                    prediction.detach().cpu(),
                    gt_boxes.detach().cpu(),
                    gt_classes.detach().cpu(),
                )
            )

    return samples


def calibrate_confidence(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[float, list[dict[str, float]]]:
    samples = collect_clean_validation(
        model,
        data_loader,
        device,
    )

    thresholds = np.round(
        np.arange(0.05, 0.51, 0.05),
        2,
    )

    results = []

    for threshold in thresholds:
        f1_scores = [
            detection_metrics(
                prediction,
                gt_boxes,
                gt_classes,
                float(threshold),
            )["f1"]
            for prediction, gt_boxes, gt_classes in samples
        ]

        results.append({
            "confidence": float(threshold),
            "mean_image_f1": float(np.mean(f1_scores)),
        })

    best = max(
        results,
        key=lambda row: (
            row["mean_image_f1"],
            row["confidence"],
        ),
    )

    return float(best["confidence"]), results


def write_batch_rows(
    writer: csv.DictWriter,
    batch: dict[str, Any],
    predictions: list[Tensor],
    split: str,
    attack_name: str,
    epsilon_pixels: int,
    defense_name: str,
    confidence: float,
) -> None:
    for image_index, prediction in enumerate(predictions):
        gt_boxes, gt_classes = get_ground_truth(
            batch,
            image_index,
        )

        metrics = detection_metrics(
            prediction,
            gt_boxes,
            gt_classes,
            confidence,
        )

        image_path = batch["im_file"][image_index]

        writer.writerow({
            "image": Path(image_path).name,
            "image_path": image_path,
            "split": split,
            "attack": attack_name,
            "epsilon_px": epsilon_pixels,
            "epsilon": epsilon_pixels / 255.0,
            "defense": defense_name,
            "confidence_threshold": confidence,
            "iou_threshold": IOU_THRESHOLD,
            **metrics,
        })


def evaluate(
    model: nn.Module,
    data_loader,
    device: torch.device,
    split: str,
    attacks: list[str],
    epsilons: list[int],
    defenses: list[str],
    confidence: float,
    output_csv: Path,
    skip_clean: bool,
) -> None:
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_csv = output_csv.with_suffix(
        output_csv.suffix + ".tmp"
    )

    try:
        with temporary_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=COLUMNS,
            )

            writer.writeheader()

            progress = tqdm(
                data_loader,
                desc=f"image detection {split}",
                unit="batch",
            )

            for raw_batch in progress:
                batch = to_device(raw_batch, device)

                clean_images = batch["img"].detach()

                clean_predictions = predict_batch(
                    model,
                    clean_images,
                )

                if not skip_clean:
                    for defense_name in defenses:
                        if defense_name == "none":
                            predictions = clean_predictions
                        else:
                            defended_images = defend(
                                clean_images,
                                defense_name,
                            )

                            predictions = predict_batch(
                                model,
                                defended_images,
                            )

                        write_batch_rows(
                            writer=writer,
                            batch=batch,
                            predictions=predictions,
                            split=split,
                            attack_name="clean",
                            epsilon_pixels=0,
                            defense_name=defense_name,
                            confidence=confidence,
                        )

                for attack_name in attacks:
                    for epsilon_pixels in epsilons:
                        progress.set_postfix(
                            attack=attack_name,
                            eps=epsilon_pixels,
                        )

                        with torch.enable_grad():
                            adversarial_images = attack(
                                model,
                                batch,
                                attack_name,
                                epsilon_pixels,
                            )

                        adversarial_predictions = predict_batch(
                            model,
                            adversarial_images,
                        )

                        for defense_name in defenses:
                            if defense_name == "none":
                                predictions = adversarial_predictions
                            else:
                                defended_images = defend(
                                    adversarial_images,
                                    defense_name,
                                )

                                predictions = predict_batch(
                                    model,
                                    defended_images,
                                )

                            write_batch_rows(
                                writer=writer,
                                batch=batch,
                                predictions=predictions,
                                split=split,
                                attack_name=attack_name,
                                epsilon_pixels=epsilon_pixels,
                                defense_name=defense_name,
                                confidence=confidence,
                            )

                file.flush()

        temporary_csv.replace(output_csv)

    except Exception:
        if temporary_csv.exists():
            temporary_csv.unlink()

        raise


def parse_list(value: str) -> list[str]:
    values = [
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    ]

    if not values:
        raise argparse.ArgumentTypeError(
            "List cannot be empty"
        )

    return values


def parse_ints(value: str) -> list[int]:
    try:
        values = [
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated integers"
        ) from error

    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "Epsilons must be positive integers"
        )

    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument(
        "--calibration-split",
        "--stats-split",
        dest="calibration_split",
        default="val",
    )

    parser.add_argument(
        "--eval-split",
        default="test",
    )

    parser.add_argument(
        "--attacks",
        type=parse_list,
        default=ATTACKS,
    )

    parser.add_argument(
        "--epsilons",
        type=parse_ints,
        default=EPS_LIST,
    )

    parser.add_argument(
        "--defenses",
        type=parse_list,
        default=DEFENSES,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--recalibrate",
        action="store_true",
    )

    parser.add_argument(
        "--skip-clean-defenses",
        action="store_true",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
    )

    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    value = value.strip().lower()

    if value == "cpu":
        return torch.device("cpu")

    if value.isdigit():
        device = torch.device(f"cuda:{value}")
    else:
        device = torch.device(value)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    return device


def load_saved_confidence(
    threshold_file: Path,
) -> tuple[float, list[dict[str, float]]] | None:
    if not threshold_file.exists():
        return None

    try:
        saved = json.loads(
            threshold_file.read_text(encoding="utf-8")
        )

        confidence = float(saved["best_confidence"])

        if not 0.0 <= confidence <= 1.0:
            return None

        return confidence, saved.get("results", [])

    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None


def main() -> None:
    args = parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = resolve_device(args.device)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    if args.confidence is not None:
        if not 0.0 <= args.confidence <= 1.0:
            raise ValueError(
                "Confidence must be between 0 and 1"
            )

    attacks = (
        ["fgsm"]
        if args.quick
        else list(args.attacks)
    )

    epsilons = (
        [1]
        if args.quick
        else list(args.epsilons)
    )

    defenses = (
        ["none", "tnorm"]
        if args.quick
        else list(args.defenses)
    )

    output_dir = Path(args.output)

    suffix = (
        f"{args.eval_split}_quick"
        if args.quick
        else args.eval_split
    )

    output_csv = (
        output_dir
        / f"image_detection_{suffix}.csv"
    )

    config_file = (
        output_dir
        / f"image_detection_{suffix}_config.json"
    )

    threshold_file = (
        output_dir
        / "confidence_calibration.json"
    )

    print("=" * 80)
    print("IMAGE-LEVEL DETECTION")
    print(f"model:       {args.model}")
    print(f"data:        {args.data}")
    print(f"device:      {device}")
    print(f"eval split:  {args.eval_split}")
    print(f"attacks:     {attacks}")
    print(f"eps:         {epsilons}")
    print(f"defense:     {defenses}")
    print(f"out:         {output_csv}")
    print("=" * 80)

    yolo = YOLO(args.model)
    model = yolo.model.to(device).float().eval()

    overrides = (
        model.args
        if isinstance(model.args, dict)
        else vars(model.args)
    )

    model.args = get_cfg(
        overrides=overrides
    )

    model.criterion = None

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    calibration_results: list[dict[str, float]] = []

    if args.confidence is not None:
        confidence = float(args.confidence)
        confidence_source = "command_line"

    elif not args.recalibrate:
        saved_calibration = load_saved_confidence(
            threshold_file
        )

        if saved_calibration is not None:
            confidence, calibration_results = saved_calibration
            confidence_source = "saved_calibration"
        else:
            confidence = DEFAULT_CONFIDENCE
            confidence_source = "default"

    else:
        confidence = DEFAULT_CONFIDENCE
        confidence_source = "default"

    should_calibrate = (
        args.recalibrate
        or (
            args.confidence is None
            and confidence_source == "default"
            and not threshold_file.exists()
        )
    )

    if should_calibrate:
        validation_loader = loader(
            Path(args.data),
            args.calibration_split,
            args.imgsz,
            args.batch,
            args.workers,
            device.type == "cuda",
        )

        confidence, calibration_results = calibrate_confidence(
            model,
            validation_loader,
            device,
        )

        confidence_source = "calibrated"

        threshold_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        threshold_file.write_text(
            json.dumps(
                {
                    "best_confidence": confidence,
                    "calibration_split": args.calibration_split,
                    "model": str(Path(args.model).resolve()),
                    "data": str(Path(args.data).resolve()),
                    "image_size": args.imgsz,
                    "iou_threshold": IOU_THRESHOLD,
                    "nms_iou": NMS_IOU,
                    "results": calibration_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        f"confidence threshold: {confidence:.2f} "
        f"({confidence_source})"
    )

    evaluation_loader = loader(
        Path(args.data),
        args.eval_split,
        args.imgsz,
        args.batch,
        args.workers,
        device.type == "cuda",
    )

    evaluate(
        model=model,
        data_loader=evaluation_loader,
        device=device,
        split=args.eval_split,
        attacks=attacks,
        epsilons=epsilons,
        defenses=defenses,
        confidence=confidence,
        output_csv=output_csv,
        skip_clean=args.skip_clean_defenses,
    )

    config_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_file.write_text(
        json.dumps(
            {
                "model": str(Path(args.model).resolve()),
                "data": str(Path(args.data).resolve()),
                "eval_split": args.eval_split,
                "calibration_split": args.calibration_split,
                "confidence": confidence,
                "confidence_source": confidence_source,
                "image_size": args.imgsz,
                "batch_size": args.batch,
                "attacks": attacks,
                "epsilons": epsilons,
                "defenses": defenses,
                "iou_threshold": IOU_THRESHOLD,
                "nms_confidence": NMS_CONF,
                "nms_iou": NMS_IOU,
                "max_detections": MAX_DETECTIONS,
                "quick": args.quick,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("DONE")
    print(output_csv)


if __name__ == "__main__":
    main()
