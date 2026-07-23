from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO

from rescue_v2_common import PROJECT_DIR, assert_role_allowed, load_protocol, sha256


OUTPUT = PROJECT_DIR / "outputs/rescue_v2/diagnosis"


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float64)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=1)
    area_box = np.maximum(box[2:] - box[:2], 0).prod()
    area_boxes = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area_box + area_boxes - intersection, 1e-12)


def letterbox_box(
    normalized_xywh: np.ndarray, raw_width: int, raw_height: int, imgsz: int
) -> np.ndarray:
    scale = min(imgsz / raw_height, imgsz / raw_width)
    resized_width = int(round(raw_width * scale))
    resized_height = int(round(raw_height * scale))
    left = int(round((imgsz - resized_width) / 2 - 0.1))
    top = int(round((imgsz - resized_height) / 2 - 0.1))
    x, y, width, height = normalized_xywh
    x1 = np.clip((x - width / 2) * raw_width * scale + left, 0, imgsz)
    y1 = np.clip((y - height / 2) * raw_height * scale + top, 0, imgsz)
    x2 = np.clip((x + width / 2) * raw_width * scale + left, 0, imgsz)
    y2 = np.clip((y + height / 2) * raw_height * scale + top, 0, imgsz)
    return np.asarray([x1, y1, x2, y2], dtype=np.float64)


def feature_cell_coverage(box: np.ndarray, strides: list[int]) -> dict[str, float]:
    width = float(max(box[2] - box[0], 0))
    height = float(max(box[3] - box[1], 0))
    values: dict[str, float] = {}
    for index, stride in enumerate(strides, start=3):
        values[f"P{index}_stride"] = int(stride)
        values[f"P{index}_cells_width"] = width / stride
        values[f"P{index}_cells_height"] = height / stride
        values[f"P{index}_cells_area"] = width * height / (stride * stride)
    values["minimum_stride"] = int(min(strides))
    values["minimum_stride_min_side_cells"] = min(width, height) / min(strides)
    return values


def classify_error(
    model_box: np.ndarray,
    correct_predictions: np.ndarray,
    wrong_predictions: np.ndarray,
    confidence: float,
) -> tuple[str, dict[str, Any]]:
    width = float(model_box[2] - model_box[0])
    height = float(model_box[3] - model_box[1])
    correct_iou = iou(model_box, correct_predictions[:, :4])
    wrong_iou = iou(model_box, wrong_predictions[:, :4])
    best_correct_index = int(np.argmax(correct_iou)) if len(correct_iou) else None
    best_wrong_index = int(np.argmax(wrong_iou)) if len(wrong_iou) else None
    best_correct_iou = (
        float(correct_iou[best_correct_index]) if best_correct_index is not None else 0.0
    )
    best_correct_conf = (
        float(correct_predictions[best_correct_index, 4])
        if best_correct_index is not None else math.nan
    )
    best_wrong_iou = (
        float(wrong_iou[best_wrong_index]) if best_wrong_index is not None else 0.0
    )
    best_wrong_conf = (
        float(wrong_predictions[best_wrong_index, 4])
        if best_wrong_index is not None else math.nan
    )
    if min(width, height) < 1.0:
        category = "A"
    elif best_correct_iou >= 0.5 and best_correct_conf < confidence:
        category = "B"
    elif best_wrong_iou >= 0.5 and best_wrong_conf >= confidence:
        category = "D"
    elif (
        best_correct_index is not None
        and best_correct_conf >= confidence
        and best_correct_iou > 0
    ):
        category = "C"
    else:
        category = "E"
    return category, {
        "best_correct_iou": best_correct_iou,
        "best_correct_confidence": best_correct_conf,
        "best_wrong_iou": best_wrong_iou,
        "best_wrong_confidence": best_wrong_conf,
        "best_wrong_class": (
            int(wrong_predictions[best_wrong_index, 5])
            if best_wrong_index is not None else -1
        ),
    }


def read_labels(path: Path) -> list[tuple[int, np.ndarray]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5:
            raise RuntimeError(f"Malformed label in {path}: {line}")
        rows.append((int(float(values[0])), np.asarray(values[1:], dtype=np.float64)))
    return rows


def unmatched_ground_truth(
    gt: np.ndarray, predictions: np.ndarray, confidence: float, threshold: float
) -> list[int]:
    active = predictions[predictions[:, 4] >= confidence]
    active = active[np.argsort(-active[:, 4])]
    matched: set[int] = set()
    for prediction in active:
        candidates = [
            index for index, row in enumerate(gt)
            if index not in matched and int(row[4]) == int(prediction[5])
        ]
        if not candidates:
            continue
        overlaps = iou(prediction[:4], gt[candidates, :4])
        best = int(np.argmax(overlaps))
        if float(overlaps[best]) >= threshold:
            matched.add(candidates[best])
    return [index for index in range(len(gt)) if index not in matched]


def bin_name(value: float) -> str:
    if value < 4:
        return "<4"
    if value < 8:
        return "4-8"
    if value < 16:
        return "8-16"
    if value < 32:
        return "16-32"
    return ">32"


def load_strides(checkpoint: Path) -> list[int]:
    model = YOLO(str(checkpoint))
    strides = sorted({int(round(float(value))) for value in model.model.stride.tolist()})
    if not strides or any(value <= 0 for value in strides):
        raise RuntimeError(f"Invalid model strides: {strides}")
    return strides


def draw_gallery(rows: list[dict[str, Any]], detections: pd.DataFrame) -> list[str]:
    gallery = OUTPUT / "fn_gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    cards: list[Image.Image] = []
    individual_paths: list[str] = []
    for row in rows:
        path = Path(row["image_path"])
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        box = np.asarray([
            row["model_x1_px_640"], row["model_y1_px_640"],
            row["model_x2_px_640"], row["model_y2_px_640"],
        ])
        draw.rectangle(box.tolist(), outline="red", width=3)
        scope = detections[
            (detections["image_path"] == row["image_path"])
            & (detections["kind"] == "prediction")
            & (detections["confidence"] >= row["operating_confidence"])
        ]
        for pred in scope.itertuples(index=False):
            draw.rectangle([pred.x1, pred.y1, pred.x2, pred.y2], outline="cyan", width=1)
        margin = max(48.0, 6.0 * max(box[2] - box[0], box[3] - box[1]))
        crop_box = (
            max(0, int(box[0] - margin)), max(0, int(box[1] - margin)),
            min(image.width, int(box[2] + margin)),
            min(image.height, int(box[3] + margin)),
        )
        crop = image.crop(crop_box)
        crop.thumbnail((280, 210), Image.Resampling.LANCZOS)
        card = Image.new("RGB", (300, 250), "white")
        card.paste(crop, ((300 - crop.width) // 2, 5))
        text = (
            f"FN {row['fn_id']} {row['class_name']} type={row['error_type']} "
            f"min={row['model_min_side_px_640']:.2f}px "
            f"IoU={row['best_iou']:.2f} conf={row['confidence_best_match']:.3f}"
        )
        ImageDraw.Draw(card).text((5, 220), text, fill="black")
        individual = gallery / f"fn_{int(row['fn_id']):03d}.png"
        card.save(individual)
        individual_paths.append(str(individual.resolve()))
        cards.append(card)
    page_paths = []
    for start in range(0, len(cards), 20):
        page_cards = cards[start:start + 20]
        page = Image.new("RGB", (1200, 1250), "#dddddd")
        for offset, card in enumerate(page_cards):
            page.paste(card, ((offset % 4) * 300, (offset // 4) * 250))
        page_path = gallery / f"page_{start // 20 + 1:02d}.png"
        page.save(page_path)
        page_paths.append(str(page_path.resolve()))
    return page_paths


def main() -> None:
    assert_role_allowed("audit")
    protocol = load_protocol()
    audit = protocol["fn_audit"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    detections_path = PROJECT_DIR / audit["source_detections"]
    manifest_path = PROJECT_DIR / audit["source_manifest"]
    thresholds_path = PROJECT_DIR / audit["source_thresholds"]
    checkpoint = PROJECT_DIR / protocol["parent_checkpoint"]
    detections = pd.read_csv(detections_path)
    manifest = pd.read_csv(manifest_path)
    threshold_payload = json.loads(thresholds_path.read_text(encoding="utf-8"))
    confidence = float(threshold_payload[audit["operating_point"]]["confidence"])
    strides = load_strides(checkpoint)
    class_names = {int(key): value for key, value in protocol["class_names"].items()}
    manifest_lookup = {
        str(Path(row.micro_val_image).resolve()): row
        for row in manifest.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for image_path, frame in detections.groupby("image_path", sort=True):
        image_path = str(Path(image_path).resolve())
        if image_path not in manifest_lookup:
            raise RuntimeError(f"Detection image is missing from manifest: {image_path}")
        manifest_row = manifest_lookup[image_path]
        gt_frame = frame[frame["kind"] == "ground_truth"].reset_index(drop=True)
        pred_frame = frame[frame["kind"] == "prediction"].reset_index(drop=True)
        gt = np.column_stack([
            gt_frame[["x1", "y1", "x2", "y2"]].to_numpy(float),
            gt_frame["class_id"].to_numpy(int),
        ])
        predictions = np.column_stack([
            pred_frame[["x1", "y1", "x2", "y2"]].to_numpy(float),
            pred_frame["confidence"].to_numpy(float),
            pred_frame["class_id"].to_numpy(int),
        ])
        labels = read_labels(Path(manifest_row.output_label))
        if len(labels) != len(gt) or [item[0] for item in labels] != gt[:, 4].astype(int).tolist():
            raise RuntimeError(f"Ground-truth order mismatch for {image_path}")
        raw_image = Image.open(manifest_row.output_image)
        raw_width, raw_height = raw_image.size
        missing = unmatched_ground_truth(
            gt, predictions, confidence, float(audit["iou_match_threshold"])
        )
        for gt_index in missing:
            class_id, normalized = labels[gt_index]
            model_box = gt[gt_index, :4]
            correct = predictions[predictions[:, 5].astype(int) == class_id]
            wrong = predictions[predictions[:, 5].astype(int) != class_id]
            category, diagnostic = classify_error(
                model_box, correct, wrong, confidence
            )
            all_iou = iou(model_box, predictions[:, :4])
            best_any = int(np.argmax(all_iou)) if len(all_iou) else None
            row: dict[str, Any] = {
                "fn_id": len(rows) + 1,
                "image_id": Path(image_path).stem,
                "image_path": image_path,
                "grouped_scene_id": manifest_row.grouped_scene_id,
                "subsequence_id": manifest_row.subsequence_id,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "raw_image_width_px": raw_width,
                "raw_image_height_px": raw_height,
                "raw_bbox_width_px": float(normalized[2] * raw_width),
                "raw_bbox_height_px": float(normalized[3] * raw_height),
                "raw_bbox_area_px": float(
                    normalized[2] * raw_width * normalized[3] * raw_height
                ),
                "operating_confidence": confidence,
                "confidence_best_match": diagnostic["best_correct_confidence"],
                "best_iou": diagnostic["best_correct_iou"],
                "best_any_iou": (
                    float(all_iou[best_any]) if best_any is not None else 0.0
                ),
                "predicted_class": (
                    int(predictions[best_any, 5]) if best_any is not None else -1
                ),
                "predicted_class_name": (
                    class_names.get(int(predictions[best_any, 5]), "none")
                    if best_any is not None else "none"
                ),
                "error_type_auto": category,
                "error_type": category,
                "manual_review_status": "pending",
                **diagnostic,
            }
            for imgsz in audit["audited_imgsz"]:
                box = letterbox_box(normalized, raw_width, raw_height, int(imgsz))
                row.update({
                    f"model_x1_px_{imgsz}": float(box[0]),
                    f"model_y1_px_{imgsz}": float(box[1]),
                    f"model_x2_px_{imgsz}": float(box[2]),
                    f"model_y2_px_{imgsz}": float(box[3]),
                    f"model_input_width_px_{imgsz}": float(box[2] - box[0]),
                    f"model_input_height_px_{imgsz}": float(box[3] - box[1]),
                    f"model_bbox_area_px_{imgsz}": float(
                        (box[2] - box[0]) * (box[3] - box[1])
                    ),
                    f"model_min_side_px_{imgsz}": float(
                        min(box[2] - box[0], box[3] - box[1])
                    ),
                    f"pixel_size_bin_{imgsz}": bin_name(
                        min(box[2] - box[0], box[3] - box[1])
                    ),
                    **{
                        f"imgsz_{imgsz}_{key}": value
                        for key, value in feature_cell_coverage(box, strides).items()
                    },
                })
            computed = np.asarray([
                row["model_x1_px_640"], row["model_y1_px_640"],
                row["model_x2_px_640"], row["model_y2_px_640"],
            ])
            if float(np.abs(computed - model_box).max()) > 1e-3:
                raise RuntimeError(
                    f"Letterbox reconstruction mismatch for {image_path} GT {gt_index}"
                )
            rows.append(row)
    if len(rows) != int(audit["expected_false_negatives"]):
        raise RuntimeError(
            f"Expected {audit['expected_false_negatives']} FN, found {len(rows)}"
        )
    review_path = OUTPUT / "manual_review.json"
    review = json.loads(review_path.read_text()) if review_path.is_file() else None
    if review:
        overrides = {int(row["fn_id"]): row for row in review.get("overrides", [])}
        for row in rows:
            override = overrides.get(int(row["fn_id"]))
            if override:
                if override["error_type"] not in {"F", "G"}:
                    raise RuntimeError("Manual overrides are restricted to F/G")
                if not str(override.get("reason", "")).strip():
                    raise RuntimeError("Manual F/G override requires a reason")
                row["error_type"] = override["error_type"]
                row["manual_review_reason"] = override["reason"]
            row["manual_review_status"] = "reviewed"
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "all_fn.csv", index=False)
    size_rows = []
    for imgsz in audit["audited_imgsz"]:
        grouped = frame.groupby([f"pixel_size_bin_{imgsz}", "class_name"]).size()
        for (size_bin, class_name), count in grouped.items():
            size_rows.append({
                "imgsz": imgsz,
                "pixel_size_measure": audit["pixel_size_measure"],
                "pixel_size_bin": size_bin,
                "class_name": class_name,
                "count": int(count),
            })
    pd.DataFrame(size_rows).to_csv(OUTPUT / "fn_by_pixel_size.csv", index=False)
    (
        frame.groupby(["error_type", "class_name"]).size().rename("count")
        .reset_index().to_csv(OUTPUT / "fn_by_error_type.csv", index=False)
    )
    pages = draw_gallery(rows, detections)
    stride_payload = {
        "status": "PASS",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "loaded_model_strides": strides,
        "hardcoded": False,
        "feature_levels": [f"P{index}" for index in range(3, 3 + len(strides))],
    }
    (OUTPUT / "feature_strides.json").write_text(
        json.dumps(stride_payload, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "PASS" if review else "PENDING_MANUAL_GALLERY_REVIEW",
        "false_negatives": len(frame),
        "all_fn_classified_once": bool(frame["error_type"].notna().all()),
        "manual_review_complete": bool(review),
        "error_types": dict(Counter(frame["error_type"])),
        "classes": dict(Counter(frame["class_name"])),
        "imgsz": audit["audited_imgsz"],
        "loaded_model_strides": strides,
        "minimum_stride": min(strides),
        "one_cell_or_less_by_imgsz": {
            str(imgsz): int(
                (
                    frame[f"imgsz_{imgsz}_minimum_stride_min_side_cells"] <= 1
                ).sum()
            )
            for imgsz in audit["audited_imgsz"]
        },
        "gallery_pages": pages,
        "test_opened": False,
    }
    (OUTPUT / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    inputs = [detections_path, manifest_path, thresholds_path, checkpoint]
    outputs = [
        OUTPUT / "all_fn.csv",
        OUTPUT / "fn_by_pixel_size.csv",
        OUTPUT / "fn_by_error_type.csv",
        OUTPUT / "feature_strides.json",
        OUTPUT / "audit_summary.json",
    ]
    completed = {
        "status": summary["status"],
        "protocol_id": protocol["protocol_id"],
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "outputs": [{"path": str(path), "sha256": sha256(path)} for path in outputs],
        "test_opened": False,
    }
    (OUTPUT / "COMPLETED.json").write_text(
        json.dumps(completed, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
