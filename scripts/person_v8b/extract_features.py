from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from audit_evaluator import average_precision, box_iou
from canonical_m4_runtime import (
    detection,
    frame_labels,
    load_image,
    load_model,
    tile_geometry,
)
from canonical_m4_tiling import frozen_tiles
from extract_feature_consistency import FeatureHook
from person_v8b.common import (
    OUTPUT_ROOT,
    ROOT,
    assert_locked,
    assert_test_sealed,
    atomic_csv,
    atomic_json,
    load_config,
    sha256_file,
)


def matched_count(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    threshold: float,
    iou: float,
) -> int:
    matched: set[int] = set()
    for prediction in sorted(
        [row for row in predictions if float(row["confidence"]) >= threshold],
        key=lambda row: -float(row["confidence"]),
    ):
        candidates = [
            (index, box_iou(prediction["box"], target["box"]))
            for index, target in enumerate(ground_truth)
            if index not in matched
        ]
        if candidates:
            index, overlap = max(candidates, key=lambda value: value[1])
            if overlap >= iou:
                matched.add(index)
    return len(matched)


def region_mask(
    boxes: list[list[float]],
    feature: torch.Tensor,
    protocol: dict[str, Any],
) -> torch.Tensor:
    batch, _channels, height, width = feature.shape
    mask = torch.zeros((batch, height, width), dtype=torch.bool, device=feature.device)
    scale, _resized_width, _resized_height, pad_left, pad_top = tile_geometry(protocol)
    input_size = float(protocol["model"]["input_size"])
    for tile_index, tile in enumerate(frozen_tiles(protocol)):
        for box in boxes:
            x1 = max(float(box[0]), float(tile.left))
            y1 = max(float(box[1]), float(tile.top))
            x2 = min(float(box[2]), float(tile.right))
            y2 = min(float(box[3]), float(tile.bottom))
            if x2 <= x1 or y2 <= y1:
                continue
            model_x1 = (x1 - tile.left) * scale + pad_left
            model_y1 = (y1 - tile.top) * scale + pad_top
            model_x2 = (x2 - tile.left) * scale + pad_left
            model_y2 = (y2 - tile.top) * scale + pad_top
            fx1 = max(0, min(width, math.floor(model_x1 / input_size * width)))
            fy1 = max(0, min(height, math.floor(model_y1 / input_size * height)))
            fx2 = max(0, min(width, math.ceil(model_x2 / input_size * width)))
            fy2 = max(0, min(height, math.ceil(model_y2 / input_size * height)))
            if fx2 > fx1 and fy2 > fy1:
                mask[tile_index, fy1:fy2, fx1:fx2] = True
    return mask


def pooled_vector(feature: torch.Tensor, mask: torch.Tensor | None) -> np.ndarray:
    values = feature.detach().float()
    if mask is None:
        return values.mean(dim=(0, 2, 3)).cpu().numpy().astype(np.float32)
    selected = values.permute(1, 0, 2, 3)[:, mask]
    if selected.numel() == 0:
        return np.zeros(values.shape[1], dtype=np.float32)
    return selected.mean(dim=1).cpu().numpy().astype(np.float32)


def confidence_entropy(confidences: np.ndarray) -> float:
    if len(confidences) == 0:
        return 0.0
    total = float(confidences.sum())
    if total <= 0:
        return 0.0
    probabilities = confidences / total
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def extract() -> dict[str, Any]:
    lock = assert_locked()
    config = load_config()
    assert_test_sealed()
    f0_path = OUTPUT_ROOT / "audit/F0_AUDIT.json"
    if not f0_path.is_file():
        raise RuntimeError("F0 audit must run before feature extraction")
    f0 = json.loads(f0_path.read_text(encoding="utf-8"))
    if f0["status"] != "PASS":
        raise RuntimeError(f"F0 audit is not PASS: {f0['status']}")
    if not torch.cuda.is_available():
        raise RuntimeError("V8b feature extraction requires CUDA")

    manifest = pd.read_csv(
        ROOT / config["immutable_inputs"]["development_manifest"]["path"]
    ).sort_values(["grouped_scene_id", "subsequence_id", "frame_id"])
    protocol = yaml.safe_load(
        (
            ROOT / config["immutable_inputs"]["tiling_protocol"]["path"]
        ).read_text(encoding="utf-8")
    )
    checkpoint = ROOT / config["immutable_inputs"]["detector_checkpoint"]["path"]
    device = torch.device("cuda:0")
    model = load_model(checkpoint, device)
    hook = FeatureHook(model)
    layers = list(config["feature_extraction"]["layers"])
    threshold = float(config["detector"]["operating_threshold"])
    floor = float(config["detector"]["confidence_floor"])
    match_iou = float(config["detector"]["match_iou"])
    raw_root = OUTPUT_ROOT / "features/raw"
    rows: list[dict[str, Any]] = []
    try:
        for frame_index, row in enumerate(manifest.itertuples(index=False), 1):
            image_path = Path(row.output_image)
            frame_id = image_path.stem
            npz_path = raw_root / str(row.grouped_scene_id) / f"{frame_id}.npz"
            metadata_path = npz_path.with_suffix(".json")
            if npz_path.is_file() and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (
                    metadata["checkpoint_sha256"]
                    != config["immutable_inputs"]["detector_checkpoint"]["sha256"]
                    or metadata["protocol_id"] != config["protocol_id"]
                ):
                    raise RuntimeError(f"Stale feature cache: {metadata_path}")
            else:
                frame = __import__(
                    "canonical_m4_runtime", fromlist=["FrameInput"]
                ).FrameInput(
                    image_path,
                    Path(row.output_label),
                    str(row.grouped_scene_id),
                    str(row.subsequence_id),
                )
                labels = [
                    value
                    for value in frame_labels(frame, protocol)
                    if int(value["class_id"]) == 0
                ]
                image = load_image(image_path, device)
                metrics, predictions, _tiles = detection(
                    model, image, labels, protocol, threshold
                )
                features = hook.features
                if features is None or len(features) != 3:
                    raise RuntimeError("P3/P4/P5 hook did not capture detection features")
                proposal_boxes = [
                    list(map(float, value["box"]))
                    for value in predictions
                    if float(value["confidence"]) >= floor
                ]
                gt_boxes = [list(map(float, value["box"])) for value in labels]
                arrays: dict[str, np.ndarray] = {}
                for layer, feature in zip(layers, features, strict=True):
                    proposal_mask = region_mask(proposal_boxes, feature, protocol)
                    gt_mask = region_mask(gt_boxes, feature, protocol)
                    arrays[f"{layer}_global"] = pooled_vector(feature, None)
                    arrays[f"{layer}_proposal"] = pooled_vector(feature, proposal_mask)
                    arrays[f"{layer}_background"] = pooled_vector(
                        feature, ~proposal_mask
                    )
                    arrays[f"{layer}_gt_oracle"] = pooled_vector(feature, gt_mask)
                confidences = np.asarray(
                    [float(value["confidence"]) for value in predictions],
                    dtype=np.float64,
                )
                filtered = [
                    value
                    for value in predictions
                    if float(value["confidence"]) >= threshold
                ]
                tp = matched_count(labels, predictions, threshold, match_iou)
                gt_count = len(labels)
                fp = len(filtered) - tp
                fn = gt_count - tp
                recall = tp / gt_count if gt_count else None
                precision = tp / len(filtered) if filtered else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if recall is not None and precision + recall > 0
                    else 0.0
                )
                evaluator_consistency = (
                    tp == int(metrics["tp"])
                    and fp == int(metrics["fp"])
                    and fn == int(metrics["fn"])
                )
                image_area = float(
                    protocol["dataset"]["image_width"]
                    * protocol["dataset"]["image_height"]
                )
                pred_areas = np.asarray(
                    [
                        max(0.0, value["box"][2] - value["box"][0])
                        * max(0.0, value["box"][3] - value["box"][1])
                        / image_area
                        for value in predictions
                    ],
                    dtype=np.float64,
                )
                gt_areas = np.asarray(
                    [
                        max(0.0, value["box"][2] - value["box"][0])
                        * max(0.0, value["box"][3] - value["box"][1])
                        / image_area
                        for value in labels
                    ],
                    dtype=np.float64,
                )
                metadata = {
                    "protocol_id": config["protocol_id"],
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "image_path": str(image_path.resolve()),
                    "grouped_scene_id": str(row.grouped_scene_id),
                    "subsequence_id": str(row.subsequence_id),
                    "frame_id": str(row.frame_id),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "gt_count": gt_count,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "unsafe": bool(
                        fn >= int(config["outcomes"]["unsafe_frame"]["fn_minimum"])
                        or (
                            gt_count > 0
                            and recall is not None
                            and recall
                            < float(config["outcomes"]["unsafe_frame"]["recall_below"])
                        )
                    ),
                    "small_gt_fraction": (
                        float((gt_areas < 0.001).mean()) if len(gt_areas) else 0.0
                    ),
                    "mean_confidence": (
                        float(confidences.mean()) if len(confidences) else 0.0
                    ),
                    "max_confidence": (
                        float(confidences.max()) if len(confidences) else 0.0
                    ),
                    "prediction_count": len(predictions),
                    "confidence_entropy": confidence_entropy(confidences),
                    "low_confidence_fraction": (
                        float((confidences < threshold).mean())
                        if len(confidences)
                        else 0.0
                    ),
                    "mean_prediction_area_ratio": (
                        float(pred_areas.mean()) if len(pred_areas) else 0.0
                    ),
                    "small_prediction_fraction": (
                        float((pred_areas < 0.001).mean()) if len(pred_areas) else 0.0
                    ),
                    "proposal_region_empty": len(proposal_boxes) == 0,
                    "nms_timeout": bool(metrics["nms_timeout"]),
                    "evaluator_consistency": (
                        "PASS" if evaluator_consistency else "FAIL"
                    ),
                    "features_finite": all(
                        bool(np.isfinite(value).all()) for value in arrays.values()
                    ),
                    "gt_region_role": "ORACLE_SUPPLEMENTARY_ONLY",
                    "ground_truth_boxes": gt_boxes,
                    "prediction_rows": predictions,
                    "test_used": False,
                }
                if (
                    metadata["nms_timeout"]
                    or not metadata["features_finite"]
                    or not evaluator_consistency
                ):
                    raise RuntimeError(f"Invalid v8b extraction for {image_path}")
                npz_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = npz_path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
                os.replace(temporary, npz_path)
                atomic_json(metadata_path, metadata)
                del image, features
            rows.append(
                {
                    **metadata,
                    "feature_path": str(npz_path.resolve()),
                    "feature_sha256": sha256_file(npz_path),
                    "metadata_path": str(metadata_path.resolve()),
                    "metadata_sha256": sha256_file(metadata_path),
                }
            )
            if frame_index % 25 == 0:
                atomic_json(
                    OUTPUT_ROOT / "runtime/F1_STATE.json",
                    {
                        "status": "RUNNING",
                        "completed_frames": frame_index,
                        "total_frames": len(manifest),
                        "current_scene": str(row.grouped_scene_id),
                        "test_access_count": 0,
                    },
                )
    finally:
        hook.close()

    fields = list(rows[0])
    feature_manifest = OUTPUT_ROOT / "features/FEATURE_MANIFEST.csv"
    atomic_csv(feature_manifest, rows, fields)
    scene_rows = []
    for scene, group in pd.DataFrame(rows).groupby("grouped_scene_id"):
        ground_truth: dict[str, list[dict[str, Any]]] = {}
        predictions: dict[str, list[dict[str, Any]]] = {}
        for item in group.to_dict("records"):
            metadata = json.loads(Path(item["metadata_path"]).read_text(encoding="utf-8"))
            image_key = metadata["image_path"]
            ground_truth[image_key] = [
                {"class_id": 0, "box": box}
                for box in metadata["ground_truth_boxes"]
            ]
            predictions[image_key] = metadata["prediction_rows"]
        tp_sum = int(group["tp"].sum())
        fn_sum = int(group["fn"].sum())
        scene_rows.append(
            {
                "grouped_scene_id": scene,
                "frames": len(group),
                "map50": average_precision(
                    ground_truth, predictions, 0, match_iou
                ),
                "recall": tp_sum / (tp_sum + fn_sum) if tp_sum + fn_sum else 1.0,
                "fn_per_frame": float(group["fn"].mean()),
                "test_used": False,
            }
        )
    detector_scene_path = OUTPUT_ROOT / "features/DETECTOR_PER_SCENE.csv"
    atomic_csv(
        detector_scene_path,
        scene_rows,
        [
            "grouped_scene_id",
            "frames",
            "map50",
            "recall",
            "fn_per_frame",
            "test_used",
        ],
    )
    completion = {
        "protocol_id": config["protocol_id"],
        "status": "PASS",
        "frames": len(rows),
        "scenes": len({row["grouped_scene_id"] for row in rows}),
        "nan_inf": 0,
        "nms_timeouts": sum(bool(row["nms_timeout"]) for row in rows),
        "evaluator_consistency": (
            "PASS"
            if all(row["evaluator_consistency"] == "PASS" for row in rows)
            else "FAIL"
        ),
        "checkpoint_sha256": sha256_file(checkpoint),
        "feature_manifest": str(feature_manifest.resolve()),
        "feature_manifest_sha256": sha256_file(feature_manifest),
        "detector_per_scene": str(detector_scene_path.resolve()),
        "detector_per_scene_sha256": sha256_file(detector_scene_path),
        "protocol_lock_sha256": sha256_file(
            ROOT / "protocol/v8b/V8B_PROTOCOL_LOCK.json"
        ),
        "test_status": "SEALED",
        "test_access_count": 0,
        "detector_training": "NOT_RUN",
        "implementation_commit": lock["implementation_commit"],
    }
    atomic_json(OUTPUT_ROOT / "features/F1_COMPLETE.json", completion)
    atomic_json(
        OUTPUT_ROOT / "runtime/F1_STATE.json",
        {
            "status": "COMPLETED",
            "completed_frames": len(rows),
            "total_frames": len(rows),
            "test_access_count": 0,
        },
    )
    return completion


def main() -> int:
    result = extract()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
