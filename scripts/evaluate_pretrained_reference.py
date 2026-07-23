from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

from audit_evaluator import average_precision, class_aware_nms, match_dataset
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


OUTPUT = OUTPUT_ROOT / "runs/R0/seed_20260722"
MANIFEST = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"
POLICY = PROJECT_DIR / "configs/rescue/candidate_execution_policy.yaml"


def main() -> None:
    ensure_branch()
    assert_test_sealed()
    protocol = load_protocol()
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completion = OUTPUT / "COMPLETED.json"
    if completion.is_file():
        print(completion)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Pretrained reference evaluation requires CUDA")
    manifest = pd.read_csv(MANIFEST)
    validation = manifest[manifest["split"] == "val"].copy()
    canonical_by_name = {name: int(key) for key, name in protocol["class_names"].items()}
    coco_to_canonical = {}
    for name, coco_ids in policy["pretrained_coco_mapping"].items():
        canonical_name = "road_vehicle" if name == "road_vehicle" else name
        for class_id in coco_ids:
            coco_to_canonical[int(class_id)] = canonical_by_name[canonical_name]
    ground_truth: dict[str, list[dict]] = {}
    predictions: dict[str, list[dict]] = {}
    detections = []
    model_path = PROJECT_DIR / "yolo11m.pt"
    model = YOLO(str(model_path))
    image_paths = validation["output_image"].astype(str).tolist()
    stream = model.predict(
        source=image_paths, stream=True, imgsz=640, conf=0.001,
        iou=0.7, device=0, batch=1, verbose=False,
    )
    label_lookup = dict(zip(validation["output_image"].astype(str), validation["output_label"].astype(str)))
    for image_path, result in zip(image_paths, stream):
        with Image.open(image_path) as image:
            width, height = image.size
        targets = []
        for line in Path(label_lookup[image_path]).read_text(encoding="utf-8").splitlines():
            class_id, xc, yc, box_width, box_height = map(float, line.split())
            box = [
                (xc - box_width / 2) * width, (yc - box_height / 2) * height,
                (xc + box_width / 2) * width, (yc + box_height / 2) * height,
            ]
            targets.append({"class_id": int(class_id), "box": box})
            detections.append({
                "image_path": image_path, "kind": "ground_truth",
                "class_id": int(class_id), "confidence": np.nan,
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
            })
        raw = []
        for box, confidence, coco_class in zip(
            result.boxes.xyxy.detach().cpu().tolist(),
            result.boxes.conf.detach().cpu().tolist(),
            result.boxes.cls.detach().cpu().tolist(),
        ):
            if int(coco_class) not in coco_to_canonical:
                continue
            raw.append({
                "class_id": coco_to_canonical[int(coco_class)],
                "confidence": float(confidence), "box": box,
            })
        remapped = class_aware_nms(raw, 0.7)
        ground_truth[image_path] = targets
        predictions[image_path] = remapped
        for prediction in remapped:
            detections.append({
                "image_path": image_path, "kind": "prediction",
                "class_id": prediction["class_id"], "confidence": prediction["confidence"],
                "x1": prediction["box"][0], "y1": prediction["box"][1],
                "x2": prediction["box"][2], "y2": prediction["box"][3],
            })
    pd.DataFrame(detections).to_csv(OUTPUT / "detections_val.csv", index=False)
    thresholds = []
    for confidence in np.round(np.arange(0.001, 0.501, 0.001), 3):
        row = match_dataset(ground_truth, predictions, float(confidence))
        precision, recall = row["precision"], row["recall"]
        row.update({
            "confidence": float(confidence),
            "f2": 5 * precision * recall / max(4 * precision + recall, 1e-12),
            "fn_per_frame": row["fn"] / len(validation),
        })
        thresholds.append(row)
    sweep = pd.DataFrame(thresholds)
    standard = sweep.sort_values(
        ["f1", "recall", "confidence"], ascending=[False, False, True]
    ).iloc[0]
    safety = sweep.sort_values(
        ["f2", "recall", "confidence"], ascending=[False, False, True]
    ).iloc[0]
    sweep.to_csv(OUTPUT / "threshold_sweep.csv", index=False)
    per_class = {
        class_id: average_precision(ground_truth, predictions, class_id)
        for class_id in range(6)
    }
    map50 = float(np.mean([value for value in per_class.values() if np.isfinite(value)]))
    scene_rows = []
    for scene, rows in validation.groupby("grouped_scene_id"):
        paths = set(rows["output_image"].astype(str))
        gt = {path: ground_truth[path] for path in paths}
        pred = {path: predictions[path] for path in paths}
        scene_ap = [
            average_precision(gt, pred, class_id)
            for class_id in range(6)
            if any(target["class_id"] == class_id for values in gt.values() for target in values)
        ]
        counts = match_dataset(gt, pred, float(safety["confidence"]))
        scene_rows.append({
            "grouped_scene_id": str(scene),
            "mAP50": float(np.mean([value for value in scene_ap if np.isfinite(value)])),
            **counts,
        })
    pd.DataFrame(scene_rows).to_csv(OUTPUT / "metrics_per_scene.csv", index=False)
    result = {
        "status": "PASS_REFERENCE_ONLY",
        "candidate": "R0", "seed": 20260722,
        "checkpoint_path": str(model_path.resolve()), "checkpoint_sha256": sha256(model_path),
        "head_classes": 80, "canonical_classes": 6,
        "mapping": policy["pretrained_coco_mapping"],
        "unavailable_canonical_class": "signal",
        "validation_mAP50": map50,
        "validation_standard_recall": float(standard["recall"]),
        "validation_standard_f1": float(standard["f1"]),
        "validation_safety_recall": float(safety["recall"]),
        "validation_safety_f2": float(safety["f2"]),
        "scene_macro_map50": float(pd.DataFrame(scene_rows)["mAP50"].mean()),
        "quality_gate_passed": False,
        "selection_eligible": False,
        "test_evaluated": False,
    }
    atomic_json(OUTPUT / "candidate_result.json", result)
    atomic_json(OUTPUT / "environment.json", environment_snapshot())
    completed_marker(
        OUTPUT, inputs=[model_path, MANIFEST, POLICY],
        outputs=[
            OUTPUT / "candidate_result.json", OUTPUT / "threshold_sweep.csv",
            OUTPUT / "metrics_per_scene.csv", OUTPUT / "detections_val.csv",
        ],
        extra={"stage": "pretrained_reference", "test_evaluated": False},
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
