from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    TEST_MARKER,
    assert_role_allowed,
    atomic_json,
    load_protocol,
    sha256,
)
from canonical_m4_runtime import (
    FrameInput,
    detection,
    frame_labels,
    load_image,
    load_model,
)
from evaluate_canonical_m4 import (
    map_metrics,
    scoped_rows,
    size_and_border_rows,
)


SOURCE_MANIFEST = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"
DESTINATION = OUTPUT_ROOT / "test/clean"


def run() -> dict:
    assert_role_allowed("test")
    if not TEST_MARKER.is_file():
        raise RuntimeError("Canonical M4 test marker is absent")
    marker = json.loads(TEST_MARKER.read_text(encoding="utf-8"))
    gate = json.loads(
        (OUTPUT_ROOT / "validation/quality_gate.json").read_text(encoding="utf-8")
    )
    protocol = load_protocol()
    checkpoint = Path(gate["checkpoint"])
    if marker["checkpoint_sha256"] != sha256(checkpoint):
        raise RuntimeError("Clean test checkpoint differs from TEST_OPENED marker")
    threshold = float(gate["safety_threshold"])
    source = pd.read_csv(SOURCE_MANIFEST)
    source = source[source["split"].eq("test")].sort_values(
        ["grouped_scene_id", "subsequence_id", "frame_id"]
    )
    if len(source) != int(protocol["dataset"]["split_frame_counts"]["test"]):
        raise RuntimeError("Canonical M4 clean test frame count mismatch")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    cache = DESTINATION / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint, torch.device("cuda:0"))
    ground_truth = {}
    predictions = {}
    frame_rows = []
    for row in source.itertuples(index=False):
        frame = FrameInput(
            Path(row.output_image),
            Path(row.output_label),
            str(row.grouped_scene_id),
            str(row.subsequence_id),
        )
        cache_path = cache / f"{frame.image_path.stem}.json"
        if cache_path.is_file():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload["checkpoint_sha256"] != sha256(checkpoint):
                raise RuntimeError("Clean test cache checkpoint mismatch")
        else:
            image = load_image(frame.image_path, torch.device("cuda:0"))
            labels = frame_labels(frame, protocol)
            metrics, detected, _tiles = detection(
                model, image, labels, protocol, threshold
            )
            payload = {
                "checkpoint_sha256": sha256(checkpoint),
                "image_path": str(frame.image_path.resolve()),
                "grouped_scene_id": frame.grouped_scene_id,
                "labels": labels,
                "predictions": detected,
                "metrics": metrics,
            }
            atomic_json(cache_path, payload)
        image_path = str(frame.image_path.resolve())
        ground_truth[image_path] = payload["labels"]
        predictions[image_path] = payload["predictions"]
        frame_rows.append({
            "grouped_scene_id": frame.grouped_scene_id,
            "subsequence_id": frame.subsequence_id,
            "image_path": image_path,
            **payload["metrics"],
        })
    map50, map5095, per_class_ap = map_metrics(ground_truth, predictions)
    per_scene, per_class = scoped_rows(
        source, ground_truth, predictions, threshold
    )
    for class_id, ap50 in per_class_ap.items():
        per_class.loc[per_class["class_id"].eq(class_id), "AP50"] = ap50
    per_size, border = size_and_border_rows(
        ground_truth, predictions, threshold, protocol
    )
    frame_metrics = pd.DataFrame(frame_rows)
    frame_metrics.to_csv(DESTINATION / "clean_test_per_frame.csv", index=False)
    per_scene.to_csv(DESTINATION / "clean_test_per_scene.csv", index=False)
    per_class.to_csv(DESTINATION / "clean_test_per_class.csv", index=False)
    per_size.to_csv(DESTINATION / "clean_test_object_sizes.csv", index=False)
    border.to_csv(DESTINATION / "clean_test_border_objects.csv", index=False)
    totals = frame_metrics[["tp", "fp", "fn"]].sum()
    precision = float(totals.tp / max(totals.tp + totals.fp, 1))
    recall = float(totals.tp / max(totals.tp + totals.fn, 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    f2 = 5 * precision * recall / max(4 * precision + recall, 1e-12)
    summary = {
        "status": "PASS",
        "checkpoint_sha256": sha256(checkpoint),
        "threshold": threshold,
        "threshold_role": "safety_maximum_F2_frozen_on_validation",
        "frames": len(source),
        "grouped_scenes": int(source["grouped_scene_id"].nunique()),
        "mAP50": map50,
        "mAP50_95": map5095,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "fn_per_frame": float(totals.fn / len(source)),
        "small_recall": float(
            per_size.loc[per_size["size"].eq("small"), "recall"].iloc[0]
        ),
        "medium_recall": float(
            per_size.loc[per_size["size"].eq("medium"), "recall"].iloc[0]
        ),
        "large_recall": float(
            per_size.loc[per_size["size"].eq("large"), "recall"].iloc[0]
        ),
        "test_open_marker_sha256": sha256(TEST_MARKER),
        "test_used_for_tuning": False,
    }
    atomic_json(DESTINATION / "clean_test_metrics.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
