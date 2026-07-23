from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    PROJECT_DIR / "configs/canonical_v3_development_pool_small_object.yaml"
)
OUTPUT_ROOT = PROJECT_DIR / "outputs/canonical_v3"
AUDIT_ROOT = OUTPUT_ROOT / "audit"
PROTOCOL_ROOT = OUTPUT_ROOT / "protocol"
M4_EVALUATION = (
    PROJECT_DIR
    / "outputs/canonical_m4/scene_cv/fold_0/seed_20260722/evaluation"
)
M4_AUGMENTATION = (
    PROJECT_DIR / "outputs/canonical_m4/tiling_audit/augmentation_object_audit.csv"
)
M4_TILING_AUDIT = (
    PROJECT_DIR / "outputs/canonical_m4/tiling_audit/tiling_audit.json"
)
M4_PROTOCOL = PROJECT_DIR / "configs/canonical_v2_m4_full_protocol.yaml"
TEST_MARKERS = [
    PROJECT_DIR / "outputs/canonical_m4/final/TEST_OPENED.json",
    OUTPUT_ROOT / "TEST_OPENED.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_protocol() -> dict[str, Any]:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    return intersection / max(first_area + second_area - intersection, 1e-12)


def read_labels(path: Path, width: int = 4112, height: int = 2504) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, center_x, center_y, box_width, box_height = map(
            float, line.split()
        )
        center_x *= width
        center_y *= height
        box_width *= width
        box_height *= height
        result.append({
            "class_id": int(class_id),
            "box": [
                center_x - box_width / 2,
                center_y - box_height / 2,
                center_x + box_width / 2,
                center_y + box_height / 2,
            ],
            "width": box_width,
            "height": box_height,
            "area_ratio": box_width * box_height / (width * height),
        })
    return result


def frozen_manifests(
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(PROJECT_DIR / protocol["data"]["source_manifest"])
    development = source[
        source["split"].isin(protocol["data"]["development_source_splits"])
    ].copy()
    development["canonical_v3_split"] = "development"
    test = source[source["split"].eq(protocol["data"]["test_source_split"])].copy()
    test["canonical_v3_split"] = "sealed_test"
    if development["grouped_scene_id"].nunique() != 15:
        raise RuntimeError("Canonical v3 development group count mismatch")
    if test["grouped_scene_id"].nunique() != 5:
        raise RuntimeError("Canonical v3 test group count mismatch")
    if set(development["grouped_scene_id"]) & set(test["grouped_scene_id"]):
        raise RuntimeError("Canonical v3 development/test scene leakage")
    atomic_csv(development, PROTOCOL_ROOT / "development_manifest.csv")
    # The sealed test manifest records provenance only. Its labels are never opened.
    atomic_csv(test, PROTOCOL_ROOT / "sealed_test_manifest.csv")
    return development, test


def development_object_table(development: pd.DataFrame) -> pd.DataFrame:
    m4 = pd.read_csv(M4_AUGMENTATION)
    model_sizes = (
        m4.groupby(["source_image", "source_gt_id"], as_index=False)
        .agg(
            model_min_side_px=(
                "model_bbox_width_px_after_augmentation",
                "max",
            ),
            model_height_proxy=(
                "model_bbox_height_px_after_augmentation",
                "max",
            ),
        )
    )
    # The limiting side determines whether a detector head receives spatial support.
    model_sizes["model_min_side_px"] = model_sizes[
        ["model_min_side_px", "model_height_proxy"]
    ].min(axis=1)
    size_lookup = {
        (str(Path(row.source_image).resolve()), int(row.source_gt_id)):
        float(row.model_min_side_px)
        for row in model_sizes.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for frame in development.itertuples(index=False):
        image_path = str(Path(frame.output_image).resolve())
        for gt_id, label in enumerate(read_labels(Path(frame.output_label))):
            area = float(label["area_ratio"])
            rows.append({
                "grouped_scene_id": str(frame.grouped_scene_id),
                "subsequence_id": str(frame.subsequence_id),
                "image_path": image_path,
                "gt_id": gt_id,
                "class_id": int(label["class_id"]),
                "raw_bbox_width_px": float(label["width"]),
                "raw_bbox_height_px": float(label["height"]),
                "raw_bbox_area_ratio": area,
                "size_group": (
                    "small" if area < 0.001
                    else ("medium" if area < 0.01 else "large")
                ),
                "m4_model_min_side_px": size_lookup.get((image_path, gt_id)),
            })
    result = pd.DataFrame(rows)
    if result["m4_model_min_side_px"].isna().any():
        missing = int(result["m4_model_min_side_px"].isna().sum())
        raise RuntimeError(f"M4 resized-size audit misses {missing} development GT")
    return result


def class_scene_matrix(
    development: pd.DataFrame,
    objects: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    names = {int(key): value for key, value in protocol["data"]["class_names"].items()}
    rows = []
    for scene, frames in development.groupby("grouped_scene_id"):
        selected = objects[objects["grouped_scene_id"].eq(scene)]
        row: dict[str, Any] = {
            "grouped_scene_id": scene,
            "frames": int(len(frames)),
            "objects": int(len(selected)),
            "small": int(selected["size_group"].eq("small").sum()),
            "medium": int(selected["size_group"].eq("medium").sum()),
            "large": int(selected["size_group"].eq("large").sum()),
        }
        for threshold in (2, 4, 8, 16):
            row[f"objects_below_{threshold}px"] = int(
                selected["m4_model_min_side_px"].lt(threshold).sum()
            )
        for class_id, name in names.items():
            row[f"class_{class_id}_{name}"] = int(
                selected["class_id"].eq(class_id).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("grouped_scene_id").reset_index(drop=True)


def partition_score(
    folds: list[list[str]],
    matrix: pd.DataFrame,
    class_columns: list[str],
    min_train_scenes: int,
) -> tuple[int, float]:
    indexed = matrix.set_index("grouped_scene_id")
    all_scenes = set(indexed.index)
    violations = 0
    vectors = []
    scale_columns = ["frames", "small", "medium", "large", *class_columns]
    totals = indexed[scale_columns].sum().replace(0, 1)
    for fold in folds:
        held = indexed.loc[fold]
        train_scenes = sorted(all_scenes - set(fold))
        train = indexed.loc[train_scenes]
        for column in class_columns:
            if held[column].sum() <= 0:
                continue
            support = int(train[column].gt(0).sum())
            violations += int(support < min_train_scenes)
            violations += int(train[column].sum() <= 0)
        vectors.append((held[scale_columns].sum() / totals).to_numpy(float))
    imbalance = float(np.asarray(vectors).std(axis=0).mean())
    return violations, imbalance


def constrained_folds(
    matrix: pd.DataFrame, protocol: dict[str, Any]
) -> tuple[list[list[str]], dict[str, Any]]:
    config = protocol["cpu_audit"]["constrained_split"]
    scenes = matrix["grouped_scene_id"].astype(str).tolist()
    class_columns = [
        column for column in matrix.columns if column.startswith("class_")
    ]
    rng = np.random.default_rng(int(config["seed"]))
    best: tuple[int, float, list[list[str]]] | None = None
    for _ in range(int(config["candidate_partitions"])):
        shuffled = list(rng.permutation(scenes))
        folds = [
            sorted(shuffled[index:index + int(config["scenes_per_fold"])])
            for index in range(0, len(shuffled), int(config["scenes_per_fold"]))
        ]
        folds.sort()
        violations, imbalance = partition_score(
            folds,
            matrix,
            class_columns,
            int(config["minimum_train_scenes_per_heldout_class"]),
        )
        candidate = (violations, imbalance, folds)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    return best[2], {
        "constraint_violations": best[0],
        "imbalance_score": best[1],
        "candidate_partitions": int(config["candidate_partitions"]),
        "seed": int(config["seed"]),
    }


def fold_audit(
    folds: list[list[str]],
    matrix: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, bool]:
    indexed = matrix.set_index("grouped_scene_id")
    all_scenes = set(indexed.index)
    names = {int(key): value for key, value in protocol["data"]["class_names"].items()}
    minimum = int(
        protocol["cpu_audit"]["constrained_split"][
            "minimum_train_scenes_per_heldout_class"
        ]
    )
    rows = []
    for fold_id, heldout in enumerate(folds):
        training = indexed.loc[sorted(all_scenes - set(heldout))]
        held = indexed.loc[heldout]
        for class_id, name in names.items():
            column = f"class_{class_id}_{name}"
            held_gt = int(held[column].sum())
            train_gt = int(training[column].sum())
            train_scenes = int(training[column].gt(0).sum())
            valid = held_gt == 0 or (train_gt > 0 and train_scenes >= minimum)
            rows.append({
                "fold": fold_id,
                "heldout_scenes": "|".join(heldout),
                "class_id": class_id,
                "class_name": name,
                "heldout_gt": held_gt,
                "train_gt": train_gt,
                "train_scene_support": train_scenes,
                "minimum_train_scene_support": minimum,
                "valid": valid,
            })
    result = pd.DataFrame(rows)
    return result, bool(result["valid"].all())


def matched_gt(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    confidence: float,
    iou_threshold: float,
) -> set[int]:
    active = sorted(
        [
            row for row in predictions
            if float(row["confidence"]) >= confidence
        ],
        key=lambda row: -float(row["confidence"]),
    )
    matched: set[int] = set()
    for prediction in active:
        candidates = [
            (index, box_iou(prediction["box"], target["box"]))
            for index, target in enumerate(ground_truth)
            if index not in matched
            and int(prediction["class_id"]) == int(target["class_id"])
        ]
        if candidates:
            index, overlap = max(candidates, key=lambda value: value[1])
            if overlap >= iou_threshold:
                matched.add(index)
    return matched


def fusion_audit(
    development: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evaluation = json.loads(
        (M4_EVALUATION / "evaluation_result.json").read_text(encoding="utf-8")
    )
    if evaluation["checkpoint_sha256"] != protocol["cpu_audit"]["fusion"][
        "checkpoint_sha256"
    ]:
        raise RuntimeError("Fusion audit checkpoint differs from frozen M4 checkpoint")
    safety = float(evaluation["safety_threshold"])
    model_prefilter = float(
        yaml.safe_load(M4_PROTOCOL.read_text(encoding="utf-8"))[
            "fusion"
        ]["confidence_prefilter"]
    )
    selected = development[
        development["grouped_scene_id"].isin(
            ["4_station_pedestrian_bridge_4", "19_vegetation_curve_19"]
        )
    ]
    boundaries_x = (1762, 2350)
    boundaries_y = (1073, 1431)
    aggregate: defaultdict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"gt": 0, "matched": 0}
    )
    nms_lost = 0
    raw_total = fused_total = 0
    for frame in selected.itertuples(index=False):
        image = str(Path(frame.output_image).resolve())
        cache_path = (
            M4_EVALUATION / "prediction_cache" / f"{Path(image).stem}.json"
        )
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        raw = list(payload["raw"])
        fused = list(payload["fused"])
        raw_total += len(raw)
        fused_total += len(fused)
        gt = read_labels(Path(frame.output_label))
        raw_oracle = matched_gt(gt, raw, -math.inf, 0.50)
        raw_prefilter = matched_gt(gt, raw, model_prefilter, 0.50)
        raw_safety = matched_gt(gt, raw, safety, 0.50)
        fused_safety = matched_gt(gt, fused, safety, 0.50)
        nms_lost += len(raw_safety - fused_safety)
        stages = {
            "tile_oracle": raw_oracle,
            "restored_global_raw_at_model_prefilter": raw_prefilter,
            "restored_global_raw_at_safety_threshold": raw_safety,
            "fused_at_safety_threshold": fused_safety,
        }
        for index, target in enumerate(gt):
            x1, y1, x2, y2 = target["box"]
            border = any(x1 < value < x2 for value in boundaries_x) or any(
                y1 < value < y2 for value in boundaries_y
            )
            scopes = ["all", "border" if border else "center"]
            for stage, matched in stages.items():
                for scope in scopes:
                    for class_id in (-1, int(target["class_id"])):
                        key = (f"{stage}:{scope}", class_id)
                        aggregate[key]["gt"] += 1
                        aggregate[key]["matched"] += int(index in matched)
    names = {int(key): value for key, value in protocol["data"]["class_names"].items()}
    rows = []
    for (scope, class_id), values in sorted(aggregate.items()):
        rows.append({
            "stage_scope": scope,
            "class_id": class_id,
            "class_name": "all" if class_id == -1 else names[class_id],
            **values,
            "recall": values["matched"] / max(values["gt"], 1),
        })
    frame = pd.DataFrame(rows)
    all_rows = frame[frame["class_id"].eq(-1)].set_index("stage_scope")
    report = {
        "status": "PASS",
        "checkpoint_sha256": evaluation["checkpoint_sha256"],
        "frames": int(len(selected)),
        "grouped_scenes": int(selected["grouped_scene_id"].nunique()),
        "safety_threshold": safety,
        "model_confidence_prefilter": model_prefilter,
        "raw_predictions": raw_total,
        "fused_predictions": fused_total,
        "predictions_removed_by_fusion": raw_total - fused_total,
        "GT_matched_before_fusion_at_safety": int(
            all_rows.loc[
                "restored_global_raw_at_safety_threshold:all", "matched"
            ]
        ),
        "GT_matched_after_fusion_at_safety": int(
            all_rows.loc["fused_at_safety_threshold:all", "matched"]
        ),
        "GT_lost_by_fusion_at_safety": int(nms_lost),
        "coordinate_restoration_roundtrip_max_abs_error": float(
            json.loads(M4_TILING_AUDIT.read_text(encoding="utf-8"))[
                "roundtrip_max_abs_error"
            ]
        ),
        "confidence_below_0_001_observable": False,
        "tile_level_mAP_is_not_comparable_to_global_mAP":
            "tile evaluator duplicates GT across overlapping tiles",
        "test_used": False,
    }
    return frame, report


def diagnostic_bundle(paths: list[Path]) -> Path:
    destination = (
        OUTPUT_ROOT / "bundles/TNormFilter_canonical_v3_cpu_audit_blocked.zip"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file():
                archive.write(path, path.relative_to(PROJECT_DIR))
    os.replace(temporary, destination)
    return destination


def main() -> None:
    protocol = load_protocol()
    if any(path.exists() for path in TEST_MARKERS):
        raise RuntimeError("Canonical v3 CPU audit blocked: test is already open")
    parent = protocol["parent_commit"]
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", parent, "HEAD"],
        cwd=PROJECT_DIR,
    ).returncode != 0:
        raise RuntimeError("Canonical v3 parent commit is not an ancestor")
    development, test = frozen_manifests(protocol)
    objects = development_object_table(development)
    matrix = class_scene_matrix(development, objects, protocol)
    folds, search = constrained_folds(matrix, protocol)
    fold_rows, folds_valid = fold_audit(folds, matrix, protocol)
    atomic_csv(objects, AUDIT_ROOT / "development_objects.csv")
    atomic_csv(matrix, AUDIT_ROOT / "class_scene_matrix.csv")
    atomic_csv(fold_rows, AUDIT_ROOT / "fold_class_support.csv")
    fold_payload = {
        "protocol_id": protocol["protocol_id"],
        "group": "grouped_scene_id",
        "folds": {str(index): value for index, value in enumerate(folds)},
        "search": search,
        "all_constraints_passed": folds_valid,
        "test_used": False,
    }
    atomic_json(PROTOCOL_ROOT / "development_folds.json", fold_payload)
    fusion_rows, fusion_report = fusion_audit(development, protocol)
    atomic_csv(fusion_rows, AUDIT_ROOT / "fusion_loss_per_class.csv")
    atomic_json(AUDIT_ROOT / "fusion_loss_report.json", fusion_report)

    old_fold_training = matrix[
        ~matrix["grouped_scene_id"].isin(
            ["4_station_pedestrian_bridge_4", "19_vegetation_curve_19"]
        )
    ]
    road_column = "class_2_road_vehicle"
    road_fold0 = {
        "heldout_road_vehicle_gt": int(
            matrix[
                matrix["grouped_scene_id"].isin(
                    ["4_station_pedestrian_bridge_4", "19_vegetation_curve_19"]
                )
            ][road_column].sum()
        ),
        "train_road_vehicle_gt": int(old_fold_training[road_column].sum()),
        "train_road_vehicle_scenes": int(old_fold_training[road_column].gt(0).sum()),
        "road_vehicle_absent_from_train": bool(
            old_fold_training[road_column].sum() == 0
        ),
    }
    class_scene_support = {
        column: int(matrix[column].gt(0).sum())
        for column in matrix.columns if column.startswith("class_")
    }
    blocking = fold_rows[~fold_rows["valid"]]
    passed = folds_valid
    report = {
        "protocol_id": protocol["protocol_id"],
        "status": "PASS" if passed else "BLOCKED_DEVELOPMENT_DATA_SUPPORT",
        "training_allowed": passed,
        "test_sealed": True,
        "test_labels_read": False,
        "development_frames": int(len(development)),
        "development_scenes": int(development["grouped_scene_id"].nunique()),
        "sealed_test_frames": int(len(test)),
        "sealed_test_scenes": int(test["grouped_scene_id"].nunique()),
        "class_scene_support": class_scene_support,
        "fold_constraints_passed": folds_valid,
        "blocking_fold_class_rows": blocking.to_dict(orient="records"),
        "old_fold_0_road_vehicle": road_fold0,
        "fusion": fusion_report,
        "next_allowed_action": (
            "start_C1_C2_development_triage"
            if passed
            else protocol["failure_policy"]["next_allowed_action"]
        ),
    }
    atomic_json(AUDIT_ROOT / "generalization_failure_audit.json", report)
    lock = {
        "status": "LOCKED_AUDIT_PASS" if passed else "LOCKED_AUDIT_BLOCKED",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "parent_commit": parent,
        "development_manifest_sha256": sha256(
            PROTOCOL_ROOT / "development_manifest.csv"
        ),
        "sealed_test_manifest_sha256": sha256(
            PROTOCOL_ROOT / "sealed_test_manifest.csv"
        ),
        "development_folds_sha256": sha256(
            PROTOCOL_ROOT / "development_folds.json"
        ),
        "audit_report_sha256": sha256(
            AUDIT_ROOT / "generalization_failure_audit.json"
        ),
        "test_sealed": True,
        "training_allowed": passed,
    }
    atomic_json(PROTOCOL_ROOT / "protocol_lock.json", lock)
    paths = [
        PROTOCOL_PATH,
        PROTOCOL_ROOT / "development_manifest.csv",
        PROTOCOL_ROOT / "sealed_test_manifest.csv",
        PROTOCOL_ROOT / "development_folds.json",
        PROTOCOL_ROOT / "protocol_lock.json",
        AUDIT_ROOT / "development_objects.csv",
        AUDIT_ROOT / "class_scene_matrix.csv",
        AUDIT_ROOT / "fold_class_support.csv",
        AUDIT_ROOT / "fusion_loss_per_class.csv",
        AUDIT_ROOT / "fusion_loss_report.json",
        AUDIT_ROOT / "generalization_failure_audit.json",
    ]
    if not passed:
        bundle = diagnostic_bundle(paths)
        report["diagnostic_bundle"] = str(bundle.resolve())
        report["diagnostic_bundle_sha256"] = sha256(bundle)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
