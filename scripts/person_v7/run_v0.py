from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from scripts.audit_evaluator import average_precision, match_dataset
from scripts.person_canonical_v5.execution_evaluate import _size_counts
from scripts.person_v3.evaluate import (
    detections_frame,
    infer_all,
    person_gt,
)
from scripts.person_v7.common import (
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_json,
    config,
    sha256,
)
from src.temporal.ratta import (
    HomographyResult,
    TemporalAggregator,
    box_iou,
    deterministic_nms,
    estimate_background_homography,
    roi_embedding,
)
from src.tracklet_verifier.verifier import (
    FEATURE_NAMES,
    MONOTONIC_DIRECTIONS,
    MonotoneRankLogistic,
    build_tracklet_features,
    label_tracklets,
)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def folds() -> dict[str, list[str]]:
    path = PROJECT / config()["data"]["folds"]
    return json.loads(path.read_text(encoding="utf-8"))["folds"]


def source_frames(fold: int, role: str) -> pd.DataFrame:
    protocol = config()
    manifest = pd.read_csv(
        PROJECT / protocol["data"]["manifest"], dtype={"frame_id": str}
    )
    development = manifest[
        manifest["split"].isin(protocol["data"]["development_splits"])
    ].copy()
    heldout = set(map(str, folds()[str(fold)]))
    if role == "train":
        source = development[
            ~development["grouped_scene_id"].astype(str).isin(heldout)
        ].copy()
    elif role == "heldout":
        source = development[
            development["grouped_scene_id"].astype(str).isin(heldout)
        ].copy()
    else:
        raise ValueError(role)
    observed = set(source["grouped_scene_id"].astype(str))
    if role == "heldout" and observed != heldout:
        raise RuntimeError("Held-out source does not match the frozen fold")
    if role == "train" and observed & heldout:
        raise RuntimeError("Verifier train source contains a held-out scene")
    test_scenes = set(
        manifest.loc[manifest["split"].eq("test"), "grouped_scene_id"].astype(str)
    )
    if set(source["grouped_scene_id"].astype(str)) & test_scenes:
        raise RuntimeError("Canonical v7 source intersects sealed test")
    source["image_path"] = source["output_image"].map(
        lambda value: str(Path(value).resolve())
    )
    source["frame_number"] = pd.to_numeric(source["frame_id"], errors="raise")
    return source.sort_values(
        ["grouped_scene_id", "subsequence_id", "frame_number", "image_path"]
    ).reset_index(drop=True)


def parse_detection_csv(
    path: Path,
    allowed_images: set[str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    frame = pd.read_csv(path)
    if not set(frame["image_path"].astype(str)) <= allowed_images:
        raise RuntimeError(f"Prediction CSV leaks outside source: {path}")
    ground_truth: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in allowed_images:
        ground_truth[image] = []
        predictions[image] = []
    for row in frame.itertuples(index=False):
        item = {
            "class_id": 0,
            "box": [float(row.x1), float(row.y1), float(row.x2), float(row.y2)],
        }
        if row.kind == "ground_truth":
            ground_truth[str(row.image_path)].append(item)
        else:
            predictions[str(row.image_path)].append(
                {**item, "confidence": float(row.confidence)}
            )
    return dict(ground_truth), dict(predictions)


def training_predictions(
    fold: int,
    source: pd.DataFrame,
    destination: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    csv_path = destination / "predictions_and_ground_truth.csv"
    allowed = set(source["image_path"])
    if csv_path.is_file():
        return parse_detection_csv(csv_path, allowed)
    fold_config = config()["baseline"][f"fold_{fold}"]
    checkpoint = PROJECT / fold_config["checkpoint"]
    if sha256(checkpoint) != fold_config["checkpoint_sha256"]:
        raise RuntimeError("B0 checkpoint hash mismatch")
    gt = person_gt(source)
    predictions, runtime_ms = infer_all(fold, checkpoint, source, destination)
    atomic_csv(csv_path, detections_frame(source, gt, predictions))
    atomic_json(
        destination / "INFERENCE_COMPLETE.json",
        {
            "fold": fold,
            "role": "train",
            "frames": len(source),
            "scenes": sorted(set(source["grouped_scene_id"].astype(str))),
            "runtime_ms": runtime_ms,
            "checkpoint_sha256": sha256(checkpoint),
            "test_used": False,
        },
    )
    return gt, predictions


def heldout_predictions(
    fold: int,
    source: pd.DataFrame,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    path = PROJECT / config()["baseline"][f"fold_{fold}"]["heldout_predictions"]
    return parse_detection_csv(path, set(source["image_path"]))


def homographies(
    fold: int,
    role: str,
    source: pd.DataFrame,
    predictions: dict[str, list[dict[str, Any]]],
    destination: Path,
) -> dict[str, HomographyResult | None]:
    cache_path = destination / "homography_audit.csv"
    matrix_path = destination / "homography_matrices.npz"
    if cache_path.is_file() and matrix_path.is_file():
        audit = pd.read_csv(cache_path)
        matrices = np.load(matrix_path)
        result: dict[str, HomographyResult | None] = {}
        for row in audit.itertuples(index=False):
            if bool(row.boundary):
                result[str(row.image_path)] = None
            else:
                matrix = matrices[str(row.matrix_key)]
                result[str(row.image_path)] = HomographyResult(
                    matrix,
                    bool(row.valid),
                    int(row.matches),
                    int(row.inliers),
                    float(row.inlier_ratio),
                )
        return result
    temporal = json.loads(
        json.dumps(
            __import__("yaml").safe_load(
                (
                    PROJECT / config()["tracklets"]["temporal_config"]
                ).read_text(encoding="utf-8")
            )["temporal"]
        )
    )
    rows = []
    matrices: dict[str, np.ndarray] = {}
    result: dict[str, HomographyResult | None] = {}
    for subsequence, sequence in source.groupby("subsequence_id", sort=False):
        previous_image = None
        previous_path = None
        for index, record in enumerate(sequence.itertuples(index=False)):
            image_path = str(record.image_path)
            current = cv2.imread(image_path)
            if current is None:
                raise RuntimeError(f"Unreadable frame: {image_path}")
            key = f"h_{len(rows):06d}"
            if previous_image is None:
                item = None
                matrix = np.eye(3)
                boundary = True
                matches = inliers = 0
                ratio = 0.0
                valid = False
            else:
                masked = [
                    row["box"]
                    for row in predictions.get(str(previous_path), [])
                    if float(row["confidence"])
                    >= float(config()["baseline"]["operating_threshold"])
                ]
                item = estimate_background_homography(
                    previous_image,
                    current,
                    masked,
                    temporal["camera_compensation"],
                )
                matrix = item.matrix
                boundary = False
                matches = item.matches
                inliers = item.inliers
                ratio = item.inlier_ratio
                valid = item.valid
            result[image_path] = item
            matrices[key] = matrix
            rows.append(
                {
                    "fold": fold,
                    "role": role,
                    "grouped_scene_id": str(record.grouped_scene_id),
                    "subsequence_id": str(subsequence),
                    "image_path": image_path,
                    "matrix_key": key,
                    "boundary": boundary,
                    "valid": valid,
                    "matches": matches,
                    "inliers": inliers,
                    "inlier_ratio": ratio,
                }
            )
            previous_image = current
            previous_path = image_path
    atomic_csv(cache_path, pd.DataFrame(rows))
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(matrix_path, **matrices)
    return result


def maximum_iou(box: list[float], targets: list[dict[str, Any]]) -> float:
    return max([box_iou(box, target["box"]) for target in targets], default=0.0)


def _raw_confidence(
    emitted: dict[str, Any],
    raw: list[dict[str, Any]],
) -> tuple[float, bool]:
    if bool(emitted.get("propagated", False)):
        return 0.0, False
    matches = [
        (box_iou(emitted["box"], row["box"]), float(row["confidence"]))
        for row in raw
    ]
    if not matches:
        return 0.0, False
    overlap, confidence = max(matches)
    if overlap < 0.999:
        raise RuntimeError("Tracker detection cannot be traced to frozen B0 proposal")
    return confidence, True


def tracklet_observations(
    fold: int,
    role: str,
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    transformations: dict[str, HomographyResult | None],
) -> pd.DataFrame:
    temporal = __import__("yaml").safe_load(
        (PROJECT / config()["tracklets"]["temporal_config"]).read_text(
            encoding="utf-8"
        )
    )["temporal"]
    rows: list[dict[str, Any]] = []
    for subsequence, sequence in source.groupby("subsequence_id", sort=False):
        aggregator = TemporalAggregator(temporal, "bytetrack")
        aggregator.reset()
        for frame_order, record in enumerate(sequence.itertuples(index=False)):
            image_path = str(record.image_path)
            image = cv2.imread(image_path)
            if image is None:
                raise RuntimeError(f"Unreadable frame: {image_path}")
            emitted, matched = aggregator.update(
                predictions.get(image_path, []),
                image,
                transformations[image_path],
            )
            match_by_track = {int(row["track_id"]): row for row in matched}
            state_by_track = {track.track_id: track for track in aggregator.tracks}
            for observation in emitted:
                track_id = int(observation["track_id"])
                state = state_by_track.get(track_id)
                if state is None:
                    continue
                if bool(observation.get("propagated", False)) and state.gap > int(
                    config()["tracklets"]["maximum_propagated_gap"]
                ):
                    continue
                detector_confidence, is_detection = _raw_confidence(
                    observation, predictions.get(image_path, [])
                )
                member = match_by_track.get(track_id, {})
                embedding = roi_embedding(image, observation["box"])
                height, width = image.shape[:2]
                row = {
                    "fold": fold,
                    "role": role,
                    "track_key": f"{subsequence}::{track_id}",
                    "track_id": track_id,
                    "grouped_scene_id": str(record.grouped_scene_id),
                    "subsequence_id": str(subsequence),
                    "frame_id": str(record.frame_id),
                    "frame_order": frame_order,
                    "image_path": image_path,
                    "detector_confidence": detector_confidence,
                    "tracker_confidence": float(observation["confidence"]),
                    "is_detection": is_detection,
                    "propagated": bool(observation.get("propagated", False)),
                    "x1": float(observation["box"][0]),
                    "y1": float(observation["box"][1]),
                    "x2": float(observation["box"][2]),
                    "y2": float(observation["box"][3]),
                    "image_width": width,
                    "image_height": height,
                    "gt_iou": maximum_iou(
                        observation["box"], ground_truth.get(image_path, [])
                    ),
                    "mu_motion": float(member.get("mu_motion", np.nan)),
                    "mu_appearance": float(member.get("mu_appearance", np.nan)),
                    "mu_scale": float(member.get("mu_scale", np.nan)),
                    "mu_border": float(member.get("mu_border", np.nan)),
                    "mu_compensated_iou": float(
                        member.get("mu_compensated_iou", np.nan)
                    ),
                }
                row.update(
                    {
                        f"embedding_{index}": float(value)
                        for index, value in enumerate(embedding)
                    }
                )
                rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No tracklet observations generated")
    return frame


def prepare_dataset(
    fold: int,
    role: str,
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observations_path = root / "tracklet_observations.csv"
    features_path = root / "tracklet_features.csv"
    labels_path = root / "tracklet_labels.csv"
    if observations_path.is_file() and features_path.is_file() and labels_path.is_file():
        return (
            pd.read_csv(observations_path),
            pd.read_csv(features_path),
            pd.read_csv(labels_path),
        )
    transformations = homographies(
        fold, role, source, predictions, root / "homography"
    )
    observations = tracklet_observations(
        fold, role, source, ground_truth, predictions, transformations
    )
    features = build_tracklet_features(observations)
    labels = label_tracklets(
        observations,
        minimum_iou=float(config()["tracklets"]["positive"]["minimum_iou"]),
        minimum_matched_frames=int(
            config()["tracklets"]["positive"]["minimum_matched_frames"]
        ),
        minimum_matched_fraction=float(
            config()["tracklets"]["positive"]["minimum_matched_fraction"]
        ),
        negative_iou=float(
            config()["tracklets"]["negative"]["maximum_iou_exclusive"]
        ),
    )
    atomic_csv(observations_path, observations)
    atomic_csv(features_path, features)
    atomic_csv(labels_path, labels)
    return observations, features, labels


def model_factory(name: str) -> Callable[[], Any]:
    protocol = config()["models"]
    if name == "V0-A_monotone_rank_logistic":
        settings = protocol["logistic"]
        return lambda: MonotoneRankLogistic(
            directions=MONOTONIC_DIRECTIONS.copy(),
            epochs=int(settings["epochs"]),
            learning_rate=float(settings["learning_rate"]),
            weight_decay=float(settings["weight_decay"]),
            ranking_weight=float(settings["ranking_weight"]),
            maximum_pairs_per_epoch=int(settings["maximum_pairs_per_epoch"]),
            seed=int(settings["seed"]),
        )
    if name == "V0-B_monotone_hist_gradient_boosting":
        settings = protocol["gradient_boosting"]
        return lambda: HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(settings["learning_rate"]),
            max_iter=int(settings["max_iter"]),
            max_leaf_nodes=int(settings["max_leaf_nodes"]),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            l2_regularization=float(settings["l2_regularization"]),
            monotonic_cst=MONOTONIC_DIRECTIONS.tolist(),
            class_weight="balanced",
            random_state=int(settings["seed"]),
        )
    raise KeyError(name)


def grouped_oof(
    name: str,
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[np.ndarray, Any, IsotonicRegression]:
    table = features.merge(labels[["track_key", "target"]], on="track_key")
    groups = table["grouped_scene_id"].astype(str).to_numpy()
    target = table["target"].to_numpy(dtype=int)
    labelled = target >= 0
    unique_groups = np.unique(groups)
    splits = min(int(config()["models"]["inner_group_folds"]), len(unique_groups))
    if splits < 2:
        raise RuntimeError("Insufficient train scenes for grouped OOF")
    raw = np.full(len(table), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=splits)
    x = table[list(FEATURE_NAMES)].to_numpy(dtype=float)
    for train_index, validation_index in splitter.split(x, groups=groups):
        train_labelled = train_index[labelled[train_index]]
        if len(np.unique(target[train_labelled])) < 2:
            raise RuntimeError("Inner grouped split lacks a verifier class")
        model = model_factory(name)()
        model.fit(x[train_labelled], target[train_labelled])
        raw[validation_index] = model.predict_proba(x[validation_index])[:, 1]
    if not np.isfinite(raw).all():
        raise RuntimeError("Grouped OOF verifier scores contain NaN/Inf")
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw[labelled], target[labelled])
    calibrated = np.asarray(calibrator.predict(raw), dtype=float)
    final = model_factory(name)()
    final.fit(x[labelled], target[labelled])
    return calibrated, final, calibrator


def track_probabilities(
    model: Any,
    calibrator: IsotonicRegression,
    features: pd.DataFrame,
) -> np.ndarray:
    raw = model.predict_proba(
        features[list(FEATURE_NAMES)].to_numpy(dtype=float)
    )[:, 1]
    return np.asarray(calibrator.predict(raw), dtype=float)


def scored_predictions(
    observations: pd.DataFrame,
    track_scores: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    alpha = float(config()["reranking"]["detector_weight_alpha"])
    nms_iou = float(config()["baseline"]["fusion_nms_iou"])
    predictions: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations.itertuples(index=False):
        track = float(track_scores[str(row.track_key)])
        score = alpha * float(row.detector_confidence) + (1.0 - alpha) * track
        predictions[str(row.image_path)].append(
            {
                "class_id": 0,
                "box": [float(row.x1), float(row.y1), float(row.x2), float(row.y2)],
                "confidence": float(score),
                "track_key": str(row.track_key),
                "propagated": bool(row.propagated),
            }
        )
    return {
        image: deterministic_nms(rows, nms_iou)
        for image, rows in predictions.items()
    }


def evaluate(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> dict[str, Any]:
    for image in source["image_path"]:
        predictions.setdefault(str(image), [])
        ground_truth.setdefault(str(image), [])
    operating = match_dataset(ground_truth, predictions, threshold, 0.50)
    size = _size_counts(ground_truth, predictions, threshold)
    small = float(size.loc[size["size"].eq("small"), "recall"].iloc[0])
    frames = len(source)
    return {
        "mAP50": float(average_precision(ground_truth, predictions, 0, 0.50)),
        "Precision": float(operating["precision"]),
        "Recall": float(operating["recall"]),
        "F1": float(operating["f1"]),
        "small_recall": small,
        "TP": int(operating["tp"]),
        "FP": int(operating["fp"]),
        "FN": int(operating["fn"]),
        "FP_per_frame": float(operating["fp"] / max(frames, 1)),
        "FN_per_frame": float(operating["fn"] / max(frames, 1)),
        "frames": frames,
        "GT": int(sum(map(len, ground_truth.values()))),
        "threshold": float(threshold),
    }


def threshold_selection(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    baseline_metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    settings = config()["reranking"]
    rows = []
    for threshold in np.linspace(0.0, 1.0, int(settings["threshold_grid_points"])):
        rows.append(evaluate(source, ground_truth, predictions, float(threshold)))
    sweep = pd.DataFrame(rows)
    standard_limit = (
        float(settings["standard"]["maximum_FP_multiplier_vs_B0"])
        * float(baseline_metrics["FP_per_frame"])
    )
    standard_pool = sweep[sweep["FP_per_frame"] <= standard_limit + 1e-12]
    if standard_pool.empty:
        raise RuntimeError("No standard threshold satisfies frozen FP constraint")
    standard_row = standard_pool.sort_values(
        ["F1", "Recall", "mAP50", "threshold"],
        ascending=[False, False, False, False],
    ).iloc[0]
    safety_pool = sweep[
        (sweep["Precision"] >= float(settings["safety"]["minimum_Precision"]))
        & (
            sweep["FP_per_frame"]
            <= float(settings["safety"]["maximum_FP_per_frame"])
        )
    ]
    if safety_pool.empty:
        safety = {"status": "UNAVAILABLE"}
    else:
        safety_row = safety_pool.sort_values(
            ["Recall", "Precision", "threshold"], ascending=[False, False, False]
        ).iloc[0]
        safety = {"status": "AVAILABLE", **safety_row.to_dict()}
    return standard_row.to_dict(), safety, sweep


def per_scene_metrics(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
    model: str,
) -> pd.DataFrame:
    rows = []
    for scene, group in source.groupby("grouped_scene_id"):
        images = set(group["image_path"].astype(str))
        gt = {image: ground_truth.get(image, []) for image in images}
        pred = {image: predictions.get(image, []) for image in images}
        rows.append(
            {
                "model": model,
                "grouped_scene_id": str(scene),
                **evaluate(group, gt, pred, threshold),
            }
        )
    return pd.DataFrame(rows)


def result_row(
    name: str,
    standard: dict[str, Any],
    safety: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": name,
        **{f"standard_{key}": value for key, value in standard.items()},
        **{
            f"safety_{key}": value
            for key, value in safety.items()
            if key != "status"
        },
        "safety_status": safety["status"],
    }


def run_fold_zero() -> None:
    assert_locked()
    root = OUTPUT / "fold_0"
    completion = root / "V0_COMPLETE.json"
    if completion.exists():
        raise RuntimeError("Canonical v7 fold 0 result already exists")
    train_source = source_frames(0, "train")
    heldout_source = source_frames(0, "heldout")
    train_scenes = set(train_source["grouped_scene_id"].astype(str))
    heldout_scenes = set(heldout_source["grouped_scene_id"].astype(str))
    if train_scenes & heldout_scenes:
        raise RuntimeError("Scene leakage between verifier train and heldout")

    train_gt, train_b0 = training_predictions(
        0, train_source, root / "train_inference"
    )
    train_observations, train_features, train_labels = prepare_dataset(
        0,
        "train",
        train_source,
        train_gt,
        train_b0,
        root / "train_tracklets",
    )
    label_counts = train_labels["label"].value_counts().to_dict()
    if label_counts.get("positive", 0) == 0 or label_counts.get("negative", 0) == 0:
        raise RuntimeError("Train tracklets do not contain both verifier classes")

    oof_results = []
    fitted: dict[str, tuple[Any, IsotonicRegression, dict[str, Any], dict[str, Any]]] = {}
    train_b0_metrics = evaluate(
        train_source,
        train_gt,
        train_b0,
        float(config()["baseline"]["operating_threshold"]),
    )
    for name in config()["models"]["candidate_order"]:
        oof_scores, final_model, calibrator = grouped_oof(
            name, train_features, train_labels
        )
        score_map = dict(zip(train_features["track_key"], oof_scores))
        predictions = scored_predictions(train_observations, score_map)
        standard, safety, sweep = threshold_selection(
            train_source, train_gt, predictions, train_b0_metrics
        )
        model_root = root / "models" / name
        model_root.mkdir(parents=True, exist_ok=True)
        with (model_root / "model.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "model": final_model,
                    "calibrator": calibrator,
                    "feature_names": FEATURE_NAMES,
                    "standard": standard,
                    "safety": safety,
                },
                handle,
            )
        atomic_csv(model_root / "threshold_sweep_train_oof.csv", sweep)
        atomic_json(
            model_root / "TRAIN_OOF_SELECTION.json",
            {
                "model": name,
                "standard": standard,
                "safety": safety,
                "selection_scope": "train_scene_grouped_OOF_only",
            },
        )
        oof_results.append(result_row(name, standard, safety))
        fitted[name] = (final_model, calibrator, standard, safety)
    oof_frame = pd.DataFrame(oof_results)
    atomic_csv(root / "TRAIN_OOF_MODEL_COMPARISON.csv", oof_frame)
    selected = (
        oof_frame.sort_values(
            [
                "standard_mAP50",
                "standard_F1",
                "standard_Recall",
                "standard_small_recall",
                "standard_FP_per_frame",
                "model",
            ],
            ascending=[False, False, False, False, True, True],
        )
        .iloc[0]["model"]
    )
    atomic_json(
        root / "PRE_HELDOUT_FREEZE.json",
        {
            "selected_model": selected,
            "selected_by": "train_scene_grouped_OOF_only",
            "feature_names": list(FEATURE_NAMES),
            "detector_weight_alpha": config()["reranking"][
                "detector_weight_alpha"
            ],
            "train_scenes": sorted(train_scenes),
            "heldout_scenes": sorted(heldout_scenes),
            "scene_intersection": [],
            "test_status": "SEALED",
        },
    )

    # Held-out fold 0 is deliberately opened only after PRE_HELDOUT_FREEZE exists.
    heldout_gt, heldout_b0 = heldout_predictions(0, heldout_source)
    heldout_observations, heldout_features, heldout_labels = prepare_dataset(
        0,
        "heldout",
        heldout_source,
        heldout_gt,
        heldout_b0,
        root / "heldout_tracklets",
    )
    baseline = evaluate(
        heldout_source,
        heldout_gt,
        heldout_b0,
        float(config()["baseline"]["operating_threshold"]),
    )
    heldout_rows = []
    scene_frames = []
    selected_parity: dict[str, Any] | None = None
    for name in config()["models"]["candidate_order"]:
        model, calibrator, standard_oof, safety_oof = fitted[name]
        probabilities = track_probabilities(model, calibrator, heldout_features)
        score_map = dict(zip(heldout_features["track_key"], probabilities))
        predictions = scored_predictions(heldout_observations, score_map)
        standard = evaluate(
            heldout_source,
            heldout_gt,
            predictions,
            float(standard_oof["threshold"]),
        )
        if safety_oof["status"] == "AVAILABLE":
            safety = {
                "status": "AVAILABLE",
                **evaluate(
                    heldout_source,
                    heldout_gt,
                    predictions,
                    float(safety_oof["threshold"]),
                ),
            }
        else:
            safety = {"status": "UNAVAILABLE"}
        heldout_rows.append(result_row(name, standard, safety))
        scene_frames.append(
            per_scene_metrics(
                heldout_source,
                heldout_gt,
                predictions,
                float(standard_oof["threshold"]),
                name,
            )
        )
        if name == selected:
            parity_path = root / "SELECTED_PREDICTIONS_AND_GROUND_TRUTH.csv"
            atomic_csv(
                parity_path,
                detections_frame(
                    heldout_source,
                    heldout_gt,
                    predictions,
                ),
            )
            repeated_gt, repeated_predictions = parse_detection_csv(
                parity_path, set(heldout_source["image_path"])
            )
            repeated = evaluate(
                heldout_source,
                repeated_gt,
                repeated_predictions,
                float(standard_oof["threshold"]),
            )
            parity_keys = (
                "mAP50",
                "Precision",
                "Recall",
                "F1",
                "small_recall",
                "FP_per_frame",
                "FN_per_frame",
            )
            differences = {
                key: abs(float(standard[key]) - float(repeated[key]))
                for key in parity_keys
            }
            selected_parity = {
                "status": (
                    "PASS"
                    if max(differences.values(), default=0.0) <= 1e-12
                    else "FAIL"
                ),
                "tolerance": 1e-12,
                "differences": differences,
                "direct_GT": standard["GT"],
                "reparsed_GT": repeated["GT"],
                "lost_GT": int(standard["GT"] - repeated["GT"]),
            }
    heldout_frame = pd.DataFrame(heldout_rows)
    atomic_csv(root / "HELDOUT_MODEL_COMPARISON.csv", heldout_frame)
    atomic_csv(root / "PER_SCENE_RESULTS.csv", pd.concat(scene_frames))
    selected_metrics = heldout_frame[heldout_frame["model"].eq(selected)].iloc[0]
    if selected_parity is None:
        raise RuntimeError("Selected verifier evaluator parity was not calculated")
    gate_config = config()["gate"]["fold_0_v0"]
    checks = {
        "mAP50": bool(selected_metrics["standard_mAP50"] >= gate_config["mAP50_min"]),
        "Recall": bool(selected_metrics["standard_Recall"] >= gate_config["Recall_min"]),
        "small_recall": bool(
            selected_metrics["standard_small_recall"]
            >= gate_config["small_recall_min"]
        ),
        "FP_per_frame": bool(
            selected_metrics["standard_FP_per_frame"]
            <= gate_config["FP_per_frame_max"]
        ),
        "F1_vs_B0": bool(selected_metrics["standard_F1"] > baseline["F1"]),
        "evaluator_consistency": selected_parity["status"] == "PASS",
        "lost_GT": selected_parity["lost_GT"] == 0,
        "NaN_Inf": bool(
            np.isfinite(
                [
                    selected_metrics["standard_mAP50"],
                    selected_metrics["standard_Precision"],
                    selected_metrics["standard_Recall"],
                    selected_metrics["standard_F1"],
                    selected_metrics["standard_small_recall"],
                    selected_metrics["standard_FP_per_frame"],
                ]
            ).all()
        ),
        "scene_leakage": not bool(train_scenes & heldout_scenes),
    }
    passed = all(checks.values())
    gate = {
        "protocol_id": config()["protocol_id"],
        "fold": 0,
        "selected_model": selected,
        "selection_scope": "train_scene_grouped_OOF_only",
        "baseline": baseline,
        "selected_metrics": selected_metrics.to_dict(),
        "evaluator_parity": selected_parity,
        "checks": checks,
        "V0_GATE": "PASS" if passed else "FAIL",
        "fold_1_status": "RELEASED" if passed else "BLOCKED",
        "crop_verifier_status": "NOT_REQUIRED" if passed else "ELIGIBLE_FOR_NEW_LOCK",
        "test_status": "SEALED",
        "attacks_status": "BLOCKED",
    }
    atomic_json(root / "V0_GATE.json", gate)
    atomic_json(root / "EVALUATOR_PARITY.json", selected_parity)
    atomic_json(
        OUTPUT / "decision_trace.json",
        {
            "protocol_id": config()["protocol_id"],
            "events": [
                {
                    "event": "TRAIN_SCENE_OOF_FREEZE",
                    "selected_model": selected,
                    "artifact": "fold_0/PRE_HELDOUT_FREEZE.json",
                },
                {
                    "event": "HELDOUT_FOLD_0_SCREENING",
                    "decision": gate["V0_GATE"],
                    "failed_checks": [
                        key for key, value in checks.items() if not value
                    ],
                },
            ],
            "fold_1_read": False,
            "test_used": False,
        },
    )
    atomic_json(
        completion,
        {
            "status": "V0_PASS" if passed else "V0_FAIL",
            "selected_model": selected,
            "gate_sha256": sha256(root / "V0_GATE.json"),
            "train_tracklets": len(train_features),
            "train_labels": label_counts,
            "heldout_tracklets": len(heldout_features),
            "heldout_label_counts_diagnostic_only": heldout_labels[
                "label"
            ].value_counts().to_dict(),
            "test_used": False,
        },
    )
    print(json.dumps(gate, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, choices=[0])
    args = parser.parse_args()
    if args.fold != 0:
        raise RuntimeError("Fold 1 runner is blocked until V0 fold 0 PASS")
    run_fold_zero()


if __name__ == "__main__":
    main()
