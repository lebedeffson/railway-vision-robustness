from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from scripts.person_v7.run_v0 import (
    evaluate,
    grouped_oof,
    heldout_predictions,
    parse_detection_csv,
    per_scene_metrics,
    result_row,
    scored_predictions,
    source_frames,
    threshold_selection,
    track_probabilities,
)
from scripts.person_v7_crop.common import (
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_json,
    config,
    sha256,
)
from src.crop_verifier.crop_model import CropMLPClassifier, FrozenYoloCropEncoder
from src.tracklet_verifier.verifier import MonotoneRankLogistic


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def crop_factory() -> Callable[[], CropMLPClassifier]:
    settings = config()["crop_classifier"]
    return lambda: CropMLPClassifier(
        hidden_units=int(settings["hidden_units"]),
        epochs=int(settings["epochs"]),
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        ranking_weight=float(settings["ranking_weight"]),
        maximum_pairs_per_epoch=int(settings["maximum_pairs_per_epoch"]),
        seed=int(settings["seed"]),
    )


def fusion_factory() -> Callable[[], MonotoneRankLogistic]:
    settings = config()["fusion"]
    return lambda: MonotoneRankLogistic(
        directions=np.asarray(settings["directions"], dtype=np.int8),
        epochs=int(settings["epochs"]),
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        ranking_weight=float(settings["ranking_weight"]),
        maximum_pairs_per_epoch=int(settings["maximum_pairs_per_epoch"]),
        seed=int(settings["seed"]),
    )


def encode_or_load(
    role: str,
    observations: pd.DataFrame,
    encoder: FrozenYoloCropEncoder,
) -> tuple[pd.DataFrame, np.ndarray]:
    root = OUTPUT / "fold_0" / f"{role}_crop_features"
    metadata_path = root / "metadata.csv"
    features_path = root / "features.npz"
    if metadata_path.is_file() and features_path.is_file():
        return pd.read_csv(metadata_path), np.load(features_path)["features"]
    metadata, features = encoder.encode(observations)
    atomic_csv(metadata_path, metadata)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(features_path, features=features)
    atomic_json(
        root / "ENCODING_COMPLETE.json",
        {
            "role": role,
            "tracklets": len(metadata),
            "feature_dimension": int(features.shape[1]),
            "encoder_checkpoint_sha256": config()["encoder"][
                "checkpoint_sha256"
            ],
            "encoder_frozen": True,
            "test_used": False,
        },
    )
    return metadata, features


def grouped_oof_estimator(
    factory: Callable[[], Any],
    metadata: pd.DataFrame,
    x: np.ndarray,
    labels: pd.DataFrame,
    inner_folds: int,
) -> tuple[np.ndarray, Any, IsotonicRegression]:
    label_map = labels.set_index("track_key")["target"].to_dict()
    target = np.asarray(
        [int(label_map[str(key)]) for key in metadata["track_key"]]
    )
    groups = metadata["grouped_scene_id"].astype(str).to_numpy()
    labelled = target >= 0
    raw = np.full(len(metadata), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=min(inner_folds, len(np.unique(groups))))
    for train_index, validation_index in splitter.split(x, groups=groups):
        train_labelled = train_index[labelled[train_index]]
        if len(np.unique(target[train_labelled])) < 2:
            raise RuntimeError("Crop inner grouped split lacks a class")
        model = factory()
        model.fit(x[train_labelled], target[train_labelled])
        raw[validation_index] = model.predict_proba(x[validation_index])[:, 1]
    if not np.isfinite(raw).all():
        raise RuntimeError("Crop OOF scores contain NaN/Inf")
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(raw[labelled], target[labelled])
    calibrated = np.asarray(calibrator.predict(raw), dtype=float)
    final = factory()
    final.fit(x[labelled], target[labelled])
    return calibrated, final, calibrator


def calibrated_scores(
    model: Any,
    calibrator: IsotonicRegression,
    x: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        calibrator.predict(model.predict_proba(x)[:, 1]),
        dtype=float,
    )


def quantize(
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    decimals = int(config()["fusion"]["score_quantization_decimals"])
    for rows in predictions.values():
        for row in rows:
            row["confidence"] = float(round(float(row["confidence"]), decimals))
    return predictions


def parity(
    root: Path,
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
    direct: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    scene_by_image = dict(
        zip(source["image_path"], source["grouped_scene_id"].astype(str))
    )
    for image in source["image_path"]:
        for target in ground_truth.get(str(image), []):
            rows.append(
                {
                    "image_path": str(image),
                    "grouped_scene_id": scene_by_image[str(image)],
                    "kind": "ground_truth",
                    "class_id": 0,
                    "confidence": 1.0,
                    "x1": target["box"][0],
                    "y1": target["box"][1],
                    "x2": target["box"][2],
                    "y2": target["box"][3],
                }
            )
        for prediction in predictions.get(str(image), []):
            rows.append(
                {
                    "image_path": str(image),
                    "grouped_scene_id": scene_by_image[str(image)],
                    "kind": "prediction",
                    "class_id": 0,
                    "confidence": prediction["confidence"],
                    "x1": prediction["box"][0],
                    "y1": prediction["box"][1],
                    "x2": prediction["box"][2],
                    "y2": prediction["box"][3],
                }
            )
    csv_path = root / "SELECTED_PREDICTIONS_AND_GROUND_TRUTH.csv"
    atomic_csv(csv_path, pd.DataFrame(rows))
    reparsed_gt, reparsed_predictions = parse_detection_csv(
        csv_path, set(source["image_path"].astype(str))
    )
    repeated = evaluate(
        source, reparsed_gt, reparsed_predictions, threshold
    )
    keys = (
        "mAP50",
        "Precision",
        "Recall",
        "F1",
        "small_recall",
        "FP_per_frame",
        "FN_per_frame",
    )
    differences = {
        key: abs(float(direct[key]) - float(repeated[key])) for key in keys
    }
    tolerance = 1e-12
    return {
        "status": (
            "PASS"
            if max(differences.values(), default=0.0) <= tolerance
            else "FAIL"
        ),
        "tolerance": tolerance,
        "differences": differences,
        "lost_GT": int(direct["GT"] - repeated["GT"]),
    }


def run() -> None:
    assert_locked()
    root = OUTPUT / "fold_0"
    if (root / "V1_V2_COMPLETE.json").exists():
        raise RuntimeError("Canonical v7 crop result already exists")
    protocol = config()
    base_root = PROJECT / "outputs/person_v7_tracklet_verifier/fold_0"
    train_observations = pd.read_csv(
        PROJECT / protocol["inputs"]["train_observations"]
    )
    train_tabular = pd.read_csv(PROJECT / protocol["inputs"]["train_features"])
    train_labels = pd.read_csv(PROJECT / protocol["inputs"]["train_labels"])
    train_source = source_frames(0, "train")
    train_gt, train_b0 = parse_detection_csv(
        base_root / "train_inference/predictions_and_ground_truth.csv",
        set(train_source["image_path"]),
    )

    encoder_config = protocol["encoder"]
    encoder = FrozenYoloCropEncoder(
        PROJECT / encoder_config["checkpoint"],
        int(encoder_config["backbone_last_layer_index"]),
        int(encoder_config["input_size"]),
        float(encoder_config["crop_context_multiplier"]),
        int(encoder_config["padding_value"]),
        int(encoder_config["batch_size"]),
    )
    train_meta, train_crop_x = encode_or_load(
        "train", train_observations, encoder
    )
    crop_oof, crop_model, crop_calibrator = grouped_oof_estimator(
        crop_factory(),
        train_meta,
        train_crop_x,
        train_labels,
        int(protocol["crop_classifier"]["inner_group_folds"]),
    )
    tabular_oof, _, _ = grouped_oof(
        protocol["inputs"]["v0_selected_model"],
        train_tabular,
        train_labels,
    )
    tabular_map = dict(zip(train_tabular["track_key"], tabular_oof))
    tabular_aligned = np.asarray(
        [tabular_map[str(key)] for key in train_meta["track_key"]]
    )
    fusion_x = np.column_stack([tabular_aligned, crop_oof])
    fusion_oof, fusion_model, fusion_calibrator = grouped_oof_estimator(
        fusion_factory(),
        train_meta,
        fusion_x,
        train_labels,
        int(protocol["fusion"]["inner_group_folds"]),
    )

    train_b0_metrics = evaluate(
        train_source, train_gt, train_b0, 0.07
    )
    train_candidates = {
        protocol["crop_classifier"]["name"]: crop_oof,
        protocol["fusion"]["name"]: fusion_oof,
    }
    train_rows = []
    thresholds: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for name, scores in train_candidates.items():
        score_map = dict(zip(train_meta["track_key"], scores))
        predictions = quantize(scored_predictions(train_observations, score_map))
        standard, safety, sweep = threshold_selection(
            train_source, train_gt, predictions, train_b0_metrics
        )
        thresholds[name] = (standard, safety)
        train_rows.append(result_row(name, standard, safety))
        atomic_csv(root / "models" / name / "threshold_sweep_train_oof.csv", sweep)
    train_comparison = pd.DataFrame(train_rows)
    atomic_csv(root / "TRAIN_OOF_MODEL_COMPARISON.csv", train_comparison)
    selected = (
        train_comparison.sort_values(
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
    models_root = root / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    with (models_root / "crop_and_fusion.pkl").open("wb") as handle:
        pickle.dump(
            {
                "crop_model": crop_model,
                "crop_calibrator": crop_calibrator,
                "fusion_model": fusion_model,
                "fusion_calibrator": fusion_calibrator,
                "selected": selected,
                "thresholds": thresholds,
            },
            handle,
        )
    atomic_json(
        root / "PRE_HELDOUT_FREEZE.json",
        {
            "selected_model": selected,
            "selected_by": "train_scene_grouped_OOF_only",
            "crop_checkpoint_sha256": encoder_config["checkpoint_sha256"],
            "encoder_frozen": True,
            "standard": thresholds[selected][0],
            "safety": thresholds[selected][1],
            "score_quantization_decimals": protocol["fusion"][
                "score_quantization_decimals"
            ],
            "test_status": "SEALED",
        },
    )

    # Held-out artifacts are read only after the amendment freeze exists.
    heldout_source = source_frames(0, "heldout")
    heldout_gt, heldout_b0 = heldout_predictions(0, heldout_source)
    heldout_observations = pd.read_csv(
        PROJECT / protocol["inputs"]["heldout_observations"]
    )
    heldout_tabular = pd.read_csv(
        PROJECT / protocol["inputs"]["heldout_features"]
    )
    heldout_meta, heldout_crop_x = encode_or_load(
        "heldout", heldout_observations, encoder
    )
    crop_heldout = calibrated_scores(
        crop_model, crop_calibrator, heldout_crop_x
    )
    with (PROJECT / protocol["inputs"]["v0_model_artifact"]).open("rb") as handle:
        v0_artifact = pickle.load(handle)
    tabular_heldout_raw = track_probabilities(
        v0_artifact["model"],
        v0_artifact["calibrator"],
        heldout_tabular,
    )
    tabular_heldout_map = dict(
        zip(heldout_tabular["track_key"], tabular_heldout_raw)
    )
    tabular_heldout = np.asarray(
        [tabular_heldout_map[str(key)] for key in heldout_meta["track_key"]]
    )
    fusion_heldout = calibrated_scores(
        fusion_model,
        fusion_calibrator,
        np.column_stack([tabular_heldout, crop_heldout]),
    )
    heldout_candidates = {
        protocol["crop_classifier"]["name"]: crop_heldout,
        protocol["fusion"]["name"]: fusion_heldout,
    }
    baseline = evaluate(heldout_source, heldout_gt, heldout_b0, 0.07)
    heldout_rows = []
    scenes = []
    selected_predictions = None
    for name, scores in heldout_candidates.items():
        score_map = dict(zip(heldout_meta["track_key"], scores))
        predictions = quantize(
            scored_predictions(heldout_observations, score_map)
        )
        standard_oof, safety_oof = thresholds[name]
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
        scenes.append(
            per_scene_metrics(
                heldout_source,
                heldout_gt,
                predictions,
                float(standard_oof["threshold"]),
                name,
            )
        )
        if name == selected:
            selected_predictions = predictions
    heldout = pd.DataFrame(heldout_rows)
    atomic_csv(root / "HELDOUT_MODEL_COMPARISON.csv", heldout)
    atomic_csv(root / "PER_SCENE_RESULTS.csv", pd.concat(scenes))
    selected_metrics = heldout[heldout["model"].eq(selected)].iloc[0]
    if selected_predictions is None:
        raise RuntimeError("Selected crop verifier predictions are missing")
    parity_result = parity(
        root,
        heldout_source,
        heldout_gt,
        selected_predictions,
        float(thresholds[selected][0]["threshold"]),
        {
            key.removeprefix("standard_"): value
            for key, value in selected_metrics.items()
            if key.startswith("standard_")
        },
    )
    gate_config = protocol["gate"]
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
        "evaluator_consistency": parity_result["status"] == "PASS",
        "lost_GT": parity_result["lost_GT"] == 0,
        "NaN_Inf": bool(
            np.isfinite(
                [
                    selected_metrics["standard_mAP50"],
                    selected_metrics["standard_Recall"],
                    selected_metrics["standard_small_recall"],
                    selected_metrics["standard_F1"],
                ]
            ).all()
        ),
    }
    passed = all(checks.values())
    gate = {
        "protocol_id": protocol["protocol_id"],
        "selected_model": selected,
        "selection_scope": "train_scene_grouped_OOF_only",
        "baseline": baseline,
        "selected_metrics": selected_metrics.to_dict(),
        "evaluator_parity": parity_result,
        "checks": checks,
        "V1_V2_GATE": "PASS" if passed else "FAIL",
        "fold_1_status": "RELEASED" if passed else "BLOCKED",
        "test_status": "SEALED",
        "attacks_status": "BLOCKED",
    }
    atomic_json(root / "V1_V2_GATE.json", gate)
    atomic_json(
        OUTPUT / "decision_trace.json",
        {
            "events": [
                {
                    "event": "V0_FAIL_RELEASED_CROP_AMENDMENT",
                    "parent_gate_sha256": sha256(
                        PROJECT / protocol["parent_gate"]
                    ),
                },
                {
                    "event": "TRAIN_SCENE_OOF_FREEZE",
                    "selected_model": selected,
                },
                {
                    "event": "HELDOUT_FOLD_0_SCREENING",
                    "decision": gate["V1_V2_GATE"],
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
        root / "V1_V2_COMPLETE.json",
        {
            "status": "PASS" if passed else "FAIL",
            "selected_model": selected,
            "train_tracklets": len(train_meta),
            "heldout_tracklets": len(heldout_meta),
            "gate_sha256": sha256(root / "V1_V2_GATE.json"),
            "test_used": False,
        },
    )
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    run()

