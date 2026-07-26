from __future__ import annotations

import itertools
import json
import os
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from scripts.temporal_safety.common import (
    config as parent_config,
    prediction_path,
    source_frames,
)
from scripts.temporal_verifier.common import (
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_csv,
    atomic_json,
    config,
    sha256,
)
from src.temporal_verifier.features import (
    FEATURE_NAMES,
    build_track_features,
    label_tracks,
    rule_acceptance,
)
from src.temporal_verifier.pipeline import (
    accepted_predictions,
    calibration_summary,
    collect_tracks,
    evaluate_system,
    fit_nested_logistic,
    load_role,
    predict_nested_logistic,
    system_deltas,
    threshold_eligible,
    two_fold_gate,
)


def role_source(role: str) -> tuple[int, pd.DataFrame, Path]:
    if role == "support":
        fold, parent_role = 0, "train"
    elif role == "screening":
        fold, parent_role = 0, "heldout"
    elif role == "confirmation":
        fold, parent_role = 1, "heldout"
    else:
        raise ValueError(role)
    return fold, source_frames(fold, parent_role), prediction_path(fold, parent_role)


def selected_parameters(tracker: str) -> dict[str, Any]:
    path = PROJECT / config()["frozen_parent"]["selected_trackers"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["selected"][tracker]["parameters"])


def dataset(
    tracker: str, role: str
) -> tuple[
    pd.DataFrame,
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    root = OUTPUT / "tracks" / tracker / role
    paths = {
        "observations": root / "track_observations.csv",
        "additions": root / "temporal_additions.csv",
        "features": root / "track_features.csv",
        "labels": root / "track_labels.csv",
    }
    fold, source, predictions_path = role_source(role)
    ground_truth, raw = load_role(source, str(predictions_path))
    if all(path.is_file() for path in paths.values()):
        return (
            source,
            ground_truth,
            raw,
            pd.read_csv(paths["observations"]),
            pd.read_csv(paths["additions"]),
            pd.read_csv(paths["features"]),
            pd.read_csv(paths["labels"]),
        )
    parent = parent_config()
    observations, additions = collect_tracks(
        tracker,
        role,
        source,
        ground_truth,
        raw,
        selected_parameters(tracker),
        parent["temporal_logic"],
        float(parent["temporal_logic"]["fusion_nms_iou"]),
    )
    features = build_track_features(observations)
    labels = label_tracks(
        observations,
        minimum_iou=float(config()["labels"]["positive"]["minimum_iou"]),
        minimum_matched_detector_frames=int(
            config()["labels"]["positive"]["minimum_matched_detector_frames"]
        ),
        minimum_matched_fraction=float(
            config()["labels"]["positive"]["minimum_matched_fraction"]
        ),
        maximum_negative_iou=float(
            config()["labels"]["negative"]["maximum_iou_exclusive"]
        ),
    )
    atomic_csv(paths["observations"], observations)
    atomic_csv(paths["additions"], additions)
    atomic_csv(paths["features"], features)
    atomic_csv(paths["labels"], labels)
    audit = {
        "tracker": tracker,
        "role": role,
        "fold": fold,
        "scenes": sorted(set(source["grouped_scene_id"].astype(str))),
        "tracks": len(features),
        "labels": labels["label"].value_counts().to_dict(),
        "inference_feature_names": list(FEATURE_NAMES),
        "ground_truth_in_inference_features": False,
        "test_used": False,
    }
    atomic_json(root / "TRACK_TABLE_AUDIT.json", audit)
    return source, ground_truth, raw, observations, additions, features, labels


def false_category(row: pd.Series) -> str:
    if row["track_length"] <= 2:
        return "short_single_noise"
    if row["interpolation_fraction"] >= 0.50:
        return "incorrect_interpolation"
    if row["duplicate_overlap_fraction"] >= 0.25:
        return "duplicate_track"
    if row["border_fraction"] >= 0.50:
        return "image_or_tile_border"
    if row["box_area_cv"] >= 0.80 or row["center_velocity_std"] >= 1.0:
        return "jumping_box"
    if row["aspect_ratio_mean"] <= 0.45 and row["center_velocity_mean"] <= 0.15:
        return "static_vertical_object"
    if row["track_length"] >= 5 and row["detection_fraction"] >= 0.60:
        return "persistent_hard_negative"
    return "other_false_track"


def false_track_audit(
    observations: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    table = features.merge(labels, on="track_key", validate="one_to_one")
    false = table[table["target"].eq(0)].copy()
    false["category"] = false.apply(false_category, axis=1)
    atomic_csv(OUTPUT / "audit/false_track_report.csv", false)
    per_scene = (
        false.groupby(["grouped_scene_id", "category"], as_index=False)
        .size()
        .rename(columns={"size": "false_tracks"})
    )
    atomic_csv(OUTPUT / "audit/false_tracks_per_scene.csv", per_scene)
    summary = {
        "false_tracks": len(false),
        "fraction_length_1_2": float((false["track_length"] <= 2).mean()),
        "fraction_without_confidence_0_25": float(
            (false["max_confidence"] < 0.25).mean()
        ),
        "fraction_unstable_size": float((false["box_area_cv"] >= 0.80).mean()),
        "fraction_almost_static": float(
            (false["center_velocity_mean"] <= 0.15).mean()
        ),
        "fraction_border": float((false["border_fraction"] >= 0.50).mean()),
        "fraction_interpolation_dominant": float(
            (false["interpolation_fraction"] >= 0.50).mean()
        ),
        "categories": false["category"].value_counts().to_dict(),
        "tile_border_note": "Only global-image border is available after tile fusion.",
    }
    atomic_json(OUTPUT / "audit/FALSE_TRACK_SUMMARY.json", summary)
    gallery = OUTPUT / "audit/false_track_gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    representative = (
        observations[
            observations["track_key"].astype(str).isin(false["track_key"].astype(str))
        ]
        .sort_values(["track_key", "detector_confidence"], ascending=[True, False])
        .drop_duplicates("track_key")
        .head(200)
    )
    for index, row in enumerate(representative.itertuples(index=False)):
        image = cv2.imread(str(row.image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        x1 = max(0, int(np.floor(row.x1)))
        y1 = max(0, int(np.floor(row.y1)))
        x2 = min(width, int(np.ceil(row.x2)))
        y2 = min(height, int(np.ceil(row.y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        cv2.imwrite(str(gallery / f"{index:04d}_{row.track_id}.jpg"), crop)


def baseline_metrics(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    raw: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    parent = parent_config()
    return evaluate_system(
        source,
        ground_truth,
        raw,
        float(parent["detector"]["operating_threshold"]),
        float(parent["detector"]["iou_threshold"]),
        float(parent["data"]["expected_nominal_fps"]),
    )


def candidate_metrics(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    raw: dict[str, list[dict[str, Any]]],
    additions: pd.DataFrame,
    accepted: set[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    parent = parent_config()
    predictions = accepted_predictions(
        raw,
        additions,
        accepted,
        float(config()["inference"]["nms_iou"]),
    )
    return evaluate_system(
        source,
        ground_truth,
        predictions,
        float(parent["detector"]["operating_threshold"]),
        float(parent["detector"]["iou_threshold"]),
        float(parent["data"]["expected_nominal_fps"]),
    )


def choose_rule(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    raw: dict[str, list[dict[str, Any]]],
    additions: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    baseline, _ = baseline_metrics(source, ground_truth, raw)
    protocol = config()["rule_verifier"]
    rows = []
    for k, window, high in itertools.product(
        protocol["k_detector_hits"],
        protocol["window_frames"],
        protocol["high_confidence_threshold"],
    ):
        accepted_mask = rule_acceptance(
            features, int(k), int(window), float(high), protocol["fixed"]
        )
        accepted = set(features.loc[accepted_mask, "track_key"].astype(str))
        metrics, _ = candidate_metrics(
            source, ground_truth, raw, additions, accepted
        )
        delta = system_deltas(baseline, metrics)
        eligible = threshold_eligible(delta, 0.10, 0.15)
        rows.append({
            "k_detector_hits": int(k),
            "window_frames": int(window),
            "high_confidence_threshold": float(high),
            "accepted_tracks": len(accepted),
            "screening_constraints_pass": eligible,
            **metrics,
            **{f"delta_{key}": value for key, value in delta.items()},
        })
    frame = pd.DataFrame(rows)
    eligible = frame[frame["screening_constraints_pass"]]
    pool = eligible if not eligible.empty else frame
    selected = pool.sort_values(
        ["f1", "recall", "false_alarms_per_minute", "high_confidence_threshold"],
        ascending=[False, False, True, False],
    ).iloc[0].to_dict()
    selected["selection_status"] = (
        "SCREENING_CONSTRAINTS_PASS"
        if bool(selected["screening_constraints_pass"])
        else "NO_RULE_MET_SCREENING_CONSTRAINTS"
    )
    return selected, frame


def accepted_by_rule(features: pd.DataFrame, selected: dict[str, Any]) -> set[str]:
    mask = rule_acceptance(
        features,
        int(selected["k_detector_hits"]),
        int(selected["window_frames"]),
        float(selected["high_confidence_threshold"]),
        config()["rule_verifier"]["fixed"],
    )
    return set(features.loc[mask, "track_key"].astype(str))


def fold_tag(frame: pd.DataFrame, fold: int, system: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "fold", fold)
    result.insert(0, "system", system)
    return result


def evaluate_two_fold(
    name: str,
    screen: tuple[Any, ...],
    confirm: tuple[Any, ...],
    accepted_screen: set[str],
    accepted_confirm: set[str],
) -> dict[str, Any]:
    source0, gt0, raw0, _, additions0, _, _ = screen
    source1, gt1, raw1, _, additions1, _, _ = confirm
    baseline0, baseline_scene0 = baseline_metrics(source0, gt0, raw0)
    baseline1, baseline_scene1 = baseline_metrics(source1, gt1, raw1)
    candidate0, candidate_scene0 = candidate_metrics(
        source0, gt0, raw0, additions0, accepted_screen
    )
    candidate1, candidate_scene1 = candidate_metrics(
        source1, gt1, raw1, additions1, accepted_confirm
    )
    gate = two_fold_gate(
        [baseline0, baseline1],
        [candidate0, candidate1],
        pd.concat([
            fold_tag(baseline_scene0, 0, "frame_detector"),
            fold_tag(baseline_scene1, 1, "frame_detector"),
        ], ignore_index=True),
        pd.concat([
            fold_tag(candidate_scene0, 0, name),
            fold_tag(candidate_scene1, 1, name),
        ], ignore_index=True),
        config()["development_gate"],
    )
    gate["candidate"] = name
    gate["per_fold"] = {
        "0": {"baseline": baseline0, "candidate": candidate0},
        "1": {"baseline": baseline1, "candidate": candidate1},
    }
    return gate


def select_logistic_threshold(
    fitted: Any,
    screen: tuple[Any, ...],
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    source, gt, raw, _, additions, features, _ = screen
    probabilities = predict_nested_logistic(fitted, features)
    baseline, _ = baseline_metrics(source, gt, raw)
    rows = []
    for threshold in np.linspace(
        0.0, 1.0, int(config()["learned_verifier"]["threshold"]["grid_points"])
    ):
        accepted = set(
            features.loc[probabilities >= threshold, "track_key"].astype(str)
        )
        metrics, _ = candidate_metrics(source, gt, raw, additions, accepted)
        delta = system_deltas(baseline, metrics)
        rows.append({
            "threshold": float(threshold),
            "accepted_tracks": len(accepted),
            "screening_constraints_pass": threshold_eligible(
                delta,
                float(
                    config()["learned_verifier"]["threshold"][
                        "absolute_recall_improvement_min"
                    ]
                ),
                float(
                    config()["learned_verifier"]["threshold"][
                        "relative_FN_reduction_min"
                    ]
                ),
            ),
            **metrics,
            **{f"delta_{key}": value for key, value in delta.items()},
        })
    frame = pd.DataFrame(rows)
    eligible = frame[frame["screening_constraints_pass"]]
    pool = eligible if not eligible.empty else frame
    selected = pool.sort_values(
        ["f1", "recall", "false_alarms_per_minute", "threshold"],
        ascending=[False, False, True, False],
    ).iloc[0].to_dict()
    selected["selection_status"] = (
        "SCREENING_CONSTRAINTS_PASS"
        if bool(selected["screening_constraints_pass"])
        else "NO_THRESHOLD_MET_SCREENING_CONSTRAINTS"
    )
    return selected, frame, probabilities


def main() -> None:
    lock = assert_locked()
    primary = config()["tracker"]["primary"]
    support = dataset(primary, "support")
    screen = dataset(primary, "screening")
    false_track_audit(support[3], support[5], support[6])

    selected_rule, rule_matrix = choose_rule(
        screen[0], screen[1], screen[2], screen[4], screen[5]
    )
    atomic_csv(OUTPUT / "rule/RULE_MATRIX.csv", rule_matrix)
    atomic_json(
        OUTPUT / "rule/PRE_CONFIRMATION_RULE_FREEZE.json",
        {
            "selected_rule": selected_rule,
            "selected_on": "fold_0_screening_only",
            "confirmation_read": False,
            "test_status": "SEALED",
        },
    )
    confirm = dataset(primary, "confirmation")
    rule_gate = evaluate_two_fold(
        "rule_verifier",
        screen,
        confirm,
        accepted_by_rule(screen[5], selected_rule),
        accepted_by_rule(confirm[5], selected_rule),
    )
    rule_gate["selected_rule"] = selected_rule
    atomic_json(OUTPUT / "rule/TWO_FOLD_RULE_GATE.json", rule_gate)

    final_stage = "rule"
    final_gate = rule_gate
    logistic_payload: dict[str, Any] | None = None
    if rule_gate["status"] == "FAIL":
        support_features, support_labels = support[5], support[6]
        fitted = fit_nested_logistic(
            support_features,
            support_labels,
            config()["learned_verifier"]["logistic"],
            int(
                config()["learned_verifier"]["nested_scene_cv"][
                    "maximum_inner_folds"
                ]
            ),
        )
        table = support_features.merge(
            support_labels[["track_key", "target"]],
            on="track_key",
            validate="one_to_one",
        )
        labelled = table["target"].to_numpy(dtype=int) >= 0
        calibration = calibration_summary(
            table.loc[labelled, "target"].to_numpy(dtype=int),
            fitted.oof_probabilities[labelled],
        )
        selected_threshold, threshold_sweep, screen_probabilities = (
            select_logistic_threshold(fitted, screen)
        )
        model_root = OUTPUT / "learned/logistic_l2"
        model_root.mkdir(parents=True, exist_ok=True)
        with (model_root / "model.pkl").open("wb") as handle:
            pickle.dump(fitted, handle)
        atomic_csv(model_root / "THRESHOLD_SWEEP.csv", threshold_sweep)
        atomic_json(
            model_root / "PRE_CONFIRMATION_FREEZE.json",
            {
                "threshold": selected_threshold,
                "features": list(FEATURE_NAMES),
                "calibration": calibration,
                "normalization": "inner_train_pipeline_only",
                "calibration_fit": "support_scene_OOF_only",
                "threshold_scope": "single_global_fold_0_screening_threshold",
                "confirmation_read_before_freeze": False,
                "model_sha256": sha256(model_root / "model.pkl"),
            },
        )
        confirm_probabilities = predict_nested_logistic(fitted, confirm[5])
        threshold = float(selected_threshold["threshold"])
        logistic_gate = evaluate_two_fold(
            "logistic_l2",
            screen,
            confirm,
            set(
                screen[5].loc[
                    screen_probabilities >= threshold, "track_key"
                ].astype(str)
            ),
            set(
                confirm[5].loc[
                    confirm_probabilities >= threshold, "track_key"
                ].astype(str)
            ),
        )
        logistic_gate["threshold_selection"] = selected_threshold
        logistic_gate["calibration"] = calibration
        atomic_json(model_root / "TWO_FOLD_LOGISTIC_GATE.json", logistic_gate)
        final_stage = "logistic_l2"
        final_gate = logistic_gate
        logistic_payload = logistic_gate

    # Sensitivity uses the already chosen primary verifier and never changes it.
    sensitivity_name = config()["tracker"]["sensitivity"]
    sensitivity_screen = dataset(sensitivity_name, "screening")
    sensitivity_confirm = dataset(sensitivity_name, "confirmation")
    if final_stage == "rule":
        accepted0 = accepted_by_rule(sensitivity_screen[5], selected_rule)
        accepted1 = accepted_by_rule(sensitivity_confirm[5], selected_rule)
    else:
        model_path = OUTPUT / "learned/logistic_l2/model.pkl"
        with model_path.open("rb") as handle:
            fitted = pickle.load(handle)
        threshold = float(
            logistic_payload["threshold_selection"]["threshold"]  # type: ignore[index]
        )
        accepted0 = set(
            sensitivity_screen[5].loc[
                predict_nested_logistic(fitted, sensitivity_screen[5])
                >= threshold,
                "track_key",
            ].astype(str)
        )
        accepted1 = set(
            sensitivity_confirm[5].loc[
                predict_nested_logistic(fitted, sensitivity_confirm[5])
                >= threshold,
                "track_key",
            ].astype(str)
        )
    sensitivity_gate = evaluate_two_fold(
        f"{sensitivity_name}_{final_stage}_sensitivity",
        sensitivity_screen,
        sensitivity_confirm,
        accepted0,
        accepted1,
    )
    atomic_json(OUTPUT / "sensitivity/BYTETRACK_GATE.json", sensitivity_gate)

    status = (
        "TWO_FOLD_PASS" if final_gate["status"] == "PASS" else "DEVELOPMENT_FAIL"
    )
    next_step = (
        "BLOCKED_MISSING_FOLDS_2_TO_4_OOF_TRACKS"
        if status == "TWO_FOLD_PASS"
        else (
            "CROP_VERIFIER_REQUIRES_NEW_AMENDMENT"
            if final_stage == "logistic_l2"
            else "LEARNED_VERIFIER_NOT_RUN"
        )
    )
    decision = {
        "protocol_id": config()["protocol_id"],
        "status": status,
        "primary_tracker": primary,
        "final_stage": final_stage,
        "final_gate": final_gate,
        "rule_gate": rule_gate["status"],
        "logistic_gate": (
            logistic_payload["status"] if logistic_payload else "NOT_RUN"
        ),
        "boosting_sensitivity": (
            "RELEASED_NOT_REQUIRED_FOR_SELECTION"
            if logistic_payload and logistic_payload["status"] == "PASS"
            else "SKIPPED_BY_DECISION_TREE"
        ),
        "next_step": next_step,
        "test_status": "SEALED",
        "test_access_count": 0,
        "v8b_unchanged": True,
        "parent_temporal_safety_unchanged": True,
        "implementation_commit": lock["implementation_commit"],
    }
    atomic_json(OUTPUT / "decision_trace.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
