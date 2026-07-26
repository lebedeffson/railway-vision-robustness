from __future__ import annotations

import inspect
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.temporal_safety.common import fold_scenes
from src.temporal_verifier.features import (
    FEATURE_NAMES,
    build_track_features,
    label_tracks,
    rule_acceptance,
)
from src.temporal_verifier.pipeline import (
    accepted_predictions,
    fit_nested_logistic,
    system_deltas,
)


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (PROJECT / "configs/temporal_verifier_v1.yaml").read_text(encoding="utf-8")
)
OUTPUT = PROJECT / "outputs/temporal_verifier_v1"


def observations() -> pd.DataFrame:
    rows = []
    for frame, confidence, gt_iou in [
        (0, 0.04, 0.6),
        (1, 0.30, 0.7),
        (2, 0.02, 0.0),
    ]:
        rows.append({
            "track_key": "ocsort::support::q::1",
            "track_id": 1,
            "grouped_scene_id": "scene-a",
            "subsequence_id": "q",
            "frame_number": frame,
            "image_path": f"f{frame}",
            "is_detector_hit": frame != 2,
            "interpolated": frame == 2,
            "missing_streak": int(frame == 2),
            "detector_confidence": confidence if frame != 2 else 0.0,
            "tracker_score": max(confidence, 0.07),
            "x1": 10.0 + frame,
            "y1": 10.0,
            "x2": 20.0 + frame,
            "y2": 30.0,
            "image_width": 100,
            "image_height": 100,
            "gt_iou": gt_iou,
            "duplicate_overlap": False,
        })
    return pd.DataFrame(rows)


def test_track_features_no_ground_truth_at_inference() -> None:
    source = inspect.getsource(build_track_features)
    assert "gt_iou" not in FEATURE_NAMES
    assert 'observations["gt_iou"]' not in source
    features = build_track_features(observations())
    assert "gt_iou" not in features
    assert np.isfinite(features[list(FEATURE_NAMES)].to_numpy()).all()


def test_track_labels_are_separate_from_inference_features() -> None:
    frame = observations()
    features = build_track_features(frame)
    labels = label_tracks(frame, 0.5, 2, 0.5, 0.3)
    assert "target" not in features
    assert labels.loc[0, "target"] == 1


def test_track_table_scene_isolation() -> None:
    screening = fold_scenes(CONFIG["data"]["screening_fold"])
    confirmation = fold_scenes(CONFIG["data"]["confirmation_fold"])
    support = set().union(
        *[
            fold_scenes(fold)
            for fold in range(5)
            if fold not in CONFIG["data"]["support_excludes_folds"]
        ]
    )
    assert not screening & confirmation
    assert not support & screening
    assert not support & confirmation
    assert len(support) == CONFIG["data"]["support_scenes"]


def test_low_confidence_cannot_birth_verified_output_alone() -> None:
    features = build_track_features(observations().iloc[[0]].copy())
    accepted = rule_acceptance(
        features, 2, 3, 0.25, CONFIG["rule_verifier"]["fixed"]
    )
    assert not bool(accepted.iloc[0])


def test_low_confidence_can_continue_confirmed_track() -> None:
    features = build_track_features(observations())
    accepted = rule_acceptance(
        features, 2, 3, 0.25, CONFIG["rule_verifier"]["fixed"]
    )
    assert bool(accepted.iloc[0])
    assert features.loc[0, "low_confidence_hits"] >= 1


def test_interpolated_frames_not_counted_as_detector_hits() -> None:
    features = build_track_features(observations())
    assert features.loc[0, "track_length"] == 3
    assert features.loc[0, "observed_frames"] == 2
    assert features.loc[0, "interpolated_frames"] == 1
    assert features.loc[0, "max_hits_window_3"] == 2


def synthetic_logistic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    labels = []
    for scene_index in range(4):
        for target in (0, 1):
            for sample in range(3):
                base = 0.2 + 0.6 * target + 0.01 * sample
                row = {
                    "track_key": f"s{scene_index}-{target}-{sample}",
                    "grouped_scene_id": f"s{scene_index}",
                }
                row.update({name: base for name in FEATURE_NAMES})
                rows.append(row)
                labels.append({"track_key": row["track_key"], "target": target})
    return pd.DataFrame(rows), pd.DataFrame(labels)


def test_verifier_nested_scene_cv_and_calibration_inner_train_only() -> None:
    features, labels = synthetic_logistic_data()
    fitted = fit_nested_logistic(
        features,
        labels,
        CONFIG["learned_verifier"]["logistic"],
        maximum_folds=4,
    )
    assert np.isfinite(fitted.oof_probabilities).all()
    assert len(fitted.labelled_index) == len(features)
    assert CONFIG["learned_verifier"]["calibration"]["fit"] == (
        "support_scene_OOF_only"
    )


def test_threshold_is_global_not_per_scene() -> None:
    threshold = CONFIG["learned_verifier"]["threshold"]
    assert threshold["global_not_per_scene"] is True
    assert threshold["selection_role"] == "screening_fold_0"


def test_false_alarm_absolute_and_relative() -> None:
    baseline = {
        "recall": 0.2,
        "FN_per_frame": 1.0,
        "false_alarms_per_minute": 10.0,
        "f1": 0.4,
    }
    candidate = {
        "recall": 0.3,
        "FN_per_frame": 0.8,
        "false_alarms_per_minute": 12.0,
        "f1": 0.38,
    }
    delta = system_deltas(baseline, candidate)
    assert delta["relative_false_alarm_increase"] == pytest.approx(0.2)
    assert candidate["false_alarms_per_minute"] == 12.0


def test_same_tracks_feed_all_verifiers() -> None:
    assert CONFIG["tracker"]["reuse_identical_tracks_for_all_verifiers"] is True
    assert CONFIG["rule_verifier"]["maximum_candidates"] == 12


def test_verifier_preserves_baseline_and_filters_only_temporal_additions() -> None:
    raw = {
        "f0": [{"class_id": 0, "box": [0, 0, 10, 10], "confidence": 0.8}]
    }
    additions = pd.DataFrame([
        {
            "track_key": "accepted",
            "image_path": "f0",
            "x1": 20,
            "y1": 20,
            "x2": 30,
            "y2": 40,
            "confidence": 0.3,
            "source": "temporal_confirmation",
            "interpolated": False,
        },
        {
            "track_key": "rejected",
            "image_path": "f0",
            "x1": 40,
            "y1": 20,
            "x2": 50,
            "y2": 40,
            "confidence": 0.3,
            "source": "temporal_confirmation",
            "interpolated": False,
        },
    ])
    output = accepted_predictions(raw, additions, {"accepted"}, 0.6)
    assert any(row["confidence"] == 0.8 for row in output["f0"])
    assert any(row.get("track_key") == "accepted" for row in output["f0"])
    assert not any(row.get("track_key") == "rejected" for row in output["f0"])


def test_test_sealed_before_gate() -> None:
    assert CONFIG["test_access"]["allowed_only_after_full_development_pass"] is True
    assert not (PROJECT / CONFIG["test_access"]["marker"]).exists()
    lock = OUTPUT / "protocol/protocol_lock.json"
    if lock.exists():
        payload = json.loads(lock.read_text(encoding="utf-8"))
        assert payload["test_status"] == "SEALED"
        assert payload["test_access_count"] == 0


def test_public_bundle_has_no_images_or_checkpoints() -> None:
    bundles = list((OUTPUT / "bundles").glob("*.zip"))
    for bundle in bundles:
        with zipfile.ZipFile(bundle) as archive:
            lowered = [name.lower() for name in archive.namelist()]
        assert not any(name.endswith((".jpg", ".jpeg", ".png")) for name in lowered)
        assert not any(name.endswith((".pt", ".pth", ".ckpt")) for name in lowered)
        assert not any("raw_predictions" in name for name in lowered)
        assert not any("model.pkl" in name for name in lowered)
