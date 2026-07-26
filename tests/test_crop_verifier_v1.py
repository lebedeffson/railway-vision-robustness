from __future__ import annotations

import inspect
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.crop_verifier.common import sha256
from scripts.temporal_safety.common import fold_scenes
from src.crop_verifier_v1.encoder import (
    context_crop,
    select_real_crop_rows,
)
from src.crop_verifier_v1.model import (
    fit_grouped_verifier,
    predict_verifier,
)


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (PROJECT / "configs/crop_verifier_v1.yaml").read_text(encoding="utf-8")
)
OUTPUT = PROJECT / "outputs/crop_verifier_v1"


def observations() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "track_key": "track",
            "grouped_scene_id": "scene",
            "frame_number": 0,
            "image_path": "a.jpg",
            "is_detector_hit": True,
            "interpolated": False,
            "detector_confidence": 0.10,
            "x1": 1,
            "y1": 2,
            "x2": 11,
            "y2": 22,
        },
        {
            "track_key": "track",
            "grouped_scene_id": "scene",
            "frame_number": 1,
            "image_path": "b.jpg",
            "is_detector_hit": False,
            "interpolated": True,
            "detector_confidence": 0.0,
            "x1": 2,
            "y1": 2,
            "x2": 12,
            "y2": 22,
        },
        {
            "track_key": "track",
            "grouped_scene_id": "scene",
            "frame_number": 2,
            "image_path": "c.jpg",
            "is_detector_hit": True,
            "interpolated": False,
            "detector_confidence": 0.50,
            "x1": 3,
            "y1": 2,
            "x2": 13,
            "y2": 22,
        },
        {
            "track_key": "track",
            "grouped_scene_id": "scene",
            "frame_number": 3,
            "image_path": "d.jpg",
            "is_detector_hit": True,
            "interpolated": False,
            "detector_confidence": 0.20,
            "x1": 4,
            "y1": 2,
            "x2": 14,
            "y2": 22,
        },
    ])


def test_three_crop_slots_use_real_detector_frames_only() -> None:
    selected = select_real_crop_rows(observations())
    assert selected["crop_slot"].tolist() == CONFIG["encoder"]["crop_slots"]
    assert selected["frame_number"].tolist() == [2, 2, 3]
    assert selected["is_detector_hit"].all()
    assert not selected["interpolated"].any()


def test_insufficient_hits_repeat_only_real_crop() -> None:
    selected = select_real_crop_rows(observations().iloc[[0, 1]])
    assert len(selected) == 3
    assert selected["frame_number"].tolist() == [0, 0, 0]
    assert selected["is_detector_hit"].all()


def test_visual_encoder_has_no_ground_truth_dependency() -> None:
    source = inspect.getsource(select_real_crop_rows)
    assert "gt_iou" not in source
    assert CONFIG["data"]["interpolated_crops_allowed"] is False


def test_context_crop_is_fixed_size_and_border_safe() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    crop = context_crop(image, [-3, -2, 5, 10], 1.75, 64, 114)
    assert crop.shape == (64, 64, 3)
    assert crop.dtype == np.uint8
    assert np.any(crop == 114)


def test_scene_disjoint_support_screening_confirmation() -> None:
    screening = fold_scenes(CONFIG["data"]["screening_fold"])
    confirmation = fold_scenes(CONFIG["data"]["confirmation_fold"])
    support = set().union(
        *[
            fold_scenes(index)
            for index in range(5)
            if index not in {
                CONFIG["data"]["screening_fold"],
                CONFIG["data"]["confirmation_fold"],
            }
        ]
    )
    assert not support & screening
    assert not support & confirmation
    assert not screening & confirmation
    assert len(support | screening | confirmation) == 15


def test_grouped_logistic_oof_and_calibration_are_finite() -> None:
    rng = np.random.default_rng(20260726)
    rows = []
    target = []
    groups = []
    for scene in range(5):
        for label in (0, 1):
            for _ in range(4):
                rows.append(rng.normal(loc=label, scale=0.2, size=8))
                target.append(label)
                groups.append(f"scene-{scene}")
    x = np.asarray(rows)
    y = np.asarray(target)
    grouped = np.asarray(groups)
    fitted = fit_grouped_verifier(
        x,
        y,
        grouped,
        CONFIG["models"]["logistic"],
        maximum_folds=5,
    )
    assert np.isfinite(fitted.oof_probabilities).all()
    predicted = predict_verifier(fitted, x[:3])
    assert predicted.shape == (3,)
    assert np.all((predicted >= 0) & (predicted <= 1))


def test_model_ablation_and_global_threshold_are_frozen() -> None:
    assert CONFIG["models"]["candidates"] == [
        "track_only",
        "visual_only",
        "combined",
    ]
    assert CONFIG["threshold_selection"]["global_not_per_scene"] is True
    assert CONFIG["models"]["sensitivity_mlp"][
        "allowed_only_after_logistic_triage_pass"
    ] is True


def test_triage_and_final_false_alarm_limits_are_distinct() -> None:
    assert CONFIG["triage_gate"]["relative_false_alarms_increase_max"] == 0.75
    assert (
        CONFIG["development_gate"]["relative_false_alarms_increase_max"] == 0.20
    )
    assert CONFIG["full_development"]["run_only_after_triage_pass"] is True


def test_detector_tracker_checkpoint_are_frozen() -> None:
    checkpoint = PROJECT / CONFIG["encoder"]["checkpoint"]
    if checkpoint.is_file():
        assert sha256(checkpoint) == CONFIG["encoder"]["checkpoint_sha256"]
    assert CONFIG["frozen_inputs"]["tracker"] == "ocsort"
    assert CONFIG["parent_status"] == "DEVELOPMENT_FAIL"


def test_crop_leakage_audit_if_available() -> None:
    audit = OUTPUT / "audit/CROP_LEAKAGE_AUDIT.json"
    if not audit.is_file():
        return
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["interpolated_crops"] == 0
    for row in payload["cross_role_checks"].values():
        assert row["scene_overlap"] == []
        assert row["crop_source_overlap_count"] == 0


def test_test_is_sealed_and_fail_closes_temporal_direction() -> None:
    assert not (PROJECT / CONFIG["test_access"]["marker"]).exists()
    assert not (PROJECT / CONFIG["test_access"]["legacy_marker"]).exists()
    decision = OUTPUT / "decision_trace.json"
    if decision.is_file():
        payload = json.loads(decision.read_text(encoding="utf-8"))
        assert payload["test_status"] == "SEALED"
        assert payload["test_access_count"] == 0
        if payload["two_fold_triage"]["status"] == "FAIL":
            assert payload["temporal_direction"] == "CLOSED_NO_PRACTICAL_GATE"
            assert payload["next_step"] == "NEW_RAILWAY_SCENES_REQUIRED"


def test_public_bundle_excludes_private_visual_payloads() -> None:
    for bundle in (OUTPUT / "bundles").glob("*.zip"):
        with zipfile.ZipFile(bundle) as archive:
            names = [name.lower() for name in archive.namelist()]
        prohibited = (
            ".jpg",
            ".jpeg",
            ".png",
            ".pt",
            ".npz",
            "model.pkl",
            "crop_metadata.csv",
            "confusion_audit_private.csv",
            "track_observations.csv",
            "raw_predictions.parquet",
            "test_opened.json",
        )
        assert not any(
            token in name for name in names for token in prohibited
        )
