from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.person_v7.run_v0 import evaluate, source_frames
from src.tracklet_verifier.verifier import (
    FEATURE_NAMES,
    MONOTONIC_DIRECTIONS,
    MonotoneRankLogistic,
    build_tracklet_features,
    label_tracklets,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/canonical_v7_person_tracklet_verifier.yaml"


def observations() -> pd.DataFrame:
    rows = []
    for track, scene, ious, detected in (
        ("s1::1", "scene_1", [0.8, 0.7, 0.0], [True, True, False]),
        ("s2::1", "scene_2", [0.0, 0.1], [True, True]),
        ("s3::1", "scene_3", [0.4, 0.6], [True, True]),
    ):
        for index, (iou, hit) in enumerate(zip(ious, detected)):
            row = {
                "track_key": track,
                "grouped_scene_id": scene,
                "subsequence_id": track.split("::")[0],
                "frame_order": index,
                "detector_confidence": 0.04 + 0.01 * index if hit else 0.0,
                "is_detection": hit,
                "x1": 10.0 + index,
                "y1": 20.0,
                "x2": 18.0 + index,
                "y2": 36.0,
                "image_width": 100,
                "image_height": 80,
                "gt_iou": iou,
                "mu_motion": 0.8 if index else np.nan,
                "mu_appearance": 0.7 if index else np.nan,
                "mu_scale": 0.9 if index else np.nan,
                "mu_border": 1.0 if index else np.nan,
                "mu_compensated_iou": 0.6 if index else np.nan,
            }
            row.update({f"embedding_{j}": float(j + index) / 10 for j in range(4)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_v7_protocol_is_sealed_and_train_oof_only() -> None:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "canonical-v7-person-tracklet-verifier-v1"
    assert protocol["test_status"] == "SEALED"
    assert protocol["attacks_status"] == "BLOCKED"
    assert protocol["data"]["test_read_permitted"] is False
    assert protocol["models"]["calibration"] == "train_scene_OOF_isotonic"
    assert tuple(protocol["features"]["ordered"]) == FEATURE_NAMES
    assert np.array_equal(
        np.asarray(
            [
                protocol["features"]["monotonic_direction"][name]
                for name in FEATURE_NAMES
            ]
        ),
        MONOTONIC_DIRECTIONS,
    )
    assert not (ROOT / protocol["test_marker"]).exists()


def test_tracklet_label_contract_excludes_ambiguous() -> None:
    labels = label_tracklets(observations())
    lookup = labels.set_index("track_key")["label"].to_dict()
    assert lookup == {
        "s1::1": "positive",
        "s2::1": "negative",
        "s3::1": "ambiguous",
    }


def test_feature_extraction_is_finite_and_complete() -> None:
    features = build_tracklet_features(observations())
    assert tuple(features[list(FEATURE_NAMES)].columns) == FEATURE_NAMES
    assert np.isfinite(features[list(FEATURE_NAMES)].to_numpy()).all()
    assert features.set_index("track_key").loc["s1::1", "hits"] == 2
    assert features.set_index("track_key").loc["s1::1", "gaps"] == 1


def test_monotone_rank_logistic_respects_frozen_signs() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, len(FEATURE_NAMES)))
    y = (x[:, 0] - x[:, 12] > 0).astype(int)
    model = MonotoneRankLogistic(
        MONOTONIC_DIRECTIONS,
        epochs=25,
        maximum_pairs_per_epoch=64,
        seed=7,
    ).fit(x, y)
    constrained = MONOTONIC_DIRECTIONS != 0
    assert np.all(
        np.sign(model.coef_[constrained])
        == MONOTONIC_DIRECTIONS[constrained]
    )
    assert np.isfinite(model.predict_proba(x)).all()


def test_fold_zero_train_and_heldout_scenes_are_disjoint() -> None:
    train = set(source_frames(0, "train")["grouped_scene_id"].astype(str))
    heldout = set(source_frames(0, "heldout")["grouped_scene_id"].astype(str))
    expected = set(
        json.loads(
            (ROOT / "outputs/person_v3/protocol/folds.json").read_text(
                encoding="utf-8"
            )
        )["folds"]["0"]
    )
    assert heldout == expected
    assert train.isdisjoint(heldout)
    assert len(train) == 12


def test_v7_service_cannot_run_after_test_opening() -> None:
    service = (
        ROOT / "systemd/tnorm-person-v7-v0.service"
    ).read_text(encoding="utf-8")
    assert "ConditionPathExists=!" in service
    assert "outputs/person_v3/test/TEST_OPENED.json" in service


def test_independent_evaluator_uses_lowercase_audit_contract() -> None:
    source = pd.DataFrame({"image_path": ["frame"], "grouped_scene_id": ["scene"]})
    gt = {"frame": [{"class_id": 0, "box": [0.0, 0.0, 10.0, 10.0]}]}
    predictions = {
        "frame": [
            {
                "class_id": 0,
                "confidence": 0.9,
                "box": [0.0, 0.0, 10.0, 10.0],
            }
        ]
    }
    metrics = evaluate(source, gt, predictions, 0.5)
    assert metrics["TP"] == 1
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0
    assert metrics["Recall"] == 1.0
