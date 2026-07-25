from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from src.temporal.ratta import (
    HomographyResult,
    TemporalAggregator,
    deterministic_nms,
    estimate_background_homography,
    product_reliability,
    warp_box,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/canonical_v6_person_temporal_tnorm.yaml"


def test_protocol_is_causal_and_test_is_sealed() -> None:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert protocol["temporal"]["causal"] is True
    assert protocol["temporal"]["window_frames"] == 5
    assert protocol["data"]["temporal_unit"] == "subsequence_id"
    assert protocol["test_status"] == "SEALED"
    assert protocol["attacks_status"] == "BLOCKED"
    assert protocol["data"]["test_read_permitted"] is False
    assert not (ROOT / protocol["test_marker"]).exists()


def test_active_data_draft_is_abandoned_before_execution() -> None:
    payload = json.loads(
        (
            ROOT / "protocol/v6/ACTIVE_DATA_DRAFT_STATUS.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "ABANDONED_BEFORE_EXECUTION"
    assert payload["training_runs"] == 0


def test_warp_box_uses_homogeneous_transform() -> None:
    transform = np.asarray([[1, 0, 5], [0, 1, -3], [0, 0, 1]], dtype=float)
    assert warp_box([10, 20, 30, 50], transform) == [15.0, 17.0, 35.0, 47.0]


def test_product_reliability_is_true_product() -> None:
    memberships = {
        "motion": 0.5,
        "appearance": 0.8,
        "scale": 0.5,
        "border": 1.0,
        "compensated_iou": 0.0,
    }
    assert np.isclose(product_reliability(memberships), 0.2)


def test_invalid_homography_has_zero_temporal_association() -> None:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    aggregator = TemporalAggregator(protocol["temporal"], "product")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = {
        "class_id": 0,
        "confidence": 0.02,
        "box": [40.0, 20.0, 50.0, 50.0],
    }
    first, _ = aggregator.update([detection], image, None)
    invalid = HomographyResult(np.eye(3), False, 0, 0, 0.0)
    second, tracks = aggregator.update([detection], image, invalid)
    assert first[0]["track_id"] != second[0]["track_id"]
    assert tracks == []


def test_blank_frames_fail_homography_quality_gate() -> None:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    blank = np.zeros((200, 300, 3), dtype=np.uint8)
    result = estimate_background_homography(
        blank,
        blank,
        [],
        protocol["temporal"]["camera_compensation"],
    )
    assert result.valid is False
    assert result.inlier_ratio == 0.0


def test_deterministic_nms_keeps_highest_score() -> None:
    predictions = [
        {"confidence": 0.2, "box": [0, 0, 10, 10]},
        {"confidence": 0.8, "box": [0, 0, 10, 10]},
    ]
    kept = deterministic_nms(predictions, 0.5)
    assert len(kept) == 1
    assert kept[0]["confidence"] == 0.8
