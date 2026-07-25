from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from person_v8b.analyze_loso import (
    FeatureDataset,
    _similarities,
    bootstrap_interval,
    build_feature_tables,
    development_gate,
    holm_adjust,
)
from person_v8b.common import ROOT, assert_test_sealed, load_config
from person_v8b.extract_features import region_mask


def synthetic_dataset() -> FeatureDataset:
    dataset = FeatureDataset.__new__(FeatureDataset)
    dataset.frame = pd.DataFrame(
        [
            {
                "image_path": f"image_{index}.png",
                "grouped_scene_id": f"scene_{index // 2}",
                "mean_confidence": 0.1 + index * 0.01,
                "max_confidence": 0.2 + index * 0.01,
                "prediction_count": index + 1,
                "confidence_entropy": 0.5,
                "low_confidence_fraction": 0.2,
                "mean_prediction_area_ratio": 0.01,
                "small_prediction_fraction": 0.5,
                "proposal_region_empty": False,
            }
            for index in range(6)
        ]
    )
    dataset.arrays = []
    for index in range(6):
        arrays = {}
        for layer_index, layer in enumerate(("P3", "P4", "P5")):
            channels = 4 + layer_index
            for region_index, region in enumerate(
                ("global", "proposal", "background")
            ):
                arrays[f"{layer}_{region}"] = np.linspace(
                    0.1,
                    0.9,
                    channels,
                    dtype=np.float64,
                ) + index * 0.01 + region_index * 0.02
            arrays[f"{layer}_gt_oracle"] = np.ones(channels)
        dataset.arrays.append(arrays)
    return dataset


def test_v8_closed_without_rewriting_locked_acquisition() -> None:
    closure = __import__("json").loads(
        (ROOT / "protocol/v8/V8_CLOSURE.json").read_text(encoding="utf-8")
    )
    assert closure["status"] == "BLOCKED_NO_NEW_DATA"
    assert closure["gpu_status"] == "GPU_NOT_STARTED"
    assert closure["test_access_count"] == 0


def test_v8b_protocol_freezes_primary_endpoint_and_forbids_detector_training() -> None:
    config = load_config()
    assert config["protocol_id"] == "canonical-v8b-person-failure-risk-v1"
    assert config["scope"]["detector_training"] == "forbidden"
    assert config["development_gate"]["primary_endpoint"] == (
        "scene_macro_MAE_fn_per_frame"
    )
    assert config["development_gate"][
        "secondary_endpoints_cannot_rescue_primary_failure"
    ]
    assert config["scope"]["adversarial_attacks"] == "out_of_scope"


def test_gt_regions_are_oracle_only_and_not_in_model_hierarchy() -> None:
    config = load_config()
    assert config["feature_extraction"]["gt_regions_forbidden_in_U0_U1_U2_U3"]
    serialized = str(config["model_hierarchy"])
    assert "gt_person" not in serialized
    assert config["interpretation"]["gt_region_claim_role"] == (
        "oracle_supplementary_only"
    )


def test_unsafe_frame_rule_is_frozen() -> None:
    rule = load_config()["outcomes"]["unsafe_frame"]
    assert rule["fn_minimum"] == 1
    assert rule["recall_below"] == pytest.approx(0.50)
    assert rule["recall_clause_requires_gt"]


def test_tnorm_similarities_match_frozen_formulas() -> None:
    left = np.asarray([0.2, 0.7, 1.0])
    right = np.asarray([0.5, 0.4, 0.8])
    result = _similarities(left, left, right, 1e-8)
    expected_product = np.sum(left * right) / np.sum(left + right - left * right)
    expected_luk = np.sum(np.maximum(0, left + right - 1)) / np.sum(
        np.minimum(1, left + right)
    )
    assert result["product"] == pytest.approx(expected_product)
    assert result["lukasiewicz"] == pytest.approx(expected_luk)


def test_feature_normalization_uses_only_fit_indices() -> None:
    dataset = synthetic_dataset()
    config = load_config()
    fit = np.asarray([0, 1, 2, 3])
    transformed = np.asarray([4])
    before = build_feature_tables(dataset, fit, transformed, config)
    for key in dataset.arrays[5]:
        dataset.arrays[5][key] = np.full_like(dataset.arrays[5][key], 1e9)
    after = build_feature_tables(dataset, fit, transformed, config)
    for level in ("U0", "U1", "U2", "U3"):
        assert np.array_equal(before[level], after[level])
        assert np.isfinite(after[level]).all()
    assert after["U3"].shape[1] > after["U2"].shape[1]


def test_region_mask_maps_only_intersecting_tile_cells() -> None:
    protocol = __import__("yaml").safe_load(
        (ROOT / "configs/canonical_v2_m4_full_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    feature = torch.zeros((4, 8, 80, 80))
    mask = region_mask([[100.0, 100.0, 200.0, 300.0]], feature, protocol)
    assert mask.shape == (4, 80, 80)
    assert int(mask[0].sum()) > 0
    assert int(mask[1:].sum()) == 0


def test_bootstrap_is_deterministic_and_paired() -> None:
    values = np.asarray([-0.3, -0.2, -0.1, -0.4])
    assert bootstrap_interval(values, 1000, 42) == bootstrap_interval(
        values, 1000, 42
    )
    low, high = bootstrap_interval(values, 1000, 42)
    assert high < 0
    assert low <= high


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    raw = [0.01, 0.04, 0.02]
    adjusted = holm_adjust(raw)
    assert all(0 <= value <= 1 for value in adjusted)
    order = np.argsort(raw)
    ordered = [adjusted[index] for index in order]
    assert ordered == sorted(ordered)


def gate_metrics(u3_mae: float) -> pd.DataFrame:
    rows = []
    for scene in range(15):
        for model, mae in (("U2", 1.0), ("U3", u3_mae)):
            rows.append(
                {
                    "model": model,
                    "grouped_scene_id": f"scene_{scene}",
                    "mae_fn": mae,
                    "brier": 0.10 if model == "U2" else 0.09,
                    "ece": 0.05 if model == "U2" else 0.04,
                }
            )
    return pd.DataFrame(rows)


def gate_bootstrap(high: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate": "U3",
                "reference": "U2",
                "metric": "mae_fn",
                "ci95_low": -0.2,
                "ci95_high": high,
            }
        ]
    )


def test_primary_gate_passes_only_with_confirmatory_effect() -> None:
    gate = development_gate(gate_metrics(0.9), gate_bootstrap(-0.05), load_config())
    assert gate["status"] == "PASS"
    assert gate["scene_wins"] == 15
    assert gate["relative_MAE_reduction_percent"] == pytest.approx(10.0)


def test_secondary_metrics_cannot_rescue_primary_gate() -> None:
    gate = development_gate(gate_metrics(0.99), gate_bootstrap(0.02), load_config())
    assert gate["status"] == "FAIL"
    assert gate["secondary_endpoints_can_rescue"] is False


def test_test_is_still_physically_sealed() -> None:
    assert_test_sealed()
    assert not (ROOT / "outputs/person_v8b/test/TEST_OPENED.json").exists()


def test_service_has_protocol_lock_and_negative_test_marker_conditions() -> None:
    unit = (ROOT / "systemd/tnorm-person-v8b-risk.service").read_text(
        encoding="utf-8"
    )
    assert "ConditionPathExists=%h/Code/андрей/TNormFilter_handoff/protocol/v8b/V8B_PROTOCOL_LOCK.json" in unit
    assert "ConditionPathExists=!%h/Code/андрей/TNormFilter_handoff/outputs/person_v8b/test/TEST_OPENED.json" in unit

