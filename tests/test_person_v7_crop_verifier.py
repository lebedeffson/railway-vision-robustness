from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.crop_verifier.crop_model import (
    CropMLPClassifier,
    context_crop,
    select_crop_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/canonical_v7_crop_verifier_amendment.yaml"


def test_crop_amendment_requires_frozen_v0_fail() -> None:
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    parent = json.loads(
        (ROOT / protocol["parent_gate"]).read_text(encoding="utf-8")
    )
    assert protocol["parent_result"] == "V0_FAIL"
    assert parent["V0_GATE"] == "FAIL"
    assert protocol["test_status"] == "SEALED"
    assert protocol["attacks_status"] == "BLOCKED"
    assert protocol["encoder"]["frozen"] is True
    assert protocol["encoder"]["crowdhuman_test_used"] is False
    assert not (ROOT / protocol["test_marker"]).exists()


def test_three_crop_selection_is_deterministic() -> None:
    group = pd.DataFrame(
        {
            "is_detection": [True, True, True, True],
            "detector_confidence": [0.01, 0.08, 0.03, 0.05],
            "frame_order": [0, 1, 2, 3],
            "image_path": ["a", "b", "c", "d"],
        }
    )
    selected = select_crop_rows(group)
    assert selected["detector_confidence"].tolist() == [0.08, 0.03, 0.01]


def test_context_crop_pads_border_without_changing_size() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[0:5, 0:5] = 255
    crop = context_crop(image, [-2.0, -2.0, 4.0, 8.0], 1.75, 64, 114)
    assert crop.shape == (64, 64, 3)
    assert crop.dtype == np.uint8
    assert np.any(crop == 114)


def test_crop_mlp_bce_ranking_fit_is_finite() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(80, 16)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = CropMLPClassifier(
        hidden_units=8,
        epochs=2,
        maximum_pairs_per_epoch=32,
        seed=11,
    ).fit(x, y)
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (80, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_crop_service_keeps_test_physically_blocked() -> None:
    service = (
        ROOT / "systemd/tnorm-person-v7-crop.service"
    ).read_text(encoding="utf-8")
    assert "ConditionPathExists=!" in service
    assert "outputs/person_v3/test/TEST_OPENED.json" in service

