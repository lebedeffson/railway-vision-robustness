from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from person_v9.proxy import (
    causal_temporal_features,
    residualize_train_only,
    run_synthetic_proxy,
)
from v8b.finalize_v8b import ROOT, finalize


def temporal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "grouped_scene_id": ["scene_a"] * 6,
            "subsequence_id": ["scene_a.0"] * 6,
            "frame_id": list(range(6)),
            "product": [0.1, 0.2, 0.4, 0.3, 0.5, 0.8],
        }
    )


def test_v8b_finalizer_reproduces_frozen_primary_numbers(tmp_path: Path) -> None:
    source = ROOT / "outputs/person_v8b/results/OOF_PREDICTIONS.csv"
    if not source.is_file():
        pytest.skip("Development OOF artifact is not installed")
    result = finalize(source, tmp_path)
    assert result["U2_scene_macro_MAE"] == pytest.approx(
        1.5954889301566773, abs=1e-9
    )
    assert result["U3_scene_macro_MAE"] == pytest.approx(
        1.7679771024802038, abs=1e-9
    )
    assert result["relative_MAE_reduction_percent"] == pytest.approx(
        -10.810991481250088, abs=1e-6
    )
    assert result["scene_wins"] == 5
    assert result["test_access_count"] == 0
    assert "/" + "home/" not in (tmp_path / "OOF_INPUT_REDACTED.csv").read_text(
        encoding="utf-8"
    )


def test_v8b_finalizer_is_independent_of_original_analysis_module() -> None:
    source = (ROOT / "scripts/v8b/finalize_v8b.py").read_text(encoding="utf-8")
    assert "from person_v8b" not in source
    assert "torch.load" not in source
    assert "cv2.imread" not in source
    assert "data/yolo_osdar23" not in source


def test_v8b_article_is_claim_safe_when_built() -> None:
    validation = (
        ROOT / "outputs/person_v8b/final/article/ARTICLE_VALIDATION.json"
    )
    if not validation.is_file():
        pytest.skip("Final article has not been built")
    payload = json.loads(validation.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["checks"]["negative_conclusion_present"]
    assert payload["checks"]["no_email"]
    assert payload["checks"]["sealed_test_disclosed"]


def test_v9_is_blocked_until_new_scenes_or_detector_oof() -> None:
    payload = json.loads(
        (ROOT / "protocol/v9/V9_PREREQUISITES.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "BLOCKED_PREREQUISITES"
    assert not payload["activation_passed"]
    assert payload["test_access_count"] == 0
    assert payload["execution_status"] == "NOT_STARTED"


def test_temporal_features_do_not_use_future_frames() -> None:
    original = temporal_frame()
    baseline = causal_temporal_features(original, ["product"])
    changed = original.copy()
    changed.loc[5, "product"] = 10_000.0
    candidate = causal_temporal_features(changed, ["product"])
    temporal_columns = [
        column
        for column in baseline.columns
        if column.startswith("product_")
    ]
    pd.testing.assert_frame_equal(
        baseline.loc[:4, temporal_columns],
        candidate.loc[:4, temporal_columns],
    )


def test_temporal_windows_reset_at_subsequence_boundary() -> None:
    frame = temporal_frame()
    frame.loc[3:, "subsequence_id"] = "scene_a.1"
    result = causal_temporal_features(frame, ["product"])
    assert result.loc[3, "product_delta"] == pytest.approx(0.0)


def test_residualization_is_fit_on_train_only() -> None:
    base_train = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    base_validation = np.asarray([[4.0], [5.0]])
    tnorm_train = np.asarray([[0.0], [0.8], [2.1], [3.2]])
    validation_a = np.asarray([[4.1], [5.2]])
    validation_b = np.asarray([[400.0], [500.0]])
    train_a, _ = residualize_train_only(
        base_train, base_validation, tnorm_train, validation_a
    )
    train_b, _ = residualize_train_only(
        base_train, base_validation, tnorm_train, validation_b
    )
    assert np.array_equal(train_a, train_b)


def test_v9_synthetic_proxy_is_not_article_evidence(tmp_path: Path) -> None:
    result = run_synthetic_proxy(tmp_path)
    assert result["status"] == "PROXY_PASS"
    assert result["article_evidence"] is False
    assert result["future_frame_access"] is False
    assert result["test_access_count"] == 0
