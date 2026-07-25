from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/temporal_safety_v1"


def test_final_numbers_match_gate_and_csv() -> None:
    if not (OUTPUT / "FINAL_DEVELOPMENT_SUMMARY.json").is_file():
        return
    summary = json.loads(
        (OUTPUT / "FINAL_DEVELOPMENT_SUMMARY.json").read_text(encoding="utf-8")
    )
    gate = json.loads(
        (OUTPUT / "triage/TRIAGE_GATE.json").read_text(encoding="utf-8")
    )
    for decision in gate["decisions"]:
        tracker = decision["tracker"]
        assert summary["recall_signal"][tracker] == decision["deltas"]["recall"]
        assert (
            summary["relative_FN_reduction"][tracker]
            == decision["deltas"]["relative_FN_reduction"]
        )
    effects = pd.read_csv(OUTPUT / "statistics/PER_SCENE_EFFECTS.csv")
    assert set(effects["tracker"]) == {"bytetrack", "ocsort"}
    assert len(effects) == 12


def test_public_zip_has_no_dataset_or_checkpoint() -> None:
    bundle = (
        OUTPUT
        / "bundles/railway_person_temporal_safety_development_fail.zip"
    )
    if not bundle.is_file():
        return
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
    assert not any("data/raw/" in name for name in names)
    assert not any("/images/" in name for name in names)
    assert not any(name.endswith(".pt") for name in names)
    assert not any("raw_predictions.parquet" in name for name in names)
    assert not any("sealed_test_manifest.csv" in name for name in names)


def test_test_stayed_sealed_after_fail() -> None:
    marker = OUTPUT / "test/TEST_OPENED.json"
    assert not marker.exists()
    if (OUTPUT / "FINAL_DEVELOPMENT_SUMMARY.json").is_file():
        summary = json.loads(
            (OUTPUT / "FINAL_DEVELOPMENT_SUMMARY.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["test_access_count"] == 0
        assert summary["test_status"] == "SEALED"
