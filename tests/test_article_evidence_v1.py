from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/article_evidence_v1"
CONFIG = yaml.safe_load(
    (ROOT / "configs/article_evidence_v1.yaml").read_text(encoding="utf-8")
)


def test_closed_test_not_accessed() -> None:
    assert not (ROOT / "outputs/temporal_safety_v1/test/TEST_OPENED.json").exists()
    assert not (ROOT / "outputs/temporal_verifier_v1/test/TEST_OPENED.json").exists()
    assert not (ROOT / "outputs/crop_verifier_v1/test/TEST_OPENED.json").exists()


def test_test_access_count_zero() -> None:
    for path in OUTPUT.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "test_access_count" in payload:
            assert payload["test_access_count"] == 0


def test_threshold_grid_exact() -> None:
    audit = json.loads(
        (OUTPUT / "threshold_baseline/THRESHOLD_AUDIT.json").read_text()
    )
    assert audit["threshold_grid"] == [
        0.5,
        0.4,
        0.3,
        0.25,
        0.2,
        0.15,
        0.1,
        0.05,
        0.01,
    ]


def test_threshold_metrics_all_methods() -> None:
    frame = pd.read_csv(OUTPUT / "threshold_baseline/THRESHOLD_METRICS.csv")
    assert set(frame["method"]) == set(CONFIG["threshold_baseline"]["methods"])
    assert len(frame) == 45


def test_threshold_metrics_all_scenes() -> None:
    frame = pd.read_csv(
        OUTPUT / "threshold_baseline/THRESHOLD_METRICS_PER_SCENE.csv"
    )
    counts = frame.groupby(["method", "threshold"]).size()
    assert set(counts) == {6}


def test_all_201_thresholds_present() -> None:
    frame = pd.read_csv(
        OUTPUT / "track_verifier/TRACK_VERIFIER_201_THRESHOLDS.csv"
    )
    assert len(frame) == 201
    assert frame["threshold"].nunique() == 201


def test_no_duplicate_thresholds() -> None:
    frame = pd.read_csv(
        OUTPUT / "track_verifier/TRACK_VERIFIER_201_THRESHOLDS.csv"
    )
    assert not frame["threshold"].duplicated().any()


def test_false_tracks_total_288() -> None:
    frame = pd.read_csv(OUTPUT / "false_tracks/FALSE_TRACKS_ALL.csv")
    assert len(frame) == 288


def test_false_track_categories_complete() -> None:
    frame = pd.read_csv(OUTPUT / "false_tracks/FALSE_TRACKS_ALL.csv")
    assert frame["category"].isin(CONFIG["false_tracks"]["semantic_categories"]).all()
    assert not frame.duplicated(["scene_id", "sequence_id", "track_id"]).any()


def test_false_track_examples_license_field_present() -> None:
    frame = pd.read_csv(OUTPUT / "false_tracks/FALSE_TRACK_EXAMPLES.csv")
    assert "publication_allowed" in frame
    assert not frame["publication_allowed"].astype(bool).any()
    assert frame["image_path"].fillna("").eq("").all()


def test_event_parameters_match_code() -> None:
    actual = json.loads(
        (OUTPUT / "operator_assistant/EVENT_PARAMETERS_ACTUAL.json").read_text()
    )
    source = yaml.safe_load((ROOT / "configs/review_assistant_v1.yaml").read_text())
    assert actual["join_time_seconds"] == source["events"]["join_time_seconds"]
    assert actual["minimum_iou"] == source["events"]["minimum_iou"]
    assert actual["reopen_window_seconds"] == source["events"]["reopen_window_seconds"]


def test_real_and_synthetic_runs_separated() -> None:
    frame = pd.read_csv(
        OUTPUT / "operator_assistant/EVENT_AGGREGATION_RUNS.csv"
    )
    assert frame["synthetic"].sum() == 1
    assert frame.loc[frame["synthetic"], "run"].tolist() == ["synthetic_demo"]


def test_runtime_has_1000_measured_frames() -> None:
    frame = pd.read_csv(OUTPUT / "runtime/RUNTIME_PER_FRAME.csv")
    final = json.loads((OUTPUT / "runtime/RUNTIME_FINAL.json").read_text())
    assert len(frame) == 1000
    assert final["warmup_frames"] == 100
    assert final["measured_frames"] == 1000


def test_runtime_all_stages_present() -> None:
    frame = pd.read_csv(OUTPUT / "runtime/RUNTIME_STAGE_SUMMARY.csv")
    required = {
        "video_decode",
        "image_preprocessing",
        "tiling",
        "detector_inference",
        "coordinate_restoration",
        "box_fusion",
        "tracking",
        "track_verification",
        "event_aggregation",
        "rendering",
        "video_encoding",
        "database_writes",
        "full_pipeline",
    }
    assert set(frame["stage"]) == required
    for column in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "max_ms", "std_ms"):
        assert frame[column].map(math.isfinite).all()


def test_article_evidence_no_local_paths() -> None:
    forbidden = ("/home/", "lebedeffson", "TNormFilter_handoff/data/")
    for path in OUTPUT.rglob("*"):
        if path.suffix.lower() not in {".csv", ".json", ".md", ".txt", ".svg"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(token.lower() in text.lower() for token in forbidden), path


def test_article_evidence_no_test_data() -> None:
    archive = OUTPUT / "article_evidence_v1_public.zip"
    with zipfile.ZipFile(archive) as handle:
        names = [name.lower() for name in handle.namelist()]
        assert not any("sealed_test" in name or "/test/" in name for name in names)
        assert not any(name.endswith((".pt", ".pth", ".mp4", ".jpg", ".jpeg")) for name in names)


def test_csv_numbers_finite() -> None:
    numeric_csvs = [
        OUTPUT / "threshold_baseline/THRESHOLD_METRICS.csv",
        OUTPUT / "track_verifier/TRACK_VERIFIER_201_THRESHOLDS.csv",
        OUTPUT / "false_tracks/FALSE_TRACKS_ALL.csv",
        OUTPUT / "runtime/RUNTIME_PER_FRAME.csv",
        OUTPUT / "tnorm/TNORM_REDUNDANCY.csv",
    ]
    for path in numeric_csvs:
        frame = pd.read_csv(path).select_dtypes("number")
        assert frame.map(math.isfinite).all().all(), path


def test_figures_png_dpi_and_svg_open() -> None:
    pngs = sorted((OUTPUT / "figures").glob("*.png"))
    svgs = sorted((OUTPUT / "figures").glob("*.svg"))
    assert len(pngs) >= 9 and len(svgs) >= 9
    for path in pngs:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            assert min(dpi) >= 299
    for path in svgs:
        assert "<svg" in path.read_text(encoding="utf-8")[:1000]


def test_manifest_valid() -> None:
    for line in (OUTPUT / "MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        digest = hashlib.sha256((OUTPUT / relative).read_bytes()).hexdigest()
        assert digest == expected


def test_public_zip_manifest() -> None:
    with zipfile.ZipFile(OUTPUT / "article_evidence_v1_public.zip") as handle:
        assert "article_evidence_v1/MANIFEST.sha256" in handle.namelist()
        assert "article_evidence_v1/FINAL_TECHNICAL_SUMMARY.md" in handle.namelist()
