from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from src.final_demo.temporal_pipeline import TemporalResearchPipeline
from src.review_assistant.database import ReviewDatabase
from src.review_assistant.event_aggregator_v2 import EvidenceEventAggregator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/operator_assistant_evidence_v2"


class StubTracker:
    def __init__(self) -> None:
        self.tracks = [
            SimpleNamespace(
                track_id=7,
                confirmed=True,
                age=2,
                hits=2,
                confidence_history=[0.1, 0.1],
            )
        ]

    def update(self, _candidates, _width, _height):
        return (
            [
                {
                    "candidate_id": "C-000000-0000",
                    "box": [1, 1, 10, 20],
                    "confidence": 0.1,
                    "raw_confidence": 0.01,
                    "track_id": 7,
                    "source": "temporal_confirmation",
                    "interpolated": False,
                }
            ],
            [],
        )


class FailingVerifier:
    threshold = 0.5

    def observe(self, *_args) -> None:
        raise RuntimeError("verifier unavailable")

    def probability(self, _state) -> float:
        return 0.0


def test_verifier_exception_bypasses_filter_without_hiding_candidate() -> None:
    pipeline = object.__new__(TemporalResearchPipeline)
    pipeline.tracker = StubTracker()
    pipeline.verifier = FailingVerifier()
    pipeline.standard_threshold = 0.07
    pipeline.last_tracker_ms = 0.0
    pipeline.last_verifier_ms = 0.0
    pipeline.frame_index = -1
    pipeline.last_trace_rows = []
    pipeline.last_fallbacks = []
    output, _ = pipeline.update(
        np.zeros((32, 32, 3), dtype=np.uint8),
        [{"box": [1, 1, 10, 20], "confidence": 0.01}],
    )
    assert len(output) == 1
    assert output[0]["processing_status"] == "VERIFIER_UNAVAILABLE"
    assert output[0]["verifier_decision"] == "BYPASS"
    assert output[0]["source"] == "verifier_fallback"
    assert pipeline.last_fallbacks[0]["frame_number"] == 0


def test_event_protocol_uses_development_only_source_ids() -> None:
    lock = json.loads(
        (OUTPUT / "EVENT_PROTOCOL_LOCK.json").read_text(encoding="utf-8")
    )
    assert lock["test_status"] == "SEALED"
    assert lock["test_access_count"] == 0
    assert lock["benchmark"]["source_sequence_id"] == "8_station_altona_8.2"
    episodes = pd.read_csv(OUTPUT / "GT_PERSON_EPISODES.csv")
    boxes = pd.read_csv(OUTPUT / "GT_PERSON_BOXES.csv")
    assert len(episodes) == 15
    assert len(boxes) == 1322
    assert set(episodes["verification_author"]) == {"SOURCE_OBJECT_UUID"}


def test_frozen_v2_inputs_match_protocol_hashes() -> None:
    lock = json.loads(
        (OUTPUT / "EVENT_PROTOCOL_LOCK.json").read_text(encoding="utf-8")
    )
    for record in lock["inputs"].values():
        path = ROOT / record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"], record["path"]


def test_preaggregation_trace_has_stable_candidate_ids() -> None:
    trace = pd.read_parquet(OUTPUT / "PRE_AGGREGATION_OBSERVATIONS.parquet")
    assert trace["candidate_id"].is_unique
    assert len(trace) >= 2410
    assert not trace["candidate_id"].astype(str).str.strip().eq("").any()
    required = {
        "scene_id",
        "source_frame_id",
        "video_frame_id",
        "detector_confidence",
        "track_id",
        "verifier_decision",
        "rejection_reason",
        "sent_to_aggregator",
    }
    assert required <= set(trace.columns)


def test_direct_and_replay_are_identical() -> None:
    replay = json.loads(
        (OUTPUT / "EVENT_BASELINE_REPLAY.json").read_text(encoding="utf-8")
    )
    assert replay["exact_event_signature_match"] is True
    assert replay["direct_unique_events"] == replay["replay_unique_events"]
    assert replay["direct_raw_detector_boxes"] == 2410


def test_all_108_full_sensitivity_configurations_present() -> None:
    frame = pd.read_csv(OUTPUT / "EVENT_SENSITIVITY_FULL.csv")
    assert len(frame) == 108
    assert (
        frame[
            [
                "join_time_seconds",
                "minimum_iou",
                "maximum_center_distance_ratio",
                "reopen_window_seconds",
            ]
        ]
        .drop_duplicates()
        .shape[0]
        == 108
    )
    assert frame.select_dtypes("number").notna().all().all()
    assert frame["true_episode_recall"].between(0, 1).all()
    assert frame["false_merge_rate"].between(0, 1).all()
    assert frame["episode_frame_coverage"].between(0, 1).all()


def test_candidate_ids_persist_in_database_schema(tmp_path: Path) -> None:
    with ReviewDatabase(tmp_path / "review.sqlite") as database:
        columns = {
            row["name"]
            for row in database.connection.execute(
                "PRAGMA table_info(event_detections)"
            )
        }
        event_columns = {
            row["name"]
            for row in database.connection.execute("PRAGMA table_info(events)")
        }
    assert {"candidate_id", "processing_status"} <= columns
    assert {"processing_status", "reopen_count"} <= event_columns


def test_no_multistream_execution_in_v2() -> None:
    lock = json.loads(
        (OUTPUT / "EVENT_PROTOCOL_LOCK.json").read_text(encoding="utf-8")
    )
    assert lock["multistream_benchmark"] == "DEFERRED"


def test_runtime_has_warmup_and_1000_measured_frames() -> None:
    final = json.loads(
        (OUTPUT / "EVENT_RUNTIME_FINAL.json").read_text(encoding="utf-8")
    )
    per_frame = pd.read_csv(OUTPUT / "EVENT_RUNTIME_PER_FRAME.csv")
    assert final["warmup_frames"] == 100
    assert final["measured_frames"] == 1000
    assert len(per_frame) == 1000
    assert per_frame.select_dtypes("number").notna().all().all()
    assert final["test_status"] == "SEALED"
    assert final["test_access_count"] == 0


def test_runtime_has_full_candidate_and_event_decomposition() -> None:
    final = json.loads(
        (OUTPUT / "EVENT_RUNTIME_FINAL.json").read_text(encoding="utf-8")
    )
    required = {
        "raw_detections",
        "tracker_observations",
        "verifier_accepted",
        "verifier_rejected",
        "aggregator_observations",
        "unique_events",
        "baseline_events",
        "temporal_only_events",
        "combined_events",
        "track_switches_merged",
        "reopened_events",
    }
    assert required <= set(final)
    assert final["raw_detections"] >= final["tracker_observations"]
    assert (
        final["verifier_accepted"] + final["verifier_rejected"]
        == final["tracker_observations"]
    )
    assert final["aggregator_observations"] == final["verifier_accepted"]


def test_all_ten_failsafe_checks_are_real_passes() -> None:
    audit = json.loads((OUTPUT / "FAILSAFE_AUDIT.json").read_text())
    checks = pd.read_csv(OUTPUT / "FAILSAFE_CHECKS.csv")
    assert audit == {
        "all_pass": True,
        "checks": 10,
        "failed": 0,
        "passed": 10,
        "test_access_count": 0,
        "test_status": "SEALED",
    }
    assert len(checks) == 10
    assert set(checks["status"]) == {"PASS"}
    forbidden = {"FAIL_CURRENT_IMPLEMENTATION", "PASS_BY_ASSUMPTION", "NOT_TESTED"}
    assert not (set(checks["status"]) & forbidden)


def test_final_figures_have_svg_and_320_dpi_png() -> None:
    names = {
        "event_count_sensitivity",
        "fragmentation_false_merge",
        "event_recall_false_events",
        "event_latency_distribution",
    }
    for name in names:
        svg = OUTPUT / "figures" / f"{name}.svg"
        png = OUTPUT / "figures" / f"{name}.png"
        assert svg.read_text(encoding="utf-8").lstrip().startswith("<?xml")
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
            assert min(dpi) >= 300


def test_public_bundle_excludes_sensitive_runtime_artifacts() -> None:
    archive = OUTPUT / "operator_assistant_evidence_v2_public.zip"
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        forbidden_suffixes = {".mp4", ".parquet", ".jpg", ".jpeg"}
        assert not any(Path(name).suffix.lower() in forbidden_suffixes for name in names)
        assert "development_event_benchmark_v2_manifest.csv" not in names
        assert "GT_PERSON_BOXES.csv" not in names
        assert "MANIFEST.sha256" in names
        for name in names:
            if Path(name).suffix.lower() in {".json", ".csv", ".md", ".txt"}:
                text = bundle.read(name).decode("utf-8")
                assert "/home/" not in text
                assert "/mnt/" not in text
