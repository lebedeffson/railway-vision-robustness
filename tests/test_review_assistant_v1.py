from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.review_assistant.clips import EventClipWriter
from src.review_assistant.config import load_config, sha256_file, verify_model_file
from src.review_assistant.database import ReviewDatabase
from src.review_assistant.event_aggregator import EventAggregator
from src.review_assistant.models import EventDetection, ReviewEvent
from src.review_assistant.processor import ReviewProcessor
from src.review_assistant.processor_v2 import FailSafeReviewProcessor
from src.review_assistant.reporting import generate_report


ROOT = Path(__file__).resolve().parents[1]


def settings() -> dict:
    return {
        "join_time_seconds": 3,
        "close_after_seconds": 3,
        "reopen_window_seconds": 30,
        "minimum_iou": 0.20,
        "maximum_center_distance_ratio": 0.10,
        "temporal_minimum_observations": 2,
    }


def candidate(
    frame: int,
    *,
    box: list[float] | None = None,
    source: str = "BASELINE",
    track_id: int | None = 1,
    confirmed: bool = False,
) -> tuple[int, float, dict]:
    return (
        frame,
        frame / 10.0,
        {
            "box": box or [10, 10, 30, 50],
            "confidence": 0.5,
            "review_source": source,
            "track_id": track_id,
            "confirmed": confirmed,
            "interpolated": False,
        },
    )


def aggregator() -> EventAggregator:
    return EventAggregator(
        settings(),
        video_id="V1",
        camera_id="C1",
        frame_width=100,
        frame_height=100,
    )


def test_raw_detections_aggregate_into_one_event() -> None:
    engine = aggregator()
    for frame in range(10):
        number, timestamp, row = candidate(frame)
        engine.observe(number, timestamp, [row])
    events = engine.finalize()
    assert len(events) == 1
    assert len(events[0].detections) == 10


def test_track_id_switch_does_not_duplicate_event() -> None:
    engine = aggregator()
    for frame, track_id in ((0, 1), (1, 7), (2, 19)):
        number, timestamp, row = candidate(frame, track_id=track_id)
        engine.observe(number, timestamp, [row])
    events = engine.finalize()
    assert len(events) == 1
    assert events[0].track_ids == {1, 7, 19}


def test_event_closes_after_timeout() -> None:
    engine = aggregator()
    number, timestamp, row = candidate(0)
    engine.observe(number, timestamp, [row])
    engine.advance(3.1)
    assert engine.accepted_events()[0].state == "CLOSED"


def test_event_reopens_within_window() -> None:
    engine = aggregator()
    number, timestamp, row = candidate(0)
    engine.observe(number, timestamp, [row])
    engine.advance(4.0)
    number, timestamp, row = candidate(100)
    engine.observe(number, timestamp, [row])
    assert len(engine.accepted_events()) == 1
    assert engine.accepted_events()[0].state == "ACTIVE"


def test_different_spatial_regions_create_different_events() -> None:
    engine = aggregator()
    first = candidate(0, box=[0, 0, 15, 30])[2]
    second = candidate(1, box=[80, 50, 99, 99])[2]
    engine.observe(0, 0.0, [first, second])
    assert len(engine.finalize()) == 2


def _video(path: Path, frames: int = 100, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (96, 64)
    )
    assert writer.isOpened()
    for frame in range(frames):
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        cv2.rectangle(image, (frame % 70, 15), (frame % 70 + 12, 50), (255, 255, 255), -1)
        writer.write(image)
    writer.release()


def _event() -> ReviewEvent:
    event = ReviewEvent(
        event_id="E000001",
        video_id="V1",
        camera_id="C1",
        start_time=4.0,
        end_time=5.0,
        state="CLOSED",
    )
    for frame in (40, 45, 50):
        event.add(
            EventDetection(
                frame_number=frame,
                timestamp=frame / 10,
                box=[10, 10, 30, 50],
                confidence=0.8,
                source="BASELINE",
                track_id=1,
            )
        )
    return event


def test_clip_contains_pre_and_post_roll(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _video(source)
    clip, thumbnail = EventClipWriter(3, 3).write(
        source, _event(), tmp_path / "event"
    )
    capture = cv2.VideoCapture(str(clip))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    assert frame_count >= 69
    assert thumbnail.is_file()
    payload = json.loads((tmp_path / "event/event.json").read_text())
    assert payload["clip_start_time"] == 1.0
    assert payload["clip_end_time"] == 8.0


def _database_with_event(tmp_path: Path) -> tuple[Path, str, str]:
    path = tmp_path / "review.sqlite"
    with ReviewDatabase(path) as database:
        video_id, _ = database.register_video(
            path=tmp_path / "source.mp4",
            sha256="a" * 64,
            camera_id="C1",
            fps=10,
            frame_count=100,
            width=96,
            height=64,
        )
        run_id, _ = database.create_run(
            video_id=video_id,
            mode="combined_queue",
            total_frames=100,
            config_sha256="b" * 64,
            resume_policy="restart_incomplete_run",
        )
        event = _event()
        event.video_id = video_id
        database.save_events(run_id, [event])
        database.complete_run(
            run_id, processing_seconds=2, track_count=1, event_count=1
        )
    return path, run_id, event.event_id


def test_operator_decision_persists(tmp_path: Path) -> None:
    path, run_id, event_id = _database_with_event(tmp_path)
    with ReviewDatabase(path) as database:
        database.review_event(
            run_id,
            event_id,
            operator="alice",
            new_status="HUMAN",
            comment="confirmed",
        )
        assert database.get_event(run_id, event_id)["review_status"] == "HUMAN"
    with ReviewDatabase(path) as reopened:
        assert reopened.get_event(run_id, event_id)["operator_comment"] == "confirmed"


def test_operator_decision_is_reversible(tmp_path: Path) -> None:
    path, run_id, event_id = _database_with_event(tmp_path)
    with ReviewDatabase(path) as database:
        database.review_event(
            run_id, event_id, operator="alice", new_status="FALSE_POSITIVE"
        )
        database.undo_last_review(run_id, event_id, "alice")
        assert database.get_event(run_id, event_id)["review_status"] == "PENDING"
        actions = {row["action"] for row in database.audit_rows()}
        assert {"REVIEW", "UNDO_REVIEW"} <= actions


def test_suppression_requires_manual_confirmation(tmp_path: Path) -> None:
    with ReviewDatabase(tmp_path / "db.sqlite") as database:
        with pytest.raises(RuntimeError, match="explicit operator"):
            database.create_suppression_rule(
                camera_id="C1",
                region=[1, 1, 10, 10],
                reference_confidence=0.2,
                reference_motion=0,
                operator="alice",
                reason="test",
                manual_confirmed=False,
            )


def test_suppressed_event_remains_auditable(tmp_path: Path) -> None:
    path, run_id, event_id = _database_with_event(tmp_path)
    config = load_config()
    with ReviewDatabase(path) as database:
        database.create_suppression_rule(
            camera_id="C1",
            region=[10, 10, 30, 50],
            reference_confidence=0.8,
            reference_motion=0,
            operator="alice",
            reason="repeated pole",
            manual_confirmed=True,
        )
        assert database.apply_suppression(
            run_id,
            event_id,
            config["suppression"],
            frame_width=96,
            frame_height=64,
        )
        event = database.get_event(run_id, event_id)
        assert event["muted"] is True
        assert len(database.list_events(run_id, muted=True)) == 1
        assert any(
            row["action"] == "SUPPRESSION_RULE_CREATED"
            for row in database.audit_rows()
        )


def test_temporal_only_event_is_marked() -> None:
    engine = aggregator()
    for frame in (0, 1):
        number, timestamp, row = candidate(frame, source="TEMPORAL_ONLY")
        engine.observe(number, timestamp, [row])
    event = engine.finalize()[0]
    assert event.source_label == "TEMPORAL_ONLY"


def test_no_autonomous_alarm_output() -> None:
    config = load_config()
    assert config["review"]["autonomous_alarm"] is False
    source = (ROOT / "src/review_assistant/processor.py").read_text()
    assert '"autonomous_alarm_output": False' in source


def test_no_safety_actuation() -> None:
    config = load_config()
    assert config["review"]["safety_actuation"] is False
    lock = json.loads(
        (ROOT / "protocol/review_assistant_v1/PRODUCT_LOCK.json").read_text()
    )
    assert lock["safety_actuation"] == "DISABLED"


def test_database_migrations(tmp_path: Path) -> None:
    with ReviewDatabase(tmp_path / "db.sqlite") as database:
        assert database.schema_version == 2
        tables = {
            row[0]
            for row in database.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {
            "videos",
            "runs",
            "events",
            "event_detections",
            "reviews",
            "camera_profiles",
            "suppression_rules",
            "audit_log",
        } <= tables


def test_resume_interrupted_video(tmp_path: Path) -> None:
    database_path = tmp_path / "db.sqlite"
    with ReviewDatabase(database_path) as database:
        video_id, _ = database.register_video(
            path=tmp_path / "v.mp4",
            sha256="c" * 64,
            camera_id="C",
            fps=10,
            frame_count=100,
            width=10,
            height=10,
        )
        run_id, resumed = database.create_run(
            video_id=video_id,
            mode="combined_queue",
            total_frames=100,
            config_sha256="d" * 64,
            resume_policy="restart_incomplete_run",
        )
        assert resumed is False
        database.update_progress(run_id, 40, 7)
        same_run, resumed = database.create_run(
            video_id=video_id,
            mode="combined_queue",
            total_frames=100,
            config_sha256="d" * 64,
            resume_policy="restart_incomplete_run",
        )
        assert same_run == run_id
        assert resumed is True
        assert database.get_run(run_id)["processed_frames"] == 0


def test_duplicate_video_detection(tmp_path: Path) -> None:
    with ReviewDatabase(tmp_path / "db.sqlite") as database:
        kwargs = dict(
            path=tmp_path / "v.mp4",
            sha256="e" * 64,
            camera_id="C",
            fps=10,
            frame_count=10,
            width=10,
            height=10,
        )
        first, duplicate_first = database.register_video(**kwargs)
        second, duplicate_second = database.register_video(**kwargs)
        assert first == second
        assert duplicate_first is False
        assert duplicate_second is True


def test_checkpoint_hash_verified(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"frozen")
    assert verify_model_file(model, sha256_file(model)) == model.resolve()
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_model_file(model, "0" * 64)


def test_long_benchmark_uses_at_least_1000_frames() -> None:
    config = load_config()
    assert config["benchmark"]["minimum_frames"] >= 1000
    script = (
        ROOT / "scripts/review_assistant/benchmark_long_video.py"
    ).read_text()
    assert "minimum_frames" in script
    assert "warmup_frames" in script


def test_report_matches_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path, run_id, event_id = _database_with_event(tmp_path)
    with ReviewDatabase(path) as database:
        database.review_event(
            run_id, event_id, operator="alice", new_status="HUMAN"
        )
    monkeypatch.setattr(
        "src.review_assistant.reporting._render_pdf",
        lambda _html, pdf: pdf.write_bytes(b"%PDF-FAKE"),
    )
    metrics = generate_report(path, run_id, tmp_path / "outputs")
    with ReviewDatabase(path) as database:
        run = database.get_run(run_id)
        events = database.list_events(run_id)
    assert metrics["raw_detections"] == run["raw_detection_count"]
    assert metrics["unique_events"] == len(events)
    assert metrics["confirmed_people"] == 1


def test_export_contains_no_local_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, run_id, _ = _database_with_event(tmp_path)
    monkeypatch.setattr(
        "src.review_assistant.reporting._render_pdf",
        lambda _html, pdf: pdf.write_bytes(b"%PDF-FAKE"),
    )
    generate_report(path, run_id, tmp_path / "outputs")
    text = (tmp_path / f"outputs/{run_id}/events.csv").read_text()
    assert str(tmp_path) not in text
    assert "password=" not in text.lower()


class FakeDetector:
    checkpoint_sha256 = "fake"

    def candidates(self, _frame: np.ndarray) -> list[dict]:
        return [
            {
                "box": [10, 10, 30, 50],
                "confidence": 0.8,
                "raw_confidence": 0.8,
                "track_id": None,
                "source": "detector",
                "confirmed": True,
                "interpolated": False,
            }
        ]

    def standard(self, rows: list[dict]) -> list[dict]:
        return rows


def test_processor_creates_operator_queue_without_alarm(tmp_path: Path) -> None:
    video = tmp_path / "short.mp4"
    _video(video, frames=8, fps=4)
    processor = ReviewProcessor(
        detector=FakeDetector(),
        temporal=None,
        database=tmp_path / "db.sqlite",
        run_root=tmp_path / "runs",
        output_root=tmp_path / "outputs",
    )
    result = processor.process(video, mode="conservative_review")
    assert result["unique_events"] == 1
    assert result["autonomous_alarm_output"] is False
    assert result["safety_actuation"] is False
    with ReviewDatabase(tmp_path / "db.sqlite") as database:
        events = database.list_events(result["run_id"])
        run = database.get_run(result["run_id"])
    assert events[0]["source_label"] == "BASELINE"
    assert Path(events[0]["clip_path"]).is_file()
    assert run["raw_detection_count"] == result["raw_detections"]


class FailingDetector:
    checkpoint_sha256 = "failing-test-detector"

    def candidates(self, _frame: np.ndarray) -> list[dict]:
        raise RuntimeError("synthetic recoverable inference failure")

    def standard(self, rows: list[dict]) -> list[dict]:
        return rows


def test_unknown_processing_error_creates_technical_review_event(
    tmp_path: Path,
) -> None:
    video = tmp_path / "short.mp4"
    _video(video, frames=3, fps=3)
    processor = FailSafeReviewProcessor(
        detector=FailingDetector(),
        temporal=None,
        database=tmp_path / "db.sqlite",
        run_root=tmp_path / "runs",
        output_root=tmp_path / "outputs",
    )
    result = processor.process(video, mode="conservative_review")
    with ReviewDatabase(tmp_path / "db.sqlite") as database:
        events = database.list_events(result["run_id"])
        audit = database.audit_rows()
    assert events
    assert events[0]["processing_status"] == "PROCESSING_FALLBACK"
    assert events[0]["review_status"] == "PENDING"
    assert any(row["action"] == "PROCESSING_FALLBACK" for row in audit)


def test_thumbnail_failure_does_not_delete_saved_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "short.mp4"
    _video(video, frames=3, fps=3)
    processor = FailSafeReviewProcessor(
        detector=FakeDetector(),
        temporal=None,
        database=tmp_path / "db.sqlite",
        run_root=tmp_path / "runs",
        output_root=tmp_path / "outputs",
    )
    monkeypatch.setattr(
        "src.review_assistant.clips.EventClipWriter.write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("thumbnail failure")
        ),
    )
    with pytest.raises(RuntimeError, match="thumbnail failure"):
        processor.process(video, mode="conservative_review")
    with ReviewDatabase(tmp_path / "db.sqlite") as database:
        runs = database.list_runs()
        events = database.list_events(runs[0]["run_id"])
    assert runs[0]["status"] == "FAILED"
    assert len(events) == 1
    assert events[0]["review_status"] == "PENDING"


def test_verifier_unavailable_event_remains_in_general_queue() -> None:
    engine = aggregator()
    row = candidate(0, source="TEMPORAL_ONLY", confirmed=True)[2]
    row.update(
        {
            "candidate_id": "fallback-1",
            "processing_status": "VERIFIER_UNAVAILABLE",
            "source": "verifier_fallback",
        }
    )
    engine.observe(0, 0.0, [row])
    event = engine.finalize()[0]
    assert event.processing_status == "VERIFIER_UNAVAILABLE"
    assert event.review_status == "PENDING"
    assert event.detections[0].candidate_id == "fallback-1"
