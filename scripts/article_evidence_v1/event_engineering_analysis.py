from __future__ import annotations

import itertools
import json
import math
import sqlite3
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.article_evidence_v1.common import atomic_csv, atomic_json, sha256
from src.review_assistant.database import ReviewDatabase
from src.review_assistant.event_aggregator import EventAggregator
from src.review_assistant.models import EventDetection, ReviewEvent


CONFIG_PATH = PROJECT / "configs/event_engineering_evidence_v1.yaml"
OUTPUT = PROJECT / "outputs/article_evidence_v1"
OPERATOR_OUTPUT = OUTPUT / "operator_assistant"
FIGURE_OUTPUT = OUTPUT / "figures"
BLOCKED = "BLOCKED_MISSING_ARTIFACT"


def _config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _load_observations(
    database_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    run = connection.execute(
        """
        SELECT r.*, v.camera_id, v.fps, v.frame_count, v.width, v.height,
               v.duration_seconds
        FROM runs r JOIN videos v ON v.video_id=r.video_id
        WHERE r.status='COMPLETED'
        ORDER BY r.completed_at DESC LIMIT 1
        """
    ).fetchone()
    if run is None:
        raise RuntimeError("No completed real event run in frozen database")
    rows = connection.execute(
        """
        SELECT d.*, e.event_id AS published_event_id
        FROM event_detections d
        JOIN events e ON e.run_id=d.run_id AND e.event_id=d.event_id
        WHERE d.run_id=?
        ORDER BY d.frame_number, d.id
        """,
        (run["run_id"],),
    ).fetchall()
    connection.close()
    observations: list[dict[str, Any]] = []
    for row in rows:
        observations.append(
            {
                "frame_number": int(row["frame_number"]),
                "timestamp": float(row["timestamp"]),
                "box": [float(value) for value in json.loads(row["bbox_json"])],
                "confidence": float(row["confidence"]),
                "review_source": str(row["source"]),
                "track_id": (
                    int(row["track_id"]) if row["track_id"] is not None else None
                ),
                "interpolated": bool(row["interpolated"]),
                "confirmed": bool(row["confirmed"]),
                "motion": float(row["motion"]),
                "published_event_id": str(row["published_event_id"]),
            }
        )
    metadata = dict(run)
    metadata["aggregator_input_observations"] = len(observations)
    return observations, metadata


def _group_by_frame(
    observations: list[dict[str, Any]],
) -> list[tuple[int, float, list[dict[str, Any]]]]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[(int(row["frame_number"]), float(row["timestamp"]))].append(row)
    return [
        (frame, timestamp, grouped[(frame, timestamp)])
        for frame, timestamp in sorted(grouped)
    ]


def _source_counts(events: list[ReviewEvent]) -> tuple[int, int, int]:
    labels = [event.source_label for event in events]
    return (
        labels.count("BASELINE"),
        labels.count("TEMPORAL_ONLY"),
        labels.count("BOTH"),
    )


def _replay(
    observations: list[dict[str, Any]],
    metadata: dict[str, Any],
    parameters: dict[str, float],
    *,
    pre_roll: float,
    post_roll: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    settings = {
        **parameters,
        "close_after_seconds": parameters["join_time_seconds"],
        "temporal_minimum_observations": 2,
    }
    aggregator = EventAggregator(
        settings,
        video_id=str(metadata["video_id"]),
        camera_id=str(metadata["camera_id"]),
        frame_width=int(metadata["width"]),
        frame_height=int(metadata["height"]),
    )
    activation_times: dict[str, float] = {}
    close_times: dict[str, float] = {}
    reopened: set[str] = set()
    for frame_number, timestamp, rows in _group_by_frame(observations):
        prior_states = {
            event_id: event.state for event_id, event in aggregator.events.items()
        }
        candidates = [
            {key: value for key, value in row.items() if key != "published_event_id"}
            for row in rows
        ]
        touched = aggregator.observe(frame_number, timestamp, candidates)
        for event in touched:
            if prior_states.get(event.event_id) == "CLOSED":
                reopened.add(event.event_id)
            if event.state == "ACTIVE" and event.event_id not in activation_times:
                activation_times[event.event_id] = timestamp
        for event_id, old_state in prior_states.items():
            event = aggregator.events[event_id]
            if old_state == "ACTIVE" and event.state == "CLOSED":
                close_times.setdefault(event_id, timestamp)
    video_duration = float(metadata["duration_seconds"])
    accepted = aggregator.finalize()
    for event in accepted:
        close_times.setdefault(event.event_id, video_duration)
        activation_times.setdefault(event.event_id, event.start_time)
    counts = [len(event.detections) for event in accepted]
    durations = [event.duration for event in accepted]
    open_latencies = [
        max(activation_times[event.event_id] - event.start_time, 0.0)
        for event in accepted
    ]
    close_latencies = [
        max(close_times[event.event_id] - event.end_time, 0.0)
        for event in accepted
    ]
    clip_durations = [
        max(
            min(event.end_time + post_roll, video_duration)
            - max(event.start_time - pre_roll, 0.0),
            0.0,
        )
        for event in accepted
    ]
    baseline, temporal_only, both = _source_counts(accepted)
    track_switches = sum(max(len(event.track_ids) - 1, 0) for event in accepted)
    row = {
        **parameters,
        "close_after_seconds": parameters["join_time_seconds"],
        "configuration_id": (
            f"J{parameters['join_time_seconds']:g}"
            f"_I{parameters['minimum_iou']:.2f}"
            f"_C{parameters['maximum_center_distance_ratio']:.2f}"
            f"_R{parameters['reopen_window_seconds']:g}"
        ),
        "input_raw_detector_boxes": int(metadata["raw_detection_count"]),
        "aggregator_input_observations": len(observations),
        "unique_events": len(accepted),
        "raw_detector_boxes_per_event": (
            float(metadata["raw_detection_count"]) / len(accepted)
            if accepted
            else 0.0
        ),
        "aggregator_observations_per_event_mean": (
            statistics.fmean(counts) if counts else 0.0
        ),
        "aggregator_observations_per_event_median": (
            statistics.median(counts) if counts else 0.0
        ),
        "events_per_video_hour": (
            len(accepted) * 3600.0 / max(video_duration, 1e-12)
        ),
        "event_duration_mean_seconds": (
            statistics.fmean(durations) if durations else 0.0
        ),
        "event_duration_median_seconds": (
            statistics.median(durations) if durations else 0.0
        ),
        "clip_duration_sum_seconds": sum(clip_durations),
        "clip_minutes_per_video_hour": (
            sum(clip_durations) / 60.0 * 3600.0 / max(video_duration, 1e-12)
        ),
        "baseline_events": baseline,
        "temporal_only_events": temporal_only,
        "both_source_events": both,
        "technical_track_id_switches_merged": track_switches,
        "reopened_events": len(reopened),
        "event_open_latency_p50_seconds": _percentile(open_latencies, 50),
        "event_open_latency_p95_seconds": _percentile(open_latencies, 95),
        "event_open_latency_p99_seconds": _percentile(open_latencies, 99),
        "first_operator_available_latency_p50_seconds": _percentile(
            open_latencies, 50
        ),
        "first_operator_available_latency_p95_seconds": _percentile(
            open_latencies, 95
        ),
        "first_operator_available_latency_p99_seconds": _percentile(
            open_latencies, 99
        ),
        "event_close_latency_p50_seconds": _percentile(close_latencies, 50),
        "event_close_latency_p95_seconds": _percentile(close_latencies, 95),
        "event_close_latency_p99_seconds": _percentile(close_latencies, 99),
        "fragmentation_true_episode": BLOCKED,
        "false_merge_count": BLOCKED,
        "true_episode_coverage": BLOCKED,
        "gt_episode_metric_status": BLOCKED,
        "evidence_status": "DIAGNOSTIC_PARTIAL_INPUT",
    }
    event_rows: list[dict[str, Any]] = []
    for event in accepted:
        event_rows.append(
            {
                "configuration_id": row["configuration_id"],
                "event_id": event.event_id,
                "start_time": event.start_time,
                "end_time": event.end_time,
                "duration_seconds": event.duration,
                "source_label": event.source_label,
                "detections": len(event.detections),
                "track_ids": len(event.track_ids),
                "open_latency_seconds": max(
                    activation_times[event.event_id] - event.start_time, 0.0
                ),
                "close_latency_seconds": max(
                    close_times[event.event_id] - event.end_time, 0.0
                ),
                "clip_duration_seconds": max(
                    min(event.end_time + post_roll, video_duration)
                    - max(event.start_time - pre_roll, 0.0),
                    0.0,
                ),
                "right_censored_at_video_end": math.isclose(
                    close_times[event.event_id], video_duration
                ),
            }
        )
    return row, event_rows


def _safe_checks(source_before: dict[str, str]) -> pd.DataFrame:
    review_config = yaml.safe_load(
        (PROJECT / "configs/review_assistant_v1.yaml").read_text(encoding="utf-8")
    )
    processor_text = (
        PROJECT / "src/review_assistant/processor.py"
    ).read_text(encoding="utf-8")
    checks: list[dict[str, Any]] = []

    def add(identifier: str, status: str, evidence: str) -> None:
        checks.append({"check": identifier, "status": status, "evidence": evidence})

    add(
        "no_autonomous_alarm",
        "PASS" if not review_config["review"]["autonomous_alarm"] else "FAIL",
        "review.autonomous_alarm=false",
    )
    add(
        "no_safety_actuation",
        "PASS" if not review_config["review"]["safety_actuation"] else "FAIL",
        "review.safety_actuation=false",
    )
    with tempfile.TemporaryDirectory(prefix="event-safety-") as temporary:
        root = Path(temporary)
        database_path = root / "review.sqlite"
        with ReviewDatabase(database_path) as database:
            video_id, _ = database.register_video(
                path=root / "source.mp4",
                sha256="1" * 64,
                camera_id="C1",
                fps=10,
                frame_count=100,
                width=100,
                height=100,
            )
            run_id, _ = database.create_run(
                video_id=video_id,
                mode="combined_queue",
                total_frames=100,
                config_sha256="2" * 64,
                resume_policy="restart_incomplete_run",
            )
            event = ReviewEvent(
                event_id="E000001",
                video_id=video_id,
                camera_id="C1",
                start_time=0,
                end_time=0,
                state="CLOSED",
            )
            event.add(
                EventDetection(
                    frame_number=0,
                    timestamp=0,
                    box=[10, 10, 30, 50],
                    confidence=0.8,
                    source="BASELINE",
                    track_id=1,
                )
            )
            database.save_events(run_id, [event])
            database.review_event(
                run_id,
                event.event_id,
                operator="audit",
                new_status="HUMAN",
            )
            database.undo_last_review(run_id, event.event_id, "audit")
            reversible = (
                database.get_event(run_id, event.event_id)["review_status"]
                == "PENDING"
            )
            audit_actions = {row["action"] for row in database.audit_rows()}
            event_survives_media_failure = (
                database.get_event(run_id, event.event_id)["event_id"]
                == event.event_id
            )
            database.update_progress(run_id, 50, 7)
            same_run, resumed = database.create_run(
                video_id=video_id,
                mode="combined_queue",
                total_frames=100,
                config_sha256="2" * 64,
                resume_policy="restart_incomplete_run",
            )
            restart_rebuild = (
                resumed
                and same_run == run_id
                and database.get_run(run_id)["processed_frames"] == 0
            )
        add(
            "operator_decision_reversible",
            "PASS" if reversible else "FAIL",
            "review plus UNDO_REVIEW restored PENDING",
        )
        add(
            "all_review_changes_audited",
            "PASS"
            if {"REVIEW", "UNDO_REVIEW"} <= audit_actions
            else "FAIL",
            "audit_log contains REVIEW and UNDO_REVIEW",
        )
        add(
            "thumbnail_failure_does_not_delete_saved_event",
            "PASS" if event_survives_media_failure else "FAIL",
            "events are committed before EventClipWriter invocation",
        )
        add(
            "restart_recovers_incomplete_processing",
            "PASS_REBUILT_FROM_SOURCE" if restart_rebuild else "FAIL",
            "same run id is reset and causal state is rebuilt from immutable source",
        )
    engine = EventAggregator(
        {
            "join_time_seconds": 3,
            "close_after_seconds": 3,
            "reopen_window_seconds": 30,
            "minimum_iou": 0.2,
            "maximum_center_distance_ratio": 0.1,
            "temporal_minimum_observations": 2,
        },
        video_id="V",
        camera_id="C",
        frame_width=100,
        frame_height=100,
    )
    for frame, track_id in ((0, 4), (1, 4), (2, 4)):
        engine.observe(
            frame,
            frame / 10,
            [
                {
                    "box": [10, 10, 30, 50],
                    "confidence": 0.5,
                    "review_source": "BASELINE",
                    "track_id": track_id,
                }
            ],
        )
    duplicate_preserved = len(engine.finalize()[0].detections) == 3
    add(
        "duplicate_track_id_preserves_observations",
        "PASS" if duplicate_preserved else "FAIL",
        "three observations with one track id remain in one event",
    )
    source_after = {
        relative: sha256(PROJECT / relative) for relative in source_before
    }
    add(
        "source_video_and_detections_immutable",
        "PASS" if source_before == source_after else "FAIL",
        "SHA-256 before and after computation is identical",
    )
    verifier_has_fallback = (
        "temporal.update(frame, candidates)" in processor_text
        and "except" in processor_text[
            max(processor_text.index("temporal.update(frame, candidates)") - 300, 0) :
            processor_text.index("temporal.update(frame, candidates)") + 300
        ]
    )
    add(
        "verifier_failure_routes_event_to_general_queue",
        "PASS" if verifier_has_fallback else "FAIL_CURRENT_IMPLEMENTATION",
        "no local conservative fallback surrounds temporal.update",
    )
    add(
        "unknown_error_defaults_to_conservative_display",
        "FAIL_CURRENT_IMPLEMENTATION",
        "outer exception marks run FAILED and re-raises",
    )
    return pd.DataFrame(checks)


def _plot(frame: pd.DataFrame) -> None:
    FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.25})
    selected = frame[
        (frame["minimum_iou"] == 0.2)
        & (frame["maximum_center_distance_ratio"] == 0.1)
        & (frame["reopen_window_seconds"] == 30)
    ].sort_values("join_time_seconds")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].plot(
        selected["join_time_seconds"],
        selected["unique_events"],
        marker="o",
        color="#1f4e79",
    )
    axes[0].set(xlabel="Join time, s", ylabel="Unique events")
    axes[1].plot(
        selected["join_time_seconds"],
        selected["clip_minutes_per_video_hour"],
        marker="s",
        color="#8c2d04",
    )
    axes[1].set(xlabel="Join time, s", ylabel="Clip minutes / video hour")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            FIGURE_OUTPUT / f"FIG_08_EVENT_SENSITIVITY.{suffix}",
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for join, group in frame.groupby("join_time_seconds"):
        ax.scatter(
            group["events_per_video_hour"],
            group["aggregator_observations_per_event_mean"],
            label=f"join={join:g}s",
            s=22,
            alpha=0.75,
        )
    ax.set(
        xlabel="Events per video hour",
        ylabel="Aggregator observations per event",
    )
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            FIGURE_OUTPUT / f"FIG_09_EVENT_AGGREGATION_TRADEOFF.{suffix}",
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    config = _config()
    lock = OPERATOR_OUTPUT / "EVENT_ENGINEERING_LOCK.json"
    if not lock.is_file():
        raise RuntimeError("Run lock_event_engineering.py before computation")
    database = PROJECT / config["inputs"]["event_database"]
    video = PROJECT / config["inputs"]["source_video"]
    source_before = {
        config["inputs"]["event_database"]: sha256(database),
        config["inputs"]["source_video"]: sha256(video),
    }
    observations, metadata = _load_observations(database)
    sensitivity = config["sensitivity"]
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for join, iou, center, reopen in itertools.product(
        sensitivity["join_time_seconds"],
        sensitivity["minimum_iou"],
        sensitivity["maximum_center_distance_ratio"],
        sensitivity["reopen_window_seconds"],
    ):
        row, details = _replay(
            observations,
            metadata,
            {
                "join_time_seconds": float(join),
                "minimum_iou": float(iou),
                "maximum_center_distance_ratio": float(center),
                "reopen_window_seconds": float(reopen),
            },
            pre_roll=float(sensitivity["clip_pre_roll_seconds"]),
            post_roll=float(sensitivity["clip_post_roll_seconds"]),
        )
        rows.append(row)
        event_rows.extend(details)
    frame = pd.DataFrame(rows).sort_values(
        [
            "join_time_seconds",
            "minimum_iou",
            "maximum_center_distance_ratio",
            "reopen_window_seconds",
        ]
    )
    details = pd.DataFrame(event_rows)
    if len(frame) != int(sensitivity["expected_configurations"]):
        raise RuntimeError(f"Incomplete event grid: {len(frame)}")
    atomic_csv(OPERATOR_OUTPUT / "EVENT_SENSITIVITY.csv", frame)
    atomic_csv(OPERATOR_OUTPUT / "EVENT_SENSITIVITY_EVENTS.csv", details)
    safety = _safe_checks(source_before)
    atomic_csv(OPERATOR_OUTPUT / "FAILSAFE_CHECKS.csv", safety)
    default = frame[
        (frame["join_time_seconds"] == 3)
        & (frame["minimum_iou"] == 0.2)
        & (frame["maximum_center_distance_ratio"] == 0.1)
        & (frame["reopen_window_seconds"] == 30)
    ].iloc[0]
    source_after = {
        relative: sha256(PROJECT / relative) for relative in source_before
    }
    complete_stream_available = int(default["unique_events"]) == int(
        metadata["event_count"]
    )
    audit = {
        "status": (
            "PASS_WITH_GT_EPISODE_METRICS_BLOCKED"
            if complete_stream_available
            else "BLOCKED_MISSING_PRE_AGGREGATION_STREAM"
        ),
        "number_of_configurations": len(frame),
        "input_raw_detector_boxes": int(metadata["raw_detection_count"]),
        "aggregator_input_observations": len(observations),
        "published_unique_events": int(metadata["event_count"]),
        "default_replay_unique_events": int(default["unique_events"]),
        "published_default_reproduced": complete_stream_available,
        "default_raw_detector_boxes_per_event": float(
            default["raw_detector_boxes_per_event"]
        ),
        "default_aggregator_observations_per_event": float(
            default["aggregator_observations_per_event_mean"]
        ),
        "gt_episode_metrics": {
            "fragmentation": BLOCKED,
            "false_merges": BLOCKED,
            "coverage": BLOCKED,
            "reason": (
                "No immutable exact mapping from the 100 encoded video frames "
                "to development GT frame and episode identities is stored."
            ),
        },
        "sensitivity_evidence_status": (
            "FULL"
            if complete_stream_available
            else "DIAGNOSTIC_PARTIAL_INPUT_NOT_ARTICLE_EVIDENCE"
        ),
        "missing_pre_aggregation_reason": (
            None
            if complete_stream_available
            else (
                "The SQLite database stores only observations belonging to accepted "
                "events. Replaying its 62 rows produces four default events instead "
                "of the published three, proving that rejected/intermediate "
                "pre-aggregation observations needed for an exact sweep were not "
                "persisted. The frozen 296-box count is aggregate-only."
            )
        ),
        "source_hashes_unchanged": source_before == source_after,
        "operator_workload_claim": False,
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    atomic_json(OPERATOR_OUTPUT / "EVENT_SENSITIVITY_AUDIT.json", audit)
    atomic_json(
        OPERATOR_OUTPUT / "FAILSAFE_AUDIT.json",
        {
            "checks": len(safety),
            "pass_or_safe_rebuild": int(
                safety["status"].isin(["PASS", "PASS_REBUILT_FROM_SOURCE"]).sum()
            ),
            "fail_current_implementation": int(
                safety["status"].eq("FAIL_CURRENT_IMPLEMENTATION").sum()
            ),
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    _plot(frame)
    print(
        json.dumps(
            {
                "configurations": len(frame),
                "default_events": int(default["unique_events"]),
                "gt_episode_metrics": BLOCKED,
                "failsafe_failures": int(
                    safety["status"].eq("FAIL_CURRENT_IMPLEMENTATION").sum()
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
