from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/article_evidence_v1"
CONFIG = yaml.safe_load(
    (ROOT / "configs/event_engineering_evidence_v1.yaml").read_text(
        encoding="utf-8"
    )
)


def test_event_engineering_lock_exact_grid() -> None:
    lock = json.loads(
        (
            OUTPUT
            / "operator_assistant/EVENT_ENGINEERING_LOCK.json"
        ).read_text(encoding="utf-8")
    )
    grid = lock["event_parameter_grid"]
    assert grid["join_time_seconds"] == [1, 2, 3, 5]
    assert grid["minimum_iou"] == [0.1, 0.2, 0.3]
    assert grid["maximum_center_distance_ratio"] == [0.05, 0.1, 0.15]
    assert grid["reopen_window_seconds"] == [10, 30, 60]
    assert grid["number_of_configurations"] == 108
    assert lock["test_status"] == "SEALED"
    assert lock["test_access_count"] == 0


def test_event_sensitivity_grid_complete_but_partial_input() -> None:
    frame = pd.read_csv(
        OUTPUT / "operator_assistant/EVENT_SENSITIVITY.csv"
    )
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
    assert set(frame["evidence_status"]) == {"DIAGNOSTIC_PARTIAL_INPUT"}
    audit = json.loads(
        (
            OUTPUT
            / "operator_assistant/EVENT_SENSITIVITY_AUDIT.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["status"] == "BLOCKED_MISSING_PRE_AGGREGATION_STREAM"
    assert audit["published_unique_events"] == 3
    assert audit["default_replay_unique_events"] == 4
    assert audit["published_default_reproduced"] is False


def test_gt_episode_metrics_are_not_imputed() -> None:
    frame = pd.read_csv(
        OUTPUT / "operator_assistant/EVENT_SENSITIVITY.csv"
    )
    for column in (
        "fragmentation_true_episode",
        "false_merge_count",
        "true_episode_coverage",
        "gt_episode_metric_status",
    ):
        assert set(frame[column]) == {"BLOCKED_MISSING_ARTIFACT"}


def test_event_source_hashes_unchanged() -> None:
    lock = json.loads(
        (
            OUTPUT
            / "operator_assistant/EVENT_ENGINEERING_LOCK.json"
        ).read_text(encoding="utf-8")
    )
    for record in lock["inputs"].values():
        path = ROOT / record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"]


def test_failsafe_checks_are_complete_and_factual() -> None:
    frame = pd.read_csv(
        OUTPUT / "operator_assistant/FAILSAFE_CHECKS.csv"
    )
    assert len(frame) == 10
    assert frame["check"].nunique() == 10
    failed = set(
        frame.loc[
            frame["status"] == "FAIL_CURRENT_IMPLEMENTATION", "check"
        ]
    )
    assert failed == {
        "verifier_failure_routes_event_to_general_queue",
        "unknown_error_defaults_to_conservative_display",
    }
    assert (
        frame.loc[frame["check"] == "no_autonomous_alarm", "status"].iloc[0]
        == "PASS"
    )
    assert (
        frame.loc[frame["check"] == "no_safety_actuation", "status"].iloc[0]
        == "PASS"
    )


def test_scaling_preserves_verified_single_stream_and_blocks_unsafe_loads() -> None:
    frame = pd.read_csv(OUTPUT / "scaling/SCALING_RESULTS.csv")
    assert set(frame["streams"]) == {1, 2, 4}
    one = frame[frame["streams"] == 1].iloc[0]
    assert one["status"] == "PASS_EXISTING_VERIFIED_RUN"
    assert int(one["measured_frames_total"]) == 1000
    assert abs(float(one["total_fps_conservative"]) - 19.20825246649) < 1e-9
    blocked = frame[frame["streams"].isin([2, 4])]
    assert set(blocked["status"]) == {"BLOCKED_RESOURCE_LIMIT"}
    assert (
        blocked["total_fps_conservative"].astype(str)
        == "BLOCKED_RESOURCE_LIMIT"
    ).all()


def test_event_engineering_figures_open() -> None:
    stems = (
        "FIG_08_EVENT_SENSITIVITY",
        "FIG_09_EVENT_AGGREGATION_TRADEOFF",
        "FIG_10_STREAM_SCALING",
    )
    for stem in stems:
        png = OUTPUT / "figures" / f"{stem}.png"
        svg = OUTPUT / "figures" / f"{stem}.svg"
        with Image.open(png) as image:
            assert min(image.info.get("dpi", (0, 0))) >= 299
        assert "<svg" in svg.read_text(encoding="utf-8")[:1000]


def test_event_engineering_never_opens_test() -> None:
    assert CONFIG["constraints"]["test_status"] == "SEALED"
    assert CONFIG["constraints"]["test_access_count"] == 0
    assert not (
        ROOT / "outputs/temporal_safety_v1/test/TEST_OPENED.json"
    ).exists()
