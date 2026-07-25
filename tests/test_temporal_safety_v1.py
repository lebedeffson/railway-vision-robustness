from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.temporal_safety.common import fold_scenes
from src.temporal_safety import ByteTrackAdapter, OCSortAdapter
from src.temporal_safety.evaluator import evaluate, run_tracker
from src.temporal_safety.metrics import gate_checks
from src.temporal_safety.sequence_loader import validate_sequence_index


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (PROJECT / "configs/temporal_safety_v1.yaml").read_text(encoding="utf-8")
)


def parameters(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "high_conf_threshold": 0.07,
        "low_conf_threshold": 0.001,
        "new_track_threshold": 0.01,
        "match_iou_threshold": 0.10,
        "track_buffer": 1,
        "minimum_confirmed_length": 2,
        "maximum_lost_frames": 1,
    }
    result.update(overrides)
    return result


def detection(confidence: float, x: float = 10.0) -> dict[str, object]:
    return {
        "class_id": 0,
        "confidence": confidence,
        "box": [x, 10.0, x + 10.0, 30.0],
    }


@pytest.mark.parametrize("tracker_class", [ByteTrackAdapter, OCSortAdapter])
def test_low_conf_detections_preserved_until_confirmation(tracker_class: type) -> None:
    tracker = tracker_class(parameters(), CONFIG["temporal_logic"])
    tracker.reset("sequence")
    first, _ = tracker.update([detection(0.05)], 100, 100)
    second, _ = tracker.update([detection(0.05, 10.5)], 100, 100)
    assert first == []
    assert len(second) == 1
    assert second[0]["source"] == "temporal_confirmation"
    assert second[0]["confidence"] >= 0.07


def test_interpolation_gap_limited_and_marked() -> None:
    tracker = ByteTrackAdapter(parameters(), CONFIG["temporal_logic"])
    tracker.reset("sequence")
    tracker.update([detection(0.08)], 100, 100)
    first_gap, _ = tracker.update([], 100, 100)
    second_gap, _ = tracker.update([], 100, 100)
    assert len(first_gap) == 1
    assert first_gap[0]["source"] == "temporal_interpolation"
    assert first_gap[0]["interpolated"] is True
    assert second_gap == []


def test_tracker_inference_api_has_no_ground_truth() -> None:
    signature = inspect.signature(ByteTrackAdapter.update)
    assert "ground_truth" not in signature.parameters
    tracker = ByteTrackAdapter(parameters(), CONFIG["temporal_logic"])
    with pytest.raises(TypeError):
        tracker.update([detection(0.08)], 100, 100, ground_truth=[])  # type: ignore[call-arg]


def test_track_confirmation_deterministic() -> None:
    outputs = []
    for _ in range(2):
        tracker = OCSortAdapter(parameters(direction_weight=0.2), CONFIG["temporal_logic"])
        tracker.reset("sequence")
        sequence = []
        for offset in [0.0, 0.5, 1.0, 1.5]:
            emitted, _ = tracker.update([detection(0.04, 10 + offset)], 100, 100)
            sequence.append(emitted)
        outputs.append(sequence)
    assert outputs[0] == outputs[1]


def test_no_future_frame_leakage() -> None:
    left = ByteTrackAdapter(parameters(), CONFIG["temporal_logic"])
    right = ByteTrackAdapter(parameters(), CONFIG["temporal_logic"])
    left.reset("sequence")
    right.reset("sequence")
    prefix = [[detection(0.04)], [detection(0.04, 10.5)]]
    left_outputs = [left.update(frame, 100, 100)[0] for frame in prefix]
    right_outputs = [right.update(frame, 100, 100)[0] for frame in prefix]
    left.update([detection(0.9, 11.0)], 100, 100)
    right.update([], 100, 100)
    assert left_outputs == right_outputs


def test_frame_order_and_scene_validation() -> None:
    frame = pd.DataFrame([
        {
            "grouped_scene_id": "a",
            "source_sequence_id": "q",
            "frame_id": str(index),
            "frame_number": index,
            "timestamp": index / 10,
            "redacted_image_id": f"id{index}",
            "width": 100,
            "height": 100,
            "person_gt_count": 0,
        }
        for index in range(3)
    ])
    audit = validate_sequence_index(frame)
    assert audit["duplicate_frames"] == 0
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_sequence_index(duplicate)


def test_scenes_never_mix_across_folds() -> None:
    groups = [fold_scenes(fold) for fold in range(5)]
    assert len(set.union(*groups)) == 15
    for left in range(5):
        for right in range(left + 1, 5):
            assert not groups[left] & groups[right]
    fit_excluded = set().union(
        *[fold_scenes(fold) for fold in CONFIG["selection"]["fit_excluded_folds"]]
    )
    assert fit_excluded == groups[0] | groups[1]


def test_tracker_grids_are_bounded() -> None:
    for name in ("bytetrack", "ocsort"):
        grid = CONFIG["tracker_grids"][name]
        assert 0 < len(grid) <= 12


def test_same_prediction_stream_can_feed_both_trackers() -> None:
    source = pd.DataFrame([
        {
            "grouped_scene_id": "a",
            "subsequence_id": "q",
            "frame_number": index,
            "image_path": f"f{index}",
            "width": 100,
            "height": 100,
        }
        for index in range(2)
    ])
    raw = {
        "f0": [detection(0.08)],
        "f1": [detection(0.04, 10.5)],
    }
    byte, _ = run_tracker(
        source,
        raw,
        ByteTrackAdapter(parameters(), CONFIG["temporal_logic"]),
        0.6,
    )
    ocsort, _ = run_tracker(
        source,
        raw,
        OCSortAdapter(parameters(direction_weight=0.2), CONFIG["temporal_logic"]),
        0.6,
    )
    assert raw["f0"][0]["confidence"] == 0.08
    assert set(byte) == set(ocsort) == {"f0", "f1"}


def test_metrics_scene_macro_and_false_alarm_units() -> None:
    source = pd.DataFrame([
        {"grouped_scene_id": "s1", "subsequence_id": "q1", "frame_number": 0, "image_path": "a"},
        {"grouped_scene_id": "s2", "subsequence_id": "q2", "frame_number": 0, "image_path": "b"},
    ])
    gt = {
        "a": [{"class_id": 0, "box": [0, 0, 10, 10]}],
        "b": [{"class_id": 0, "box": [0, 0, 10, 10]}],
    }
    predictions = {
        "a": [{"class_id": 0, "box": [0, 0, 10, 10], "confidence": 0.9}],
        "b": [{"class_id": 0, "box": [20, 20, 30, 30], "confidence": 0.9}],
    }
    result, scenes = evaluate(source, gt, predictions, 0.07, 0.5, 10)
    assert result["recall"] == pytest.approx(0.5)
    assert result["false_alarms_per_minute"] == pytest.approx(300)
    assert scenes["recall"].mean() == pytest.approx(0.5)


def test_false_alarm_and_worst_scene_gate() -> None:
    baseline = {
        "recall": 0.2,
        "FN_per_frame": 1.0,
        "false_alarms_per_minute": 10.0,
        "f1": 0.4,
    }
    candidate = {
        "recall": 0.3,
        "FN_per_frame": 0.8,
        "false_alarms_per_minute": 12.0,
        "f1": 0.38,
        "maximum_scene_recall_degradation": 0.0,
    }
    checks = gate_checks(
        baseline,
        candidate,
        CONFIG["triage_gate"],
        CONFIG["triage_gate"]["maximum_scene_recall_degradation"],
    )
    assert all(checks.values())


def test_test_sealed_before_gate() -> None:
    lock = PROJECT / "outputs/temporal_safety_v1/protocol/protocol_lock.json"
    if lock.exists():
        payload = json.loads(lock.read_text(encoding="utf-8"))
        assert payload["test_status"] == "SEALED"
        assert payload["test_access_count"] == 0
    assert not (PROJECT / CONFIG["test_access"]["marker"]).exists()
