from __future__ import annotations

import json
from pathlib import Path

import pytest

from canonical_m4_tiling import assign_ground_truth_to_tiles
from person_v8.audit_active_data import audit
from person_v8.build_splits import choose_screening_roles, constrained_folds
from person_v8.common import ROOT, load_config, read_yolo_labels
from person_v8.lock_execution import create_execution_lock
from person_v8.run_screening import check_authorization


def test_v8_protocol_is_data_first_and_test_sealed() -> None:
    config = load_config()
    assert config["protocol_id"] == "canonical-v8-person-active-data-v1"
    assert config["status"] == "WAITING_FOR_NEW_DATA"
    assert config["test"] == {
        "source": "original_test",
        "sealed": True,
        "readable": False,
        "reusable": False,
        "access_count_required": 0,
        "opened_marker": "outputs/person_v8/test/TEST_OPENED.json",
    }
    forbidden = set(config["model"]["forbidden"])
    assert {
        "NWD",
        "QFL",
        "GroupDRO",
        "MixStyle",
        "SWAD",
        "P2",
        "attention",
        "temporal_fusion",
        "tracklet_verifier",
        "instance_pasting",
    } <= forbidden


def test_v8_acquisition_minimums_are_frozen() -> None:
    acquisition = load_config()["acquisition"]
    assert acquisition["independent_new_scene_minimum"] == 3
    assert acquisition["selected_frame_minimum"] == 300
    assert acquisition["person_box_minimum"] == 500
    assert acquisition["small_or_distant_fraction_minimum"] == pytest.approx(0.40)


def test_person_label_parser_accepts_person_and_empty_background(
    tmp_path: Path,
) -> None:
    positive = tmp_path / "positive.txt"
    positive.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    labels = read_yolo_labels(positive, 100, 50)
    assert labels == [{"class_id": 0, "box": [40.0, 15.0, 60.0, 35.0]}]
    negative = tmp_path / "negative.txt"
    negative.write_text("", encoding="utf-8")
    assert read_yolo_labels(negative, 100, 50) == []


@pytest.mark.parametrize(
    "line,error",
    [
        ("1 0.5 0.5 0.2 0.2\n", "invalid person class"),
        ("0 1.0 0.5 0.2 0.2\n", "box extends outside image"),
        ("0 0.5 0.5 -0.2 0.2\n", "coordinates outside"),
        ("0 0.5 0.5 0.2\n", "expected 5"),
    ],
)
def test_person_label_parser_rejects_invalid_labels(
    tmp_path: Path, line: str, error: str
) -> None:
    path = tmp_path / "invalid.txt"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        read_yolo_labels(path, 100, 50)


def test_frozen_tiling_preserves_border_person() -> None:
    tiling = load_config(ROOT / "configs/canonical_v2_m4_full_protocol.yaml")
    labels = [{"class_id": 0, "box": [1755.0, 100.0, 1775.0, 150.0]}]
    assigned = assign_ground_truth_to_tiles(labels, tiling)
    retained = [
        label
        for tile_labels in assigned.values()
        for label in tile_labels
        if label["source_gt_id"] == 0
    ]
    assert retained


def test_screening_roles_require_distinct_new_scenes() -> None:
    rows = [
        {
            "grouped_scene_id": f"new_{index}",
            "frames": str(100 + index),
            "person_gt": str(200 + index * 5),
            "small_gt": str(100 + index * 3),
        }
        for index in range(4)
    ]
    heldout, confirmation, support = choose_screening_roles(rows)
    assert heldout != confirmation
    assert heldout not in support
    assert confirmation not in support
    assert len(support) == 2


def test_constrained_folds_are_deterministic_and_scene_disjoint() -> None:
    rows = [
        {
            "grouped_scene_id": f"scene_{index:02d}",
            "frames": 10 + index,
            "person_gt": 20 + index,
            "small_gt": 10 + index,
            "medium_gt": 5,
            "large_gt": 1,
        }
        for index in range(15)
    ]
    first, first_score = constrained_folds(rows, 5, 20260725, 100)
    second, second_score = constrained_folds(rows, 5, 20260725, 100)
    assert first == second
    assert first_score == second_score
    flattened = [scene for fold in first for scene in fold]
    assert len(flattened) == len(set(flattened)) == len(rows)


def test_missing_acquisition_data_is_waiting_not_fail(tmp_path: Path) -> None:
    result = audit(tmp_path / "missing.csv", tmp_path / "missing-corrections.csv")
    assert result["status"] == "WAITING_FOR_NEW_DATA"
    assert result["training_authorized"] is False
    assert result["test_access_count"] == 0


def test_execution_lock_is_blocked_before_cpu_gate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Execution lock requires|Execution inputs"):
        create_execution_lock(tmp_path / "manifest.csv", tmp_path / "corrections.csv")


def test_screening_is_blocked_without_execution_lock() -> None:
    execution_lock = ROOT / "outputs/person_v8/protocol/V8_EXECUTION_LOCK.json"
    assert not execution_lock.exists()
    with pytest.raises(RuntimeError, match="TRAINING_BLOCKED"):
        check_authorization()


def test_systemd_requires_execution_lock_and_intact_test_seal() -> None:
    unit = (ROOT / "systemd/tnorm-person-v8-screening.service").read_text(
        encoding="utf-8"
    )
    assert (
        "ConditionPathExists=%h/Code/андрей/TNormFilter_handoff/"
        "outputs/person_v8/protocol/V8_EXECUTION_LOCK.json"
    ) in unit
    assert (
        "ConditionPathExists=!%h/Code/андрей/TNormFilter_handoff/"
        "outputs/person_v8/test/TEST_OPENED.json"
    ) in unit
    assert "--check-only" in unit


def test_acquisition_schema_and_template_match() -> None:
    schema = json.loads(
        (ROOT / "protocol/v8/schemas/acquisition_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    header = (
        ROOT / "protocol/v8/templates/acquisition_manifest.csv"
    ).read_text(encoding="utf-8").splitlines()[0].split(",")
    assert schema["required_columns"] == header

