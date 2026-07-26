from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest
import yaml

from scripts.new_scenes.audit_new_scenes import (
    _annotation_digest,
    _maximum_consecutive,
    _timestamp_key,
)
from scripts.new_scenes.authorize_independent_data import authorize
from scripts.new_scenes.common import sha256, verify_parent_state


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (PROJECT / "configs/new_scenes_v1.yaml").read_text(encoding="utf-8")
)
NEXT_CONFIG = yaml.safe_load(
    (PROJECT / "configs/independent_data_v1.yaml").read_text(encoding="utf-8")
)
OUTPUT = PROJECT / "outputs/new_scenes_v1"


def test_parent_negative_results_and_release_are_immutable() -> None:
    state = verify_parent_state()
    assert state["v8b"]["status"] == "FINAL_NEGATIVE_RESULT"
    assert state["temporal_safety_v1"]["status"] == "DEVELOPMENT_FAIL"
    assert state["temporal_verifier_v1"]["status"] == "DEVELOPMENT_FAIL"
    assert state["crop_verifier_v1"]["status"] == "CLOSED_NO_PRACTICAL_GATE"
    assert state["release_v0_11"]["tag"] == (
        "v0.11-crop-verifier-closed-no-practical-gate"
    )
    assert state["test_status"] == "SEALED"
    assert state["test_access_count"] == 0
    for key in (
        "v8b",
        "temporal_safety_v1",
        "temporal_verifier_v1",
        "crop_verifier_v1",
    ):
        artifact = PROJECT / state[key]["artifact"]
        assert sha256(artifact) == state[key]["sha256"]


def test_acquisition_budget_and_diversity_are_frozen() -> None:
    gate = CONFIG["acquisition_gate"]
    assert CONFIG["protocol_id"] == "railway-person-new-scenes-v1"
    assert CONFIG["scope"] == "data_only"
    assert gate["independent_scenes_minimum"] == 8
    assert gate["independent_scenes_maximum"] == 12
    assert gate["distinct_camera_or_capture_points_minimum"] == 3
    assert gate["illumination_conditions_minimum"] == 2
    assert gate["frames_per_scene_minimum"] == 100
    assert gate["frames_per_scene_maximum"] == 300
    assert gate["total_frames_minimum"] == 1500
    assert gate["total_frames_maximum"] == 3000
    assert gate["continuous_fragment_minimum_frames"] == 20
    assert gate["continuous_fragment_target_maximum_frames"] == 100
    assert set(gate["required_strata"]) == {
        "person_present",
        "no_person_hard_negative",
        "small_person",
        "occlusion",
        "entry_exit",
    }


def test_scene_sequence_and_source_video_are_indivisible() -> None:
    independence = CONFIG["scene_independence"]
    assert independence["one_scene_per_sequence"] is True
    assert independence["one_split_per_scene"] is True
    assert independence["one_split_per_sequence"] is True
    assert independence["adjacent_old_video_fragments_are_new_scenes"] is False
    cpu_gate = CONFIG["cpu_gate"]
    assert cpu_gate["scene_leakage"] == 0
    assert cpu_gate["sequence_leakage"] == 0
    assert cpu_gate["source_video_fragment_leakage"] == 0
    assert cpu_gate["source_video_split_leakage"] == 0
    assert cpu_gate["duplicate_frame_number"] == 0
    assert cpu_gate["non_monotonic_timestamp"] == 0


@pytest.mark.parametrize(
    "schema_name,template_name",
    [
        ("NEW_SCENES_MANIFEST", "NEW_SCENES_MANIFEST"),
        ("NEW_SCENES_ANNOTATIONS", "NEW_SCENES_ANNOTATIONS"),
        ("HARD_NEGATIVE_AUDIT", "HARD_NEGATIVE_AUDIT"),
        ("ANNOTATION_REVIEW_LOG", "ANNOTATION_REVIEW_LOG"),
    ],
)
def test_schema_and_template_headers_match(
    schema_name: str, template_name: str
) -> None:
    schema = json.loads(
        (
            PROJECT
            / f"protocol/new_scenes_v1/schemas/{schema_name}.schema.json"
        ).read_text(encoding="utf-8")
    )
    with (
        PROJECT / f"protocol/new_scenes_v1/templates/{template_name}.csv"
    ).open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert schema["required_columns"] == header


def test_frame_sequence_helpers_are_deterministic() -> None:
    assert _maximum_consecutive([9, 5, 6, 6, 7, 12]) == 3
    assert _maximum_consecutive([]) == 0
    assert _timestamp_key("1720000000.5") == pytest.approx(1720000000.5)
    assert _timestamp_key("2026-07-26T12:00:00Z") == pytest.approx(
        1785067200.0
    )
    with pytest.raises(ValueError):
        _timestamp_key("")


def test_annotation_digest_is_order_independent() -> None:
    rows = [
        {"annotation_id": "b", "image_id": "image", "x1": "2"},
        {"annotation_id": "a", "image_id": "image", "x1": "1"},
    ]
    assert _annotation_digest(rows) == _annotation_digest(list(reversed(rows)))


def test_independent_experiment_is_staged_and_blocked() -> None:
    assert NEXT_CONFIG["protocol_id"] == "railway-person-independent-data-v1"
    assert NEXT_CONFIG["status"] == "BLOCKED_BY_NEW_SCENES_GATE"
    assert NEXT_CONFIG["candidates"]["B2"]["run_only_after"] == (
        "B1_DETECTOR_GATE_PASS"
    )
    assert NEXT_CONFIG["candidates"]["B3"]["run_only_after"] == (
        "B2_TEMPORAL_GATE_PASS"
    )
    detector_gate = NEXT_CONFIG["detector_gate"]
    assert detector_gate["absolute_macro_recall_improvement_min"] == 0.10
    assert detector_gate["relative_FN_per_frame_reduction_min"] == 0.15
    assert detector_gate["absolute_F1_improvement_min"] == 0.05
    assert detector_gate["relative_false_alarms_increase_max"] == 0.20


def test_authorization_is_blocked_without_passing_data_audit() -> None:
    audit_path = OUTPUT / "NEW_SCENES_AUDIT.json"
    if audit_path.is_file():
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        assert payload["status"] != "PASS"
    with pytest.raises(
        RuntimeError,
        match="not locked|audit is missing|audit must PASS",
    ):
        authorize()


def test_test_is_physically_sealed() -> None:
    assert CONFIG["test"]["status"] == "SEALED"
    assert CONFIG["test"]["access_count"] == 0
    assert NEXT_CONFIG["test"]["status"] == "SEALED"
    assert NEXT_CONFIG["test"]["access_count"] == 0
    assert not (PROJECT / CONFIG["test"]["marker"]).exists()
    assert not (PROJECT / CONFIG["test"]["legacy_marker"]).exists()


def test_public_bundle_has_no_dataset_or_model_payload() -> None:
    prohibited_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".pt",
        ".pth",
        ".onnx",
        ".parquet",
        ".npz",
        ".npy",
    }
    prohibited_tokens = (
        "/data/new_scenes_v1/",
        "new_scenes_annotations.csv",
        "hard_negative_audit.csv",
        "annotation_review_log.csv",
        "test_opened.json",
    )
    for bundle in (OUTPUT / "bundles").glob("*.zip"):
        with zipfile.ZipFile(bundle) as archive:
            names = [name.lower() for name in archive.namelist()]
        assert not any(Path(name).suffix in prohibited_suffixes for name in names)
        assert not any(
            token in name for name in names for token in prohibited_tokens
        )
