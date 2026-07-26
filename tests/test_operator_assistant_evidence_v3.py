from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/operator_assistant_evidence_v3"


def test_v3_test_status_sealed_and_access_zero() -> None:
    lock = json.loads((OUTPUT / "protocol/EVENT_SEMANTICS_LOCK.json").read_text())
    assert lock["test_status"] == "SEALED"
    assert lock["test_access_count"] == 0
    assert lock["training"] == "FORBIDDEN"


def test_event_semantics_lock_defines_hierarchy() -> None:
    lock = json.loads((OUTPUT / "protocol/EVENT_SEMANTICS_LOCK.json").read_text())
    assert lock["event_unit"] == "hazard_situation"
    assert lock["child_unit"] == "person_episode"
    assert lock["immutable"] is True


def test_minimum_five_scene_selection_is_frozen() -> None:
    scenes = pd.read_csv(OUTPUT / "protocol/DEVELOPMENT_SCENE_SELECTION.csv")
    assert scenes.scene_id.nunique() == 5
    assert scenes.included.all()
    assert set(scenes.reason) == {"FROZEN_TECHNICAL_ELIGIBILITY"}


def test_every_v2_gt_person_has_explicit_stage_trace() -> None:
    trace = pd.read_csv(OUTPUT / "diagnostics/PERSON_EPISODE_STAGE_TRACE.csv")
    assert len(trace) == 15
    assert trace.gt_person_episode_id.nunique() == 15
    allowed = {
        "NO_DETECTOR_MATCH",
        "NO_TRACKER_MATCH",
        "VERIFIER_REJECTED_ALL",
        "NOT_ROUTED_TO_AGGREGATOR",
        "PERSON_EPISODE_NOT_CREATED",
        "NOT_ASSIGNED_TO_HAZARD_EVENT",
        "PARTIAL_COVERAGE",
        "COVERED",
    }
    assert set(trace.first_failure_stage) <= allowed
    assert (trace.first_failure_stage == "NO_DETECTOR_MATCH").sum() == 1
    assert (trace.first_failure_stage == "NO_TRACKER_MATCH").sum() == 1


def test_person_episode_children_are_conserved_in_every_scene() -> None:
    for audit in OUTPUT.glob("streams/*/EVENT_REPLAY_AUDIT.json"):
        payload = json.loads(audit.read_text())
        assert payload["exact_match"] is True
        assert payload["child_conservation"] is True


def test_hazard_gt_missing_blocks_semantic_evaluation() -> None:
    agreement = json.loads(
        (OUTPUT / "annotations/HAZARD_ANNOTATION_AGREEMENT.json").read_text()
    )
    final = json.loads((OUTPUT / "FINAL_STATUS.json").read_text())
    assert agreement["status"] == "BLOCKED_PENDING_TWO_AUTHOR_ANNOTATIONS"
    assert final["hazard_annotation_complete"] is False


def test_all_six_baselines_use_identical_scene_inputs() -> None:
    frame = pd.read_csv(OUTPUT / "baselines/BASELINE_RESULTS_PER_SCENE.csv")
    assert frame.method.nunique() == 6
    assert len(frame) == 30
    for _, group in frame.groupby("scene_id"):
        assert group.identical_input_sha256.nunique() == 1


def test_b4_b5_sensitivity_have_108_rows_and_no_selection() -> None:
    for name in ["EVENT_SENSITIVITY_B4.csv", "EVENT_SENSITIVITY_B5.csv"]:
        frame = pd.read_csv(OUTPUT / f"sensitivity/{name}")
        assert len(frame) == 108
        assert frame.baseline_configuration.sum() == 1
        assert frame.children_preserved.all()
    audit = json.loads(
        (OUTPUT / "sensitivity/EVENT_SENSITIVITY_AUDIT.json").read_text()
    )
    assert audit["baseline_selected_from_sweep"] is False
    assert audit["per_scene_rows"] == 1080


def test_all_v3_failsafe_checks_pass() -> None:
    checks = pd.read_csv(OUTPUT / "FAILSAFE_CHECKS.csv")
    assert not checks.empty
    assert set(checks.status) == {"PASS"}


def test_runtime_is_honest_about_unfinished_parity_modes() -> None:
    frame = pd.read_csv(OUTPUT / "runtime/RUNTIME_COMPARISON.csv")
    assert frame.loc[frame["mode"] == "FULL_SYNC", "measured_frames"].iloc[0] == 1000
    assert set(frame.loc[frame["mode"] != "FULL_SYNC", "status"]) == {
        "BLOCKED_NOT_IMPLEMENTED_WITH_PARITY_EVIDENCE"
    }


def test_public_bundle_has_no_media_or_absolute_paths() -> None:
    with zipfile.ZipFile(OUTPUT / "operator_assistant_evidence_v3_public.zip") as archive:
        for name in archive.namelist():
            assert Path(name).suffix.lower() not in {".mp4", ".jpg", ".jpeg", ".pt", ".pkl"}
            data = archive.read(name)
            assert b"/home/" not in data
            assert b"/mnt/" not in data
