from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/operator_assistant_b7"


def test_b7_lock_is_final_sealed_and_gt_evaluation_only() -> None:
    lock = json.loads((OUTPUT / "B7_PROTOCOL_LOCK.json").read_text())
    tracked = json.loads(
        (ROOT / "protocol/operator_assistant_b7/B7_PROTOCOL_LOCK.json").read_text()
    )
    assert lock == tracked
    assert lock["final_experiment"] is True
    assert lock["test_status"] == "SEALED"
    assert lock["test_access_count"] == 0
    assert lock["gt_person_usage"] == "EVALUATION_ONLY"
    assert lock["immutable"] is True


def test_b6_audit_restores_pair_counts_and_observability() -> None:
    micro = json.loads(
        (OUTPUT / "b6_audit/B6_ASSOCIATION_MICRO.json").read_text()
    )
    assert micro["tp_links"] == 44
    assert micro["fp_links"] == 16
    assert micro["fn_links"] == 750
    assert micro["predicted_positive_pairs"] == 60
    assert micro["true_positive_pairs"] == 794
    assert micro["association_f1"] > 0
    audit = json.loads((OUTPUT / "b6_audit/B6_AUDIT.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["gt_used_for_model_training"] is False
    assert audit["coefficients_exported"] == 95
    assert audit["normalization_rows"] == 95
    assert audit["pseudopair_rows"] > 0


def test_all_four_methods_and_five_scenes_present() -> None:
    frame = pd.read_csv(OUTPUT / "B7_RESULTS_PER_SCENE.csv")
    assert frame.scene_id.nunique() == 5
    assert set(frame.method) == {
        "B6-FULL",
        "B7-FLOW",
        "B7-HARD",
        "B7-CONSENSUS",
    }
    assert len(frame) == 20


def test_hard_negative_mix_matches_lock() -> None:
    frame = pd.read_csv(OUTPUT / "B7_HARD_PSEUDO_PAIR_INDEX.csv")
    expected = {
        "hard_within_scene": 0.50,
        "simultaneous_distinct_track": 0.25,
        "temporally_adjacent_incompatible": 0.15,
        "easy_different_scene": 0.10,
    }
    for _, group in frame.groupby("held_scene"):
        negative = group[group.label == 0]
        for category, fraction in expected.items():
            actual = (negative.category == category).mean()
            assert abs(actual - fraction) <= 0.001


def test_flow_enforces_one_predecessor_and_one_successor() -> None:
    edges = pd.read_csv(OUTPUT / "B7_FLOW_EDGES.csv")
    selected = edges[edges.selected]
    for _, group in selected.groupby(["scene_id", "method"]):
        assert group.left_fragment_id.value_counts().max() <= 1
        assert group.right_fragment_id.value_counts().max() <= 1
    consensus = selected[selected.method == "B7-CONSENSUS"]
    assert (consensus.consensus_frequency >= 0.80).all()


def test_scene_normalization_is_robust_and_gt_free() -> None:
    lock = json.loads((OUTPUT / "B7_PROTOCOL_LOCK.json").read_text())
    assert lock["scene_normalization"]["method"] == "median_mad"
    assert lock["scene_normalization"]["gt_used"] is False
    frame = pd.read_csv(OUTPUT / "B7_SCENE_NORMALIZATION.csv")
    assert frame.scene_id.nunique() == 5
    assert len(frame) == 5 * 9
    assert (frame.mad_scale > 0).all()


def test_consensus_uses_32_runs_and_stability_is_reported() -> None:
    frame = pd.read_csv(OUTPUT / "B7_STABILITY_PER_SCENE.csv")
    consensus = frame[frame.method == "B7-CONSENSUS"]
    assert len(consensus) == 5
    assert (consensus.perturbed_runs == 32).all()
    assert consensus.median_perturbation_ari.notna().all()
    assert consensus.p10_perturbation_ari.notna().all()


def test_replay_conservation_and_final_stop() -> None:
    replay = json.loads((OUTPUT / "B7_REPLAY_AUDIT.json").read_text())
    cross_process = json.loads(
        (OUTPUT / "B7_CROSS_PROCESS_DETERMINISM.json").read_text()
    )
    assert replay["exact_match"] is True
    assert replay["all_fragments_preserved"] is True
    assert cross_process["independent_replay_runs"] == 2
    assert cross_process["exact_match"] is True
    decision = json.loads((OUTPUT / "B7_DECISION.json").read_text())
    tracked_decision = json.loads(
        (ROOT / "protocol/operator_assistant_b7/B7_FINAL_DECISION.json").read_text()
    )
    assert decision == tracked_decision
    assert decision["compute_status"] == "FROZEN_AFTER_B7"
    assert decision["b8_allowed"] is False
    assert decision["further_threshold_tuning_allowed"] is False
    freeze = json.loads((OUTPUT / "COMPUTE_FREEZE.json").read_text())
    assert freeze["further_model_experiments_allowed"] is False


def test_scientific_and_operational_gates_are_separate() -> None:
    decision = json.loads((OUTPUT / "B7_DECISION.json").read_text())
    assert decision["scientific_decision"] in {"PASS", "FAIL"}
    assert decision["operational_decision"] in {"PASS", "FAIL"}
    assert "scientific_conditions" in decision
    assert "operational_conditions" in decision


def test_public_bundle_is_safe_and_manifest_valid() -> None:
    archive_path = OUTPUT / "operator_assistant_b7_public.zip"
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert "B7_HIERARCHY.jsonl" in names
        manifest = archive.read("MANIFEST.sha256").decode().splitlines()
        for line in manifest:
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        for name in names:
            assert Path(name).suffix.lower() not in {
                ".pt",
                ".pth",
                ".mp4",
                ".jpg",
                ".jpeg",
                ".parquet",
            }
            payload = archive.read(name)
            assert b"/home/" not in payload
            assert b"/mnt/" not in payload
            assert b"railway_test" not in payload.lower()
