from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/operator_assistant_b6"


def test_b6_lock_keeps_test_sealed_and_gt_evaluation_only() -> None:
    lock = json.loads((OUTPUT / "B6_PROTOCOL_LOCK.json").read_text())
    tracked_lock = json.loads(
        (ROOT / "protocol/operator_assistant_b6/B6_PROTOCOL_LOCK.json").read_text()
    )
    assert lock == tracked_lock
    assert lock["test_status"] == "SEALED"
    assert lock["test_access_count"] == 0
    assert lock["gt_person_usage"] == "EVALUATION_ONLY"
    assert lock["detector_tracker_verifier_training"] == "FORBIDDEN"
    assert lock["immutable"] is True


def test_b6_has_five_loso_scenes_and_all_methods() -> None:
    frame = pd.read_csv(OUTPUT / "B6_RESULTS_PER_SCENE.csv")
    assert frame.scene_id.nunique() == 5
    assert set(frame.method) == {"B1", "B4", "B5", "B6-G", "B6-A", "B6-FULL"}
    assert len(frame) == 30


def test_pseudo_pairs_never_use_gt() -> None:
    audit = pd.read_csv(OUTPUT / "PSEUDO_PAIR_AUDIT.csv")
    assert len(audit) == 15
    assert (audit.gt_fields_used == 0).all()
    assert (audit.positive_pairs_used > 0).all()
    assert (audit.negative_pairs_used > 0).all()


def test_fragment_conservation_and_hierarchy() -> None:
    fragments = pd.read_parquet(OUTPUT / "TRACK_FRAGMENTS.parquet")
    replay = json.loads((OUTPUT / "B6_REPLAY_AUDIT.json").read_text())
    assert fragments.fragment_id.is_unique
    assert replay["all_fragments_preserved"] is True
    bundles = [
        json.loads(line)
        for line in (OUTPUT / "B6_HIERARCHY.jsonl").read_text().splitlines()
    ]
    represented = {
        fragment_id
        for bundle in bundles
        for episode in bundle["person_episodes"]
        for fragment_id in episode["track_fragment_ids"]
    }
    assert represented == set(fragments.fragment_id)
    assert all(
        bundle["semantics"] == "SPATIOTEMPORAL_REVIEW_CARD_NOT_HAZARD_GROUND_TRUTH"
        for bundle in bundles
    )


def test_b6_metrics_include_bcubed_and_uncertainty() -> None:
    frame = pd.read_csv(OUTPUT / "B6_RESULTS_PER_SCENE.csv")
    for column in [
        "bcubed_precision",
        "bcubed_recall",
        "bcubed_f1",
        "abstention_rate",
        "association_f1",
        "cross_person_merge_rate",
        "same_person_split_recovery",
        "person_episode_coverage",
    ]:
        assert column in frame
        assert frame[column].notna().all()
    perturbations = pd.read_csv(OUTPUT / "PERTURBATION_STABILITY.csv")
    assert len(perturbations) == 5 * 3 * 120


def test_direct_and_replay_are_identical() -> None:
    replay = json.loads((OUTPUT / "B6_REPLAY_AUDIT.json").read_text())
    cross_process = json.loads(
        (OUTPUT / "B6_CROSS_PROCESS_DETERMINISM.json").read_text()
    )
    assert replay["exact_match"] is True
    assert all(record["exact_match"] for record in replay["records"])
    assert cross_process["independent_replay_runs"] == 2
    assert cross_process["exact_match"] is True


def test_compute_is_frozen_after_b6() -> None:
    freeze = json.loads((OUTPUT / "COMPUTE_FREEZE.json").read_text())
    decision = json.loads((OUTPUT / "B6_DECISION.json").read_text())
    tracked_decision = json.loads(
        (ROOT / "protocol/operator_assistant_b6/B6_FINAL_DECISION.json").read_text()
    )
    assert decision == tracked_decision
    assert freeze["status"] == "FROZEN_AFTER_B6"
    assert freeze["further_model_experiments_allowed"] is False
    assert freeze["further_threshold_tuning_allowed"] is False
    assert freeze["test_status"] == "SEALED"
    assert freeze["test_access_count"] == 0


def test_b6_public_bundle_is_safe_and_manifested() -> None:
    archive_path = OUTPUT / "operator_assistant_b6_public.zip"
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert "MANIFEST.sha256" in names
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
