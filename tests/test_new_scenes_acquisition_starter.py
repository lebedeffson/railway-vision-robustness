from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "acquisition/new_scenes_v1"
STARTER = ROOT / "starter"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_supplied_starter_internal_manifest_is_intact() -> None:
    manifest = STARTER / "MANIFEST.sha256"
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = STARTER / relative.removeprefix("./")
        assert path.is_file()
        assert sha256(path) == expected
        entries.append(relative)
    assert len(entries) == 18


def test_protocol_references_match_frozen_project_configs() -> None:
    for reference, project in (
        ("new_scenes_v1.yaml", "configs/new_scenes_v1.yaml"),
        ("independent_data_v1.yaml", "configs/independent_data_v1.yaml"),
    ):
        supplied = yaml.safe_load(
            (STARTER / "protocol_reference" / reference).read_text(
                encoding="utf-8"
            )
        )
        frozen = yaml.safe_load((PROJECT / project).read_text(encoding="utf-8"))
        assert supplied == frozen


def test_scene_slot_plan_meets_prospective_counts_only() -> None:
    slots = rows(STARTER / "ACQUISITION_SCENE_SLOTS.csv")
    assert len(slots) == 10
    assert Counter(row["source_id"] for row in slots) == {
        "railgoerl24": 4,
        "raileye3d": 6,
    }
    assert Counter(row["split_role"] for row in slots) == {
        "train_support": 6,
        "heldout": 2,
        "confirmation": 2,
    }
    assert all(row["status"].startswith("PENDING_") for row in slots)
    assert all(int(row["target_frames"]) == 200 for row in slots)
    assert all(int(row["continuous_fragment_min"]) == 20 for row in slots)


def test_static_railbench_candidates_exclude_test_and_osdar23() -> None:
    candidates = rows(
        STARTER / "metadata/railbench_static_candidates_train_val.csv"
    )
    assert len(candidates) == 240
    assert {row["split"] for row in candidates} <= {"train", "val"}
    assert all(
        row["counts_toward_temporal_scene_gate"].lower() == "false"
        for row in candidates
    )
    assert all(
        "osdar23" not in row["original_dataset"].lower()
        for row in candidates
    )


def test_terms_gated_sources_are_not_downloaded_by_project_wrapper() -> None:
    wrapper = (ROOT / "download_verified_open_sources.sh").read_text(
        encoding="utf-8"
    )
    assert "Annotated_RGB_data.7z" in wrapper
    assert "raileye3d_dataset.git" in wrapper
    assert "railbench_data" not in wrapper
    assert "zenodo.org" not in wrapper
    assert "RAWPED" not in wrapper


def test_ingestion_does_not_authorize_training_or_test() -> None:
    audit = json.loads((ROOT / "INGESTION_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["starter_package"]["accepted_scenes_contributed"] == 0
    assert audit["project_state"] == {
        "status": "WAITING_FOR_NEW_SCENES",
        "training_authorized": False,
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    assert audit["source_verification"]["railbench_object"]["status"] == (
        "BLOCKED_PENDING_EXPLICIT_TERMS_ACCEPTANCE"
    )
