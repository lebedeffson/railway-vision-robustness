from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "configs/canonical_v5_person_data_first.yaml"
LOCK_ROOT = PROJECT / "protocols/canonical_v5_person_data_first_v1"
LOCK = LOCK_ROOT / "protocol_lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=PROJECT, text=True
    ).strip()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate(protocol: dict[str, Any]) -> dict[str, str]:
    marker = PROJECT / protocol["claim_boundary"]["test_marker"]
    if marker.exists():
        raise RuntimeError(f"Sealed test marker already exists: {marker}")
    if not protocol["claim_boundary"]["test_sealed"]:
        raise RuntimeError("Canonical v5 must freeze with test sealed")
    if protocol["matrix"]["execution_order"] != ["D1_fold0", "D1_fold1"]:
        raise RuntimeError("Unexpected v5 candidate execution order")
    if protocol["matrix"]["D2"]["role"] != "deferred_not_part_of_v1":
        raise RuntimeError("RT-DETR must remain outside v5-v1")
    forbidden = set(protocol["external_pretraining"]["forbidden"])
    if forbidden != {"NWD", "QFL", "GroupDRO", "MixStyle", "SWAD"}:
        raise RuntimeError("Data-first loss exclusions changed")
    hashes: dict[str, str] = {}
    for key in (
        "development_manifest",
        "folds",
        "railway_tile_manifest",
        "coco_initialization",
    ):
        path = PROJECT / protocol["frozen_inputs"][key]
        expected = protocol["frozen_inputs"][f"{key}_sha256"]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen input changed: {key}: {actual} != {expected}"
            )
        hashes[key] = actual
    return hashes


def lock() -> dict[str, Any]:
    if git("status", "--short"):
        raise RuntimeError("Commit v5 protocol and code before locking")
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ancestor_check = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            protocol["parent_commit"],
            "HEAD",
        ],
        cwd=PROJECT,
        check=False,
    )
    if ancestor_check.returncode != 0:
        raise RuntimeError(
            "The pre-v5 proxy closure commit is not an ancestor of HEAD"
        )
    hashes = validate(protocol)
    payload = {
        "status": "LOCKED",
        "protocol_id": protocol["protocol_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit_at_lock": git("rev-parse", "HEAD"),
        "protocol_path": str(CONFIG.relative_to(PROJECT)),
        "protocol_sha256": sha256(CONFIG),
        "frozen_input_hashes": hashes,
        "test_sealed": True,
        "test_marker_absent": True,
        "crowdhuman_terms_acceptance_recorded": False,
        "data_acquisition_allowed": False,
        "training_allowed": False,
    }
    atomic_json(LOCK, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.check:
        print(json.dumps(validate(protocol), indent=2))
        return
    print(json.dumps(lock(), indent=2))


if __name__ == "__main__":
    main()
