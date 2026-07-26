from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from scripts.new_scenes.audit_new_scenes import audit
from scripts.new_scenes.common import (
    LOCK,
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_json,
    config,
    sha256,
)


PUBLIC_FILES = (
    "configs/new_scenes_v1.yaml",
    "configs/independent_data_v1.yaml",
    "protocol/new_scenes_v1/ACQUISITION_LOCK.json",
    "protocol/new_scenes_v1/ACQUISITION_LOCK.sha256",
    "protocol/new_scenes_v1/PARENT_STATE_FREEZE.json",
    "protocol/new_scenes_v1/PROTOCOL.md",
    "protocol/new_scenes_v1/schemas/NEW_SCENES_MANIFEST.schema.json",
    "protocol/new_scenes_v1/schemas/NEW_SCENES_ANNOTATIONS.schema.json",
    "protocol/new_scenes_v1/schemas/HARD_NEGATIVE_AUDIT.schema.json",
    "protocol/new_scenes_v1/schemas/ANNOTATION_REVIEW_LOG.schema.json",
    "protocol/new_scenes_v1/templates/NEW_SCENES_MANIFEST.csv",
    "protocol/new_scenes_v1/templates/NEW_SCENES_ANNOTATIONS.csv",
    "protocol/new_scenes_v1/templates/HARD_NEGATIVE_AUDIT.csv",
    "protocol/new_scenes_v1/templates/ANNOTATION_REVIEW_LOG.csv",
    "protocol/independent_data_v1/PROTOCOL_DRAFT.md",
    "scripts/new_scenes/__init__.py",
    "scripts/new_scenes/common.py",
    "scripts/new_scenes/audit_new_scenes.py",
    "scripts/new_scenes/lock_protocol.py",
    "scripts/new_scenes/authorize_independent_data.py",
    "scripts/new_scenes/finalize.py",
    "tests/test_new_scenes_v1.py",
)


def build_bundle(destination: Path) -> dict[str, object]:
    files = [(PROJECT / relative, relative) for relative in PUBLIC_FILES]
    for relative in (
        "outputs/new_scenes_v1/NEW_SCENES_AUDIT.json",
        "outputs/new_scenes_v1/HARD_NEGATIVE_COUNTS.csv",
        "outputs/new_scenes_v1/DATASET_CARD.md",
        "outputs/new_scenes_v1/STATUS.json",
    ):
        path = PROJECT / relative
        if path.is_file():
            files.append((path, relative))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="new-scenes-public-") as temporary:
        staging = Path(temporary) / "railway-person-new-scenes-v1"
        for source, relative in files:
            if not source.is_file():
                raise RuntimeError(f"Missing public artifact: {relative}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifest = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            manifest.append(
                f"{sha256(path)}  {path.relative_to(staging).as_posix()}"
            )
        (staging / "MANIFEST.sha256").write_text(
            "\n".join(manifest) + "\n", encoding="utf-8"
        )
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(item for item in staging.rglob("*") if item.is_file()):
                archive.write(
                    path,
                    Path("railway-person-new-scenes-v1")
                    / path.relative_to(staging),
                )
    return {
        "path": str(destination.relative_to(PROJECT)),
        "sha256": sha256(destination),
        "files": len(files) + 1,
    }


def main() -> None:
    lock = assert_locked()
    status = audit()
    summary = {
        "protocol_id": config()["protocol_id"],
        "status": status["status"],
        "training_authorized": False,
        "independent_data_protocol": status["independent_data_protocol"],
        "implementation_commit": lock["implementation_commit"],
        "acquisition_lock_sha256": sha256(LOCK),
        "parent_release": config()["frozen_parent"]["release"],
        "parent_release_recalculated": False,
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    atomic_json(OUTPUT / "FINAL_SUMMARY.json", summary)
    name = "railway_person_new_scenes_waiting_for_data.zip"
    bundle = build_bundle(OUTPUT / "bundles" / name)
    summary["public_bundle"] = bundle
    atomic_json(OUTPUT / "FINAL_SUMMARY.json", summary)
    (OUTPUT / "bundles/MANIFEST.sha256").write_text(
        f"{bundle['sha256']}  {name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
