from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
DESTINATION = (
    PROJECT
    / "outputs/bundles/railway_vision_robustness_v7_tracklet_verifier_FAIL.zip"
)

FILES = (
    "configs/canonical_v7_person_tracklet_verifier.yaml",
    "configs/canonical_v7_crop_verifier_amendment.yaml",
    "protocol/v7/V7_PROTOCOL.md",
    "protocol/v7/V7_PROTOCOL_LOCK.json",
    "protocol/v7/V7_CROP_AMENDMENT.md",
    "protocol/v7/V7_CROP_PROTOCOL_LOCK.json",
    "reports/v7/V7_V0_FINAL_REPORT.md",
    "reports/v7/V7_CROP_FINAL_REPORT.md",
    "outputs/person_v7_tracklet_verifier/fold_0/V0_GATE.json",
    "outputs/person_v7_tracklet_verifier/fold_0/V0_COMPLETE.json",
    "outputs/person_v7_tracklet_verifier/fold_0/PRE_HELDOUT_FREEZE.json",
    "outputs/person_v7_tracklet_verifier/fold_0/TRAIN_OOF_MODEL_COMPARISON.csv",
    "outputs/person_v7_tracklet_verifier/fold_0/HELDOUT_MODEL_COMPARISON.csv",
    "outputs/person_v7_tracklet_verifier/fold_0/PER_SCENE_RESULTS.csv",
    "outputs/person_v7_tracklet_verifier/fold_0/EVALUATOR_PARITY.json",
    "outputs/person_v7_tracklet_verifier/decision_trace.json",
    "outputs/person_v7_crop_verifier/fold_0/V1_V2_GATE.json",
    "outputs/person_v7_crop_verifier/fold_0/V1_V2_COMPLETE.json",
    "outputs/person_v7_crop_verifier/fold_0/PRE_HELDOUT_FREEZE.json",
    "outputs/person_v7_crop_verifier/fold_0/TRAIN_OOF_MODEL_COMPARISON.csv",
    "outputs/person_v7_crop_verifier/fold_0/HELDOUT_MODEL_COMPARISON.csv",
    "outputs/person_v7_crop_verifier/fold_0/PER_SCENE_RESULTS.csv",
    "outputs/person_v7_crop_verifier/decision_trace.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    missing = [relative for relative in FILES if not (PROJECT / relative).is_file()]
    if missing:
        raise RuntimeError(f"Missing canonical v7 bundle inputs: {missing}")
    with tempfile.TemporaryDirectory(prefix="canonical-v7-bundle-") as temporary:
        root = Path(temporary) / "railway_vision_robustness_v7_FAIL"
        checksums = []
        for relative in FILES:
            source = PROJECT / relative
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            checksums.append(f"{sha256(target)}  {relative}")
        (root / "SHA256SUMS").write_text(
            "\n".join(checksums) + "\n", encoding="utf-8"
        )
        (root / "BUNDLE_STATUS.json").write_text(
            json.dumps(
                {
                    "protocols": [
                        "canonical-v7-person-tracklet-verifier-v1",
                        "canonical-v7-person-tracklet-crop-verifier-v1",
                    ],
                    "scientific_status": "FAIL",
                    "fold_1_read": False,
                    "test_status": "SEALED",
                    "attacks_status": "BLOCKED",
                    "contains_images": False,
                    "contains_crops": False,
                    "contains_checkpoints": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        temporary_zip = DESTINATION.with_suffix(".zip.tmp")
        with zipfile.ZipFile(
            temporary_zip, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root.parent))
        temporary_zip.replace(DESTINATION)
    with zipfile.ZipFile(DESTINATION) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt v7 bundle member: {bad}")
    print(
        json.dumps(
            {
                "bundle": str(DESTINATION),
                "sha256": sha256(DESTINATION),
                "members": len(zipfile.ZipFile(DESTINATION).namelist()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

