from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from scripts.temporal_safety.common import CONFIG, OUTPUT, PROJECT, sha256


def main() -> None:
    gate = json.loads(
        (OUTPUT / "triage/TRIAGE_GATE.json").read_text(encoding="utf-8")
    )
    if gate["status"] != "DEVELOPMENT_FAIL":
        raise RuntimeError("Failure bundle is only valid after development FAIL")
    files = [
        CONFIG,
        PROJECT / "protocol/temporal_safety_v1/PROTOCOL.md",
        OUTPUT / "protocol/protocol_lock.json",
        OUTPUT / "data/SEQUENCE_INDEX_AUDIT.json",
        OUTPUT / "baseline/RAW_PREDICTIONS_AUDIT.json",
        OUTPUT / "selection/TRACKER_GRID_RESULTS.csv",
        OUTPUT / "selection/SELECTED_TRACKER_CONFIGS.json",
        OUTPUT / "triage/TRIAGE_GATE.json",
        OUTPUT / "triage/PER_SCENE_RESULTS.csv",
        OUTPUT / "decision_trace.json",
        OUTPUT / "development/DEVELOPMENT_GATE.json",
        OUTPUT / "reports/TEMPORAL_SAFETY_DEVELOPMENT_REPORT.md",
    ]
    destination = OUTPUT / "bundles/railway_person_temporal_safety_development_fail.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        manifest = []
        for path in files:
            if not path.is_file():
                raise RuntimeError(f"Required bundle file is missing: {path}")
            relative = path.relative_to(PROJECT)
            archive.write(path, relative.as_posix())
            manifest.append(f"{sha256(path)}  {relative.as_posix()}")
        archive.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    with zipfile.ZipFile(destination) as archive:
        prohibited = ("data/raw/", "images/", ".pt", "sealed_test_manifest.csv")
        for name in archive.namelist():
            if any(token in name for token in prohibited):
                raise RuntimeError(f"Public bundle contains prohibited path: {name}")
    (destination.parent / "MANIFEST.sha256").write_text(
        f"{sha256(destination)}  {destination.name}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

