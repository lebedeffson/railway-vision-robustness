from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from rescue_common import PROJECT_DIR, OUTPUT_ROOT, atomic_json, now, run_text, sha256


def main() -> None:
    destination = OUTPUT_ROOT / "bundles/TNormFilter_rescue_v1_pre_gpu_audit.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    roots = [
        PROJECT_DIR / "configs/rescue",
        OUTPUT_ROOT / "protocol",
        OUTPUT_ROOT / "audit",
        OUTPUT_ROOT / "evaluator",
        OUTPUT_ROOT / "micro_overfit",
        OUTPUT_ROOT / "pipeline_status.json",
        PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv",
        PROJECT_DIR / "data/yolo_osdar23_rescue_v1/data.yaml",
        PROJECT_DIR / "systemd/tnorm-rescue-v1.service",
    ]
    allowed = {".json", ".csv", ".yaml", ".yml", ".txt", ".md", ".png", ".jpg", ".service"}
    files = []
    for root in roots:
        candidates = [root] if root.is_file() else root.rglob("*")
        files.extend(
            path for path in candidates
            if path.is_file() and path.suffix.lower() in allowed
            and OUTPUT_ROOT / "micro_overfit/dataset" not in path.parents
        )
    with tempfile.TemporaryDirectory() as directory:
        staging = Path(directory)
        checksum_rows = []
        for source in sorted(set(files)):
            relative = source.relative_to(PROJECT_DIR)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            checksum_rows.append(f"{sha256(target)}  {relative.as_posix()}")
        (staging / "checksums.sha256").write_text(
            "\n".join(checksum_rows) + "\n", encoding="utf-8"
        )
        atomic_json(staging / "manifest.json", {
            "artifact": "pre_gpu_audit_not_final_scientific_result",
            "created_at": now(),
            "git_commit": run_text(["git", "rev-parse", "HEAD"]),
            "git_branch": run_text(["git", "branch", "--show-current"]),
            "pipeline_status": "blocked_infrastructure",
            "test_opened": False,
            "weights_included": False,
            "files": len(checksum_rows),
        })
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging))
    checksum = sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{checksum}  {destination.name}\n", encoding="utf-8"
    )
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Pre-GPU rescue bundle CRC verification failed")
    print(json.dumps({
        "bundle": str(destination.resolve()), "sha256": checksum,
        "status": "PASS_PRE_GPU_AUDIT_ONLY",
    }, indent=2))


if __name__ == "__main__":
    main()
