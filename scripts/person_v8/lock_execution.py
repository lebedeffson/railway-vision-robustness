from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from person_v8.common import OUTPUT_ROOT, ROOT, assert_test_sealed, atomic_json, load_config, sha256_file


EXECUTION_LOCK = OUTPUT_ROOT / "protocol/V8_EXECUTION_LOCK.json"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def create_execution_lock(
    manifest: Path,
    corrections: Path,
) -> dict[str, object]:
    if _git("status", "--porcelain"):
        raise RuntimeError("Execution lock requires a clean source worktree")
    config = load_config()
    assert_test_sealed(config)
    acquisition_lock = ROOT / "protocol/v8/V8_ACQUISITION_LOCK.json"
    required = [
        acquisition_lock,
        manifest,
        corrections,
        OUTPUT_ROOT / "audit/CPU_GATE.json",
        OUTPUT_ROOT / "audit/scene_statistics.csv",
        OUTPUT_ROOT / "audit/frame_statistics.csv",
        OUTPUT_ROOT / "audit/fold_support.csv",
        OUTPUT_ROOT / "protocol/screening_split.json",
        OUTPUT_ROOT / "protocol/folds.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Execution inputs are missing: {missing}")
    gate = json.loads((OUTPUT_ROOT / "audit/CPU_GATE.json").read_text(encoding="utf-8"))
    if gate["status"] != "PASS":
        raise RuntimeError(f"Cannot lock execution after CPU gate {gate['status']}")
    files = {str(path.resolve()): sha256_file(path) for path in required}
    b0 = ROOT / config["immutable_inputs"]["b0_initialization"]["path"]
    if not b0.exists() or sha256_file(b0) != config["immutable_inputs"]["b0_initialization"]["sha256"]:
        raise RuntimeError("Frozen B0 initialization is missing or has the wrong hash")
    files[str(b0.resolve())] = sha256_file(b0)
    payload: dict[str, object] = {
        "protocol_id": config["protocol_id"],
        "lock_kind": "EXECUTION",
        "source_commit": _git("rev-parse", "HEAD"),
        "training_authorized": True,
        "test_status": "SEALED",
        "test_access_count": 0,
        "attack_status": "BLOCKED",
        "files": files,
    }
    atomic_json(EXECUTION_LOCK, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/person_v8/acquisition_manifest.csv",
    )
    parser.add_argument(
        "--corrections",
        type=Path,
        default=ROOT / "data/person_v8/correction_log.csv",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            create_execution_lock(args.manifest, args.corrections),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

