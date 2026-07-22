from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hash(payload: dict[str, Any], expected: str, source: Path) -> None:
    observed = payload.get("checkpoint_sha256")
    if observed != expected:
        raise RuntimeError(
            f"Checkpoint provenance mismatch in {source}: {observed} != {expected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify one frozen checkpoint across calibration, clean test and attacks"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--clean-test-summary", type=Path)
    parser.add_argument("--matrix-config", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = sha256(args.checkpoint)
    thresholds = load(args.thresholds)
    verify_hash(thresholds, expected, args.thresholds)
    if thresholds.get("selection_split") != "val" or thresholds.get("test_used_for_selection"):
        raise RuntimeError("Threshold selection was not frozen exclusively on validation")

    checked: list[dict[str, Any]] = [{
        "role": "canonical_v2_threshold_calibration",
        "path": str(args.thresholds.resolve()),
        "checkpoint_sha256": expected,
    }]
    if args.clean_test_summary is not None:
        summary = load(args.clean_test_summary)
        verify_hash(summary, expected, args.clean_test_summary)
        if summary.get("evaluated_splits") != ["test"]:
            raise RuntimeError("Canonical clean-test evaluation contains a non-test split")
        if summary.get("threshold_mode") != "frozen_validation_thresholds":
            raise RuntimeError("Canonical clean test did not use frozen validation thresholds")
        checked.append({
            "role": "canonical_v2_clean_test",
            "path": str(args.clean_test_summary.resolve()),
            "checkpoint_sha256": expected,
        })
    for config_path in args.matrix_config:
        config = load(config_path)
        verify_hash(config, expected, config_path)
        checked.append({
            "role": f"canonical_v2_{config.get('split', 'unknown')}_attacks",
            "path": str(config_path.resolve()),
            "checkpoint_sha256": expected,
            "confidence": config.get("confidence"),
        })

    confidence = float(thresholds["safety"]["confidence"])
    for config_path in args.matrix_config:
        config = load(config_path)
        if abs(float(config.get("confidence")) - confidence) > 1e-12:
            raise RuntimeError(f"Attack confidence differs from frozen safety threshold: {config_path}")

    payload = {
        "status": "PASS",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": expected,
        "frozen_safety_confidence": confidence,
        "checked_artifacts": checked,
        "test_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
