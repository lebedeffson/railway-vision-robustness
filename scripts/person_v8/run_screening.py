from __future__ import annotations

import argparse
import json
from pathlib import Path

from person_v8.common import OUTPUT_ROOT, ROOT, assert_test_sealed, load_config, sha256_file


def check_authorization() -> dict[str, object]:
    config = load_config()
    assert_test_sealed(config)
    execution_lock = OUTPUT_ROOT / "protocol/V8_EXECUTION_LOCK.json"
    if not execution_lock.exists():
        raise RuntimeError(
            "TRAINING_BLOCKED: V8_EXECUTION_LOCK.json does not exist; "
            "new independent scenes have not passed the CPU gate"
        )
    lock = json.loads(execution_lock.read_text(encoding="utf-8"))
    if not lock.get("training_authorized"):
        raise RuntimeError("TRAINING_BLOCKED: execution lock denies training")
    if lock.get("test_status") != "SEALED" or lock.get("test_access_count") != 0:
        raise RuntimeError("TRAINING_BLOCKED: test seal is not intact")
    for raw_path, expected in lock["files"].items():
        path = ROOT / raw_path if not raw_path.startswith("/") else Path(raw_path)
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"TRAINING_BLOCKED: execution input hash mismatch: {path}")
    return {
        "protocol_id": config["protocol_id"],
        "status": "TRAINING_AUTHORIZED",
        "test_status": "SEALED",
        "attack_status": "BLOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", default=False)
    args = parser.parse_args()
    result = check_authorization()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.check_only:
        raise RuntimeError(
            "The v8 GPU implementation is intentionally unavailable until "
            "new data are acquired and the execution lock is frozen"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
