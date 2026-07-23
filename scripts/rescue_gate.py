from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from rescue_common import PROJECT_DIR, OUTPUT_ROOT, load_protocol, sha256


Role = Literal["test", "attack", "finalization"]
QUALITY_GATE = OUTPUT_ROOT / "final/quality_gate.json"


def load_gate(path: Path = QUALITY_GATE) -> dict:
    if not path.is_file():
        raise RuntimeError("Rescue quality gate is absent; test and downstream stages are sealed")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_gate_allows(role: Role, path: Path = QUALITY_GATE) -> dict:
    gate = load_gate(path)
    protocol = load_protocol()
    expected_split = protocol["split_manifest_sha256"]
    if gate.get("protocol_id") != protocol["protocol_id"]:
        raise RuntimeError(f"{role} blocked: rescue protocol mismatch")
    if gate.get("split_manifest_sha256") != expected_split:
        raise RuntimeError(f"{role} blocked: split manifest provenance mismatch")
    if gate.get("quality_gate_passed") is not True:
        raise RuntimeError(f"{role} blocked: rescue quality gate did not pass")
    checkpoint = Path(gate.get("checkpoint_path", ""))
    if not checkpoint.is_file():
        raise RuntimeError(f"{role} blocked: frozen checkpoint is missing")
    if sha256(checkpoint) != gate.get("checkpoint_sha256"):
        raise RuntimeError(f"{role} blocked: frozen checkpoint hash mismatch")
    if gate.get("test_opened") is not False:
        raise RuntimeError(f"{role} blocked: test-open state is not sealed")
    return gate


def open_test_once(path: Path = QUALITY_GATE) -> Path:
    gate = assert_gate_allows("test", path)
    protocol = load_protocol()
    marker = PROJECT_DIR / protocol["test_open_marker"]
    if marker.exists():
        raise RuntimeError("Canonical rescue test was already opened")
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "status": "OPENED_ONCE",
        "protocol_id": gate["protocol_id"],
        "checkpoint_sha256": gate["checkpoint_sha256"],
        "split_manifest_sha256": gate["split_manifest_sha256"],
    }, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker)
    return marker
