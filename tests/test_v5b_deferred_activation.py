from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_current_v5_matrix_remains_unchanged_and_star_free() -> None:
    execution = yaml.safe_load(
        (ROOT / "configs/person_v5/execution_v5.yaml").read_text(encoding="utf-8")
    )
    primary = execution["matrix"]["primary"]
    assert list(primary) == ["V5-A", "V5-B", "V5-C", "V5-D"]
    assert primary["V5-A"]["runtime_name"] == "V5-A"
    assert primary["V5-B"]["runtime_name"] == "V5-B"
    assert primary["V5-C"]["runtime_name"] == "V5-C-fraction25"
    assert primary["V5-D"]["runtime_name"] == "V5-D-fraction25"
    assert "star" not in str(primary).lower()


def test_current_service_cannot_start_v5b() -> None:
    service = (
        ROOT / "systemd/tnorm-person-v5-candidates.service"
    ).read_text(encoding="utf-8").lower()
    assert "starcoord" not in service
    assert "v5b" not in service
    assert "run_execution_pipeline" in service


def test_v5b_is_documented_as_deferred_not_active() -> None:
    document = (
        ROOT / "protocol/v5b/V5B_DEFERRED_ACTIVATION.md"
    ).read_text(encoding="utf-8")
    assert "Status: `INACTIVE`" in document
    assert "TWO_FOLD_GATE.json" in document
    assert "feat/person-canonical-v5b-starcoord" in document
    assert "does not contain P2" in document
    assert not (ROOT / "protocol/v5b/V5B_PROTOCOL_LOCK.json").exists()
