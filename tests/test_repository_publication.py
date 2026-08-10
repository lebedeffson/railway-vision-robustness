from __future__ import annotations

from pathlib import Path

from scripts.operator_assistant_b7.verify_public_evidence import verify


ROOT = Path(__file__).resolve().parents[1]


def test_public_repository_metadata_is_canonical() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "SCIENTIFIC_FAIL" in readme
    assert "OPERATIONAL_FAIL" in readme
    assert "FROZEN_AFTER_B7" in readme
    assert "Test access count: `0`" in readme
    assert "124112200072-2" in readme
    assert "f25a751c52104b0353995d2206afbe92a4182ff2" in readme
    assert "License decision required from repository owner" in readme


def test_tracked_public_evidence_verifies() -> None:
    result = verify(ROOT / "artifacts/operator_assistant_b7_public.zip")
    assert result["manifest"] == "PASS"
    assert result["test_status"] == "SEALED"
    assert result["test_access_count"] == 0
