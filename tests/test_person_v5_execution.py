from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.person_canonical_v5.execution_analysis import holm  # noqa: E402
from scripts.person_canonical_v5.finalize_execution import finalize  # noqa: E402


def test_execution_lock_freezes_primary_and_sensitivity_pasting() -> None:
    lock = json.loads(
        (ROOT / "protocol/v5/V5_EXECUTION_LOCK.json").read_text(encoding="utf-8")
    )
    assert lock["implementation_commit"] == "a50fc967f42d3607fe684f2ae545feb7ea433305"
    assert lock["primary_pasting_fraction"] == 0.25
    assert lock["sensitivity_pasting_fraction"] == 0.50
    assert lock["sensitivity_folds"] == [0]
    assert lock["test_status"] == "SEALED"
    assert lock["attacks_status"] == "BLOCKED"


def test_execution_config_uses_one_fixed_candidate_threshold() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/person_v5/execution_v5.yaml").read_text(encoding="utf-8")
    )
    assert config["evaluation"]["confidence_threshold"] == 0.07
    assert "threshold_rule" not in config["matrix"]
    assert config["selection"]["sensitivity_candidates_excluded"] == [
        "V5-C50", "V5-D50"
    ]


def test_primary_matrix_does_not_select_50_percent_pasting() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/person_v5/execution_v5.yaml").read_text(encoding="utf-8")
    )
    primary = config["matrix"]["primary"]
    assert primary["V5-C"]["runtime_name"] == "V5-C-fraction25"
    assert primary["V5-D"]["runtime_name"] == "V5-D-fraction25"
    assert all(row["folds"] == [0, 1] for row in primary.values())
    sensitivity = config["matrix"]["sensitivity"]
    assert all(row["folds"] == [0] for row in sensitivity.values())
    assert all(row["descriptive_only"] for row in sensitivity.values())


def test_holm_is_monotone_and_never_below_raw_p() -> None:
    frame = pd.DataFrame({
        "raw_p": [0.01, 0.04, 0.03, 0.20],
        "metric": ["a", "b", "c", "d"],
    })
    adjusted = holm(frame)
    ordered = adjusted.sort_values("raw_p")
    assert ordered["Holm_p"].is_monotonic_increasing
    assert (adjusted["Holm_p"] >= adjusted["raw_p"]).all()
    assert (adjusted["Holm_p"] <= 1.0).all()


def test_service_requires_both_execution_locks_and_sealed_test() -> None:
    service = (
        ROOT / "systemd/tnorm-person-v5-candidates.service"
    ).read_text(encoding="utf-8")
    assert "protocol/v5/V5_EXECUTION_LOCK.json" in service
    assert "protocol/v5/V5_RUNTIME_LOCK.json" in service
    assert "ConditionPathExists=!" in service
    assert "run_execution_pipeline" in service


def test_runner_never_opens_test_or_runs_attacks() -> None:
    source = (
        ROOT / "scripts/person_canonical_v5/run_execution_pipeline.py"
    ).read_text(encoding="utf-8")
    assert "TEST_OPENED.json" not in source
    assert "attack_runs=0" not in source
    assert "scripts.run_attack" not in source
    assert "TEST_NOT_OPENED" in source
    assert "ATTACKS_BLOCKED" in source


def test_terminal_finalizer_waits_without_gate() -> None:
    assert finalize()["status"] == "WAITING_FOR_TERMINAL_GATE"


def test_finalizer_timer_is_non_mutating_until_terminal_gate() -> None:
    timer = (
        ROOT / "systemd/tnorm-person-v5-finalize.timer"
    ).read_text(encoding="utf-8")
    service = (
        ROOT / "systemd/tnorm-person-v5-finalize.service"
    ).read_text(encoding="utf-8")
    assert "OnUnitActiveSec=2min" in timer
    assert "ConditionPathExists=!" in service
    assert "finalize_execution" in service
