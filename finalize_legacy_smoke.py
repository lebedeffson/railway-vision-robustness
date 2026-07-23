from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
PILOT = PROJECT_DIR / "outputs/final_practice/deadline/pilot/pilot_gate.json"
RECOVERY = PROJECT_DIR / "outputs/final_practice/audit/legacy_recovery_recalculation.json"
OUTPUT = PROJECT_DIR / "outputs/final_practice/deadline/pilot/legacy_smoke_disposition.json"
ALLOWED_LEGACY_FAILURES = {"membership_saturation_below_20_percent"}
CRITICAL_CHECKS = {
    "no_sequence_leakage", "normalization_fit_clean_validation_only",
    "separate_P3_P4_P5", "required_metrics_present", "no_massive_nan",
    "similarities_in_physical_range", "at_least_one_tnorm_nonconstant",
    "actual_linf_respects_epsilon", "attack_losses_are_finite",
    "adaptive_gradient_is_nonzero", "g_raw_and_g_clipped_are_separate",
    "D0_D3_and_R0_R3_train", "bootstrap_group_is_sequence_id",
}


def main() -> None:
    if not PILOT.is_file() or not RECOVERY.is_file():
        subprocess.run(
            [str(PROJECT_DIR / ".venv/bin/python"), "prepare_deadline_validation.py"],
            cwd=PROJECT_DIR, check=False,
        )
    if not PILOT.is_file() or not RECOVERY.is_file():
        raise RuntimeError("Legacy smoke artifacts are incomplete")
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    checks = pilot.get("checks", {})
    failed = {name for name, passed in checks.items() if not passed}
    critical_failed = {name for name in CRITICAL_CHECKS if not checks.get(name, False)}
    if critical_failed or not failed <= ALLOWED_LEGACY_FAILURES:
        raise RuntimeError(
            f"Legacy smoke has critical failures: critical={sorted(critical_failed)}, "
            f"all={sorted(failed)}"
        )
    payload = {
        "status": "PASS" if not failed else "ACCEPTED_AS_LEGACY_LIMITATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "legacy_pilot_status_preserved": pilot.get("status"),
        "failed_checks": sorted(failed),
        "critical_checks_passed": True,
        "canonical_pilot_gate_affected": False,
        "rationale": (
            "Legacy compatibility smoke evidence is archived but cannot gate the "
            "separately fitted canonical v2 model and five-scene validation pilot."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
