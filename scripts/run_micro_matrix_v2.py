from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from rescue_common import atomic_json
from rescue_v2_common import PROJECT_DIR, assert_role_allowed, load_protocol


ROOT = PROJECT_DIR / "outputs/rescue_v2"
MICRO = ROOT / "micro"
PYTHON = PROJECT_DIR / ".venv/bin/python"


def load_result(candidate: str) -> dict[str, Any] | None:
    path = MICRO / candidate / "result.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_process_running(candidate: str) -> bool:
    expected = f"--candidate\x00{candidate}".encode()
    own_pid = os.getpid()
    for command_line in Path("/proc").glob("[0-9]*/cmdline"):
        if int(command_line.parent.name) == own_pid:
            continue
        try:
            value = command_line.read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b"run_micro_candidate_v2.py" in value and expected in value:
            return True
    return False


def wait_for_candidate(candidate: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = load_result(candidate)
        if result is not None:
            return result
        if not candidate_process_running(candidate):
            return run_candidate(candidate)
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for {candidate}")


def run_candidate(candidate: str) -> dict[str, Any]:
    existing = load_result(candidate)
    if existing is not None:
        return existing
    script = (
        "scripts/run_micro_candidate_v2.py"
        if candidate in {"M1", "M2"}
        else "scripts/run_micro_view_candidate_v2.py"
    )
    try:
        subprocess.run(
            [str(PYTHON), "-u", script, "--candidate", candidate],
            cwd=PROJECT_DIR,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        failure = {
            "candidate": candidate,
            "status": "EXECUTION_FAILED",
            "micro_gate_passed": False,
            "returncode": error.returncode,
            "error": f"{type(error).__name__}: {error}",
            "test_evaluated": False,
        }
        destination = MICRO / candidate
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / "result.json", failure)
        return failure
    result = load_result(candidate)
    if result is None:
        raise RuntimeError(f"{candidate} completed without result.json")
    return result


def summarize(results: dict[str, dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    order = protocol["micro_selection"]["order"][:-1]
    passing = [
        candidate for candidate in order
        if bool(results[candidate]["micro_gate_passed"])
    ]
    eligible = [
        candidate for candidate in passing
        if bool(results[candidate].get(
            "eligible_for_full_training_without_deployable_roi", True
        ))
    ]
    selected = eligible[: int(protocol["micro_selection"]["maximum_winners"])]
    m5_required = not passing
    status = "PASS" if selected else (
        "M5_REQUIRED" if m5_required else "PROTOCOL_AMENDMENT_REQUIRED"
    )
    return {
        "status": status,
        "protocol_id": protocol["protocol_id"],
        "candidate_order": order,
        "results": {
            candidate: {
                key: results[candidate].get(key)
                for key in (
                    "status", "micro_gate_passed", "mAP50", "recall",
                    "small_recall", "medium_recall", "large_recall",
                    "checkpoint_sha256",
                    "eligible_for_full_training_without_deployable_roi",
                    "error",
                )
            }
            for candidate in order
        },
        "passing_candidates": passing,
        "full_training_eligible_candidates": eligible,
        "selected_candidates": selected,
        "m5_required": m5_required,
        "test_opened": False,
        "attacks_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-m1", action="store_true")
    parser.add_argument("--wait-timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    assert_role_allowed("micro")
    protocol = load_protocol()
    ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    results["M1"] = (
        wait_for_candidate("M1", args.wait_timeout_seconds)
        if args.wait_for_m1 else run_candidate("M1")
    )
    atomic_json(ROOT / "micro_matrix_status.json", {
        "status": "running", "completed": ["M1"], "test_opened": False
    })
    for candidate in ("M2", "M3", "M4"):
        results[candidate] = run_candidate(candidate)
        atomic_json(ROOT / "micro_matrix_status.json", {
            "status": "running",
            "completed": list(results),
            "test_opened": False,
        })
    summary = summarize(results, protocol)
    atomic_json(ROOT / "micro_selection.json", summary)
    atomic_json(ROOT / "micro_matrix_status.json", {
        "status": summary["status"],
        "completed": list(results),
        "test_opened": False,
    })
    print(json.dumps(summary, indent=2))
    if summary["m5_required"]:
        raise SystemExit(20)
    if not summary["selected_candidates"]:
        raise SystemExit(21)


if __name__ == "__main__":
    main()
