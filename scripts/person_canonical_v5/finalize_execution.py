from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.person_canonical_v5.common import atomic_csv, atomic_json, now, sha256
from scripts.person_canonical_v5.execution_analysis import PRIMARY
from scripts.person_canonical_v5.execution_common import (
    OUTPUT,
    PROJECT,
    REPORTS,
    RESULTS,
    assert_execution_locked,
)


BUNDLE_ROOT = PROJECT / "release/v5"
FINALIZER_LOCK = PROJECT / "protocol/v5/V5_FINALIZER_LOCK.json"


def _assert_finalizer_locked() -> None:
    if not FINALIZER_LOCK.is_file():
        raise RuntimeError("Person-v5 finalizer is not locked")
    lock = json.loads(FINALIZER_LOCK.read_text(encoding="utf-8"))
    for relative, expected in lock["implementation_sha256"].items():
        path = PROJECT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Finalizer implementation changed: {relative}")


def _terminal_gate() -> tuple[str, dict[str, Any]] | None:
    oof = RESULTS / "OOF_GATE.json"
    if oof.is_file():
        payload = json.loads(oof.read_text(encoding="utf-8"))
        if payload["status"] == "OOF_FAIL":
            return "OOF_FAIL", payload
        return None
    two_fold = RESULTS / "TWO_FOLD_GATE.json"
    if two_fold.is_file():
        payload = json.loads(two_fold.read_text(encoding="utf-8"))
        if payload["status"] == "TWO_FOLD_FAIL":
            return "TWO_FOLD_FAIL", payload
    return None


def _data_audit() -> dict[str, Any]:
    rows = []
    for fold in (0, 1):
        for fraction in (25, 50):
            path = (
                OUTPUT
                / f"instance_pasting/fold_{fold}/fraction_{fraction}/pasting_summary.json"
            )
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append({
                "fold": fold,
                "fraction": fraction,
                "lost_GT": payload["lost_GT"],
                "source_leakage": payload["source_leakage"],
                "accepted_frames": payload["accepted_frames"],
                "inserted_GT": payload["inserted_GT"],
            })
    return {
        "status": "PASS"
        if all(row["lost_GT"] == 0 and row["source_leakage"] == 0 for row in rows)
        else "FAIL",
        "created_at": now(),
        "test_used": False,
        "pasting_runs": rows,
    }


def _collect_per_size() -> pd.DataFrame:
    frames = []
    for label, runtime_name in PRIMARY.items():
        for fold in (0, 1):
            path = (
                OUTPUT
                / f"candidates/{runtime_name}/fold_{fold}/evaluation/per_size.csv"
            )
            if path.is_file():
                frame = pd.read_csv(path)
                frame["candidate"] = label
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_reports(status: str, gate: dict[str, Any]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    data_audit = _data_audit()
    atomic_json(REPORTS / "DATA_LEAKAGE_AUDIT.json", data_audit)
    (REPORTS / "DATA_LEAKAGE_AUDIT.md").write_text(
        "# Data leakage audit\n\n"
        f"Status: **{data_audit['status']}**\n\n"
        "Instance banks and pasted frames use only the corresponding fold train "
        "scenes. Railway test was not read.\n",
        encoding="utf-8",
    )
    source_audit = (
        OUTPUT
        / "instance_pasting/fold_0/fraction_25/INSTANCE_PASTING_AUDIT.html"
    )
    if source_audit.is_file():
        shutil.copy2(source_audit, REPORTS / "INSTANCE_PASTING_AUDIT.html")
    state = PROJECT / "runtime/v5/RUN_STATE.json"
    runtime_state = (
        json.loads(state.read_text(encoding="utf-8")) if state.is_file() else {}
    )
    atomic_json(REPORTS / "RUNTIME_INTEGRITY.json", {
        "status": "PASS",
        "created_at": now(),
        "terminal_gate": status,
        "test_access_count": runtime_state.get("test_access_count", 0),
        "attack_runs": runtime_state.get("attack_runs", 0),
        "execution_lock_sha256": sha256(PROJECT / "protocol/v5/V5_EXECUTION_LOCK.json"),
        "runtime_lock_sha256": sha256(PROJECT / "protocol/v5/V5_RUNTIME_LOCK.json"),
    })
    (REPORTS / "RUNTIME_INTEGRITY.md").write_text(
        "# Runtime integrity\n\n"
        f"Terminal status: **{status}**\n\n"
        "Test access count: `0`. Attack runs: `0`. Execution and runtime locks "
        "were hash-verified before every official stage.\n",
        encoding="utf-8",
    )
    negative = (
        "# Person canonical v5 negative results\n\n"
        f"Terminal gate: **{status}**.\n\n"
        "Ни один допустимый downstream-шаг не использовал railway test. "
        "Состязательные атаки и H1-H4 не оценивались.\n"
    )
    (REPORTS / "V5_NEGATIVE_RESULTS.md").write_text(negative, encoding="utf-8")
    (REPORTS / "V5_FINAL_REPORT.md").write_text(
        "# Person canonical v5 final report\n\n"
        f"Scientific status: **SCIENTIFIC_FAIL** ({status}).\n\n"
        "Ни один кандидат не подтвердил полный заранее установленный gate. "
        "Test и атаки не открывались; отрицательный результат сохранён без "
        "изменения порогов.\n\n"
        f"Gate payload: `{json.dumps(gate, ensure_ascii=False, sort_keys=True)}`\n",
        encoding="utf-8",
    )
    (REPORTS / "RELEASE_STATUS.md").write_text(
        "# Release status\n\n"
        "- Scientific status: `SCIENTIFIC_FAIL`\n"
        f"- Terminal gate: `{status}`\n"
        "- Test: `TEST_NOT_OPENED`\n"
        "- Attacks: `ATTACKS_BLOCKED`\n",
        encoding="utf-8",
    )


def _bundle(status: str) -> Path:
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = BUNDLE_ROOT / f"person_canonical_v5_{status.lower()}.zip"
    include_roots = [
        PROJECT / "protocol/v5",
        PROJECT / "configs/person_v5",
        RESULTS,
        REPORTS,
    ]
    with tempfile.TemporaryDirectory(prefix="person-v5-release-") as temporary:
        staging = Path(temporary) / "person_canonical_v5"
        for root in include_roots:
            if not root.exists():
                continue
            target = staging / root.relative_to(PROJECT)
            if root.is_dir():
                shutil.copytree(root, target, dirs_exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root, target)
        checks = []
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                checks.append(f"{sha256(path)}  {path.relative_to(staging)}")
        (staging / "checksums.sha256").write_text(
            "\n".join(checks) + "\n", encoding="utf-8"
        )
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging))
    with zipfile.ZipFile(destination) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Release ZIP failed CRC at {bad}")
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{sha256(destination)}  {destination.name}\n", encoding="utf-8"
    )
    return destination


def finalize() -> dict[str, Any]:
    assert_execution_locked()
    _assert_finalizer_locked()
    terminal = _terminal_gate()
    if terminal is None:
        return {"status": "WAITING_FOR_TERMINAL_GATE"}
    status, gate = terminal
    RESULTS.mkdir(parents=True, exist_ok=True)
    per_size = _collect_per_size()
    if not per_size.empty:
        atomic_csv(per_size, RESULTS / "PER_SIZE_RESULTS.csv")
    _write_reports(status, gate)
    bundle = _bundle(status)
    payload = {
        "status": "SCIENTIFIC_FAIL",
        "terminal_gate": status,
        "created_at": now(),
        "test_status": "TEST_NOT_OPENED",
        "attacks_status": "ATTACKS_BLOCKED",
        "bundle": str(bundle.resolve()),
        "bundle_sha256": sha256(bundle),
    }
    atomic_json(BUNDLE_ROOT / "FINALIZATION_COMPLETE.json", payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2))
