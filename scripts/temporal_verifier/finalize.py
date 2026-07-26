from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from scripts.temporal_verifier.common import (
    CONFIG,
    LOCK,
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_json,
    config,
    sha256,
)


PUBLIC_FILES = (
    "configs/temporal_verifier_v1.yaml",
    "protocol/temporal_verifier_v1/PROTOCOL.md",
    "src/temporal_verifier/__init__.py",
    "src/temporal_verifier/features.py",
    "src/temporal_verifier/pipeline.py",
    "scripts/temporal_verifier/__init__.py",
    "scripts/temporal_verifier/common.py",
    "scripts/temporal_verifier/lock_protocol.py",
    "scripts/temporal_verifier/run_verifier.py",
    "scripts/temporal_verifier/finalize.py",
    "tests/test_temporal_verifier_v1.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_output_files() -> list[Path]:
    candidates = [
        LOCK,
        LOCK.with_suffix(".sha256"),
        OUTPUT / "decision_trace.json",
        OUTPUT / "audit/FALSE_TRACK_SUMMARY.json",
        OUTPUT / "audit/false_track_report.csv",
        OUTPUT / "audit/false_tracks_per_scene.csv",
        OUTPUT / "rule/RULE_MATRIX.csv",
        OUTPUT / "rule/PRE_CONFIRMATION_RULE_FREEZE.json",
        OUTPUT / "rule/TWO_FOLD_RULE_GATE.json",
        OUTPUT / "learned/logistic_l2/THRESHOLD_SWEEP.csv",
        OUTPUT / "learned/logistic_l2/PRE_CONFIRMATION_FREEZE.json",
        OUTPUT / "learned/logistic_l2/TWO_FOLD_LOGISTIC_GATE.json",
        OUTPUT / "sensitivity/BYTETRACK_GATE.json",
        OUTPUT / "FINAL_SUMMARY.json",
        OUTPUT / "reports/FINAL_REPORT.md",
    ]
    return [path for path in candidates if path.is_file()]


def report_text(decision: dict[str, Any], audit: dict[str, Any]) -> str:
    gate = decision["final_gate"]
    delta = gate["deltas"]
    checks = gate["checks"]
    failed = ", ".join(gate["failed_conditions"]) or "none"
    return f"""# Railway person temporal verifier v1

## Decision

```text
status: {decision["status"]}
primary tracker: {decision["primary_tracker"]}
final verifier stage: {decision["final_stage"]}
test: SEALED
test access count: 0
```

The frame detector is frozen. The verifier filters only temporal-confirmed or
interpolated outputs; it never removes the original frame-detector predictions.

## Two-fold development result

| Metric | Effect |
| --- | ---: |
| Recall | {delta["recall"]:+.6f} |
| Relative FN/frame reduction | {100.0 * delta["relative_FN_reduction"]:+.2f}% |
| Relative false-alarm change | {100.0 * delta["relative_false_alarm_increase"]:+.2f}% |
| F1 | {delta["F1"]:+.6f} |
| Improved scene fraction | {100.0 * gate["improved_scene_fraction"]:.1f}% |
| Worst-scene Recall delta | {gate["worst_scene_recall_delta"]:+.6f} |

Failed gate conditions: `{failed}`.

## False-track audit

The development support split contained {audit["false_tracks"]} unambiguous
false tracks. The public package contains aggregate and tabular diagnostics,
but excludes the private crop gallery and all source images.

## Gate checks

```json
{json.dumps(checks, indent=2, sort_keys=True)}
```

## Scope

This is a development-only result. A PASS on two folds still does not authorize
test access until full development/OOF evidence is available. A FAIL requires a
new crop-verifier amendment; it must not trigger post-hoc retuning under this
protocol ID.
"""


def ensure_public_path(path: Path) -> None:
    lowered = str(path).lower()
    forbidden = (
        ".pt",
        ".pth",
        ".ckpt",
        ".jpg",
        ".jpeg",
        ".png",
        "raw_predictions",
        "track_observations",
        "temporal_additions",
        "model.pkl",
        "test_opened",
    )
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"Forbidden public bundle path: {path}")


def build_bundle(destination: Path) -> dict[str, Any]:
    files: list[tuple[Path, str]] = []
    for relative in PUBLIC_FILES:
        source = PROJECT / relative
        if not source.is_file():
            raise RuntimeError(f"Missing public source: {relative}")
        files.append((source, relative))
    for source in safe_output_files():
        relative = str(source.relative_to(PROJECT))
        files.append((source, relative))
    for _, relative in files:
        ensure_public_path(Path(relative))

    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[str] = []
    with tempfile.TemporaryDirectory(prefix="temporal-verifier-bundle-") as temporary:
        staging = Path(temporary) / "railway-person-temporal-verifier-v1"
        for source, relative in files:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            relative = path.relative_to(staging).as_posix()
            manifest_rows.append(f"{sha256(path)}  {relative}")
        manifest = staging / "MANIFEST.sha256"
        manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(item for item in staging.rglob("*") if item.is_file()):
                archive.write(
                    path,
                    Path("railway-person-temporal-verifier-v1")
                    / path.relative_to(staging),
                )
    return {
        "path": str(destination.relative_to(PROJECT)),
        "sha256": sha256(destination),
        "files": len(files) + 1,
    }


def main() -> None:
    lock = assert_locked()
    decision_path = OUTPUT / "decision_trace.json"
    if not decision_path.is_file():
        raise RuntimeError("Verifier decision trace is missing")
    decision = read_json(decision_path)
    if decision["test_status"] != "SEALED" or decision["test_access_count"] != 0:
        raise RuntimeError("Test isolation was violated")
    audit = read_json(OUTPUT / "audit/FALSE_TRACK_SUMMARY.json")
    report = OUTPUT / "reports/FINAL_REPORT.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(report_text(decision, audit), encoding="utf-8")

    summary = {
        "protocol_id": config()["protocol_id"],
        "status": decision["status"],
        "primary_tracker": decision["primary_tracker"],
        "final_stage": decision["final_stage"],
        "final_gate": decision["final_gate"],
        "false_track_summary": audit,
        "implementation_commit": lock["implementation_commit"],
        "protocol_lock_sha256": sha256(LOCK),
        "config_sha256": sha256(CONFIG),
        "test_status": "SEALED",
        "test_access_count": 0,
        "public_claim": (
            "development gate passed; test remains sealed pending full development"
            if decision["status"] == "TWO_FOLD_PASS"
            else "development gate failed; test was not opened"
        ),
    }
    atomic_json(OUTPUT / "FINAL_SUMMARY.json", summary)
    bundle_name = (
        "railway_person_temporal_verifier_two_fold_pass.zip"
        if decision["status"] == "TWO_FOLD_PASS"
        else "railway_person_temporal_verifier_development_fail.zip"
    )
    bundle = build_bundle(OUTPUT / "bundles" / bundle_name)
    summary["public_bundle"] = bundle
    atomic_json(OUTPUT / "FINAL_SUMMARY.json", summary)
    sidecar = OUTPUT / "bundles/MANIFEST.sha256"
    sidecar.write_text(
        f"{bundle['sha256']}  {bundle_name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
