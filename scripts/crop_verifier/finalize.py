from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from scripts.crop_verifier.common import (
    CONFIG,
    LOCK,
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_json,
    config,
    sha256,
)


PUBLIC_SOURCE = (
    "configs/crop_verifier_v1.yaml",
    "protocol/crop_verifier_v1/PROTOCOL.md",
    "src/crop_verifier_v1/__init__.py",
    "src/crop_verifier_v1/encoder.py",
    "src/crop_verifier_v1/model.py",
    "scripts/crop_verifier/__init__.py",
    "scripts/crop_verifier/common.py",
    "scripts/crop_verifier/lock_protocol.py",
    "scripts/crop_verifier/extract_embeddings.py",
    "scripts/crop_verifier/run_crop_verifier.py",
    "scripts/crop_verifier/finalize.py",
    "tests/test_crop_verifier_v1.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def public_outputs() -> list[Path]:
    candidates = [
        LOCK,
        LOCK.with_suffix(".sha256"),
        OUTPUT / "protocol/PRE_CONFIRMATION_FREEZE.json",
        OUTPUT / "audit/CROP_LEAKAGE_AUDIT.json",
        OUTPUT / "audit/CONFUSION_AUDIT.json",
        OUTPUT / "audit/CONFUSION_COUNTS.csv",
        OUTPUT / "audit/FALSE_CATEGORY_EFFECT.csv",
        OUTPUT / "results/SUPPORT_OOF_MODEL_COMPARISON.csv",
        OUTPUT / "results/FOLD0_MODEL_SELECTION.csv",
        OUTPUT / "results/TWO_FOLD_MODEL_COMPARISON.csv",
        OUTPUT / "results/FALSE_ALARM_STAGES.csv",
        OUTPUT / "results/FULL_DEVELOPMENT_GATE.json",
        OUTPUT / "results/FULL_DEVELOPMENT_PER_SCENE.csv",
        OUTPUT / "models/track_only/TWO_FOLD_TRIAGE_GATE.json",
        OUTPUT / "models/visual_only/TWO_FOLD_TRIAGE_GATE.json",
        OUTPUT / "models/combined/TWO_FOLD_TRIAGE_GATE.json",
        OUTPUT / "decision_trace.json",
        OUTPUT / "FINAL_SUMMARY.json",
        OUTPUT / "reports/FINAL_REPORT.md",
    ]
    for role in ("support", "screening", "confirmation"):
        candidates.append(OUTPUT / f"embeddings/{role}/EMBEDDING_AUDIT.json")
    return [path for path in candidates if path.is_file()]


def report(decision: dict[str, Any]) -> str:
    gate = decision["two_fold_triage"]
    delta = gate["deltas"]
    failed = ", ".join(gate["failed_conditions"]) or "none"
    full = decision["full_development_gate"]["status"]
    return f"""# Railway person crop verifier v1

## Final status

```text
status: {decision["status"]}
selected model: {decision["selected_model"]}
global threshold: {decision["selected_threshold"]}
two-fold triage: {gate["status"]}
full development: {full}
test: SEALED
test access count: 0
```

The detector, OC-SORT tracker, low-confidence predictions and parent verifier
results remained frozen.

## Two-fold triage

| Metric | Effect |
|---|---:|
| Recall | {delta["recall"]:+.6f} |
| Relative FN/frame reduction | {100.0 * delta["relative_FN_reduction"]:+.2f}% |
| Relative false-alarm change | {100.0 * delta["relative_false_alarm_increase"]:+.2f}% |
| F1 | {delta["F1"]:+.6f} |

Failed conditions: `{failed}`.

## Interpretation

The three prospectively frozen L2-logistic variants were compared:
track-only, frozen visual embedding only, and their concatenation. Confirmation
fold 1 could not change the model or threshold chosen on support OOF plus fold 0.

If triage failed, full 15-scene development and test were not run. Under the
frozen stop rule the temporal direction is closed and new railway scenes are
required. The private confusion gallery, embeddings, crop metadata, model
pickle, source images, detector checkpoint and test data are excluded from the
public package.
"""


def validate_public_name(relative: str) -> None:
    lowered = relative.lower()
    forbidden = (
        ".jpg",
        ".jpeg",
        ".png",
        ".pt",
        ".pth",
        ".ckpt",
        ".npz",
        "model.pkl",
        "crop_metadata.csv",
        "confusion_audit_private.csv",
        "track_observations.csv",
        "temporal_additions.csv",
        "raw_predictions.parquet",
        "test_opened.json",
    )
    if any(token in lowered for token in forbidden):
        raise RuntimeError(f"Forbidden public payload: {relative}")


def build_bundle(destination: Path) -> dict[str, Any]:
    files: list[tuple[Path, str]] = []
    for relative in PUBLIC_SOURCE:
        source = PROJECT / relative
        if not source.is_file():
            raise RuntimeError(f"Missing public source: {relative}")
        files.append((source, relative))
    for source in public_outputs():
        files.append((source, str(source.relative_to(PROJECT))))
    for _, relative in files:
        validate_public_name(relative)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crop-verifier-public-") as temporary:
        staging = Path(temporary) / "railway-person-crop-verifier-v1"
        for source, relative in files:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifest_rows = []
        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            manifest_rows.append(
                f"{sha256(path)}  {path.relative_to(staging).as_posix()}"
            )
        (staging / "MANIFEST.sha256").write_text(
            "\n".join(manifest_rows) + "\n", encoding="utf-8"
        )
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(item for item in staging.rglob("*") if item.is_file()):
                archive.write(
                    path,
                    Path("railway-person-crop-verifier-v1")
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
        raise RuntimeError("Crop verifier decision is missing")
    decision = read_json(decision_path)
    if decision["test_status"] != "SEALED" or decision["test_access_count"] != 0:
        raise RuntimeError("Test isolation was violated")
    report_path = OUTPUT / "reports/FINAL_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report(decision), encoding="utf-8")
    summary = {
        "protocol_id": config()["protocol_id"],
        "status": decision["status"],
        "selected_model": decision["selected_model"],
        "selected_threshold": decision["selected_threshold"],
        "two_fold_triage": decision["two_fold_triage"],
        "full_development_gate": decision["full_development_gate"],
        "implementation_commit": lock["implementation_commit"],
        "config_sha256": sha256(CONFIG),
        "protocol_lock_sha256": sha256(LOCK),
        "temporal_direction": decision["temporal_direction"],
        "next_step": decision["next_step"],
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    atomic_json(OUTPUT / "FINAL_SUMMARY.json", summary)
    status = (
        "development_pass"
        if decision["status"] == "DEVELOPMENT_PASS"
        else "closed_no_practical_gate"
    )
    bundle_name = f"railway_person_crop_verifier_{status}.zip"
    bundle = build_bundle(OUTPUT / "bundles" / bundle_name)
    summary["public_bundle"] = bundle
    atomic_json(OUTPUT / "FINAL_SUMMARY.json", summary)
    (OUTPUT / "bundles/MANIFEST.sha256").write_text(
        f"{bundle['sha256']}  {bundle_name}\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
