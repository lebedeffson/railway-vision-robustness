from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "outputs/bundles/TNormFilter_legacy_baseline.zip"
VAL = PROJECT_DIR / "outputs/final_practice/unified_diagnostics_val_raw.csv"
CHECKPOINT = PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/weights/best.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    required = [
        VAL, VAL.with_suffix(".json"), CHECKPOINT,
        PROJECT_DIR / "outputs/final_practice/audit/nms_timeout_audit.json",
        PROJECT_DIR / "outputs/final_practice/audit/nms_audit_summary.json",
        PROJECT_DIR / "outputs/final_practice/audit/nms_timeout_recheck.csv",
        PROJECT_DIR / "outputs/final_practice/audit/legacy_recovery_recalculation.json",
        PROJECT_DIR / "outputs/final_practice/deadline/pilot/pilot_gate.json",
        PROJECT_DIR / "outputs/final_practice/deadline/pilot/legacy_smoke_disposition.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Legacy baseline is incomplete: {missing}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tnorm_legacy_") as directory:
        root = Path(directory) / "TNormFilter_legacy_baseline"
        mappings = [
            (VAL, "raw/unified_diagnostics_validation_legacy.csv"),
            (VAL.with_suffix(".json"), "configs/legacy_matrix.json"),
            (PROJECT_DIR / "outputs/final_practice/09_statistics_val/model_comparison_m0_m4.csv", "statistics/model_comparison_m0_m4.csv"),
            (PROJECT_DIR / "outputs/final_practice/audit/nms_timeout_audit.json", "audit/nms_timeout_audit.json"),
            (PROJECT_DIR / "outputs/final_practice/audit/nms_audit_summary.json", "audit/nms_audit_summary.json"),
            (PROJECT_DIR / "outputs/final_practice/audit/nms_timeout_recheck.csv", "audit/nms_timeout_recheck.csv"),
            (PROJECT_DIR / "outputs/final_practice/audit/nms_timeout_cases.csv", "audit/nms_timeout_cases.csv"),
            (PROJECT_DIR / "outputs/final_practice/audit/legacy_recovery_recalculation.json", "audit/legacy_recovery_recalculation.json"),
            (PROJECT_DIR / "outputs/final_practice/deadline/pilot/pilot_gate.json", "audit/canonical_smoke_pilot_gate.json"),
            (PROJECT_DIR / "outputs/final_practice/deadline/pilot/legacy_smoke_disposition.json", "audit/legacy_smoke_disposition.json"),
            (PROJECT_DIR / "outputs/final_practice/interim_audit/go_no_go.json", "audit/interim_go_no_go.json"),
            (PROJECT_DIR / "outputs/final_practice/legacy_validation.complete.json", "audit/legacy_validation.complete.json"),
            (PROJECT_DIR / "outputs/final_practice/audit/split_audit.json", "split/split_audit.json"),
            (PROJECT_DIR / "outputs/final_practice/audit/split_manifest.csv", "split/split_manifest.csv"),
            (PROJECT_DIR / "outputs/final_practice/audit/checkpoint_selection.csv", "checkpoint/checkpoint_selection.csv"),
            (PROJECT_DIR / "outputs/final_practice/audit/checkpoint_provenance.json", "checkpoint/checkpoint_provenance.json"),
            (PROJECT_DIR / "config/revision_q1_protocol.yaml", "configs/revision_q1_protocol.yaml"),
        ]
        for source, relative in mappings:
            copy(source, root / relative)
        (root / "README.md").write_text(
            "# TNormFilter legacy baseline\n\n"
            "Validation-only historical compatibility experiment. The feature columns named "
            "Product, Goedel and Lukasiewicz are legacy compatibility scores, not the canonical "
            "pointwise T-norm evidence. No legacy test matrix is included.\n",
            encoding="utf-8",
        )
        (root / "known_limitations.md").write_text(
            "# Known limitations\n\n"
            "- Legacy compatibility columns are not canonical pointwise T-norms.\n"
            "- Validation contains three independent grouped scenes.\n"
            "- The detector has low clean recall and many strong attacks exhibit floor effects.\n"
            "- This archive is historical baseline evidence and must not populate canonical claims.\n",
            encoding="utf-8",
        )
        files = sorted(path for path in root.rglob("*") if path.is_file())
        checksums = {path.relative_to(root).as_posix(): sha256(path) for path in files}
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True
            ).strip(),
            "role": "legacy_validation_baseline_not_primary_article_evidence",
            "split": "validation", "frames": 198, "independent_scenes": 3,
            "checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
            "files": checksums,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        files = sorted(path for path in root.rglob("*") if path.is_file())
        (root / "checksums.sha256").write_text(
            "".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
            encoding="utf-8",
        )
        temporary = OUTPUT.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root.parent).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Legacy ZIP integrity check failed")
        temporary.replace(OUTPUT)
    OUTPUT.with_suffix(".zip.sha256").write_text(
        f"{sha256(OUTPUT)}  {OUTPUT.name}\n", encoding="utf-8"
    )
    print(json.dumps({"zip": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
