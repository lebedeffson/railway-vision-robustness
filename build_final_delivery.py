from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from pipeline_status import STATUS_PATH, load as load_pipeline_status, sync as sync_pipeline_status
from pipeline_status import update as update_pipeline_status


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/bundles/TNormFilter_final_practice.zip"
FINAL_ROOT = PROJECT_DIR / "outputs/final_practice"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if path.is_file():
            copy_file(path, destination / path.relative_to(source))


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    ).stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def stage_payload(staging: Path, output: Path) -> None:
    copy_file(PROJECT_DIR / "AGENTS.md", staging / "AGENTS.md")
    copy_file(STATUS_PATH, staging / "pipeline_status.json")
    for source, relative in (
        (PROJECT_DIR / "config/raw_frame_exclusions.json", "configs/raw_frame_exclusions.json"),
        (PROJECT_DIR / "data/yolo_osdar23/data.yaml", "configs/data.yaml"),
        (PROJECT_DIR / "data/yolo_osdar23/manifest.csv", "audit/dataset_manifest.csv"),
        (PROJECT_DIR / "data/yolo_osdar23/split_groups.txt", "audit/split_groups.txt"),
        (PROJECT_DIR / "outputs/training/yolo11m_baseline_stage1/args.yaml", "configs/stage1_args.yaml"),
        (PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/args.yaml", "configs/stage2_args.yaml"),
        (PROJECT_DIR / "outputs/training/yolo11m_baseline_stage1/results.csv", "logs/stage1_results.csv"),
        (PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/results.csv", "logs/stage2_results.csv"),
        (PROJECT_DIR / "outputs/training/yolo11m_baseline_stage1/weights/best.pt", "checkpoints/stage1_best.pt"),
        (PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/weights/best.pt", "checkpoints/stage2_best.pt"),
        (FINAL_ROOT / "validation_freeze.json", "configs/validation_freeze.json"),
        (FINAL_ROOT / "preflight.json", "configs/preflight.json"),
        (FINAL_ROOT / "TNormFilter_final_practice_report.pdf", "report/TNormFilter_final_practice_report.pdf"),
        (FINAL_ROOT / "report_summary.json", "report/report_summary.json"),
        (PROJECT_DIR.parent / "RZD (2).pdf", "report/RZD_original_intermediate_report.pdf"),
        (PROJECT_DIR / "data/raw/2_station_berliner_tor_2.1/readme.md", "audit/osdar23_readme.md"),
        (PROJECT_DIR / "data/raw/2_station_berliner_tor_2.1/license.md", "audit/osdar23_license.md"),
    ):
        copy_file(source, staging / relative)
    copy_tree(FINAL_ROOT / "configs", staging / "configs")
    copy_tree(FINAL_ROOT / "audit", staging / "audit")
    copy_tree(FINAL_ROOT / "00_audit", staging / "audit/yolo_audit")
    copy_tree(FINAL_ROOT / "raw", staging / "raw")
    copy_tree(FINAL_ROOT / "tables", staging / "tables")
    copy_tree(FINAL_ROOT / "figures", staging / "figures")
    copy_tree(FINAL_ROOT / "09_statistics", staging / "statistics/test")
    copy_tree(FINAL_ROOT / "09_statistics_val", staging / "statistics/validation")
    for raw_file in (
        FINAL_ROOT / "unified_diagnostics_raw.csv",
        FINAL_ROOT / "unified_diagnostics_raw.json",
        FINAL_ROOT / "unified_diagnostics_val_raw.csv",
        FINAL_ROOT / "unified_diagnostics_val_raw.json",
    ):
        copy_file(raw_file, staging / "raw" / raw_file.name)
    for test in (PROJECT_DIR / "tests").glob("*.py"):
        copy_file(test, staging / "tests" / test.name)
    for script in PROJECT_DIR.glob("*.py"):
        copy_file(script, staging / "tests/reproducibility_scripts" / script.name)
    test_result = FINAL_ROOT / "tests/test_results.txt"
    copy_file(test_result, staging / "tests/test_results.txt")

    journal = subprocess.run(
        ["journalctl", "--user", "-u", "tnorm-wait-train.service", "-n", "500", "--no-pager"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    ).stdout
    (staging / "logs").mkdir(parents=True, exist_ok=True)
    (staging / "logs/training_service.log").write_text(journal, encoding="utf-8")

    commit = git("rev-parse", "HEAD")
    git_info = (
        f"commit={commit}\n"
        f"branch={git('branch', '--show-current')}\n"
        f"status_porcelain_begin\n{git('status', '--short')}\nstatus_porcelain_end\n"
    )
    (staging / "git_info.txt").write_text(git_info, encoding="utf-8")
    readme = """# TNormFilter final practice

This bundle contains the sequence-level OSDaR23 audit, clean YOLO11m baseline,
FGSM/PGD and adaptive Product-PGD diagnostics, P3/P4/P5 feature consistency,
M0-M4 statistics, latency evidence, figures, PDF report and reproducibility code.

Claim boundary: T-norm values are evaluated as complementary diagnostic features.
The Product filter is not claimed to be a universal white-box defense. JPEG and
median are excluded from adaptive robustness rankings because BPDA is not used.

The raw OSDaR23 images are intentionally not included.

Dataset sources: [OSDaR23 DOI](https://doi.org/10.57806/9mv146r0),
[research report](https://doi.org/10.48755/dzsf.230012.01),
[labeling guide](https://doi.org/10.48755/dzsf.230012.05), and
[ASAM OpenLABEL](https://www.asam.net/standards/detail/openlabel/).
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")

    audit = json.loads((FINAL_ROOT / "audit/split_audit.json").read_text(encoding="utf-8"))
    status = load_pipeline_status()
    run_summary = (
        "# Run summary\n\n"
        f"- Commit: `{commit}`\n"
        f"- Dataset audit: {audit['status']}; {audit['scenes_checked']}/{audit['scenes_expected']} scenes checked\n"
        f"- Retained readable frames: {audit['frames_found_and_readable']} of {audit['frames_expected']} OpenLABEL references\n"
        f"- Pre-split documented exclusions: {audit['configured_exclusions']}\n"
        f"- Statistical unit: sequence_id\n"
        f"- Pipeline states: " + ", ".join(
            f"{name}={entry['status']}" for name, entry in status["stages"].items()
        ) + "\n"
        "- Negative result retained: the simple validation threshold policy did not generalize to test.\n"
    )
    (staging / "run_summary.md").write_text(run_summary, encoding="utf-8")

    dataset_manifest = PROJECT_DIR / "data/yolo_osdar23/manifest.csv"
    split_groups = PROJECT_DIR / "data/yolo_osdar23/split_groups.txt"
    checkpoint = PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/weights/best.pt"
    manifest = {
        "project": "TNormFilter OSDaR23 final practice",
        "commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "versions": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "ultralytics": package_version("ultralytics"),
            "cuda": torch.version.cuda,
        },
        "dataset_hash": sha256(dataset_manifest),
        "dataset": {
            "name": "Open Sensor Data for Rail 2023 (OSDaR23)",
            "version": "1.1.0",
            "doi": "https://doi.org/10.57806/9mv146r0",
            "research_report": "https://doi.org/10.48755/dzsf.230012.01",
            "labeling_guide": "https://doi.org/10.48755/dzsf.230012.05",
            "annotation_license": "CC0 1.0",
            "sensor_license": "CC BY-SA 3.0 DE",
        },
        "split_hash": sha256(split_groups if split_groups.is_file() else dataset_manifest),
        "checkpoint": str(checkpoint.relative_to(PROJECT_DIR)),
        "checkpoint_sha256": sha256(checkpoint),
        "seeds": [42, 123, 999],
        "pipeline": status["stages"],
        "archive_target": str(output),
        "files": [],
    }
    manifest["files"] = [
        {
            "path": path.relative_to(staging).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(staging.rglob("*")) if path.is_file()
    ]
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksum_files = [path for path in sorted(staging.rglob("*")) if path.is_file()]
    (staging / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(staging).as_posix()}\n" for path in checksum_files),
        encoding="utf-8",
    )


def build(output: Path) -> dict[str, object]:
    checkpoint = PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/weights/best.pt"
    report = FINAL_ROOT / "TNormFilter_final_practice_report.pdf"
    if checkpoint.is_file() and not report.is_file():
        subprocess.run(
            [str(PROJECT_DIR / ".venv/bin/python"), "assemble_final_outputs.py"],
            cwd=PROJECT_DIR, check=True,
        )
        subprocess.run(
            [str(PROJECT_DIR / ".venv/bin/python"), "build_final_report.py"],
            cwd=PROJECT_DIR, check=True,
        )
    sync_pipeline_status()
    update_pipeline_status("bundle", "running", inputs=[PROJECT_DIR / "build_final_delivery.py"])
    required = [checkpoint, report, STATUS_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Final delivery prerequisites missing: {missing}")
    with tempfile.TemporaryDirectory(prefix="tnorm_final_bundle_") as directory:
        staging = Path(directory)
        stage_payload(staging, output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            bad_member = archive.testzip()
            names = set(archive.namelist())
        if bad_member is not None:
            raise RuntimeError(f"ZIP integrity failure: {bad_member}")
        required_names = {
            "README.md", "AGENTS.md", "run_summary.md", "manifest.json",
            "pipeline_status.json", "git_info.txt", "checksums.sha256",
            "report/TNormFilter_final_practice_report.pdf",
        }
        if required_names - names:
            raise RuntimeError(f"ZIP missing required members: {sorted(required_names - names)}")
        temporary.replace(output)
    bundle_hash = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{bundle_hash}  {output.name}\n", encoding="utf-8"
    )
    update_pipeline_status("bundle", "success", outputs=[output])
    return {
        "status": "PASS",
        "zip": str(output),
        "bytes": output.stat().st_size,
        "sha256": bundle_hash,
        "files": len(names),
        "testzip": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trained final-practice delivery ZIP")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
