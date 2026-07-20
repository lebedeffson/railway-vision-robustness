from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/bundles/TNormFilter_final_practice.zip"


ROOT_FILES = [
    "AGENTS.md",
    "HANDOFF.md",
    "FINAL_PRACTICE_PROTOCOL.md",
    "requirements.txt",
]


ROOT_GLOBS = ["*.py", "tests/*.py"]


OPTIONAL_FILES = [
    "data/yolo_osdar23/data.yaml",
    "data/yolo_osdar23/manifest.csv",
    "data/yolo_osdar23/split_groups.txt",
    "outputs/training/yolo11m_baseline_stage1/results.csv",
    "outputs/training/yolo11m_baseline_stage1/results.png",
    "outputs/training/yolo11m_baseline_stage1/args.yaml",
    "outputs/training/yolo11m_baseline_stage1/weights/best.pt",
    "outputs/training/yolo11m_baseline_stage2/results.csv",
    "outputs/training/yolo11m_baseline_stage2/results.png",
    "outputs/training/yolo11m_baseline_stage2/args.yaml",
    "outputs/training/yolo11m_baseline_stage2/weights/best.pt",
]


OUTPUT_TREES = [
    "outputs/evaluation",
    "outputs/diagnostics",
    "outputs/final_practice",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    files: set[Path] = set()
    for value in ROOT_FILES + OPTIONAL_FILES:
        path = PROJECT_DIR / value
        if path.is_file():
            files.add(path)
    for pattern in ROOT_GLOBS:
        files.update(path for path in PROJECT_DIR.glob(pattern) if path.is_file())
    for tree in OUTPUT_TREES:
        root = PROJECT_DIR / tree
        if root.is_dir():
            files.update(path for path in root.rglob("*") if path.is_file())
    return sorted(files)


def build(output: Path) -> dict[str, object]:
    trained_model = (
        PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/weights/best.pt"
    )
    if not trained_model.is_file():
        raise RuntimeError(
            "Final delivery is forbidden before training: stage2 weights/best.pt is missing"
        )
    files = selected_files()
    manifest_rows = [
        {
            "path": str(path.relative_to(PROJECT_DIR)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    manifest = {
        "status": "TRAINED_DELIVERY",
        "project": "TNormFilter OSDaR23 final practice",
        "file_count": len(files),
        "files": manifest_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(PROJECT_DIR))
        archive.writestr("DELIVERY_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        bad_member = archive.testzip()
    if bad_member is not None:
        raise RuntimeError(f"ZIP integrity failure: {bad_member}")
    bundle_hash = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{bundle_hash}  {output.name}\n", encoding="utf-8"
    )
    return {
        "status": "PASS",
        "zip": str(output),
        "bytes": output.stat().st_size,
        "sha256": bundle_hash,
        "files": len(files) + 1,
        "testzip": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trained final-practice delivery ZIP")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))


if __name__ == "__main__":
    main()
