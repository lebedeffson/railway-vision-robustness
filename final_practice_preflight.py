from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
from pathlib import Path

from checkpoint_selection import export_selection, selected_checkpoint


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/final_practice/preflight.json"


REQUIRED_INPUTS = {
    "dataset_yaml": "data/yolo_osdar23/data.yaml",
    "dataset_manifest": "data/yolo_osdar23/manifest.csv",
    "trained_model": str(selected_checkpoint().relative_to(PROJECT_DIR)),
}

LEGACY_RESULTS = {
    "feature_val": "outputs/diagnostics/feature_consistency/feature_consistency_val.csv",
    "feature_test": "outputs/diagnostics/feature_consistency/feature_consistency_test.csv",
    "detection_val": "outputs/diagnostics/image_detection/image_detection_val.csv",
    "detection_test": "outputs/diagnostics/image_detection/image_detection_test.csv",
}

MODULES = ["torch", "ultralytics", "cv2", "numpy", "pandas", "sklearn", "scipy", "yaml", "tqdm"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=PROJECT_DIR, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_report() -> dict[str, object]:
    required = {
        name: {"path": value, "exists": (PROJECT_DIR / value).is_file()}
        for name, value in REQUIRED_INPUTS.items()
    }
    legacy = {
        name: {"path": value, "exists": (PROJECT_DIR / value).is_file()}
        for name, value in LEGACY_RESULTS.items()
    }
    modules: dict[str, dict[str, object]] = {}
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            modules[name] = {
                "available": True,
                "version": getattr(module, "__version__", None),
            }
        except Exception as error:
            modules[name] = {
                "available": False,
                "error": f"{type(error).__name__}: {error}",
            }
    missing_required = [name for name, item in required.items() if not item["exists"]]
    missing_modules = [name for name, result in modules.items() if not result["available"]]
    input_archive = PROJECT_DIR.parent / "project.zip"
    return {
        "status": "READY" if not missing_required and not missing_modules else "BLOCKED_INPUTS",
        "project_dir": str(PROJECT_DIR),
        "baseline_commit": git_value("rev-list", "-n", "1", "diagnostics_v1_current"),
        "baseline_tag": "diagnostics_v1_current" if git_value("tag", "--list", "diagnostics_v1_current") else None,
        "received_archive": {
            "path": str(input_archive),
            "exists": input_archive.is_file(),
            "sha256": sha256(input_archive) if input_archive.is_file() else None,
        },
        "required_inputs": required,
        "legacy_results": legacy,
        "python_modules": modules,
        "missing_required_inputs": missing_required,
        "missing_python_modules": missing_modules,
        "next_gate": (
            "Run audit_final_practice.py; do not launch attacks before audit PASS"
            if not missing_required and not missing_modules
            else "Restore dataset, trained best.pt, raw CSVs if available, and install requirements"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed final-practice readiness check")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    export_selection(PROJECT_DIR / "outputs/final_practice/audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
