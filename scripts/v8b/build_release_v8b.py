from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs/person_v8b/final"
BUNDLES = FINAL / "bundles"
ARCHIVE_NAME = "railway_vision_robustness_v8b_final_public.zip"
PREFIX = "railway_vision_robustness_v8b_final_public"
FIXED_TIME = (2026, 7, 26, 0, 0, 0)
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
    ".cff",
    ".sha256",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files() -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []

    def add(path: Path, target: str | None = None) -> None:
        if not path.is_file():
            raise RuntimeError(f"Required release file is missing: {path}")
        entries.append((path, target or path.relative_to(ROOT).as_posix()))

    add(FINAL / "PUBLIC_README.md", "README.md")
    add(
        FINAL / "PUBLIC_CONFIG_V8B.yaml",
        "configs/canonical_v8b_person_failure_risk.yaml",
    )
    add(FINAL / "CONFIG_REDACTION.json")
    add(ROOT / "configs/canonical_v9_person_residual_temporal.yaml")
    for path in sorted((ROOT / "protocol/v8b").glob("*")):
        if path.is_file():
            add(path)
    for path in sorted((ROOT / "protocol/v9").glob("*")):
        if path.is_file():
            add(path)
    for path in sorted((ROOT / "reports/v8b").glob("*")):
        if path.is_file():
            add(path)
    for path in sorted((ROOT / "scripts/v8b").glob("*.py")):
        add(path)
    for path in sorted((ROOT / "scripts/person_v9").glob("*.py")):
        add(path)
    add(ROOT / "tests/test_v8b_finalization_v9.py")
    for path in sorted(FINAL.rglob("*")):
        if not path.is_file() or BUNDLES in path.parents:
            continue
        add(path, f"final/{path.relative_to(FINAL).as_posix()}")
    return sorted(entries, key=lambda item: item[1])


def scan_public_text(entries: Iterable[tuple[Path, str]]) -> list[str]:
    failures: list[str] = []
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    secret = re.compile(
        r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
    )
    forbidden = (
        "/" + "home/",
        "CrowdHuman/images",
        "data/" + "yolo_osdar23",
        "outputs/" + "person_v3/",
    )
    for path, target in entries:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(value in text for value in forbidden):
            failures.append(f"{target}: absolute/private path")
        if email.search(text):
            failures.append(f"{target}: email")
        if secret.search(text):
            failures.append(f"{target}: secret-like value")
    return failures


def write_zip(entries: list[tuple[Path, str]], archive: Path) -> str:
    payloads: list[tuple[str, bytes]] = [
        (target, path.read_bytes()) for path, target in entries
    ]
    manifest = "\n".join(
        f"{sha256_bytes(payload)}  {target}" for target, payload in payloads
    ) + "\n"
    payloads.append(("PACKAGE_MANIFEST.sha256", manifest.encode("utf-8")))
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as handle:
        for target, payload in payloads:
            info = zipfile.ZipInfo(f"{PREFIX}/{target}", FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, payload)
    return manifest


def validate_archive(archive: Path) -> dict[str, object]:
    with zipfile.ZipFile(archive) as handle:
        bad = handle.testzip()
        names = handle.namelist()
        text_failures = []
        for name in names:
            suffix = Path(name).suffix.lower()
            if suffix not in TEXT_SUFFIXES:
                continue
            text = handle.read(name).decode("utf-8", errors="replace")
            if "/" + "home/" in text:
                text_failures.append(f"{name}: absolute path")
            if re.search(
                r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text
            ):
                text_failures.append(f"{name}: email")
        required = {
            f"{PREFIX}/final/article/article_v8b_final.docx",
            f"{PREFIX}/final/article/article_v8b_final.pdf",
            f"{PREFIX}/final/article/article_v8b_final.md",
            f"{PREFIX}/final/FINAL_METRICS.json",
            f"{PREFIX}/final/FINALIZATION_AUDIT.json",
            f"{PREFIX}/final/OOF_INPUT_REDACTED.csv",
            f"{PREFIX}/protocol/v8b/V8B_PROTOCOL_LOCK.json",
            f"{PREFIX}/protocol/v9/V9_PREREQUISITES.json",
            f"{PREFIX}/PACKAGE_MANIFEST.sha256",
        }
        checks = {
            "zip_integrity": bad is None,
            "required_files": required <= set(names),
            "no_absolute_paths_or_emails": not text_failures,
            "no_dataset_images": not any(
                Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}
                and "/final/figures/" not in name
                for name in names
            ),
            "no_checkpoint": not any(
                Path(name).suffix.lower() in {".pt", ".pth", ".ckpt"}
                for name in names
            ),
            "no_raw_feature_tensor": not any(
                Path(name).suffix.lower() in {".npz", ".npy"}
                for name in names
            ),
            "no_test_results": not any("/test/" in name.lower() for name in names),
        }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "text_failures": text_failures,
        "files": len(names),
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
    }


def main() -> int:
    finalization = json.loads(
        (FINAL / "FINALIZATION_AUDIT.json").read_text(encoding="utf-8")
    )
    article = json.loads(
        (FINAL / "article/ARTICLE_VALIDATION.json").read_text(encoding="utf-8")
    )
    prerequisites = json.loads(
        (ROOT / "protocol/v9/V9_PREREQUISITES.json").read_text(encoding="utf-8")
    )
    if finalization["status"] != "PASS" or article["status"] != "PASS":
        raise RuntimeError("Release blocked by V8b finalization/article audit")
    if finalization["test_access_count"] != 0:
        raise RuntimeError("Release blocked: test access is nonzero")
    if prerequisites["activation_passed"]:
        raise RuntimeError("Unexpected V9 activation in V8b negative release")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    (FINAL / "RELEASE_INFO.json").write_text(
        json.dumps(
            {
                "repository": "railway-vision-robustness",
                "version": "0.8b",
                "tag": "v0.8b-negative-result",
                "commit": commit,
                "v8b_status": "FINALIZED_NEGATIVE_RESULT",
                "v9_status": "BLOCKED_PREREQUISITES",
                "test_status": "SEALED",
                "test_access_count": 0,
                "license_added_by_release": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    public_readme = """# Canonical V8b public reproducibility package

This package contains the finalized negative-result study, redacted OOF rows,
independent post-lock finalization code, six article tables, five figures,
MD/DOCX/PDF manuscripts, V8b protocol evidence, and the blocked prospective
V9 design.

The package excludes dataset images, detector checkpoints, raw feature
tensors, local absolute paths, CrowdHuman materials, sealed-test content,
credentials, and service memory. The V8b test was not opened. V9 was not
started.

Run the finalizer against `final/OOF_INPUT_REDACTED.csv` as described in
`final/REPRODUCE.md`. The expected primary result is U2 MAE 1.5954889302,
U3 MAE 1.7679771025, relative reduction -10.81099148%, and five U3 scene
wins.
"""
    (FINAL / "PUBLIC_README.md").write_text(public_readme, encoding="utf-8")
    private_config_path = ROOT / "configs/canonical_v8b_person_failure_risk.yaml"
    public_config = yaml.safe_load(private_config_path.read_text(encoding="utf-8"))
    public_config["immutable_inputs"]["detector_checkpoint"]["path"] = (
        "FROZEN_B0_CHECKPOINT_NOT_INCLUDED"
    )
    public_config["immutable_inputs"]["development_manifest"]["path"] = (
        "DEVELOPMENT_MANIFEST_NOT_INCLUDED"
    )
    public_config["immutable_inputs"]["sealed_test_manifest"]["path"] = (
        "SEALED_TEST_MANIFEST_NOT_INCLUDED"
    )
    public_config_path = FINAL / "PUBLIC_CONFIG_V8B.yaml"
    public_config_path.write_text(
        yaml.safe_dump(public_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (FINAL / "CONFIG_REDACTION.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "scientific_fields_changed": False,
                "redacted_fields": [
                    "immutable_inputs.detector_checkpoint.path",
                    "immutable_inputs.development_manifest.path",
                    "immutable_inputs.sealed_test_manifest.path",
                ],
                "original_config_sha256": sha256_file(private_config_path),
                "public_config_sha256": sha256_file(public_config_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    entries = selected_files()
    failures = scan_public_text(entries)
    if failures:
        raise RuntimeError(f"Public redaction failed: {failures}")
    BUNDLES.mkdir(parents=True, exist_ok=True)
    archive = BUNDLES / ARCHIVE_NAME
    write_zip(entries, archive)
    validation = validate_archive(archive)
    (BUNDLES / "RELEASE_VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (BUNDLES / "MANIFEST.sha256").write_text(
        f"{validation['archive_sha256']}  {ARCHIVE_NAME}\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    if validation["status"] != "PASS":
        raise RuntimeError("V8b public release validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
