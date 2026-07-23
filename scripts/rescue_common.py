from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_DIR / "configs/rescue/canonical_v2_rescue_v1.yaml"
OUTPUT_ROOT = PROJECT_DIR / "outputs/rescue"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    payload = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "canonical-v2-rescue-v1":
        raise RuntimeError("Unexpected rescue protocol")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command, cwd=PROJECT_DIR, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return completed.stdout.strip()


def environment_snapshot() -> dict[str, Any]:
    try:
        import torch
        import ultralytics

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        ultralytics_version = ultralytics.__version__
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as error:  # pragma: no cover - diagnostic fallback
        torch_version = cuda_version = ultralytics_version = gpu = None
        import_error = f"{type(error).__name__}: {error}"
    else:
        import_error = None
    return {
        "created_at": now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch_version,
        "cuda": cuda_version,
        "ultralytics": ultralytics_version,
        "gpu": gpu,
        "nvidia_smi": run_text([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
        "git_commit": run_text(["git", "rev-parse", "HEAD"]),
        "git_branch": run_text(["git", "branch", "--show-current"]),
        "git_status": run_text(["git", "status", "--porcelain=v1"]),
        "protocol_path": str(PROTOCOL_PATH.resolve()),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "import_error": import_error,
        "command": sys.argv,
    }


def assert_test_sealed() -> None:
    protocol = load_protocol()
    if protocol.get("test_sealed") is not True:
        raise RuntimeError("Rescue protocol does not seal test")
    marker = PROJECT_DIR / protocol["test_open_marker"]
    if marker.exists():
        raise RuntimeError(f"Test-open marker already exists: {marker}")


def write_protocol_lock() -> Path:
    protocol = load_protocol()
    manifest = PROJECT_DIR / protocol["split_manifest"]
    actual = sha256(manifest)
    if actual != protocol["split_manifest_sha256"]:
        raise RuntimeError(f"Frozen split hash mismatch: {actual}")
    destination = OUTPUT_ROOT / "protocol"
    destination.mkdir(parents=True, exist_ok=True)
    copied = destination / PROTOCOL_PATH.name
    source_bytes = PROTOCOL_PATH.read_bytes()
    if copied.exists() and copied.read_bytes() != source_bytes:
        raise RuntimeError("Existing rescue protocol lock differs from committed protocol")
    if not copied.exists():
        copied.write_bytes(source_bytes)
    lock = destination / "protocol_lock.json"
    atomic_json(lock, {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "parent_commit": protocol["parent_commit"],
        "current_commit": run_text(["git", "rev-parse", "HEAD"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "split_manifest_sha256": actual,
        "test_sealed": True,
        "created_at": now(),
    })
    atomic_json(destination / "environment.json", environment_snapshot())
    return lock


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError as error:
        raise ValueError(f"Image path has no images component: {image_path}") from error
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def completed_marker(
    output: Path, *, inputs: list[Path], outputs: list[Path], extra: dict[str, Any] | None = None
) -> Path:
    missing = [str(path) for path in outputs if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot complete stage with missing outputs: {missing}")
    marker = output / "COMPLETED.json"
    atomic_json(marker, {
        "status": "PASS",
        "protocol_id": load_protocol()["protocol_id"],
        "created_at": now(),
        "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in inputs],
        "outputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in outputs],
        **(extra or {}),
    })
    return marker


def ensure_branch() -> None:
    branch = run_text(["git", "branch", "--show-current"])
    if branch != "feat/canonical-v2-rescue-v1":
        raise RuntimeError(f"Rescue must run on its frozen branch, found {branch!r}")


def configure_low_priority() -> None:
    try:
        os.nice(10)
    except OSError:
        pass
