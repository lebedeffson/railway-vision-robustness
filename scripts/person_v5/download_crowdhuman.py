from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


PROJECT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT / "configs/canonical_v5_person_data_first.yaml"
LOCK = PROJECT / "protocols/canonical_v5_person_data_first_v1/protocol_lock.json"
MANIFEST = (
    PROJECT
    / "protocols/canonical_v5_person_data_first_v1/"
    "data_acquisition_manifest.json"
)
MIRROR = "https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def planned_files(protocol: dict[str, Any]) -> dict[str, int]:
    files = {
        str(name): int(size)
        for name, size in protocol["crowdhuman"]["transport"][
            "expected_files"
        ].items()
    }
    if any("test" in name.lower() for name in files):
        raise RuntimeError("CrowdHuman test must not enter v5 acquisition")
    return files


def download(url: str, destination: Path, expected_size: int) -> None:
    part = destination.with_suffix(destination.suffix + ".part")
    current = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "railway-vision-robustness/1.0"}
    if current:
        headers["Range"] = f"bytes={current}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        if current and error.code == 416 and current == expected_size:
            part.replace(destination)
            return
        raise
    status = getattr(response, "status", response.getcode())
    if current and status != 206:
        current = 0
        part.unlink(missing_ok=True)
    mode = "ab" if current else "wb"
    with response, part.open(mode) as handle:
        shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
    actual_size = part.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Incomplete download {destination.name}: "
            f"{actual_size} != {expected_size}"
        )
    part.replace(destination)


def acquire(destination: Path, workers: int = 3) -> dict[str, Any]:
    if not LOCK.is_file():
        raise RuntimeError("Canonical v5 protocol lock is missing")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "LOCKED" or not lock["test_sealed"]:
        raise RuntimeError("Canonical v5 lock is not sealed")
    protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if sha256(CONFIG) != lock["protocol_sha256"]:
        raise RuntimeError("Canonical v5 protocol changed after lock")
    files = planned_files(protocol)
    destination.mkdir(parents=True, exist_ok=True)
    def acquire_one(item: tuple[str, int]) -> tuple[str, dict[str, Any]]:
        name, expected_size = item
        target = destination / name
        print(
            f"download_start name={name} existing_bytes="
            f"{target.with_suffix(target.suffix + '.part').stat().st_size if target.with_suffix(target.suffix + '.part').exists() else 0}",
            file=sys.stderr,
            flush=True,
        )
        if not target.is_file() or target.stat().st_size != expected_size:
            download(
                f"{MIRROR}/{name}?download=true",
                target,
                expected_size,
            )
        record = {
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "transport_url": f"{MIRROR}/{name}?download=true",
        }
        print(
            f"download_complete name={name} bytes={record['bytes']}",
            file=sys.stderr,
            flush=True,
        )
        return name, record

    records = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(int(workers), len(files)))
    ) as executor:
        for name, record in executor.map(acquire_one, files.items()):
            records[name] = record
    payload = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "terms_acknowledged_by_invoking_user": True,
        "terms_scope": "non-commercial research and education only",
        "redistribution_of_images": "prohibited",
        "official_identity": protocol["crowdhuman"]["identity"],
        "transport": "sshao0516/CrowdHuman Hugging Face mirror",
        "test_downloaded": False,
        "files": records,
    }
    atomic_json(MANIFEST, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=PROJECT / "data/crowdhuman_downloads",
    )
    parser.add_argument(
        "--accept-noncommercial-research-terms",
        action="store_true",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not args.accept_noncommercial_research_terms:
        raise SystemExit(
            "CrowdHuman terms were not accepted; no network request was made"
        )
    print(
        json.dumps(
            acquire(args.destination.resolve(), workers=args.workers),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
