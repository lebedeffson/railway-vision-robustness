from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / "outputs/person_v4_proxy"
PROTOCOL = PROJECT / "protocols/person_v4_train_only_proxy_v1"
BUNDLE = OUTPUT / "bundles/TNormFilter_person_v4_train_only_proxy_failed.zip"
CHECKPOINT_INDEX = OUTPUT / "results/final_checkpoint_index.json"
CLEANUP_MANIFEST = OUTPUT / "results/checkpoint_cleanup.json"


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


def selected_checkpoints() -> list[Path]:
    selected: list[Path] = []
    for split in ("proxy_0", "proxy_1", "proxy_2"):
        selected.append(
            OUTPUT / f"runs/{split}/A0/stage3/weights/best.pt"
        )
        selected.append(
            OUTPUT / f"runs/{split}/A3/stage3/weights/swad.pt"
        )
    return selected


def create_checkpoint_index() -> dict[str, Any]:
    records = []
    for path in selected_checkpoints():
        if not path.is_file():
            raise RuntimeError(f"Selected checkpoint missing: {path}")
        records.append(
            {
                "path": str(path.relative_to(PROJECT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    payload = {
        "status": "PASS",
        "selected_checkpoint_count": len(records),
        "weights_in_bundle": False,
        "checkpoints": records,
    }
    atomic_json(CHECKPOINT_INDEX, payload)
    return payload


def bundle_sources() -> list[Path]:
    paths = [
        PROJECT / "configs/person_v4_train_only_proxy.yaml",
        PROJECT / "configs/person_v4_train_only_proxy_runtime.yaml",
        OUTPUT / "runtime_contract.json",
        CHECKPOINT_INDEX,
    ]
    paths.extend(sorted(PROTOCOL.glob("*.json")))
    paths.extend(sorted(PROTOCOL.glob("*.csv")))
    paths.extend(sorted((OUTPUT / "results").glob("*.csv")))
    paths.extend(sorted((OUTPUT / "results").glob("*.json")))
    for marker in sorted((OUTPUT / "runs").rglob("TRAINING_COMPLETE.json")):
        paths.append(marker)
    for marker in sorted((OUTPUT / "runs").rglob("SWAD_SELECTION.json")):
        paths.append(marker)
    return list(dict.fromkeys(path for path in paths if path.is_file()))


def build_bundle() -> dict[str, Any]:
    gate = json.loads(
        (OUTPUT / "results/proxy_gate.json").read_text(encoding="utf-8")
    )
    if gate["status"] != "FAIL" or gate["test_opened"]:
        raise RuntimeError("Proxy is not a sealed completed FAIL")
    create_checkpoint_index()
    sources = bundle_sources()
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="person-v4-proxy-bundle-") as temp:
        staging = Path(temp)
        checksums = []
        for source in sources:
            relative = source.relative_to(PROJECT)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            checksums.append(f"{sha256(target)}  {relative.as_posix()}")
        (staging / "checksums.sha256").write_text(
            "\n".join(checksums) + "\n", encoding="utf-8"
        )
        manifest = {
            "status": "FAIL",
            "claim": "train-only compute filter; not article evidence",
            "test_opened": False,
            "weights_included": False,
            "files": len(sources),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(
            BUNDLE, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging))
    with zipfile.ZipFile(BUNDLE) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Proxy bundle CRC failed: {bad}")
    sidecar = BUNDLE.with_suffix(BUNDLE.suffix + ".sha256")
    sidecar.write_text(
        f"{sha256(BUNDLE)}  {BUNDLE.name}\n", encoding="utf-8"
    )
    return {
        "bundle": str(BUNDLE),
        "sha256": sha256(BUNDLE),
        "files": len(sources),
    }


def prune() -> dict[str, Any]:
    if not BUNDLE.is_file() or not zipfile.is_zipfile(BUNDLE):
        raise RuntimeError("Build and verify proxy bundle before pruning")
    selected = {path.resolve() for path in selected_checkpoints()}
    removable = [
        path
        for path in sorted((OUTPUT / "runs").rglob("*.pt"))
        if path.resolve() not in selected
    ]
    records = [
        {
            "path": str(path.relative_to(PROJECT)),
            "bytes": path.stat().st_size,
        }
        for path in removable
    ]
    payload = {
        "status": "PLANNED",
        "reason": "remove non-selected proxy epoch/intermediate checkpoints",
        "selected_checkpoints_preserved": len(selected),
        "files_removed": len(records),
        "bytes_removed": sum(item["bytes"] for item in records),
        "removed": records,
    }
    atomic_json(CLEANUP_MANIFEST, payload)
    for path in removable:
        path.unlink()
    payload["status"] = "COMPLETE"
    payload["selected_checkpoints_present_after_cleanup"] = sum(
        path.is_file() for path in selected_checkpoints()
    )
    atomic_json(CLEANUP_MANIFEST, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune", action="store_true")
    args = parser.parse_args()
    result = build_bundle()
    if args.prune:
        result["cleanup"] = prune()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

