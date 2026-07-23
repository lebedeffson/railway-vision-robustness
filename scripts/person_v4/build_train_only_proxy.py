from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve()
DEFAULT_PROJECT = HERE.parents[2]
CONFIG_RELATIVE = Path("configs/person_v4_train_only_proxy.yaml")
LOCKED_CODE = (
    Path("scripts/person_v4/losses.py"),
    Path("scripts/person_v4/train.py"),
    Path("scripts/person_v4/proxy.py"),
    Path("scripts/person_v4/swad.py"),
    Path("scripts/person_v4/build_train_only_proxy.py"),
    Path("scripts/person_v4/run_proxy_cpu_checks.py"),
    Path("tests/test_person_v4_math.py"),
    Path("tests/test_person_v4_proxy.py"),
)


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


def centered_block(rows: list[dict[str, str]], cap: int) -> list[dict[str, str]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("subsequence_id", ""),
            row.get("frame_id", ""),
            row.get("source_image", ""),
        ),
    )
    if len(ordered) <= cap:
        return ordered
    start = (len(ordered) - int(cap)) // 2
    return ordered[start : start + int(cap)]


def freeze(
    project_root: Path,
    data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    config_path = project_root / CONFIG_RELATIVE
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = protocol["inputs"]
    resolved = {
        "development_manifest": data_root
        / inputs["development_manifest"],
        "outer_folds": data_root / inputs["outer_folds"],
        "tile_manifest": data_root / inputs["tile_manifest"],
        "initialization": data_root / inputs["initialization"],
    }
    for key, path in resolved.items():
        expected = inputs[f"{key}_sha256"]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen proxy input changed: {key}: {actual} != {expected}"
            )
    folds = json.loads(
        resolved["outer_folds"].read_text(encoding="utf-8")
    )
    outer_fold = str(inputs["outer_fold"])
    forbidden = set(folds["folds"][outer_fold])
    declared_forbidden = set(
        protocol["forbidden"]["outer_fold0_heldout_scenes"]
    )
    if forbidden != declared_forbidden:
        raise RuntimeError("Outer fold-0 exclusion no longer matches protocol")
    allowed = {
        scene
        for key, scenes in folds["folds"].items()
        if key != outer_fold
        for scene in scenes
    }
    requested = {
        scene
        for split in protocol["proxy_splits"]
        for scene in (
            *split["source_scenes"],
            split["heldout_scene"],
        )
    }
    if not requested <= allowed or requested & forbidden:
        raise RuntimeError("Proxy split escaped outer fold-0 training scope")

    by_scene: dict[str, list[dict[str, str]]] = {
        scene: [] for scene in requested
    }
    manifest_path = resolved["development_manifest"]
    with manifest_path.open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            scene = row["grouped_scene_id"]
            if scene in by_scene:
                by_scene[scene].append(row)
    if any(not rows for rows in by_scene.values()):
        missing = sorted(
            scene for scene, rows in by_scene.items() if not rows
        )
        raise RuntimeError(f"Proxy scene has no development rows: {missing}")

    selection = protocol["selection"]
    output_rows: list[dict[str, str]] = []
    selected_role_paths: list[tuple[str, str, str, str]] = []
    split_payload = []
    for split in protocol["proxy_splits"]:
        source_scenes = list(split["source_scenes"])
        heldout = str(split["heldout_scene"])
        if heldout in source_scenes or len(set(source_scenes)) != 4:
            raise RuntimeError("Proxy split role contract is invalid")
        counts: Counter[str] = Counter()
        for scene in source_scenes:
            selected = centered_block(
                by_scene[scene],
                int(selection["source_frame_cap_per_scene"]),
            )
            counts[f"source:{scene}"] = len(selected)
            for row in selected:
                selected_role_paths.append(
                    (split["id"], "source", scene, row["output_image"])
                )
                output_rows.append(
                    {
                        "proxy_split": split["id"],
                        "role": "source",
                        "grouped_scene_id": scene,
                        "subsequence_id": row["subsequence_id"],
                        "frame_id": row["frame_id"],
                        "frame_stem": Path(row["output_image"]).stem,
                    }
                )
        selected = centered_block(
            by_scene[heldout], int(selection["heldout_frame_cap"])
        )
        counts[f"heldout:{heldout}"] = len(selected)
        for row in selected:
            selected_role_paths.append(
                (split["id"], "heldout", heldout, row["output_image"])
            )
            output_rows.append(
                {
                    "proxy_split": split["id"],
                    "role": "heldout",
                    "grouped_scene_id": heldout,
                    "subsequence_id": row["subsequence_id"],
                    "frame_id": row["frame_id"],
                    "frame_stem": Path(row["output_image"]).stem,
                }
            )
        split_payload.append(
            {
                "id": split["id"],
                "source_scenes": source_scenes,
                "heldout_scene": heldout,
                "frame_counts": dict(sorted(counts.items())),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    frame_manifest = output_root / "proxy_frame_manifest.csv"
    with frame_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    selected_roles = set(selected_role_paths)
    by_source: dict[str, list[dict[str, str]]] = {}
    with resolved["tile_manifest"].open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            source = row["source_image"]
            if any(key[3] == source for key in selected_roles):
                by_source.setdefault(source, []).append(row)
    tile_rows: list[dict[str, str]] = []
    missing_tile_sources = []
    for split_id, role, scene, source in sorted(selected_roles):
        matches = by_source.get(source, [])
        if not matches:
            missing_tile_sources.append(source)
            continue
        for row in sorted(matches, key=lambda value: value["tile_id"]):
            tile_rows.append(
                {
                    "proxy_split": split_id,
                    "role": role,
                    "grouped_scene_id": scene,
                    "frame_stem": Path(source).stem,
                    "tile_id": row["tile_id"],
                    "tile_stem": Path(row["tile_image"]).stem,
                }
            )
    if missing_tile_sources:
        raise RuntimeError(
            "Selected proxy frames are missing frozen tiles: "
            f"{missing_tile_sources[:3]}"
        )
    tile_manifest = output_root / "proxy_tile_manifest.csv"
    with tile_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tile_rows[0]))
        writer.writeheader()
        writer.writerows(tile_rows)
    splits_path = output_root / "proxy_splits.json"
    atomic_json(
        splits_path,
        {
            "protocol_id": protocol["protocol_id"],
            "selection": selection,
            "splits": split_payload,
            "test_used": False,
            "external_fold0_used": False,
        },
    )
    audit = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "outer_fold": int(outer_fold),
        "outer_fold0_heldout_scenes": sorted(forbidden),
        "outer_fold0_heldout_images_opened": False,
        "outer_fold0_heldout_labels_opened": False,
        "official_validation_opened": False,
        "test_opened": False,
        "selected_scene_count": len(requested),
        "selected_row_count": len(output_rows),
        "selected_tile_count": len(tile_rows),
        "selected_frames_without_tiles": len(missing_tile_sources),
        "proxy_split_count": len(split_payload),
        "source_scenes_per_split": 4,
        "heldout_scenes_per_split": 1,
        "input_hashes": {
            key: sha256(path) for key, path in resolved.items()
        },
    }
    audit_path = output_root / "isolation_audit.json"
    atomic_json(audit_path, audit)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
    ).strip()
    lock = {
        "status": "LOCKED",
        "protocol_id": protocol["protocol_id"],
        "git_commit_at_freeze": commit,
        "protocol_path": str(CONFIG_RELATIVE),
        "protocol_sha256": sha256(config_path),
        "proxy_splits_sha256": sha256(splits_path),
        "proxy_frame_manifest_sha256": sha256(frame_manifest),
        "proxy_tile_manifest_sha256": sha256(tile_manifest),
        "isolation_audit_sha256": sha256(audit_path),
        "code_sha256": {
            str(path): sha256(project_root / path)
            for path in LOCKED_CODE
        },
        "test_sealed": True,
        "active_A3_may_be_interrupted": False,
    }
    atomic_json(output_root / "protocol_lock.json", lock)
    atomic_json(
        output_root / "execution_status.json",
        {
            "status": "READY_NOT_STARTED",
            "reason": "active_A3_fold0_must_finish_unchanged",
            "GPU_proxy_started": False,
            "active_A3_interrupted": False,
            "test_opened": False,
            "next_allowed_action": (
                "run_CPU_checks_then_wait_for_active_A3_verdict"
            ),
        },
    )
    return {"lock": lock, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=DEFAULT_PROJECT
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    data_root = (args.data_root or project_root).resolve()
    output_root = (
        args.output_root
        or project_root
        / "protocols/person_v4_train_only_proxy_v1"
    ).resolve()
    print(
        json.dumps(
            freeze(project_root, data_root, output_root),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
