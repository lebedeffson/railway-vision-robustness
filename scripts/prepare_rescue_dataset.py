from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

from rescue_common import (
    PROJECT_DIR,
    OUTPUT_ROOT,
    assert_test_sealed,
    atomic_json,
    completed_marker,
    ensure_branch,
    load_protocol,
    sha256,
)


DESTINATION = PROJECT_DIR / "data/yolo_osdar23_rescue_v1"
AUDIT = OUTPUT_ROOT / "audit"


def link_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise RuntimeError(f"Rescue image target differs: {destination}")
    os.link(source, destination)


def unique_lines(path: Path) -> tuple[list[str], int]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: set[str] = set()
    result = []
    removed = 0
    for line in lines:
        if line in seen:
            removed += 1
            continue
        seen.add(line)
        result.append(line)
    return result, removed


def main() -> None:
    ensure_branch()
    assert_test_sealed()
    protocol = load_protocol()
    split_audit = json.loads((AUDIT / "split_audit.json").read_text(encoding="utf-8"))
    visual = json.loads((AUDIT / "visual_audit.json").read_text(encoding="utf-8"))
    if split_audit.get("status") != "PASS" or visual.get("visual_audit_passed") is not True:
        raise RuntimeError("Rescue dataset repair requires accepted data and visual audits")
    source_manifest = PROJECT_DIR / protocol["split_manifest"]
    manifest = pd.read_csv(source_manifest)
    output_rows = []
    removed_rows = []
    for row in manifest.itertuples(index=False):
        split = str(row.split)
        source_image = Path(row.output_image)
        source_label = Path(row.output_label)
        destination_image = DESTINATION / "images" / split / source_image.name
        destination_label = DESTINATION / "labels" / split / source_label.name
        link_image(source_image, destination_image)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        lines, removed = unique_lines(source_label)
        if split != "train" and removed:
            raise RuntimeError(f"Validation/test duplicate repair was not pre-authorized: {source_label}")
        content = "\n".join(lines) + ("\n" if lines else "")
        if destination_label.exists() and destination_label.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Existing repaired label differs: {destination_label}")
        destination_label.write_text(content, encoding="utf-8")
        if removed:
            removed_rows.append({
                "split": split, "grouped_scene_id": row.grouped_scene_id,
                "source_label": str(source_label), "destination_label": str(destination_label),
                "removed_exact_duplicate_lines": removed,
                "source_sha256": sha256(source_label),
                "destination_sha256": sha256(destination_label),
            })
        output_rows.append({
            **row._asdict(),
            "output_image": str(destination_image.resolve()),
            "output_label": str(destination_label.resolve()),
            "annotations": len(lines),
            "rescue_label_repair": "removed_exact_duplicate" if removed else "unchanged",
        })
    if sum(row["removed_exact_duplicate_lines"] for row in removed_rows) != 1:
        raise RuntimeError(f"Expected exactly one duplicate label line repair, found {removed_rows}")
    output_manifest = DESTINATION / "manifest.csv"
    pd.DataFrame(output_rows).to_csv(output_manifest, index=False)
    pd.DataFrame(removed_rows).to_csv(AUDIT / "label_repairs.csv", index=False)
    source_yaml = yaml.safe_load((PROJECT_DIR / protocol["dataset"]).read_text(encoding="utf-8"))
    source_yaml.update({
        "path": str(DESTINATION.resolve()),
        "train": "images/train", "val": "images/val", "test": "images/test",
    })
    data_yaml = DESTINATION / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(source_yaml, sort_keys=False), encoding="utf-8")
    repair = {
        "status": "PASS",
        "source_split_manifest": str(source_manifest.resolve()),
        "source_split_manifest_sha256": sha256(source_manifest),
        "rescue_manifest": str(output_manifest.resolve()),
        "rescue_manifest_sha256": sha256(output_manifest),
        "rescue_data_yaml": str(data_yaml.resolve()),
        "rescue_data_yaml_sha256": sha256(data_yaml),
        "split_assignments_changed": False,
        "images_changed": False,
        "labels_changed": 1,
        "duplicate_label_lines_removed": 1,
        "test_labels_changed": False,
        "test_model_evaluation_performed": False,
        "repairs": removed_rows,
    }
    atomic_json(AUDIT / "data_repair.json", repair)
    completed_marker(
        OUTPUT_ROOT / "dataset",
        inputs=[source_manifest, PROJECT_DIR / protocol["dataset"]],
        outputs=[output_manifest, data_yaml, AUDIT / "label_repairs.csv", AUDIT / "data_repair.json"],
        extra={"stage": "rescue_dataset_repair", "test_model_evaluation_performed": False},
    )
    print(json.dumps(repair, indent=2))


if __name__ == "__main__":
    main()
