from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scripts.person_v3.common import (
    AUDIT_ROOT,
    DATASET_ROOT,
    FOLDS_PATH,
    LOCK_PATH,
    OUTPUT_ROOT,
    PROJECT_DIR,
    PROTOCOL_PATH,
    PROTOCOL_ROOT,
    assert_test_sealed,
    atomic_json,
    atomic_text,
    load_protocol,
    sha256,
)


M4_TILE_MANIFEST = (
    PROJECT_DIR / "outputs/canonical_m4/tiling_audit/tile_manifest.csv"
)
M4_OBJECT_AUDIT = (
    PROJECT_DIR / "outputs/canonical_m4/tiling_audit/augmentation_object_audit.csv"
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def filtered_person_label(source: Path) -> str:
    rows = [
        line
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and int(float(line.split()[0])) == 0
    ]
    return "\n".join(rows) + ("\n" if rows else "")


def person_scene_matrix(
    development: pd.DataFrame,
    objects: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for scene, frames in development.groupby("grouped_scene_id"):
        frame_paths = {
            str(Path(value).resolve()) for value in frames["output_image"]
        }
        selected = objects[
            objects["source_image"].astype(str).isin(frame_paths)
            & objects["class_id"].eq(0)
        ].copy()
        selected = (
            selected.sort_values("visible_fraction", ascending=False)
            .drop_duplicates(["source_image", "source_gt_id"])
        )
        area = (
            selected["tile_bbox_width_px"]
            * selected["tile_bbox_height_px"]
            / (4112 * 2504)
        )
        minimum_side = selected[
            [
                "model_bbox_width_px_after_augmentation",
                "model_bbox_height_px_after_augmentation",
            ]
        ].min(axis=1)
        rows.append({
            "grouped_scene_id": str(scene),
            "frames": int(len(frames)),
            "person_positive_frames": int(
                selected["source_image"].nunique()
            ),
            "person_negative_frames": int(
                len(frames) - selected["source_image"].nunique()
            ),
            "person_GT": int(len(selected)),
            "small_person": int((area < 0.001).sum()),
            "medium_person": int(((area >= 0.001) & (area < 0.01)).sum()),
            "large_person": int((area >= 0.01).sum()),
            "person_below_2px": int((minimum_side < 2).sum()),
            "person_below_4px": int((minimum_side < 4).sum()),
            "person_below_8px": int((minimum_side < 8).sum()),
            "person_border_GT": int(
                selected["visible_fraction"].lt(1.0).sum()
            ),
        })
    return pd.DataFrame(rows).sort_values("grouped_scene_id").reset_index(drop=True)


def choose_folds(
    matrix: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[list[list[str]], float]:
    positive = matrix[matrix["person_GT"].gt(0)].reset_index(drop=True)
    negative = matrix[matrix["person_GT"].eq(0)].reset_index(drop=True)
    if len(positive) != 10 or len(negative) != 5:
        raise RuntimeError(
            f"Expected 10 positive and 5 negative scenes, got "
            f"{len(positive)}/{len(negative)}"
        )
    columns = [
        "frames", "person_GT", "small_person", "medium_person", "large_person"
    ]
    positive_values = positive[columns].to_numpy(float)
    negative_values = negative[columns].to_numpy(float)
    totals = np.maximum(
        positive_values.sum(axis=0) + negative_values.sum(axis=0), 1.0
    )
    config = protocol["folds"]
    rng = np.random.default_rng(int(config["seed"]))
    best: tuple[float, list[list[str]]] | None = None
    for _ in range(int(config["candidate_partitions"])):
        positive_indices = rng.permutation(10).reshape(5, 2)
        negative_indices = rng.permutation(5)
        fold_values = positive_values[positive_indices].sum(axis=1)
        fold_values += negative_values[negative_indices]
        score = float((fold_values / totals).std(axis=0).mean())
        folds = [
            sorted([
                *positive.iloc[pair]["grouped_scene_id"].astype(str).tolist(),
                str(negative.iloc[negative_indices[index]]["grouped_scene_id"]),
            ])
            for index, pair in enumerate(positive_indices)
        ]
        folds.sort()
        candidate = (score, folds)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[0]


def write_dataset_view(
    development: pd.DataFrame,
    folds: list[list[str]],
) -> pd.DataFrame:
    tile_manifest = pd.read_csv(M4_TILE_MANIFEST)
    development_scenes = set(development["grouped_scene_id"].astype(str))
    tiles = tile_manifest[
        tile_manifest["grouped_scene_id"].astype(str).isin(development_scenes)
    ].copy()
    images = DATASET_ROOT / "images/development"
    labels = DATASET_ROOT / "labels/development"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    rows = []
    for tile in tiles.itertuples(index=False):
        destination_image = images / Path(tile.tile_image).name
        destination_label = labels / f"{Path(tile.tile_image).stem}.txt"
        if not destination_image.exists():
            os.symlink(Path(tile.tile_image).resolve(), destination_image)
        payload = filtered_person_label(Path(tile.tile_label))
        if not destination_label.is_file() or (
            destination_label.read_text(encoding="utf-8") != payload
        ):
            atomic_text(destination_label, payload)
        rows.append({
            "grouped_scene_id": str(tile.grouped_scene_id),
            "source_image": str(Path(tile.source_image).resolve()),
            "tile_id": str(tile.tile_id),
            # Do not resolve the image symlink: Ultralytics derives the label
            # path by replacing /images/ with /labels/ in this person-only view.
            "tile_image": str(destination_image.absolute()),
            "tile_label": str(destination_label.absolute()),
            "person_GT": sum(bool(line.strip()) for line in payload.splitlines()),
        })
    frame = pd.DataFrame(rows)
    fold_root = DATASET_ROOT / "folds"
    all_scenes = set(development_scenes)
    for fold_id, heldout in enumerate(folds):
        root = fold_root / f"fold_{fold_id}"
        train_scenes = all_scenes - set(heldout)
        train = frame[frame["grouped_scene_id"].isin(train_scenes)]
        validation = frame[frame["grouped_scene_id"].isin(heldout)]
        train_list = root / "train.txt"
        validation_list = root / "val.txt"
        atomic_text(
            train_list,
            "\n".join(train["tile_image"].astype(str)) + "\n",
        )
        atomic_text(
            validation_list,
            "\n".join(validation["tile_image"].astype(str)) + "\n",
        )
        data = {
            "path": str(DATASET_ROOT.resolve()),
            "train": str(train_list.resolve()),
            "val": str(validation_list.resolve()),
            "nc": 1,
            "names": {0: "person"},
        }
        atomic_text(root / "data.yaml", yaml.safe_dump(data, sort_keys=False))
    atomic_text(
        DATASET_ROOT / "all_development.txt",
        "\n".join(frame["tile_image"].astype(str)) + "\n",
    )
    return frame


def main() -> None:
    protocol = load_protocol()
    assert_test_sealed()
    if sha256(PROJECT_DIR / protocol["data"]["source_manifest"]) != protocol[
        "data"
    ]["source_manifest_sha256"]:
        raise RuntimeError("Person v3 source manifest hash mismatch")
    if sha256(PROJECT_DIR / protocol["pipeline"]["initialization"]) != protocol[
        "pipeline"
    ]["initialization_sha256"]:
        raise RuntimeError("Person v3 initialization hash mismatch")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol["parent_commit"], "HEAD"],
        cwd=PROJECT_DIR,
    ).returncode != 0:
        raise RuntimeError("Person v3 parent commit is not an ancestor")
    source = pd.read_csv(PROJECT_DIR / protocol["data"]["source_manifest"])
    development = source[
        source["split"].isin(protocol["data"]["development_source_splits"])
    ].copy()
    sealed_test = source[source["split"].eq("test")].copy()
    if development["grouped_scene_id"].nunique() != 15:
        raise RuntimeError("Person v3 development scene count mismatch")
    if sealed_test["grouped_scene_id"].nunique() != 5:
        raise RuntimeError("Person v3 sealed test scene count mismatch")
    development["person_v3_split"] = "development"
    sealed_test["person_v3_split"] = "sealed_test"
    atomic_csv(development, PROTOCOL_ROOT / "development_manifest.csv")
    atomic_csv(sealed_test, PROTOCOL_ROOT / "sealed_test_manifest.csv")
    objects = pd.read_csv(M4_OBJECT_AUDIT)
    matrix = person_scene_matrix(development, objects)
    folds, score = choose_folds(matrix, protocol)
    all_scenes = set(matrix["grouped_scene_id"])
    support_rows = []
    for fold_id, heldout in enumerate(folds):
        held = matrix[matrix["grouped_scene_id"].isin(heldout)]
        train = matrix[matrix["grouped_scene_id"].isin(all_scenes - set(heldout))]
        support_rows.append({
            "fold": fold_id,
            "heldout_scenes": "|".join(heldout),
            "heldout_person_positive_scenes": int(held["person_GT"].gt(0).sum()),
            "train_person_positive_scenes": int(train["person_GT"].gt(0).sum()),
            "heldout_person_GT": int(held["person_GT"].sum()),
            "train_person_GT": int(train["person_GT"].sum()),
            "heldout_small_person": int(held["small_person"].sum()),
            "heldout_medium_person": int(held["medium_person"].sum()),
            "heldout_large_person": int(held["large_person"].sum()),
            "scene_intersection": len(set(heldout) & (all_scenes - set(heldout))),
        })
    support = pd.DataFrame(support_rows)
    valid = bool(
        support["heldout_person_positive_scenes"].ge(1).all()
        and support["train_person_positive_scenes"].ge(8).all()
        and support["scene_intersection"].eq(0).all()
    )
    if not valid:
        raise RuntimeError("Person v3 constrained scene folds are invalid")
    atomic_csv(matrix, AUDIT_ROOT / "class_scene_matrix.csv")
    atomic_csv(support, AUDIT_ROOT / "fold_support.csv")
    atomic_json(FOLDS_PATH, {
        "protocol_id": protocol["protocol_id"],
        "seed": protocol["folds"]["seed"],
        "search_candidates": protocol["folds"]["candidate_partitions"],
        "balance_score": score,
        "folds": {str(index): value for index, value in enumerate(folds)},
        "test_used": False,
    })
    tiles = write_dataset_view(development, folds)
    audit = {
        "status": "PASS",
        "protocol_id": protocol["protocol_id"],
        "development_frames": int(len(development)),
        "development_scenes": int(development["grouped_scene_id"].nunique()),
        "sealed_test_frames": int(len(sealed_test)),
        "sealed_test_scenes": int(sealed_test["grouped_scene_id"].nunique()),
        "test_labels_read": False,
        "person_positive_scenes": int(matrix["person_GT"].gt(0).sum()),
        "person_negative_scenes": int(matrix["person_GT"].eq(0).sum()),
        "person_GT": int(matrix["person_GT"].sum()),
        "person_positive_frames": int(matrix["person_positive_frames"].sum()),
        "person_negative_frames": int(matrix["person_negative_frames"].sum()),
        "tiles": int(len(tiles)),
        "person_positive_tiles": int(tiles["person_GT"].gt(0).sum()),
        "person_background_tiles": int(tiles["person_GT"].eq(0).sum()),
        "folds_valid": valid,
        "original_annotations_modified": False,
    }
    atomic_json(AUDIT_ROOT / "dataset_view_audit.json", audit)
    lock = {
        "status": "LOCKED",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "parent_commit": protocol["parent_commit"],
        "development_manifest_sha256": sha256(
            PROTOCOL_ROOT / "development_manifest.csv"
        ),
        "sealed_test_manifest_sha256": sha256(
            PROTOCOL_ROOT / "sealed_test_manifest.csv"
        ),
        "folds_sha256": sha256(FOLDS_PATH),
        "dataset_view_audit_sha256": sha256(
            AUDIT_ROOT / "dataset_view_audit.json"
        ),
        "test_sealed": True,
        "training_allowed": True,
    }
    atomic_json(LOCK_PATH, lock)
    print(json.dumps({**audit, "protocol_lock": lock}, indent=2))


if __name__ == "__main__":
    main()
