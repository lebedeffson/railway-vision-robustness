from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit_final_practice import canonical_path, load_manifest


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "outputs/final_practice/unified_diagnostics_val_raw.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/final_practice/audit"
MANIFEST = PROJECT_DIR / "data/yolo_osdar23/manifest.csv"
SIGNATURE = [
    "attack", "adaptive", "epsilon", "steps", "restart", "seed", "defense", "layer",
]


def expected_rows(config: dict) -> int:
    total = 0
    for attack, _epsilon, _steps, adaptive in config["conditions"]:
        restarts = 1 if attack == "fgsm" else len(config["seeds"])
        defenses = (
            [name for name in config["defenses"] if name in {"none", "tnorm"}]
            if adaptive else config["defenses"]
        )
        total += restarts * len(defenses) * 3
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Deadline matrix integrity gate")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config or args.input.with_suffix(".json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = pd.read_csv(args.input, low_memory=False)
    required = {"sequence_id", "image_path", "split", *SIGNATURE}
    missing_columns = sorted(required - set(data))
    expected_images = {
        canonical_path(row["image_path"])
        for row in load_manifest(MANIFEST)
        if row["split"] == args.split
    }
    data["canonical_image"] = data["image_path"].map(canonical_path)
    actual_images = set(data["canonical_image"])
    per_image = data.groupby("canonical_image").size()
    rows_per_image = expected_rows(config)
    incomplete = per_image[per_image != rows_per_image]
    duplicate_mask = data.duplicated(["canonical_image", *SIGNATURE], keep=False)
    reference = data[data["canonical_image"] == per_image.index[0]][SIGNATURE].drop_duplicates()
    reference_keys = {tuple(row) for row in reference.itertuples(index=False, name=None)}
    missing_rows: list[dict[str, object]] = []
    for image, group in data.groupby("canonical_image"):
        existing = {
            tuple(row) for row in group[SIGNATURE].drop_duplicates().itertuples(index=False, name=None)
        }
        for signature in sorted(reference_keys - existing, key=str):
            missing_rows.append({
                "image_path": image,
                **dict(zip(SIGNATURE, signature, strict=True)),
            })
    numeric = data.select_dtypes(include=[np.number])
    nan_counts = {name: int(value) for name, value in numeric.isna().sum().items() if value}
    inf_counts = {
        name: int(np.isinf(numeric[name].to_numpy(float)).sum())
        for name in numeric
        if np.isinf(numeric[name].to_numpy(float)).any()
    }
    sequence_by_image = data.groupby("canonical_image")["sequence_id"].nunique()
    model = Path(config["model"]).resolve()
    official = (
        PROJECT_DIR / "outputs/training/yolo11m_baseline_stage2/weights/best.pt"
    ).resolve()
    payload = {
        "status": "PASS",
        "input": str(args.input),
        "split": args.split,
        "rows_expected": len(expected_images) * rows_per_image,
        "rows_actual": len(data),
        "rows_per_image_expected": rows_per_image,
        "images_expected": len(expected_images),
        "images_actual": len(actual_images),
        "missing_images": sorted(expected_images - actual_images),
        "unexpected_images": sorted(actual_images - expected_images),
        "incomplete_images": {str(key): int(value) for key, value in incomplete.items()},
        "duplicate_rows": int(duplicate_mask.sum()),
        "missing_condition_rows": len(missing_rows),
        "missing_columns": missing_columns,
        "nan_counts": nan_counts,
        "inf_counts": inf_counts,
        "images_with_multiple_sequence_ids": int((sequence_by_image != 1).sum()),
        "sequences": int(data["sequence_id"].nunique()),
        "checkpoint": str(model),
        "checkpoint_is_official_stage2_best": model == official,
    }
    hard_failures = (
        payload["rows_expected"] != payload["rows_actual"]
        or payload["missing_images"] or payload["unexpected_images"]
        or payload["incomplete_images"] or payload["duplicate_rows"]
        or payload["missing_condition_rows"] or missing_columns or inf_counts
        or payload["images_with_multiple_sequence_ids"]
        or not payload["checkpoint_is_official_stage2_best"]
    )
    payload["status"] = "FAIL" if hard_failures else "PASS"
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(missing_rows, columns=["image_path", *SIGNATURE]).to_csv(
        args.output / "missing_conditions.csv", index=False
    )
    (args.output / "matrix_integrity.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if hard_failures:
        raise SystemExit("Matrix integrity gate failed")


if __name__ == "__main__":
    main()
