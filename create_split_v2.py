from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE = PROJECT_DIR / "data/yolo_osdar23"
DESTINATION = PROJECT_DIR / "data/yolo_osdar23_v2"
OUTPUT = PROJECT_DIR / "outputs/canonical_v2/split"
SEED = 20260722
TARGET_COUNTS = {"train": 10, "val": 5, "test": 5}
SMALL_AREA = 0.001
MEDIUM_AREA = 0.01


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def frame_objects(path: Path) -> tuple[Counter[int], Counter[str], int]:
    classes: Counter[int] = Counter(); sizes: Counter[str] = Counter(); total = 0
    if not path.is_file():
        return classes, sizes, total
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(fields[0]); area = float(fields[3]) * float(fields[4])
        size = "small" if area < SMALL_AREA else "medium" if area < MEDIUM_AREA else "large"
        classes[class_id] += 1; sizes[size] += 1; total += 1
    return classes, sizes, total


def group_statistics(manifest: pd.DataFrame) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for sequence_id, rows in manifest.groupby("sequence_id", sort=True):
        classes: Counter[int] = Counter(); sizes: Counter[str] = Counter(); objects = 0
        for image_name in rows["output_image"]:
            counts, size_counts, total = frame_objects(label_path(Path(image_name)))
            classes.update(counts); sizes.update(size_counts); objects += total
        result[str(sequence_id)] = {
            "frames": len(rows), "objects": objects, "classes": classes,
            "sizes": sizes, "small_fraction": sizes["small"] / max(objects, 1),
        }
    return result


def split_score(candidate: dict[str, list[str]], stats: dict[str, dict]) -> float:
    all_classes = sorted({key for value in stats.values() for key in value["classes"]})
    total_frames = sum(value["frames"] for value in stats.values())
    total_objects = Counter()
    for value in stats.values(): total_objects.update(value["classes"])
    global_small = sum(v["small_fraction"] * v["objects"] for v in stats.values()) / max(sum(v["objects"] for v in stats.values()), 1)
    score = 0.0
    for split, groups in candidate.items():
        ratio = TARGET_COUNTS[split] / 20
        frames = sum(stats[group]["frames"] for group in groups)
        score += 3.0 * ((frames - total_frames * ratio) / max(total_frames * ratio, 1)) ** 2
        objects = Counter()
        for group in groups: objects.update(stats[group]["classes"])
        for class_id in all_classes:
            if objects[class_id] == 0:
                score += 10_000.0
            else:
                target = total_objects[class_id] * ratio
                score += ((objects[class_id] - target) / max(target, 1)) ** 2
        object_count = sum(stats[group]["objects"] for group in groups)
        small = sum(stats[group]["small_fraction"] * stats[group]["objects"] for group in groups) / max(object_count, 1)
        score += 2.0 * (small - global_small) ** 2
    return score


def find_split(stats: dict[str, dict], iterations: int) -> tuple[dict[str, list[str]], float]:
    groups = sorted(stats)
    if len(groups) != 20:
        raise RuntimeError(f"Expected 20 grouped independent scenes, found {len(groups)}")
    randomizer = random.Random(SEED); best = None; best_score = float("inf")
    for _ in range(iterations):
        shuffled = groups.copy(); randomizer.shuffle(shuffled)
        candidate = {
            "train": sorted(shuffled[:10]), "val": sorted(shuffled[10:15]),
            "test": sorted(shuffled[15:20]),
        }
        score = split_score(candidate, stats)
        if score < best_score: best, best_score = candidate, score
    if best is None: raise RuntimeError("Split search failed")
    return best, best_score


def link(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    try:
        os.link(source, destination); return "hardlink"
    except OSError:
        shutil.copy2(source, destination); return "copy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create model-result-independent 10/5/5 grouped-scene split")
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen_manifest = OUTPUT / "split_v2_manifest.csv"
    frozen_summary = OUTPUT / "split_v2_summary.json"
    frozen_hash = OUTPUT / "split_v2_hash.txt"
    if not args.force and all(path.is_file() for path in (
        frozen_manifest, frozen_summary, frozen_hash, DESTINATION / "data.yaml"
    )):
        summary = json.loads(frozen_summary.read_text(encoding="utf-8"))
        expected = frozen_hash.read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256(frozen_manifest.read_bytes()).hexdigest()
        if summary.get("status") != "PASS" or actual != expected:
            raise RuntimeError("Frozen split v2 artifacts failed idempotence verification")
        print(json.dumps({**summary, "resume": "verified_frozen_split"}, indent=2))
        return
    manifest = pd.read_csv(SOURCE / "manifest.csv")
    stats = group_statistics(manifest)
    split, score = find_split(stats, args.iterations)
    assignment = {group: name for name, groups in split.items() for group in groups}
    rows = []
    for row in manifest.itertuples(index=False):
        target_split = assignment[str(row.sequence_id)]
        source_image = Path(row.output_image); source_label = label_path(source_image)
        target_image = DESTINATION / "images" / target_split / source_image.name
        target_label = DESTINATION / "labels" / target_split / source_label.name
        image_method = link(source_image, target_image); link(source_label, target_label)
        rows.append({
            **row._asdict(), "original_split": row.split, "split": target_split,
            "subsequence_id": str(row.sequence),
            "grouped_scene_id": str(row.sequence_id),
            "output_image": str(target_image), "output_label": str(target_label),
            "transfer_method": image_method,
        })
    output_manifest = pd.DataFrame(rows)
    output_manifest.to_csv(DESTINATION / "manifest.csv", index=False)
    output_manifest.to_csv(OUTPUT / "split_v2_manifest.csv", index=False)
    source_yaml = yaml.safe_load((SOURCE / "data.yaml").read_text(encoding="utf-8"))
    source_yaml.update({
        "path": str(DESTINATION.resolve()), "train": "images/train",
        "val": "images/val", "test": "images/test",
    })
    (DESTINATION / "data.yaml").write_text(yaml.safe_dump(source_yaml, sort_keys=False), encoding="utf-8")
    intersections = {
        f"{left}_{right}": sorted(set(split[left]) & set(split[right]))
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    class_ids = sorted({class_id for value in stats.values() for class_id in value["classes"]})
    class_counts = {}
    class_scene_counts = {}
    size_object_counts = {}
    scene_contributions = {}
    missing_classes = {}
    for split_name, groups in split.items():
        counts = Counter()
        sizes: Counter[str] = Counter()
        for group in groups: counts.update(stats[group]["classes"])
        for group in groups: sizes.update(stats[group]["sizes"])
        class_counts[split_name] = {str(class_id): counts[class_id] for class_id in class_ids}
        class_scene_counts[split_name] = {
            str(class_id): sum(stats[group]["classes"][class_id] > 0 for group in groups)
            for class_id in class_ids
        }
        size_object_counts[split_name] = {
            name: int(sizes[name]) for name in ("small", "medium", "large")
        }
        split_frames = sum(stats[group]["frames"] for group in groups)
        split_objects = sum(stats[group]["objects"] for group in groups)
        scene_contributions[split_name] = [
            {
                "grouped_scene_id": group,
                "frames": stats[group]["frames"],
                "frame_fraction": stats[group]["frames"] / max(split_frames, 1),
                "objects": stats[group]["objects"],
                "object_fraction": stats[group]["objects"] / max(split_objects, 1),
            }
            for group in groups
        ]
        missing_classes[split_name] = [class_id for class_id in class_ids if counts[class_id] == 0]
    summary = {
        "status": (
            "PASS" if not any(intersections.values()) and not any(missing_classes.values())
            else "FAIL"
        ),
        "selection_uses_model_or_attack_results": False, "random_seed": SEED,
        "search_iterations": args.iterations, "score": score,
        "scene_counts": {key: len(value) for key, value in split.items()},
        "frame_counts": output_manifest.groupby("split").size().to_dict(),
        "object_counts": output_manifest.groupby("split")["annotations"].sum().to_dict(),
        "split_sequences": split, "intersections": intersections,
        "grouping_contract": {
            "independent_unit": "grouped_scene_id",
            "sequence_id_aliases_grouped_scene_id": True,
            "raw_unit": "subsequence_id",
        },
        "class_object_counts": class_counts,
        "class_grouped_scene_counts": class_scene_counts,
        "size_object_counts": size_object_counts,
        "scene_contributions": scene_contributions,
        "missing_classes": missing_classes,
        "small_object_area_ratio": SMALL_AREA,
        "medium_object_area_ratio": MEDIUM_AREA,
        "day_night_weather": "not_used_metadata_unavailable_in_frozen_manifest",
    }
    (OUTPUT / "split_v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256((OUTPUT / "split_v2_manifest.csv").read_bytes()).hexdigest()
    (OUTPUT / "split_v2_hash.txt").write_text(f"{digest}  split_v2_manifest.csv\n", encoding="utf-8")
    if summary["status"] != "PASS" or min(summary["scene_counts"].values()) < 5:
        raise RuntimeError("Split v2 acceptance gate failed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
