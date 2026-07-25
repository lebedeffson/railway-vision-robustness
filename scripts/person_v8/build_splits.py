from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

from person_v8.common import (
    OUTPUT_ROOT,
    ROOT,
    assert_test_sealed,
    atomic_json,
    load_config,
    read_csv,
    write_csv,
)


def _numeric(row: dict[str, str], field: str) -> int:
    return int(float(row.get(field, "0") or 0))


def choose_screening_roles(
    scene_rows: list[dict[str, str]],
) -> tuple[str, str, list[str]]:
    if len(scene_rows) < 3:
        raise ValueError("At least three new scenes are required")
    totals = {
        field: sum(_numeric(row, field) for row in scene_rows)
        for field in ("frames", "person_gt", "small_gt")
    }
    target = {field: value / len(scene_rows) for field, value in totals.items()}

    def distance(row: dict[str, str]) -> float:
        return sum(
            abs(_numeric(row, field) - target[field]) / max(1.0, target[field])
            for field in target
        )

    ordered = sorted(scene_rows, key=lambda row: row["grouped_scene_id"])
    heldout, confirmation = min(
        itertools.permutations(ordered, 2),
        key=lambda pair: (
            distance(pair[0]) + distance(pair[1]),
            abs(_numeric(pair[0], "person_gt") - _numeric(pair[1], "person_gt")),
            pair[0]["grouped_scene_id"],
            pair[1]["grouped_scene_id"],
        ),
    )
    excluded = {heldout["grouped_scene_id"], confirmation["grouped_scene_id"]}
    support = [
        row["grouped_scene_id"]
        for row in ordered
        if row["grouped_scene_id"] not in excluded
    ]
    return heldout["grouped_scene_id"], confirmation["grouped_scene_id"], support


def _old_scene_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    rows = read_csv(ROOT / config["immutable_inputs"]["prior_development_manifest"]["path"])
    grouped: dict[str, dict[str, int]] = {}
    for row in rows:
        scene = row.get("grouped_scene_id") or row.get("group")
        if not scene:
            continue
        stats = grouped.setdefault(
            scene,
            {
                "frames": 0,
                "person_gt": 0,
                "small_gt": 0,
                "medium_gt": 0,
                "large_gt": 0,
            },
        )
        stats["frames"] += 1
        # Exact person counts come from labels later in execution. The legacy
        # fold is balanced with frozen person-v3 support data where available.
    support_path = ROOT / "outputs/person_v3/audit/class_scene_matrix.csv"
    if support_path.exists():
        for row in read_csv(support_path):
            scene = row.get("grouped_scene_id") or row.get("scene_id") or row.get("scene")
            if scene not in grouped:
                continue
            for source, target in (
                ("person", "person_gt"),
                ("person_gt", "person_gt"),
                ("person_GT", "person_gt"),
                ("person_small", "small_gt"),
                ("small_person", "small_gt"),
                ("person_medium", "medium_gt"),
                ("medium_person", "medium_gt"),
                ("person_large", "large_gt"),
                ("large_person", "large_gt"),
            ):
                if row.get(source):
                    grouped[scene][target] = _numeric(row, source)
    return [
        {"grouped_scene_id": scene, "source": "prior_development", **stats}
        for scene, stats in sorted(grouped.items())
    ]


def _balance_score(
    folds: list[list[dict[str, Any]]],
    fields: tuple[str, ...],
) -> float:
    score = 0.0
    for field in fields:
        values = [sum(float(row.get(field, 0)) for row in fold) for fold in folds]
        mean = sum(values) / len(values)
        if mean:
            score += sum(((value - mean) / mean) ** 2 for value in values)
    counts = [len(fold) for fold in folds]
    mean_count = sum(counts) / len(counts)
    score += sum(((value - mean_count) / mean_count) ** 2 for value in counts)
    return score


def constrained_folds(
    rows: list[dict[str, Any]], fold_count: int, seed: int, searches: int
) -> tuple[list[list[str]], float]:
    if len(rows) < fold_count:
        raise ValueError("Fewer scenes than requested folds")
    fields = ("frames", "person_gt", "small_gt", "medium_gt", "large_gt")
    rng = random.Random(seed)
    best_folds: list[list[dict[str, Any]]] | None = None
    best_score = math.inf
    ordered = sorted(rows, key=lambda row: row["grouped_scene_id"])
    for _ in range(searches):
        shuffled = ordered.copy()
        rng.shuffle(shuffled)
        folds = [[] for _ in range(fold_count)]
        for index, row in enumerate(shuffled):
            folds[index % fold_count].append(row)
        score = _balance_score(folds, fields)
        if score < best_score:
            best_score = score
            best_folds = folds
    assert best_folds is not None
    result = [
        sorted(row["grouped_scene_id"] for row in fold)
        for fold in best_folds
    ]
    result.sort(key=lambda fold: tuple(fold))
    return result, best_score


def build() -> dict[str, Any]:
    config = load_config()
    assert_test_sealed(config)
    gate_path = OUTPUT_ROOT / "audit/CPU_GATE.json"
    if not gate_path.exists():
        raise RuntimeError("CPU_GATE.json is missing; acquisition audit must run first")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["status"] != "PASS":
        raise RuntimeError(f"CPU acquisition gate is not PASS: {gate['status']}")
    new_rows = read_csv(OUTPUT_ROOT / "audit/scene_statistics.csv")
    heldout, confirmation, support = choose_screening_roles(new_rows)
    old_rows = _old_scene_rows(config)
    new_stats = [
        {"source": "new_acquisition", **row}
        for row in new_rows
    ]
    fold_rows: list[dict[str, Any]] = [*old_rows, *new_stats]
    folds, balance = constrained_folds(
        fold_rows,
        int(config["split"]["oof_folds"]),
        int(config["split"]["seed"]),
        int(config["split"]["search_candidates"]),
    )
    old_scenes = sorted(row["grouped_scene_id"] for row in old_rows)
    screening = {
        "protocol_id": config["protocol_id"],
        "training_scenes": sorted([*old_scenes, *support]),
        "screening_heldout_scenes": [heldout],
        "untouched_confirmation_scenes": [confirmation],
        "test_used": False,
    }
    oof = {
        "protocol_id": config["protocol_id"],
        "seed": int(config["split"]["seed"]),
        "search_candidates": int(config["split"]["search_candidates"]),
        "balance_score": balance,
        "folds": {str(index): scenes for index, scenes in enumerate(folds)},
        "test_used": False,
    }
    atomic_json(OUTPUT_ROOT / "protocol/screening_split.json", screening)
    atomic_json(OUTPUT_ROOT / "protocol/folds.json", oof)

    by_scene = {row["grouped_scene_id"]: row for row in fold_rows}
    support_rows = []
    for fold_index, scenes in enumerate(folds):
        for scene in scenes:
            row = by_scene[scene]
            support_rows.append(
                {
                    "fold": fold_index,
                    "grouped_scene_id": scene,
                    "source": row["source"],
                    "frames": row.get("frames", 0),
                    "person_gt": row.get("person_gt", 0),
                    "small_gt": row.get("small_gt", 0),
                    "medium_gt": row.get("medium_gt", 0),
                    "large_gt": row.get("large_gt", 0),
                }
            )
    write_csv(
        OUTPUT_ROOT / "audit/fold_support.csv",
        support_rows,
        [
            "fold",
            "grouped_scene_id",
            "source",
            "frames",
            "person_gt",
            "small_gt",
            "medium_gt",
            "large_gt",
        ],
    )
    return {"screening": screening, "oof": oof}


def main() -> int:
    argparse.ArgumentParser().parse_args()
    result = build()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
