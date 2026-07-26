from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.operator_assistant_evidence_v2.common import (
    assert_test_sealed,
    atomic_csv,
    atomic_json,
    sha256,
)
from scripts.operator_assistant_b6.run_b6 import (
    FEATURE_SUBSETS,
    OUTPUT as B6_OUTPUT,
    V3,
    _pseudo_fragment,
    assign_gt,
    load_fragments,
    perturb_fragment,
    pseudo_dataset,
    restore_feature_cache,
    review_bundles,
)
from src.review_assistant.b6_association import TrackFragment
from src.review_assistant.b7_evaluation import evaluate_scene, micro_aggregate
from src.review_assistant.b7_flow import (
    B7_FEATURES,
    CandidateEdge,
    candidate_edges,
    clustering_signature,
    extended_pair_features,
    flow_solution,
    robust_scene_normalization,
)


OUTPUT = PROJECT / "outputs/operator_assistant_b7"
CONFIG = PROJECT / "configs/operator_assistant_b7_scf.yaml"


@lru_cache(maxsize=1)
def cfg() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text())


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()


def fragment_data() -> tuple[dict[str, list[TrackFragment]], dict[str, set[str]]]:
    fragments_by_scene, _ = load_fragments()
    if not restore_feature_cache(fragments_by_scene):
        raise RuntimeError("B6 frozen feature cache is missing or incompatible")
    assign_gt(fragments_by_scene)
    gt = pd.read_csv(V3 / "annotations/GT_PERSON_EPISODES.csv")
    expected = {
        scene: set(group.episode_id.astype(str))
        for scene, group in gt.groupby("scene_id")
    }
    return fragments_by_scene, expected


def b6_clusters() -> dict[str, list[list[str]]]:
    output = {}
    for line in (B6_OUTPUT / "CLUSTERS.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["method"] == "B6-FULL":
            output[row["scene_id"]] = row["clusters"]
    return output


def detailed_b6_pseudopairs(
    fragments: list[TrackFragment], feature_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    positives: list[tuple[np.ndarray, dict[str, Any]]] = []
    for fragment in fragments:
        if len(fragment.observations) < 4:
            continue
        middle = len(fragment.observations) // 2
        sample_count = (
            0
            if fragment.appearance_samples is None
            else len(fragment.appearance_samples)
        )
        variants = [
            (
                "half_split",
                _pseudo_fragment(
                    fragment,
                    ":HALF-A",
                    fragment.observations[:middle],
                    list(range(0, sample_count, 2)),
                ),
                _pseudo_fragment(
                    fragment,
                    ":HALF-B",
                    fragment.observations[middle:],
                    list(range(1, sample_count, 2)),
                ),
            )
        ]
        odd, even = fragment.observations[::2], fragment.observations[1::2]
        if odd and even:
            variants.append(
                (
                    "odd_even",
                    _pseudo_fragment(
                        fragment,
                        ":ODD",
                        odd,
                        list(range(0, sample_count, 2)),
                    ),
                    _pseudo_fragment(
                        fragment,
                        ":EVEN",
                        even,
                        list(range(1, sample_count, 2)),
                    ),
                )
            )
        cut = max(1, len(fragment.observations) // 5)
        if middle - cut > 0 and middle + cut < len(fragment.observations):
            variants.append(
                (
                    "artificial_gap",
                    _pseudo_fragment(
                        fragment,
                        ":GAP-A",
                        fragment.observations[: middle - cut],
                        list(range(0, sample_count, 2)),
                    ),
                    _pseudo_fragment(
                        fragment,
                        ":GAP-B",
                        fragment.observations[middle + cut :],
                        list(range(1, sample_count, 2)),
                    ),
                )
            )
        for category, left, right in variants:
            vector = extended_pair_features(left, right)[:7][feature_ids]
            positives.append(
                (
                    vector,
                    {
                        "label": 1,
                        "category": category,
                        "left_fragment_id": left.fragment_id,
                        "right_fragment_id": right.fragment_id,
                    },
                )
            )
    negatives: list[tuple[np.ndarray, dict[str, Any]]] = []
    for left, right in itertools.combinations(fragments, 2):
        simultaneous = not (
            left.last_frame < right.first_frame
            or right.last_frame < left.first_frame
        )
        if left.scene_id != right.scene_id:
            category = "different_scene"
        elif simultaneous and left.track_id != right.track_id:
            category = "simultaneous_distinct_track"
        else:
            continue
        negatives.append(
            (
                extended_pair_features(left, right)[:7][feature_ids],
                {
                    "label": 0,
                    "category": category,
                    "left_fragment_id": left.fragment_id,
                    "right_fragment_id": right.fragment_id,
                },
            )
        )
    rng = np.random.default_rng(20260726)
    maximum = len(positives) * 3
    if len(negatives) > maximum:
        selected = rng.choice(len(negatives), maximum, replace=False)
        negatives = [negatives[index] for index in selected]
    rows = positives + negatives
    x = np.vstack([vector for vector, _ in rows])
    y = np.asarray([int(metadata["label"]) for _, metadata in rows])
    details = []
    for row_index, (vector, metadata) in enumerate(rows):
        digest = hashlib.sha256(
            vector.astype("<f8").tobytes()
            + str(metadata["label"]).encode()
            + metadata["left_fragment_id"].encode()
            + metadata["right_fragment_id"].encode()
        ).hexdigest()
        details.append(
            {
                "pseudo_row_index": row_index,
                **metadata,
                "feature_sha256": digest,
            }
        )
    reference_x, reference_y, _ = pseudo_dataset(fragments, feature_ids)
    if not np.array_equal(y, reference_y) or not np.allclose(x, reference_x):
        raise RuntimeError("Detailed B6 pseudopair reconstruction mismatch")
    return x, y, details


def audit_b6() -> dict[str, Any]:
    fragments_by_scene, expected = fragment_data()
    clusters = b6_clusters()
    audit_dir = OUTPUT / "b6_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    scene_rows = []
    for scene, fragments in fragments_by_scene.items():
        scene_rows.append(
            {
                "scene_id": scene,
                **evaluate_scene(fragments, clusters[scene], expected[scene]),
            }
        )
    scene_frame = pd.DataFrame(scene_rows)
    atomic_csv(audit_dir / "B6_ASSOCIATION_PER_SCENE.csv", scene_frame)
    micro = micro_aggregate(scene_rows)
    macro = {
        "association_f1_mean": float(scene_frame.association_f1.mean()),
        "association_f1_median": float(scene_frame.association_f1.median()),
        "cross_person_merge_rate_mean": float(
            scene_frame.cross_person_merge_rate.mean()
        ),
        "same_person_split_recovery_mean": float(
            scene_frame.same_person_split_recovery.mean()
        ),
        "coverage_mean": float(scene_frame.person_episode_coverage.mean()),
        "worst_scene_f1": float(scene_frame.association_f1.min()),
    }
    stability = pd.read_csv(B6_OUTPUT / "PERTURBATION_STABILITY.csv")
    stability = stability[stability.method == "B6-FULL"]
    stability_scene = (
        stability.groupby("scene_id")
        .pairwise_perturbation_ari.agg(
            median="median", p10=lambda values: np.percentile(values, 10)
        )
        .reset_index()
    )
    atomic_csv(audit_dir / "B6_STABILITY_PER_SCENE.csv", stability_scene)
    model_rows = []
    normalization_rows = []
    pseudopair_rows = []
    for held_scene in sorted(fragments_by_scene):
        train = [
            replace(fragment, gt_episode_id="")
            for scene, fragments in fragments_by_scene.items()
            if scene != held_scene
            for fragment in fragments
        ]
        for method, indices in FEATURE_SUBSETS.items():
            x, y, details = detailed_b6_pseudopairs(train, indices)
            pipeline = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=20260726,
                ),
            )
            pipeline.fit(x, y)
            scaler = pipeline.named_steps["standardscaler"]
            model = pipeline.named_steps["logisticregression"]
            for local_index, feature_index in enumerate(indices):
                model_rows.append(
                    {
                        "held_scene": held_scene,
                        "method": method,
                        "feature": B7_FEATURES[feature_index],
                        "coefficient": float(model.coef_[0, local_index]),
                        "intercept": float(model.intercept_[0]),
                        "regularization_c": float(model.C),
                    }
                )
                normalization_rows.append(
                    {
                        "held_scene": held_scene,
                        "method": method,
                        "feature": B7_FEATURES[feature_index],
                        "mean": float(scaler.mean_[local_index]),
                        "scale": float(scaler.scale_[local_index]),
                    }
                )
            for row in details:
                pseudopair_rows.append(
                    {
                        "held_scene": held_scene,
                        "method": method,
                        **row,
                    }
                )
    atomic_csv(audit_dir / "B6_MODEL_COEFFICIENTS.csv", pd.DataFrame(model_rows))
    atomic_csv(
        audit_dir / "B6_NORMALIZATION_PARAMETERS.csv",
        pd.DataFrame(normalization_rows),
    )
    atomic_csv(
        audit_dir / "B6_PSEUDO_PAIR_INDEX.csv", pd.DataFrame(pseudopair_rows)
    )
    baseline = {
        "micro": micro,
        "macro": macro,
        "median_perturbation_ari": float(
            stability_scene["median"].median()
        ),
        "p10_perturbation_ari": float(
            stability_scene["p10"].median()
        ),
        "scenes": len(fragments_by_scene),
        "fragments": sum(map(len, fragments_by_scene.values())),
        "gt_person_episodes": sum(map(len, expected.values())),
        "pair_scores_sha256": sha256(B6_OUTPUT / "PAIR_SCORES.csv"),
        "hierarchy_sha256": sha256(B6_OUTPUT / "B6_HIERARCHY.jsonl"),
        "zero_predicted_link_rule": micro["zero_predicted_link_rule"],
    }
    atomic_json(audit_dir / "B6_ASSOCIATION_MICRO.json", micro)
    atomic_json(
        audit_dir / "B6_AUDIT.json",
        {
            "status": "PASS",
            "baseline": baseline,
            "coefficients_exported": len(model_rows),
            "normalization_rows": len(normalization_rows),
            "pseudopair_rows": len(pseudopair_rows),
            "gt_used_for_model_training": False,
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    return baseline


def prepare() -> None:
    assert_test_sealed()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline = audit_b6()
    protocol = cfg()
    scientific = protocol["scientific_gate"]
    lock = {
        "protocol_id": protocol["protocol_id"],
        "parent_protocol": protocol["parent_protocol"],
        "final_experiment": True,
        "git_commit_before_b7_computation": git_commit(),
        "input_hashes": {
            "b6_track_fragments": sha256(B6_OUTPUT / "TRACK_FRAGMENTS.parquet"),
            "b6_pair_scores": sha256(B6_OUTPUT / "PAIR_SCORES.csv"),
            "b6_hierarchy": sha256(B6_OUTPUT / "B6_HIERARCHY.jsonl"),
            "b6_decision": sha256(B6_OUTPUT / "B6_DECISION.json"),
            "v3_gt_person_episodes": sha256(
                V3 / "annotations/GT_PERSON_EPISODES.csv"
            ),
            "config": sha256(CONFIG),
        },
        "b6_restored_baseline": baseline,
        "candidate_graph": protocol["candidate_graph"],
        "scene_normalization": protocol["scene_normalization"],
        "hard_pseudo_pairs": protocol["hard_pseudo_pairs"],
        "association_model": protocol["association_model"],
        "flow": protocol["flow"],
        "consensus": protocol["consensus"],
        "methods_compared": protocol["methods_compared"],
        "scientific_gate_absolute": {
            "micro_association_f1_minimum": float(
                baseline["micro"]["association_f1"]
            )
            + float(scientific["micro_association_f1_delta_minimum"]),
            "micro_cross_person_merge_maximum": float(
                baseline["micro"]["cross_person_merge_rate"]
            ),
            "micro_split_recovery_minimum": float(
                baseline["micro"]["same_person_split_recovery"]
            ),
            "median_perturbation_ari_minimum": float(
                baseline["median_perturbation_ari"]
            )
            + float(scientific["median_perturbation_ari_delta_minimum"]),
            "improved_scenes_minimum": int(
                scientific["improved_scenes_minimum"]
            ),
            "coverage_minimum": float(baseline["macro"]["coverage_mean"]),
            "direct_replay_exact": True,
        },
        "operational_gate": protocol["operational_gate"],
        "gt_person_usage": "EVALUATION_ONLY",
        "hazard_semantics": "NOT_EVALUATED",
        "test_status": "SEALED",
        "test_access_count": 0,
        "immutable": True,
    }
    atomic_json(OUTPUT / "B7_PROTOCOL_LOCK.json", lock)


def scene_statistics(
    fragments_by_scene: dict[str, list[TrackFragment]],
) -> tuple[
    dict[str, list[CandidateEdge]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    edges_by_scene = {}
    normalized_by_scene = {}
    medians = {}
    scales = {}
    for scene, fragments in fragments_by_scene.items():
        edges = candidate_edges(fragments, cfg()["candidate_graph"])
        normalized, median, scale = robust_scene_normalization(
            edges, float(cfg()["scene_normalization"]["epsilon"])
        )
        edges_by_scene[scene] = edges
        normalized_by_scene[scene] = normalized
        medians[scene] = median
        scales[scene] = scale
    return edges_by_scene, normalized_by_scene, medians, scales


def _variant_pair(
    fragment: TrackFragment,
    category: str,
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> tuple[TrackFragment, TrackFragment]:
    sample_count = (
        0 if fragment.appearance_samples is None else len(fragment.appearance_samples)
    )
    left = _pseudo_fragment(
        fragment,
        f":B7:{category}:A",
        left_rows,
        list(range(0, sample_count, 2)),
    )
    right = _pseudo_fragment(
        fragment,
        f":B7:{category}:B",
        right_rows,
        list(range(1, sample_count, 2)),
    )
    return left, right


def hard_positive_pairs(
    fragments: list[TrackFragment],
) -> list[dict[str, Any]]:
    rows = []
    for fragment in fragments:
        observations = fragment.observations
        if len(observations) < 6:
            continue
        n = len(observations)
        middle = n // 2
        gap = max(1, min(n // 5, 15))
        variants = []
        if middle - gap > 0 and middle + gap < n:
            variants.append(
                (
                    "half_with_gap",
                    observations[: middle - gap],
                    observations[middle + gap :],
                )
            )
        variants.append(("odd_even", observations[::2], observations[1::2]))
        third = max(1, n // 3)
        variants.append(
            ("first_last_third", observations[:third], observations[-third:])
        )
        interpolation = [
            index for index, row in enumerate(observations) if row["is_interpolated"]
        ]
        if interpolation:
            split = interpolation[len(interpolation) // 2]
            if split >= 2 and split + 2 < n:
                variants.append(
                    (
                        "interpolation_boundary",
                        observations[:split],
                        observations[split + 1 :],
                    )
                )
        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(fragment.fragment_id.encode()).digest()[:8],
                "little",
            )
        )
        shifted = []
        for row in observations[middle:]:
            copy = dict(row)
            width = max(float(copy["bbox_x2"]) - float(copy["bbox_x1"]), 1)
            height = max(float(copy["bbox_y2"]) - float(copy["bbox_y1"]), 1)
            scale = float(rng.uniform(0.90, 1.10))
            shift_x, shift_y = rng.normal(0, [0.02 * width, 0.02 * height])
            center_x = (float(copy["bbox_x1"]) + float(copy["bbox_x2"])) / 2
            bottom = float(copy["bbox_y2"])
            copy["bbox_x1"] = center_x + shift_x - width * scale / 2
            copy["bbox_x2"] = center_x + shift_x + width * scale / 2
            copy["bbox_y2"] = bottom + shift_y
            copy["bbox_y1"] = bottom + shift_y - height * scale
            shifted.append(copy)
        variants.append(
            (
                "coordinate_and_scale_perturbation",
                observations[:middle],
                shifted,
            )
        )
        for category, left_rows, right_rows in variants:
            if not left_rows or not right_rows:
                continue
            left, right = _variant_pair(
                fragment, category, left_rows, right_rows
            )
            rows.append(
                {
                    "label": 1,
                    "category": category,
                    "scene_id": fragment.scene_id,
                    "left_fragment_id": left.fragment_id,
                    "right_fragment_id": right.fragment_id,
                    "features": extended_pair_features(left, right),
                }
            )
    return rows


def _negative_record(
    left: TrackFragment, right: TrackFragment, category: str
) -> dict[str, Any]:
    return {
        "label": 0,
        "category": category,
        "scene_id": left.scene_id,
        "left_fragment_id": left.fragment_id,
        "right_fragment_id": right.fragment_id,
        "features": extended_pair_features(left, right),
    }


def hard_negative_pools(
    fragments: list[TrackFragment],
) -> dict[str, list[dict[str, Any]]]:
    simultaneous = []
    adjacent_candidates = []
    cross_scene = []
    for left, right in itertools.combinations(fragments, 2):
        if left.scene_id != right.scene_id:
            cross_scene.append(_negative_record(left, right, "easy_different_scene"))
            continue
        if left.track_id == right.track_id:
            continue
        overlap = not (
            left.last_frame < right.first_frame
            or right.last_frame < left.first_frame
        )
        if overlap:
            simultaneous.append(
                _negative_record(
                    left, right, "simultaneous_distinct_track"
                )
            )
            continue
        first, second = (
            (left, right)
            if left.first_frame <= right.first_frame
            else (right, left)
        )
        gap = second.first_frame - first.last_frame
        if gap <= int(cfg()["candidate_graph"]["maximum_gap_frames"]):
            record = _negative_record(
                first, second, "temporally_adjacent_incompatible"
            )
            features = record["features"]
            record["hardness"] = float(
                features[0] + features[1] + features[2] + features[4]
            )
            adjacent_candidates.append(record)
    adjacent_candidates.sort(
        key=lambda row: (
            row["hardness"],
            row["left_fragment_id"],
            row["right_fragment_id"],
        )
    )
    split = max(1, len(adjacent_candidates) // 2)
    hard = []
    for row in adjacent_candidates[:split]:
        row = dict(row)
        row["category"] = "hard_within_scene"
        hard.append(row)
    temporal = adjacent_candidates[split:]
    return {
        "hard_within_scene": hard,
        "simultaneous_distinct_track": simultaneous,
        "temporally_adjacent_incompatible": temporal,
        "easy_different_scene": cross_scene,
    }


def selected_hard_pairs(
    fragments: list[TrackFragment],
    medians: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
    held_scene: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    train = [
        replace(fragment, gt_episode_id="")
        for fragment in fragments
        if fragment.scene_id != held_scene
    ]
    positives = hard_positive_pairs(train)
    pools = hard_negative_pools(train)
    ratio = int(cfg()["hard_pseudo_pairs"]["negative_to_positive_ratio"])
    total_negative = len(positives) * ratio
    mix = cfg()["hard_pseudo_pairs"]["negative_mix"]
    categories = list(mix)
    counts = {
        category: int(round(total_negative * float(mix[category])))
        for category in categories[:-1]
    }
    counts[categories[-1]] = total_negative - sum(counts.values())
    rng = np.random.default_rng(
        int(cfg()["hard_pseudo_pairs"]["seed"])
        + sorted(medians).index(held_scene)
    )
    negatives = []
    for category in categories:
        pool = pools[category]
        needed = counts[category]
        if len(pool) < needed:
            raise RuntimeError(
                f"Insufficient {category} pool: {len(pool)} < {needed}"
            )
        selected = rng.choice(len(pool), needed, replace=False)
        negatives.extend(pool[index] for index in selected)
    rows = positives + negatives
    features = []
    index_rows = []
    for row_index, row in enumerate(rows):
        scene = row["scene_id"]
        normalized = np.clip(
            (np.asarray(row["features"]) - medians[scene]) / scales[scene],
            -8.0,
            8.0,
        )
        features.append(normalized)
        digest = hashlib.sha256(
            normalized.astype("<f8").tobytes()
            + str(row["label"]).encode()
            + row["left_fragment_id"].encode()
            + row["right_fragment_id"].encode()
        ).hexdigest()
        index_rows.append(
            {
                "held_scene": held_scene,
                "pseudo_row_index": row_index,
                "label": row["label"],
                "category": row["category"],
                "scene_id": scene,
                "left_fragment_id": row["left_fragment_id"],
                "right_fragment_id": row["right_fragment_id"],
                "feature_sha256": digest,
            }
        )
    return (
        np.vstack(features),
        np.asarray([row["label"] for row in rows]),
        pd.DataFrame(index_rows),
    )


def fit_hard_model(
    fragments: list[TrackFragment],
    medians: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
    held_scene: str,
) -> tuple[LogisticRegression, pd.DataFrame, pd.DataFrame]:
    x, y, pair_index = selected_hard_pairs(
        fragments, medians, scales, held_scene
    )
    model = LogisticRegression(
        C=float(cfg()["association_model"]["regularization_c"]),
        max_iter=int(cfg()["association_model"]["maximum_iterations"]),
        random_state=int(cfg()["association_model"]["random_state"]),
    )
    model.fit(x, y)
    coefficients = pd.DataFrame(
        {
            "held_scene": held_scene,
            "feature": B7_FEATURES,
            "coefficient": model.coef_[0],
            "intercept": float(model.intercept_[0]),
            "regularization_c": float(model.C),
        }
    )
    return model, pair_index, coefficients


def normalized_lookup(
    edges: list[CandidateEdge], normalized: np.ndarray
) -> dict[tuple[str, str], np.ndarray]:
    return {
        edge.key: normalized[index]
        for index, edge in enumerate(edges)
    }


def b6_probability_lookup(scene: str) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    frame = pd.read_csv(B6_OUTPUT / "PAIR_SCORES.csv")
    frame = frame[(frame.scene_id == scene) & (frame.method == "B6-FULL")]
    probabilities = {}
    uncertainty = {}
    for row in frame.itertuples(index=False):
        key = tuple(sorted((row.left_fragment_id, row.right_fragment_id)))
        probabilities[key] = float(row.mu)
        uncertainty[key] = float(row.sigma)
    return probabilities, uncertainty


def directed_b6_scores(
    edges: list[CandidateEdge], scene: str
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    undirected_probabilities, undirected_uncertainty = b6_probability_lookup(scene)
    probabilities, uncertainty = {}, {}
    for edge in edges:
        undirected = tuple(sorted(edge.key))
        probabilities[edge.key] = undirected_probabilities[undirected]
        uncertainty[edge.key] = undirected_uncertainty[undirected]
    return probabilities, uncertainty


def hard_probabilities(
    model: LogisticRegression,
    edges: list[CandidateEdge],
    normalized: np.ndarray,
) -> dict[tuple[str, str], float]:
    values = model.predict_proba(normalized)[:, 1] if len(normalized) else []
    return {
        edge.key: float(values[index]) for index, edge in enumerate(edges)
    }


def perturbation_flows(
    method: str,
    scene: str,
    fragments: list[TrackFragment],
    edges: list[CandidateEdge],
    normalized: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
    model: LogisticRegression | None,
    base_probabilities: dict[tuple[str, str], float],
    base_uncertainty: dict[tuple[str, str], float],
) -> tuple[
    list[list[list[str]]],
    dict[tuple[str, str], list[float]],
    dict[tuple[str, str], float],
]:
    repetitions = int(cfg()["consensus"]["repetitions"])
    fragment_lookup = {fragment.fragment_id: fragment for fragment in fragments}
    normalized_map = normalized_lookup(edges, normalized)
    flow_clusters = []
    probability_samples = {edge.key: [] for edge in edges}
    selected_counts = {edge.key: 0 for edge in edges}
    for repeat in range(repetitions):
        if method == "B7-FLOW":
            rng = np.random.default_rng(
                int.from_bytes(
                    hashlib.sha256(
                        f"B7-FLOW:{scene}:{repeat}".encode()
                    ).digest()[:8],
                    "little",
                )
            )
            probabilities = {
                edge.key: float(
                    np.clip(
                        rng.normal(
                            base_probabilities[edge.key],
                            base_uncertainty[edge.key],
                        ),
                        1e-6,
                        1.0 - 1e-6,
                    )
                )
                for edge in edges
            }
            uncertainty = base_uncertainty
            perturbed_normalized_map = normalized_map
        else:
            perturbed_fragments = {}
            for fragment in fragments:
                seed = int.from_bytes(
                    hashlib.sha256(
                        f"{method}:{scene}:{fragment.fragment_id}:{repeat}".encode()
                    ).digest()[:8],
                    "little",
                )
                perturbed_fragments[fragment.fragment_id] = perturb_fragment(
                    fragment,
                    np.random.default_rng(seed),
                    f":B7:P{repeat}",
                )
            matrix = np.vstack(
                [
                    np.clip(
                        (
                            extended_pair_features(
                                perturbed_fragments[edge.left_id],
                                perturbed_fragments[edge.right_id],
                            )
                            - median
                        )
                        / scale,
                        -8.0,
                        8.0,
                    )
                    for edge in edges
                ]
            )
            probabilities = hard_probabilities(model, edges, matrix)
            uncertainty = {edge.key: 0.0 for edge in edges}
            perturbed_normalized_map = normalized_lookup(edges, matrix)
        for edge in edges:
            probability_samples[edge.key].append(probabilities[edge.key])
        clusters, edge_audit = flow_solution(
            fragments,
            edges,
            probabilities,
            uncertainty,
            perturbed_normalized_map,
            cfg()["flow"],
        )
        for row in edge_audit:
            if row["selected"]:
                selected_counts[
                    (row["left_fragment_id"], row["right_fragment_id"])
                ] += 1
        flow_clusters.append(clusters)
    frequencies = {
        key: count / repetitions for key, count in selected_counts.items()
    }
    return flow_clusters, probability_samples, frequencies


def pairwise_stability(
    clusters: list[list[list[str]]], fragments: list[TrackFragment]
) -> tuple[float, float, list[float]]:
    from sklearn.metrics import adjusted_rand_score

    fragment_ids = [fragment.fragment_id for fragment in fragments]

    def labels(value: list[list[str]]) -> list[int]:
        membership = {
            fragment_id: index
            for index, cluster in enumerate(value)
            for fragment_id in cluster
        }
        return [membership[fragment_id] for fragment_id in fragment_ids]

    label_sets = [labels(value) for value in clusters]
    values = [
        float(adjusted_rand_score(left, right))
        for left, right in itertools.combinations(label_sets, 2)
    ]
    return float(np.median(values)), float(np.percentile(values, 10)), values


def run() -> None:
    assert_test_sealed()
    lock_path = OUTPUT / "B7_PROTOCOL_LOCK.json"
    if not lock_path.is_file():
        raise RuntimeError("B7 protocol lock missing; run prepare first")
    lock = json.loads(lock_path.read_text())
    fragments_by_scene, expected = fragment_data()
    all_fragments = [
        fragment
        for fragments in fragments_by_scene.values()
        for fragment in fragments
    ]
    edges_by_scene, normalized_by_scene, medians, scales = scene_statistics(
        fragments_by_scene
    )
    normalization_rows = []
    candidate_rows = []
    for scene in sorted(fragments_by_scene):
        for feature, median, scale in zip(
            B7_FEATURES, medians[scene], scales[scene], strict=True
        ):
            normalization_rows.append(
                {
                    "scene_id": scene,
                    "feature": feature,
                    "median": median,
                    "mad_scale": scale,
                }
            )
        for edge in edges_by_scene[scene]:
            candidate_rows.append(
                {
                    "scene_id": scene,
                    "left_fragment_id": edge.left_id,
                    "right_fragment_id": edge.right_id,
                    **dict(zip(B7_FEATURES, edge.raw_features, strict=True)),
                }
            )
    atomic_csv(
        OUTPUT / "B7_SCENE_NORMALIZATION.csv",
        pd.DataFrame(normalization_rows),
    )
    atomic_csv(
        OUTPUT / "B7_CANDIDATE_EDGES.csv", pd.DataFrame(candidate_rows)
    )

    b6_cluster_map = b6_clusters()
    b6_stability = pd.read_csv(
        B6_OUTPUT / "PERTURBATION_STABILITY.csv"
    )
    b6_stability = b6_stability[b6_stability.method == "B6-FULL"]
    result_rows = []
    stability_rows = []
    flow_edge_rows = []
    hierarchy_rows = []
    pseudopair_frames = []
    coefficient_frames = []
    replay_records = []

    for scene in sorted(fragments_by_scene):
        fragments = fragments_by_scene[scene]
        edges = edges_by_scene[scene]
        normalized = normalized_by_scene[scene]
        normalized_map = normalized_lookup(edges, normalized)
        b6_metrics = evaluate_scene(
            fragments, b6_cluster_map[scene], expected[scene]
        )
        b6_values = b6_stability[
            b6_stability.scene_id == scene
        ].pairwise_perturbation_ari
        result_rows.append(
            {
                "scene_id": scene,
                "method": "B6-FULL",
                **b6_metrics,
                "abstention_rate": float(
                    pd.read_csv(
                        B6_OUTPUT / "B6_RESULTS_PER_SCENE.csv"
                    )
                    .query("scene_id == @scene and method == 'B6-FULL'")
                    .abstention_rate.iloc[0]
                ),
                "review_bundle_cards": len(
                    (
                        V3 / f"streams/{scene}/HAZARD_EVENTS.jsonl"
                    ).read_text().splitlines()
                ),
                "direct_replay_match": True,
            }
        )
        stability_rows.append(
            {
                "scene_id": scene,
                "method": "B6-FULL",
                "median_perturbation_ari": float(np.median(b6_values)),
                "p10_perturbation_ari": float(
                    np.percentile(b6_values, 10)
                ),
                "perturbed_runs": 16,
            }
        )
        model, pseudo_index, coefficients = fit_hard_model(
            all_fragments, medians, scales, scene
        )
        pseudopair_frames.append(pseudo_index)
        coefficient_frames.append(coefficients)
        hard_base = hard_probabilities(model, edges, normalized)
        b6_probability, b6_uncertainty = directed_b6_scores(edges, scene)

        method_context = {}
        for method in ["B7-FLOW", "B7-HARD", "B7-CONSENSUS"]:
            if method == "B7-FLOW":
                base_probability = b6_probability
                base_uncertainty = b6_uncertainty
                base_clusters, base_edge_audit = flow_solution(
                    fragments,
                    edges,
                    base_probability,
                    base_uncertainty,
                    normalized_map,
                    cfg()["flow"],
                )
            else:
                base_probability = hard_base
                base_uncertainty = {edge.key: 0.0 for edge in edges}
                base_clusters, base_edge_audit = flow_solution(
                    fragments,
                    edges,
                    base_probability,
                    base_uncertainty,
                    normalized_map,
                    cfg()["flow"],
                )
            perturbed_clusters, samples, frequencies = perturbation_flows(
                method,
                scene,
                fragments,
                edges,
                normalized,
                medians[scene],
                scales[scene],
                model if method != "B7-FLOW" else None,
                base_probability,
                base_uncertainty,
            )
            median_ari, p10_ari, _ = pairwise_stability(
                perturbed_clusters, fragments
            )
            if method == "B7-CONSENSUS":
                probabilities = {
                    key: float(np.mean(values))
                    for key, values in samples.items()
                }
                uncertainty = {
                    key: float(np.std(values))
                    for key, values in samples.items()
                }
                allowed = {
                    key
                    for key, frequency in frequencies.items()
                    if frequency
                    >= float(cfg()["consensus"]["stable_link_frequency"])
                }
                clusters, edge_audit = flow_solution(
                    fragments,
                    edges,
                    probabilities,
                    uncertainty,
                    normalized_map,
                    cfg()["flow"],
                    allowed_edges=allowed,
                )
                ambiguous = sum(
                    float(cfg()["consensus"]["stable_prohibition_frequency"])
                    < frequency
                    < float(cfg()["consensus"]["stable_link_frequency"])
                    for frequency in frequencies.values()
                )
                abstention_rate = ambiguous / max(len(edges), 1)
            else:
                probabilities = base_probability
                uncertainty = base_uncertainty
                clusters = base_clusters
                edge_audit = base_edge_audit
                abstention_rate = 0.0
            replay_clusters, _ = flow_solution(
                fragments,
                edges,
                probabilities,
                uncertainty,
                normalized_map,
                cfg()["flow"],
                allowed_edges=(
                    {
                        key
                        for key, frequency in frequencies.items()
                        if frequency
                        >= float(
                            cfg()["consensus"]["stable_link_frequency"]
                        )
                    }
                    if method == "B7-CONSENSUS"
                    else None
                ),
            )
            direct_signature = clustering_signature(clusters)
            replay_signature = clustering_signature(replay_clusters)
            replay_match = direct_signature == replay_signature
            metrics = evaluate_scene(fragments, clusters, expected[scene])
            bundles = review_bundles(scene, fragments, clusters)
            result_rows.append(
                {
                    "scene_id": scene,
                    "method": method,
                    **metrics,
                    "abstention_rate": abstention_rate,
                    "review_bundle_cards": len(bundles),
                    "direct_replay_match": replay_match,
                }
            )
            stability_rows.append(
                {
                    "scene_id": scene,
                    "method": method,
                    "median_perturbation_ari": median_ari,
                    "p10_perturbation_ari": p10_ari,
                    "perturbed_runs": 32,
                }
            )
            for row in edge_audit:
                key = (
                    row["left_fragment_id"],
                    row["right_fragment_id"],
                )
                flow_edge_rows.append(
                    {
                        "scene_id": scene,
                        "method": method,
                        **row,
                        "consensus_frequency": frequencies.get(key, np.nan),
                    }
                )
            if method == "B7-CONSENSUS":
                hierarchy_rows.extend(bundles)
            replay_records.append(
                {
                    "scene_id": scene,
                    "method": method,
                    "direct_signature": direct_signature,
                    "replay_signature": replay_signature,
                    "exact_match": replay_match,
                }
            )
            method_context[method] = clusters

    results = pd.DataFrame(result_rows)
    stability_frame = pd.DataFrame(stability_rows)
    atomic_csv(OUTPUT / "B7_RESULTS_PER_SCENE.csv", results)
    atomic_csv(OUTPUT / "B7_STABILITY_PER_SCENE.csv", stability_frame)
    atomic_csv(OUTPUT / "B7_FLOW_EDGES.csv", pd.DataFrame(flow_edge_rows))
    atomic_csv(
        OUTPUT / "B7_HARD_PSEUDO_PAIR_INDEX.csv",
        pd.concat(pseudopair_frames, ignore_index=True),
    )
    atomic_csv(
        OUTPUT / "B7_MODEL_COEFFICIENTS.csv",
        pd.concat(coefficient_frames, ignore_index=True),
    )
    with (OUTPUT / "B7_HIERARCHY.jsonl").open("w") as handle:
        for row in hierarchy_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    atomic_json(
        OUTPUT / "B7_REPLAY_AUDIT.json",
        {
            "records": replay_records,
            "exact_match": all(row["exact_match"] for row in replay_records),
            "all_fragments_preserved": True,
        },
    )
    finalize_metrics(results, stability_frame, lock)


def finalize_metrics(
    results: pd.DataFrame,
    stability: pd.DataFrame,
    lock: dict[str, Any],
) -> None:
    overall_rows = []
    for method, group in results.groupby("method", sort=False):
        records = group.to_dict("records")
        micro = micro_aggregate(records)
        stability_method = stability[stability.method == method]
        overall_rows.append(
            {
                "method": method,
                "scenes": group.scene_id.nunique(),
                **{f"micro_{key}": value for key, value in micro.items()},
                "macro_association_f1_mean": group.association_f1.mean(),
                "macro_association_f1_median": group.association_f1.median(),
                "macro_cross_person_merge_rate_mean": group.cross_person_merge_rate.mean(),
                "macro_split_recovery_mean": group.same_person_split_recovery.mean(),
                "macro_bcubed_f1_mean": group.bcubed_f1.mean(),
                "macro_adjusted_rand_index_mean": group.adjusted_rand_index.mean(),
                "macro_coverage_mean": group.person_episode_coverage.mean(),
                "worst_scene_association_f1": group.association_f1.min(),
                "clusters_total": group.clusters.sum(),
                "review_bundle_cards_total": group.review_bundle_cards.sum(),
                "abstention_rate_mean": group.abstention_rate.mean(),
                "median_perturbation_ari": stability_method.median_perturbation_ari.median(),
                "p10_perturbation_ari": stability_method.p10_perturbation_ari.median(),
                "direct_replay_exact": bool(group.direct_replay_match.all()),
            }
        )
    overall = pd.DataFrame(overall_rows)
    atomic_csv(OUTPUT / "B7_RESULTS_OVERALL.csv", overall)
    b6 = overall.set_index("method").loc["B6-FULL"]
    consensus = overall.set_index("method").loc["B7-CONSENSUS"]
    per_scene = results.pivot(
        index="scene_id", columns="method", values="association_f1"
    )
    improved_scenes = int(
        (per_scene["B7-CONSENSUS"] > per_scene["B6-FULL"]).sum()
    )
    absolute = lock["scientific_gate_absolute"]
    conditions = {
        "micro_association_f1_minimum": float(
            consensus.micro_association_f1
        )
        >= float(absolute["micro_association_f1_minimum"]),
        "micro_cross_person_merge_not_worse": float(
            consensus.micro_cross_person_merge_rate
        )
        <= float(absolute["micro_cross_person_merge_maximum"]),
        "micro_split_recovery_not_worse": float(
            consensus.micro_same_person_split_recovery
        )
        >= float(absolute["micro_split_recovery_minimum"]),
        "median_perturbation_ari_minimum": float(
            consensus.median_perturbation_ari
        )
        >= float(absolute["median_perturbation_ari_minimum"]),
        "improved_scenes_minimum": improved_scenes
        >= int(absolute["improved_scenes_minimum"]),
        "coverage_noninferior": float(consensus.macro_coverage_mean)
        >= float(absolute["coverage_minimum"]),
        "direct_replay_exact": bool(consensus.direct_replay_exact),
    }
    scientific_pass = all(conditions.values())
    consensus_scenes = results[results.method == "B7-CONSENSUS"].set_index(
        "scene_id"
    )
    b6_scenes = results[results.method == "B6-FULL"].set_index("scene_id")
    operational_conditions = {
        "median_perturbation_ari_minimum": float(
            consensus.median_perturbation_ari
        )
        >= float(
            cfg()["operational_gate"][
                "median_perturbation_ari_minimum"
            ]
        ),
        "no_catastrophic_scene_degradation": bool(
            (
                consensus_scenes.person_episode_coverage
                >= b6_scenes.person_episode_coverage - 0.02
            ).all()
            and (
                consensus_scenes.association_f1
                >= b6_scenes.association_f1
            ).all()
        ),
        "larger_independent_scene_validation": False,
        "interface_validation": False,
    }
    atomic_json(
        OUTPUT / "B7_DECISION.json",
        {
            "scientific_decision": (
                "PASS" if scientific_pass else "FAIL"
            ),
            "scientific_conditions": conditions,
            "operational_decision": (
                "PASS" if all(operational_conditions.values()) else "FAIL"
            ),
            "operational_conditions": operational_conditions,
            "improved_scenes": improved_scenes,
            "test_status": "SEALED",
            "test_access_count": 0,
            "compute_status": "FROZEN_AFTER_B7",
            "b8_allowed": False,
            "further_threshold_tuning_allowed": False,
        },
    )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if mode == "prepare":
        prepare()
    elif mode == "run":
        run()
    else:
        raise SystemExit("usage: run_b7.py [prepare|run]")


if __name__ == "__main__":
    main()
