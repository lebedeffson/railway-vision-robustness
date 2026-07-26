from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.operator_assistant_evidence_v2.common import (
    assert_test_sealed,
    atomic_csv,
    atomic_json,
    atomic_text,
    sha256,
)
from src.review_assistant.b6_association import (
    PAIR_FEATURES,
    TrackFragment,
    complete_link_clusters,
    pair_features,
    split_fragments,
)
from src.review_assistant.event_aggregator import box_iou


warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.*",
)


OUTPUT = PROJECT / "outputs/operator_assistant_b6"
V3 = PROJECT / "outputs/operator_assistant_evidence_v3"
CONFIG = PROJECT / "configs/operator_assistant_b6.yaml"
INDEX = PROJECT / "outputs/temporal_safety_v1/data/frame_sequence_index.csv"
DINO = Path(
    "/home/lebedeffson/.cache/huggingface/hub/models--facebook--dinov2-small/"
    "snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056"
)
VDA_REPO = Path("/home/lebedeffson/.cache/operator-b6-models/Video-Depth-Anything")
VDA_WEIGHT = Path("/home/lebedeffson/.cache/operator-b6-models/video_depth_anything_vits.pth")


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text())


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()


def _safe_torchvision_import() -> None:
    original = torch.library.register_fake

    def safe_register(op: str, *args: Any, **kwargs: Any):
        decorator = original(op, *args, **kwargs)

        def apply(function: Any):
            try:
                return decorator(function)
            except RuntimeError:
                return function

        return apply

    torch.library.register_fake = safe_register


def prepare() -> None:
    assert_test_sealed()
    if not DINO.joinpath("model.safetensors").is_file() or not VDA_WEIGHT.is_file():
        raise RuntimeError("Frozen B6 models missing")
    cfg = config()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = {
        "protocol_id": cfg["protocol_id"],
        "parent_protocol": cfg["parent_protocol"],
        "final_experiment": True,
        "git_commit_before_computation": git_commit(),
        "inputs": {
            "v3_manifest": sha256(V3 / "MANIFEST.sha256"),
            "v3_gt_person_episodes": sha256(V3 / "annotations/GT_PERSON_EPISODES.csv"),
            "v3_gt_person_boxes": sha256(V3 / "annotations/GT_PERSON_BOXES.parquet"),
            "dinov2_model": sha256(DINO / "model.safetensors"),
            "video_depth_anything_small": sha256(VDA_WEIGHT),
            "video_depth_repository_commit": cfg["models"]["video_depth_repository_commit"],
            "config": sha256(CONFIG),
        },
        "fragment_rule": cfg["fragment"],
        "pseudo_pair_rule": cfg["pseudo_pairs"],
        "uncertainty_rule": cfg["uncertainty"],
        "models_compared": cfg["models_compared"],
        "success_gate": cfg["success_gate"],
        "gt_person_usage": "EVALUATION_ONLY",
        "test_status": "SEALED",
        "test_access_count": 0,
        "training": "ASSOCIATION_LOGISTIC_ON_PSEUDOPAIRS_ONLY",
        "detector_tracker_verifier_training": "FORBIDDEN",
        "immutable": True,
    }
    atomic_json(OUTPUT / "B6_PROTOCOL_LOCK.json", lock)


def load_fragments() -> tuple[dict[str, list[TrackFragment]], dict[str, pd.DataFrame]]:
    cfg = config()
    fragments_by_scene = {}
    traces = {}
    for path in sorted(V3.glob("streams/*/PRE_AGGREGATION_OBSERVATIONS.parquet")):
        scene = path.parent.name
        trace = pd.read_parquet(path)
        accepted = trace[trace.sent_to_aggregator].copy()
        traces[scene] = accepted
        fragments_by_scene[scene] = split_fragments(
            accepted.to_dict("records"), cfg["fragment"], 4112, 2504
        )
    return fragments_by_scene, traces


def restore_feature_cache(
    fragments_by_scene: dict[str, list[TrackFragment]],
) -> bool:
    path = OUTPUT / "TRACK_FRAGMENTS.parquet"
    if not path.is_file():
        return False
    cache = pd.read_parquet(path)
    required = {
        "fragment_id",
        "appearance",
        "appearance_samples",
        "depth_median",
        "depth_slope",
        "depth_sequence",
    }
    if not required.issubset(cache.columns):
        return False
    lookup = {fragment.fragment_id: fragment for values in fragments_by_scene.values() for fragment in values}
    if set(cache.fragment_id) != set(lookup):
        return False
    for row in cache.itertuples(index=False):
        fragment = lookup[row.fragment_id]
        fragment.appearance = np.asarray(json.loads(row.appearance), dtype=np.float32)
        fragment.appearance_samples = np.asarray(
            json.loads(row.appearance_samples), dtype=np.float32
        )
        fragment.appearance_spread = float(row.appearance_spread)
        fragment.depth_median = float(row.depth_median)
        fragment.depth_slope = float(row.depth_slope)
        fragment.depth_sequence = np.asarray(
            json.loads(row.depth_sequence), dtype=np.float32
        )
    return True


def image_map() -> dict[tuple[str, int], Path]:
    index = pd.read_csv(INDEX)
    selected = pd.read_csv(V3 / "protocol/DEVELOPMENT_SCENE_SELECTION.csv")
    sequences = dict(zip(selected.scene_id, selected.source_sequence_id))
    output = {}
    for scene, sequence in sequences.items():
        rows = index[index.source_sequence_id == sequence].sort_values("frame_number")
        for frame_id, row in enumerate(rows.itertuples(index=False)):
            output[(scene, frame_id)] = Path(row.image_path)
    return output


def extract_depth(
    fragments_by_scene: dict[str, list[TrackFragment]],
    images: dict[tuple[str, int], Path],
) -> None:
    _safe_torchvision_import()
    sys.path.insert(0, str(VDA_REPO))
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    model.load_state_dict(torch.load(VDA_WEIGHT, map_location="cpu"), strict=True)
    model = model.cuda().eval()
    for scene, fragments in fragments_by_scene.items():
        frame_ids = sorted({frame for fragment in fragments for frame in range(fragment.first_frame, fragment.last_frame + 1)})
        maximum = max(frame_ids, default=-1)
        frames = []
        for frame_id in range(maximum + 1):
            frame = cv2.imread(str(images[(scene, frame_id)]))
            frame = cv2.resize(frame, (640, 392), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        depths, _ = model.infer_video_depth(
            np.stack(frames), 10, input_size=280, device="cuda", fp32=False
        )
        for fragment in fragments:
            values = []
            points = []
            for row in fragment.observations:
                frame_id = int(row["video_frame_id"])
                x = int(np.clip(((row["bbox_x1"] + row["bbox_x2"]) / 2) / 4112 * 640, 0, 639))
                y = int(np.clip(row["bbox_y2"] / 2504 * 392, 0, 391))
                values.append(float(depths[frame_id, y, x]))
                points.append(frame_id)
            scale = max(float(np.median(depths)), 1e-9)
            normalized = np.asarray(values) / scale
            fragment.depth_median = float(np.median(normalized))
            fragment.depth_sequence = normalized.astype(np.float32)
            fragment.depth_slope = (
                float(np.polyfit(points, normalized, 1)[0]) if len(points) > 1 else 0.0
            )
        np.savez_compressed(
            OUTPUT / f"depth_{scene}.npz",
            fragment_id=np.asarray([item.fragment_id for item in fragments]),
            depth_median=np.asarray([item.depth_median for item in fragments]),
            depth_slope=np.asarray([item.depth_slope for item in fragments]),
        )
        del depths
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()


def extract_appearance(
    fragments_by_scene: dict[str, list[TrackFragment]],
    images: dict[tuple[str, int], Path],
) -> None:
    _safe_torchvision_import()
    from transformers.models.dinov2.modeling_dinov2 import Dinov2Model

    model = Dinov2Model.from_pretrained(DINO).cuda().eval()
    tasks = []
    count = int(config()["features"]["appearance_samples"])
    for fragments in fragments_by_scene.values():
        for fragment in fragments:
            real = [row for row in fragment.observations if not row["is_interpolated"]]
            selected = sorted(real, key=lambda row: float(row["detector_confidence"]), reverse=True)[:count]
            for row in selected:
                tasks.append((fragment, row))
    tasks.sort(key=lambda item: (item[0].scene_id, int(item[1]["video_frame_id"])))
    embeddings: dict[str, list[np.ndarray]] = {}
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    for start in range(0, len(tasks), 32):
        batch = []
        refs = tasks[start : start + 32]
        for fragment, row in refs:
            frame = cv2.imread(str(images[(fragment.scene_id, int(row["video_frame_id"]))]))
            crop = frame[
                max(int(row["bbox_y1"]), 0) : max(int(row["bbox_y2"]), 1),
                max(int(row["bbox_x1"]), 0) : max(int(row["bbox_x2"]), 1),
            ]
            if crop.size == 0:
                crop = np.zeros((224, 224, 3), dtype=np.uint8)
            crop = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
            crop = (crop.astype(np.float32) / 255.0 - mean) / std
            batch.append(crop.transpose(2, 0, 1))
        tensor = torch.from_numpy(np.stack(batch)).cuda()
        with torch.no_grad(), torch.autocast("cuda"):
            vectors = model(pixel_values=tensor).last_hidden_state[:, 0].float().cpu().numpy()
        for (fragment, _), vector in zip(refs, vectors, strict=True):
            embeddings.setdefault(fragment.fragment_id, []).append(vector)
    for fragments in fragments_by_scene.values():
        for fragment in fragments:
            vectors = embeddings.get(fragment.fragment_id, [np.zeros(384)])
            normalized = np.stack(vectors)
            median = np.median(normalized, axis=0)
            fragment.appearance = median / max(np.linalg.norm(median), 1e-12)
            fragment.appearance_samples = normalized
            fragment.appearance_spread = float(
                np.median(np.linalg.norm(normalized - median, axis=1))
            )
    del model
    torch.cuda.empty_cache()


def assign_gt(fragments_by_scene: dict[str, list[TrackFragment]]) -> None:
    gt = pd.read_parquet(V3 / "annotations/GT_PERSON_BOXES.parquet")
    for scene, fragments in fragments_by_scene.items():
        scene_gt = gt[gt.scene_id == scene]
        for fragment in fragments:
            matches = []
            for row in fragment.observations:
                frame_gt = scene_gt[scene_gt.video_frame_id == int(row["video_frame_id"])]
                for candidate in frame_gt.itertuples(index=False):
                    if box_iou(
                        [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]],
                        [candidate.bbox_x1, candidate.bbox_y1, candidate.bbox_x2, candidate.bbox_y2],
                    ) >= 0.5:
                        matches.append(str(candidate.episode_id))
            if matches:
                fragment.gt_episode_id = max(set(matches), key=matches.count)


def _pseudo_fragment(
    source: TrackFragment, suffix: str, observations: list[dict[str, Any]], sample_indices: list[int]
) -> TrackFragment:
    samples = source.appearance_samples
    selected_samples = (
        samples[sample_indices]
        if samples is not None and len(samples) and sample_indices
        else samples
    )
    appearance = source.appearance
    spread = source.appearance_spread
    if selected_samples is not None and len(selected_samples):
        median = np.median(selected_samples, axis=0)
        appearance = median / max(float(np.linalg.norm(median)), 1e-12)
        spread = float(np.median(np.linalg.norm(selected_samples - median, axis=1)))
    depth = source.depth_sequence
    if depth is not None and len(depth) == len(source.observations):
        position_by_candidate = {
            str(row["candidate_id"]): index
            for index, row in enumerate(source.observations)
        }
        positions = [
            position_by_candidate[str(row["candidate_id"])] for row in observations
        ]
        selected_depth = depth[positions]
    else:
        selected_depth = depth
    frames = [int(row["video_frame_id"]) for row in observations]
    depth_median = (
        float(np.median(selected_depth))
        if selected_depth is not None and len(selected_depth)
        else source.depth_median
    )
    depth_slope = (
        float(np.polyfit(frames, selected_depth, 1)[0])
        if selected_depth is not None and len(selected_depth) > 1 and len(set(frames)) > 1
        else source.depth_slope
    )
    return TrackFragment(
        fragment_id=source.fragment_id + suffix,
        scene_id=source.scene_id,
        track_id=source.track_id,
        observations=observations,
        appearance=appearance,
        appearance_samples=selected_samples,
        appearance_spread=spread,
        depth_median=depth_median,
        depth_slope=depth_slope,
        depth_sequence=selected_depth,
    )


def pseudo_dataset(
    fragments: list[TrackFragment], feature_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    positives = []
    positive_kinds = {"half_split": 0, "odd_even": 0, "artificial_gap": 0}
    for fragment in fragments:
        if len(fragment.observations) < 4:
            continue
        middle = len(fragment.observations) // 2
        sample_count = 0 if fragment.appearance_samples is None else len(fragment.appearance_samples)
        left = _pseudo_fragment(
            fragment, ":HALF-A", fragment.observations[:middle], list(range(0, sample_count, 2))
        )
        right = _pseudo_fragment(
            fragment, ":HALF-B", fragment.observations[middle:], list(range(1, sample_count, 2))
        )
        positives.append(pair_features(left, right)[feature_ids])
        positive_kinds["half_split"] += 1
        odd = fragment.observations[::2]
        even = fragment.observations[1::2]
        if odd and even:
            left = _pseudo_fragment(fragment, ":ODD", odd, list(range(0, sample_count, 2)))
            right = _pseudo_fragment(fragment, ":EVEN", even, list(range(1, sample_count, 2)))
            positives.append(pair_features(left, right)[feature_ids])
            positive_kinds["odd_even"] += 1
        cut = max(1, len(fragment.observations) // 5)
        if middle - cut > 0 and middle + cut < len(fragment.observations):
            left = _pseudo_fragment(
                fragment, ":GAP-A", fragment.observations[: middle - cut], list(range(0, sample_count, 2))
            )
            right = _pseudo_fragment(
                fragment, ":GAP-B", fragment.observations[middle + cut :], list(range(1, sample_count, 2))
            )
            positives.append(pair_features(left, right)[feature_ids])
            positive_kinds["artificial_gap"] += 1
    negatives = []
    negative_kinds = {"different_scene": 0, "simultaneous_distinct_track": 0}
    for left, right in itertools.combinations(fragments, 2):
        simultaneous = not (left.last_frame < right.first_frame or right.last_frame < left.first_frame)
        if left.scene_id != right.scene_id:
            negatives.append(pair_features(left, right)[feature_ids])
            negative_kinds["different_scene"] += 1
        elif simultaneous and left.track_id != right.track_id:
            negatives.append(pair_features(left, right)[feature_ids])
            negative_kinds["simultaneous_distinct_track"] += 1
    rng = np.random.default_rng(int(config()["pseudo_pairs"]["seed"]))
    maximum = len(positives) * int(config()["pseudo_pairs"]["maximum_negative_ratio"])
    if len(negatives) > maximum:
        negatives = [negatives[index] for index in rng.choice(len(negatives), maximum, replace=False)]
    audit = {
        **positive_kinds,
        **negative_kinds,
        "positive_pairs_used": len(positives),
        "negative_pairs_used": len(negatives),
        "gt_fields_used": 0,
    }
    return (
        np.vstack(positives + negatives),
        np.asarray([1] * len(positives) + [0] * len(negatives)),
        audit,
    )


FEATURE_SUBSETS = {
    "B6-G": [1, 2, 3, 4, 5, 6],
    "B6-A": [0, 1, 3, 4, 5, 6],
    "B6-FULL": list(range(7)),
}


def perturb_fragment(
    source: TrackFragment, rng: np.random.Generator, suffix: str
) -> TrackFragment:
    indexed_rows = list(enumerate(source.observations))
    if len(indexed_rows) > 2:
        weakest = min(
            indexed_rows, key=lambda item: float(item[1]["output_confidence"])
        )
        indexed_rows = [item for item in indexed_rows if item[0] != weakest[0]]
    if len(indexed_rows) > 3:
        trim_start = int(rng.integers(0, 2))
        trim_end = int(rng.integers(0, 2))
        indexed_rows = indexed_rows[
            trim_start : len(indexed_rows) - trim_end if trim_end else None
        ]
    keep_probability = float(rng.uniform(0.70, 0.90))
    retained = [item for item in indexed_rows if rng.random() <= keep_probability]
    if not retained:
        retained = indexed_rows[:1]
    rows = []
    noise_ratio = float(config()["pseudo_pairs"]["coordinate_noise_std"])
    for _, original in retained:
        row = dict(original)
        width = max(float(row["bbox_x2"]) - float(row["bbox_x1"]), 1.0)
        height = max(float(row["bbox_y2"]) - float(row["bbox_y1"]), 1.0)
        noise = rng.normal(0, [width, height, width, height]) * noise_ratio
        for key, value in zip(
            ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"], noise, strict=True
        ):
            row[key] = float(row[key]) + float(value)
        rows.append(row)
    samples = source.appearance_samples
    sample_indices: list[int] = []
    if samples is not None and len(samples):
        sample_size = min(len(samples), max(1, int(rng.integers(1, len(samples) + 1))))
        sample_indices = sorted(
            rng.choice(len(samples), sample_size, replace=False).tolist()
        )
    fragment = _pseudo_fragment(
        source,
        suffix,
        [row for _, row in zip(retained, rows, strict=True)],
        sample_indices,
    )
    if source.depth_sequence is not None and len(source.depth_sequence) == len(
        source.observations
    ):
        indices = [index for index, _ in retained]
        fragment.depth_sequence = source.depth_sequence[indices]
        fragment.depth_median = float(np.median(fragment.depth_sequence))
        frames = [int(row["video_frame_id"]) for row in fragment.observations]
        if len(frames) > 1 and len(set(frames)) > 1:
            fragment.depth_slope = float(
                np.polyfit(frames, fragment.depth_sequence, 1)[0]
            )
    return fragment


def score_pairs(
    model: Any,
    fragments: list[TrackFragment],
    indices: list[int],
    seed: int,
) -> tuple[dict[tuple[str, str], tuple[float, float]], dict[tuple[str, str], list[float]]]:
    output: dict[tuple[str, str], tuple[float, float]] = {}
    repeats: dict[tuple[str, str], list[float]] = {}
    repetition_count = int(config()["uncertainty"]["repetitions"])
    perturbed: dict[str, list[TrackFragment]] = {}
    for fragment in fragments:
        variants = []
        for repeat in range(repetition_count):
            fragment_seed = int.from_bytes(
                hashlib.sha256(
                    f"{seed}:{fragment.fragment_id}:{repeat}".encode()
                ).digest()[:8],
                "little",
            )
            variants.append(
                perturb_fragment(
                    fragment,
                    np.random.default_rng(fragment_seed),
                    f":P{repeat}",
                )
            )
        perturbed[fragment.fragment_id] = variants
    keys = []
    matrices = []
    for left, right in itertools.combinations(fragments, 2):
        if left.scene_id != right.scene_id:
            continue
        key = tuple(sorted((left.fragment_id, right.fragment_id)))
        keys.append(key)
        matrices.append(
            np.stack(
                [
                    pair_features(
                        perturbed[left.fragment_id][repeat],
                        perturbed[right.fragment_id][repeat],
                    )[indices]
                    for repeat in range(repetition_count)
                ]
            )
        )
    if not matrices:
        return output, repeats
    probabilities = model.predict_proba(np.concatenate(matrices, axis=0))[:, 1]
    probabilities = probabilities.reshape(len(keys), repetition_count)
    for key, values in zip(keys, probabilities, strict=True):
        values_list = values.astype(float).tolist()
        output[key] = (float(np.mean(values)), float(np.std(values)))
        repeats[key] = values_list
    return output, repeats


def cluster_labels(fragments: list[TrackFragment], clusters: list[list[str]]) -> np.ndarray:
    membership = {fragment_id: index for index, cluster in enumerate(clusters) for fragment_id in cluster}
    return np.asarray([membership[fragment.fragment_id] for fragment in fragments])


def evaluate_clustering(
    fragments: list[TrackFragment],
    clusters: list[list[str]],
    expected_episode_ids: set[str],
) -> dict[str, float]:
    valid = [fragment for fragment in fragments if fragment.gt_episode_id]
    predicted = cluster_labels(fragments, clusters)
    positions = [fragments.index(fragment) for fragment in valid]
    predicted = predicted[positions]
    gt_values = [fragment.gt_episode_id for fragment in valid]
    gt_index = {value: index for index, value in enumerate(sorted(set(gt_values)))}
    truth = np.asarray([gt_index[value] for value in gt_values])
    y_true, y_pred = [], []
    for left, right in itertools.combinations(range(len(valid)), 2):
        y_true.append(int(truth[left] == truth[right]))
        y_pred.append(int(predicted[left] == predicted[right]))
    if y_true:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
    else:
        precision = recall = f1 = 0.0
    merged = sum(y_pred)
    cross = sum(pred and not true for pred, true in zip(y_pred, y_true, strict=True))
    represented = {fragment.gt_episode_id for fragment in valid}
    b3_precision = []
    b3_recall = []
    for index in range(len(valid)):
        predicted_members = predicted == predicted[index]
        truth_members = truth == truth[index]
        overlap = int(np.logical_and(predicted_members, truth_members).sum())
        b3_precision.append(overlap / max(int(predicted_members.sum()), 1))
        b3_recall.append(overlap / max(int(truth_members.sum()), 1))
    bcubed_precision = float(np.mean(b3_precision)) if b3_precision else 0.0
    bcubed_recall = float(np.mean(b3_recall)) if b3_recall else 0.0
    bcubed_f1 = (
        2 * bcubed_precision * bcubed_recall / (bcubed_precision + bcubed_recall)
        if bcubed_precision + bcubed_recall
        else 0.0
    )
    return {
        "association_precision": float(precision),
        "association_recall": float(recall),
        "association_f1": float(f1),
        "cross_person_merge_rate": cross / max(merged, 1),
        "same_person_split_recovery": float(recall),
        "bcubed_precision": bcubed_precision,
        "bcubed_recall": bcubed_recall,
        "bcubed_f1": bcubed_f1,
        "adjusted_rand_index": float(adjusted_rand_score(truth, predicted)),
        "person_episode_coverage": len(represented & expected_episode_ids)
        / max(len(expected_episode_ids), 1),
        "clusters": len(clusters),
    }


def baseline_clusters(method: str, fragments: list[TrackFragment], scene: str) -> list[list[str]]:
    if method in {"B1", "B5"}:
        groups: dict[int, list[str]] = {}
        for fragment in fragments:
            groups.setdefault(fragment.track_id, []).append(fragment.fragment_id)
        return list(groups.values())
    if method == "B4":
        events = [
            json.loads(line)
            for line in (V3 / f"streams/{scene}/HAZARD_EVENTS.jsonl").read_text().splitlines()
        ]
        candidate_to_fragment = {
            candidate: fragment.fragment_id
            for fragment in fragments
            for candidate in fragment.candidate_ids
        }
        event_memberships: dict[str, dict[int, int]] = {}
        for event_index, event in enumerate(events):
            for candidate in event["candidate_ids"]:
                fragment_id = candidate_to_fragment.get(candidate)
                if fragment_id:
                    counts = event_memberships.setdefault(fragment_id, {})
                    counts[event_index] = counts.get(event_index, 0) + 1
        groups_by_event: dict[int, list[str]] = {}
        represented = set()
        for fragment_id, counts in event_memberships.items():
            event_index = max(counts, key=lambda key: (counts[key], -key))
            groups_by_event.setdefault(event_index, []).append(fragment_id)
            represented.add(fragment_id)
        groups = [sorted(group) for _, group in sorted(groups_by_event.items())]
        groups.extend([[fragment.fragment_id] for fragment in fragments if fragment.fragment_id not in represented])
        return groups
    raise ValueError(method)


def review_bundles(
    scene: str, fragments: list[TrackFragment], clusters: list[list[str]]
) -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in (V3 / f"streams/{scene}/HAZARD_EVENTS.jsonl").read_text().splitlines()
    ]
    candidate_event: dict[str, int] = {}
    for event_index, event in enumerate(events):
        for candidate_id in event["candidate_ids"]:
            candidate_event[candidate_id] = event_index
    fragment_lookup = {fragment.fragment_id: fragment for fragment in fragments}
    assigned: dict[str, list[dict[str, Any]]] = {}
    for cluster_index, cluster in enumerate(clusters):
        counts: dict[int, int] = {}
        for fragment_id in cluster:
            for candidate_id in fragment_lookup[fragment_id].candidate_ids:
                if candidate_id in candidate_event:
                    event_index = candidate_event[candidate_id]
                    counts[event_index] = counts.get(event_index, 0) + 1
        if counts:
            bundle_id = f"{scene}:RB{max(counts, key=lambda key: (counts[key], -key)):04d}"
        else:
            bundle_id = f"{scene}:RB-UNASSIGNED-{cluster_index:04d}"
        assigned.setdefault(bundle_id, []).append(
            {
                "person_episode_id": f"{scene}:B6-PE{cluster_index:04d}",
                "track_fragment_ids": sorted(cluster),
                "candidate_ids": sorted(
                    candidate_id
                    for fragment_id in cluster
                    for candidate_id in fragment_lookup[fragment_id].candidate_ids
                ),
            }
        )
    return [
        {
            "review_bundle_id": bundle_id,
            "scene_id": scene,
            "semantics": "SPATIOTEMPORAL_REVIEW_CARD_NOT_HAZARD_GROUND_TRUTH",
            "person_episodes": children,
        }
        for bundle_id, children in sorted(assigned.items())
    ]


def run() -> None:
    assert_test_sealed()
    if not (OUTPUT / "B6_PROTOCOL_LOCK.json").is_file():
        raise RuntimeError("B6 protocol lock missing")
    fragments_by_scene, _ = load_fragments()
    cache_restored = restore_feature_cache(fragments_by_scene)
    if not cache_restored:
        images = image_map()
        extract_depth(fragments_by_scene, images)
        extract_appearance(fragments_by_scene, images)
    assign_gt(fragments_by_scene)
    gt_episodes = pd.read_csv(V3 / "annotations/GT_PERSON_EPISODES.csv")
    expected_by_scene = {
        scene: set(group.episode_id.astype(str))
        for scene, group in gt_episodes.groupby("scene_id")
    }
    all_fragments = [fragment for values in fragments_by_scene.values() for fragment in values]
    feature_rows = []
    for fragment in all_fragments:
        feature_rows.append(
            {
                "fragment_id": fragment.fragment_id,
                "scene_id": fragment.scene_id,
                "track_id": fragment.track_id,
                "start_frame": fragment.first_frame,
                "end_frame": fragment.last_frame,
                "real_observation_count": fragment.real_count,
                "interpolated_count": fragment.interpolated_count,
                "candidate_ids": json.dumps(fragment.candidate_ids),
                "appearance": json.dumps(fragment.appearance.tolist() if fragment.appearance is not None else []),
                "appearance_samples": json.dumps(
                    fragment.appearance_samples.tolist()
                    if fragment.appearance_samples is not None
                    else []
                ),
                "appearance_spread": fragment.appearance_spread,
                "depth_median": fragment.depth_median,
                "depth_slope": fragment.depth_slope,
                "depth_sequence": json.dumps(
                    fragment.depth_sequence.tolist()
                    if fragment.depth_sequence is not None
                    else []
                ),
                "gt_episode_id_evaluation_only": fragment.gt_episode_id,
                "signature": fragment.signature(),
            }
        )
    pd.DataFrame(feature_rows).to_parquet(OUTPUT / "TRACK_FRAGMENTS.parquet", index=False)
    result_rows = []
    score_rows = []
    cluster_records = []
    perturbation_rows = []
    pseudo_audit_rows = []
    hierarchy_records = []
    for held_scene, held_fragments in fragments_by_scene.items():
        train = [fragment for scene, values in fragments_by_scene.items() if scene != held_scene for fragment in values]
        for method in ["B1", "B4", "B5"]:
            clusters = baseline_clusters(method, held_fragments, held_scene)
            bundles = review_bundles(held_scene, held_fragments, clusters)
            result_rows.append(
                {
                    "scene_id": held_scene,
                    "method": method,
                    **evaluate_clustering(
                        held_fragments, clusters, expected_by_scene[held_scene]
                    ),
                    "review_bundle_cards": len(bundles),
                    "abstention_rate": 0.0,
                    "direct_replay_match": True,
                }
            )
        for model_name, indices in FEATURE_SUBSETS.items():
            x, y, pseudo_audit = pseudo_dataset(train, indices)
            pseudo_audit_rows.append(
                {"held_scene": held_scene, "method": model_name, **pseudo_audit}
            )
            model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=20260726))
            model.fit(x, y)
            scores, repeats = score_pairs(model, held_fragments, indices, 20260726)
            clusters, abstentions = complete_link_clusters(
                held_fragments,
                scores,
                config()["uncertainty"]["safe_link_threshold"],
                config()["uncertainty"]["cannot_link_threshold"],
            )
            replay_clusters, replay_abstentions = complete_link_clusters(
                held_fragments,
                dict(scores),
                config()["uncertainty"]["safe_link_threshold"],
                config()["uncertainty"]["cannot_link_threshold"],
            )
            direct_signature = hashlib.sha256(
                json.dumps(sorted(clusters), sort_keys=True).encode()
            ).hexdigest()
            replay_signature = hashlib.sha256(
                json.dumps(sorted(replay_clusters), sort_keys=True).encode()
            ).hexdigest()
            replay_match = (
                sorted(clusters) == sorted(replay_clusters)
                and abstentions == replay_abstentions
            )
            bundles = review_bundles(held_scene, held_fragments, clusters)
            metrics = evaluate_clustering(
                held_fragments, clusters, expected_by_scene[held_scene]
            )
            total_pairs = max(len(scores), 1)
            result_rows.append(
                {
                    "scene_id": held_scene,
                    "method": model_name,
                    **metrics,
                    "review_bundle_cards": len(bundles),
                    "abstention_rate": abstentions / total_pairs,
                    "direct_replay_match": replay_match,
                }
            )
            for key, (mu, sigma) in scores.items():
                score_rows.append({"scene_id": held_scene, "method": model_name, "left_fragment_id": key[0], "right_fragment_id": key[1], "mu": mu, "sigma": sigma, "safe_score": mu - 1.96 * sigma, "upper_score": mu + 1.96 * sigma})
            repeat_labels = []
            for repeat in range(16):
                repeat_scores = {key: (values[repeat], 0.0) for key, values in repeats.items()}
                repeat_clusters, _ = complete_link_clusters(
                    held_fragments, repeat_scores,
                    config()["uncertainty"]["safe_link_threshold"],
                    config()["uncertainty"]["cannot_link_threshold"],
                )
                repeat_labels.append(cluster_labels(held_fragments, repeat_clusters))
            pairwise_aris = [
                float(adjusted_rand_score(left, right))
                for left, right in itertools.combinations(repeat_labels, 2)
            ]
            for comparison, ari in enumerate(pairwise_aris):
                perturbation_rows.append(
                    {
                        "scene_id": held_scene,
                        "method": model_name,
                        "comparison": comparison,
                        "pairwise_perturbation_ari": ari,
                    }
                )
            cluster_records.append(
                {
                    "scene_id": held_scene,
                    "method": model_name,
                    "clusters": clusters,
                    "direct_signature": direct_signature,
                    "replay_signature": replay_signature,
                    "direct_replay_match": replay_match,
                    "median_pairwise_perturbation_ari": float(np.median(pairwise_aris)),
                }
            )
            if model_name == "B6-FULL":
                hierarchy_records.extend(bundles)
    results = pd.DataFrame(result_rows)
    atomic_csv(OUTPUT / "B6_RESULTS_PER_SCENE.csv", results)
    overall = results.groupby("method", as_index=False).agg(
        scenes=("scene_id", "nunique"),
        association_precision=("association_precision", "mean"),
        association_recall=("association_recall", "mean"),
        association_f1=("association_f1", "mean"),
        cross_person_merge_rate=("cross_person_merge_rate", "mean"),
        same_person_split_recovery=("same_person_split_recovery", "mean"),
        bcubed_precision=("bcubed_precision", "mean"),
        bcubed_recall=("bcubed_recall", "mean"),
        bcubed_f1=("bcubed_f1", "mean"),
        adjusted_rand_index=("adjusted_rand_index", "mean"),
        person_episode_coverage=("person_episode_coverage", "mean"),
        clusters=("clusters", "sum"),
        review_bundle_cards=("review_bundle_cards", "sum"),
        abstention_rate=("abstention_rate", "mean"),
    )
    atomic_csv(OUTPUT / "B6_RESULTS_OVERALL.csv", overall)
    atomic_csv(OUTPUT / "PAIR_SCORES.csv", pd.DataFrame(score_rows))
    atomic_csv(OUTPUT / "PERTURBATION_STABILITY.csv", pd.DataFrame(perturbation_rows))
    atomic_csv(OUTPUT / "PSEUDO_PAIR_AUDIT.csv", pd.DataFrame(pseudo_audit_rows))
    with (OUTPUT / "CLUSTERS.jsonl").open("w") as handle:
        for row in cluster_records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (OUTPUT / "B6_HIERARCHY.jsonl").open("w") as handle:
        for row in hierarchy_records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    atomic_json(
        OUTPUT / "B6_REPLAY_AUDIT.json",
        {
            "records": [
                {
                    "scene_id": row["scene_id"],
                    "method": row["method"],
                    "direct_signature": row["direct_signature"],
                    "replay_signature": row["replay_signature"],
                    "exact_match": row["direct_replay_match"],
                }
                for row in cluster_records
            ],
            "exact_match": all(row["direct_replay_match"] for row in cluster_records),
            "all_fragments_preserved": sum(
                len(cluster)
                for row in cluster_records
                if row["method"] == "B6-FULL"
                for cluster in row["clusters"]
            )
            == len(all_fragments),
        },
    )
    evaluate_gate(results, overall, pd.DataFrame(cluster_records))


def evaluate_gate(results: pd.DataFrame, overall: pd.DataFrame, clusters: pd.DataFrame) -> None:
    b5 = results[results.method == "B5"].set_index("scene_id")
    full = results[results.method == "B6-FULL"].set_index("scene_id")
    b1 = results[results.method == "B1"].set_index("scene_id")
    median_ari = float(
        clusters[
            clusters.method == "B6-FULL"
        ].median_pairwise_perturbation_ari.median()
    )
    conditions = {
        "coverage_noninferior_to_b5": bool((full.person_episode_coverage >= b5.person_episode_coverage).all()),
        "cross_person_merge_below_b5": bool(full.cross_person_merge_rate.mean() < b5.cross_person_merge_rate.mean()),
        "association_f1_above_b5": bool(full.association_f1.mean() > b5.association_f1.mean()),
        "split_recovery_above_b1": bool(full.same_person_split_recovery.mean() > b1.same_person_split_recovery.mean()),
        "replay_exact": bool(full.direct_replay_match.all()),
        "fragments_preserved": bool(
            json.loads((OUTPUT / "B6_REPLAY_AUDIT.json").read_text())[
                "all_fragments_preserved"
            ]
        ),
        "median_perturbation_ari_minimum": median_ari >= 0.90,
        "card_increase_vs_b5_maximum": float(full.review_bundle_cards.sum())
        <= float(b5.review_bundle_cards.sum()) * 1.20,
        "improved_scenes_minimum": int((full.association_f1 > b5.association_f1).sum()) >= 4,
        "worst_scene_coverage_degradation_maximum": float((b5.person_episode_coverage - full.person_episode_coverage).max()) <= 0.02,
    }
    passed = all(conditions.values())
    atomic_json(
        OUTPUT / "B6_DECISION.json",
        {
            "decision": "PASS" if passed else "FAIL",
            "conditions": conditions,
            "median_perturbation_ari": median_ari,
            "test_status": "SEALED",
            "test_access_count": 0,
            "compute_status": "FROZEN_AFTER_B6",
            "further_experiments_allowed": False,
        },
    )


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if mode == "prepare":
        prepare()
    elif mode == "run":
        run()
    else:
        raise SystemExit("usage: run_b6.py [prepare|run]")


if __name__ == "__main__":
    main()
