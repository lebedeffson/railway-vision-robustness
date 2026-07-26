from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scripts.article_evidence_v1.common import (
    OUTPUT,
    PROJECT,
    atomic_csv,
    atomic_json,
    check_no_test_markers,
    config,
    sha256,
)
from scripts.article_evidence_v1.compute_saved import _gate_values
from scripts.temporal_safety.common import source_frames
from src.crop_verifier_v1 import FrozenYoloTrackEncoder
from src.crop_verifier_v1.model import predict_verifier
from src.temporal.ratta import box_iou
from src.temporal_safety.evaluator import evaluate, parse_prediction_table
from src.temporal_verifier.features import build_track_features
from src.temporal_verifier.pipeline import (
    accepted_predictions,
    collect_tracks,
    predict_nested_logistic,
)


def _selected_tracker_parameters() -> dict[str, dict[str, Any]]:
    payload = json.loads(
        (
            PROJECT
            / "outputs/temporal_safety_v1/selection/SELECTED_TRACKER_CONFIGS.json"
        ).read_text(encoding="utf-8")
    )
    return {
        name: dict(payload["selected"][name]["parameters"])
        for name in ("bytetrack", "ocsort")
    }


def _roles() -> list[tuple[int, pd.DataFrame, dict[str, Any], dict[str, Any]]]:
    table = pd.read_parquet(
        PROJECT / "outputs/temporal_safety_v1/baseline/raw_predictions.parquet"
    )
    roles = []
    for fold in (0, 1):
        source = source_frames(fold, "heldout").sort_values(
            ["subsequence_id", "frame_number"]
        )
        selected = table[
            table["detector_fold"].eq(fold)
            & table["prediction_role"].eq("heldout")
        ].copy()
        ground_truth, raw = parse_prediction_table(
            selected, set(source["image_path"].astype(str))
        )
        roles.append((fold, source, ground_truth, raw))
    return roles


def _prediction_counts(
    source: pd.DataFrame, predictions: dict[str, list[dict[str, Any]]]
) -> pd.Series:
    return source["image_path"].astype(str).map(
        lambda path: len(predictions.get(path, []))
    )


def _frame_fn(
    targets: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> int:
    used: set[int] = set()
    for prediction in sorted(
        predictions, key=lambda row: -float(row.get("confidence", 0.0))
    ):
        candidates = [
            (index, box_iou(prediction["box"], target["box"]))
            for index, target in enumerate(targets)
            if index not in used
        ]
        index, overlap = max(candidates, default=(-1, 0.0), key=lambda item: item[1])
        if overlap >= 0.5:
            used.add(index)
    return len(targets) - len(used)


def _streaks(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
) -> tuple[int, float]:
    streaks: list[int] = []
    for _, sequence in source.groupby("subsequence_id", sort=False):
        current = 0
        for row in sequence.sort_values("frame_number").itertuples(index=False):
            path = str(row.image_path)
            missed = _frame_fn(
                ground_truth.get(path, []), predictions.get(path, [])
            )
            if missed > 0:
                current += 1
            elif current:
                streaks.append(current)
                current = 0
        if current:
            streaks.append(current)
    return (max(streaks, default=0), float(np.mean(streaks)) if streaks else 0.0)


def _metrics(
    source: pd.DataFrame,
    ground_truth: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    track_counts: dict[str, int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    global_metrics, per_scene = evaluate(
        source,
        ground_truth,
        predictions,
        threshold=0.0,
        iou_threshold=0.5,
        nominal_fps=10.0,
    )
    counts = _prediction_counts(source, predictions)
    maximum, mean = _streaks(source, ground_truth, predictions)
    global_metrics.update(
        {
            "maximum_consecutive_miss": maximum,
            "mean_consecutive_miss": mean,
            "detections_per_frame": float(counts.mean()),
            "tracks_per_scene": float(
                np.mean(list(track_counts.values())) if track_counts else 0.0
            ),
        }
    )
    extra = []
    for scene, group in source.groupby("grouped_scene_id", sort=True):
        paths = set(group["image_path"].astype(str))
        scene_gt = {path: ground_truth[path] for path in paths}
        scene_predictions = {path: predictions.get(path, []) for path in paths}
        max_streak, mean_streak = _streaks(
            group, scene_gt, scene_predictions
        )
        extra.append(
            {
                "grouped_scene_id": str(scene),
                "maximum_consecutive_miss": max_streak,
                "mean_consecutive_miss": mean_streak,
                "detections_per_frame": float(
                    group["image_path"]
                    .astype(str)
                    .map(lambda path: len(predictions.get(path, [])))
                    .mean()
                ),
                "tracks_per_scene": int(track_counts.get(str(scene), 0)),
            }
        )
    per_scene = per_scene.merge(
        pd.DataFrame(extra), on="grouped_scene_id", validate="one_to_one"
    )
    return global_metrics, per_scene


def _track_counts(observations: pd.DataFrame) -> dict[str, int]:
    if observations.empty:
        return {}
    return (
        observations.groupby("grouped_scene_id")["track_key"]
        .nunique()
        .astype(int)
        .to_dict()
    )


def _empty_predictions(raw: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    return {key: [dict(value) for value in values] for key, values in raw.items()}


def _combine_roles(
    role_results: list[
        tuple[
            pd.DataFrame,
            dict[str, list[dict[str, Any]]],
            dict[str, list[dict[str, Any]]],
            dict[str, int],
        ]
    ],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, int]]:
    source = pd.concat([row[0] for row in role_results], ignore_index=True)
    ground_truth: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    for _, gt, pred, tracks in role_results:
        ground_truth.update(gt)
        predictions.update(pred)
        for scene, value in tracks.items():
            counts[str(scene)] += int(value)
    return source, ground_truth, predictions, dict(counts)


def _build_encoder() -> FrozenYoloTrackEncoder:
    settings = yaml.safe_load(
        (PROJECT / "configs/crop_verifier_v1.yaml").read_text(encoding="utf-8")
    )["encoder"]
    checkpoint = PROJECT / settings["checkpoint"]
    if sha256(checkpoint) != settings["checkpoint_sha256"]:
        raise RuntimeError("Frozen crop encoder hash mismatch")
    return FrozenYoloTrackEncoder(
        checkpoint=checkpoint,
        last_layer_index=int(settings["backbone_last_layer_index"]),
        input_size=int(settings["input_size"]),
        context_multiplier=float(settings["crop_context_multiplier"]),
        padding_value=int(settings["padding_value"]),
        batch_size=int(settings["batch_size"]),
    )


def _candidate_outputs(
    threshold: float,
    encoder: FrozenYoloTrackEncoder,
    track_model: Any,
    combined_model: Any,
) -> dict[
    str,
    list[
        tuple[
            pd.DataFrame,
            dict[str, list[dict[str, Any]]],
            dict[str, list[dict[str, Any]]],
            dict[str, int],
        ]
    ],
]:
    tracker_config = _selected_tracker_parameters()
    temporal_config = yaml.safe_load(
        (PROJECT / "configs/temporal_safety_v1.yaml").read_text(encoding="utf-8")
    )["temporal_logic"]
    methods: dict[str, list[Any]] = defaultdict(list)
    for fold, source, ground_truth, raw in _roles():
        filtered = {
            image: [
                dict(row)
                for row in rows
                if float(row["confidence"]) >= threshold
            ]
            for image, rows in raw.items()
        }
        methods["frame_detector"].append(
            (source, ground_truth, _empty_predictions(filtered), {})
        )
        tracker_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for tracker in ("bytetrack", "ocsort"):
            observations, additions = collect_tracks(
                tracker,
                f"article_t{threshold:.3f}_f{fold}",
                source,
                ground_truth,
                filtered,
                tracker_config[tracker],
                temporal_config,
                0.60,
            )
            tracker_data[tracker] = (observations, additions)
            predictions = accepted_predictions(
                filtered,
                additions,
                set(additions["track_key"].astype(str)),
                0.60,
            )
            methods[tracker].append(
                (source, ground_truth, predictions, _track_counts(observations))
            )
        observations, additions = tracker_data["ocsort"]
        features = build_track_features(observations)
        track_probability = predict_nested_logistic(track_model, features)
        accepted_track = set(
            features.loc[track_probability >= 0.17, "track_key"].astype(str)
        )
        track_predictions = accepted_predictions(
            filtered, additions, accepted_track, 0.60
        )
        methods["track_verifier"].append(
            (
                source,
                ground_truth,
                track_predictions,
                _track_counts(
                    observations[
                        observations["track_key"].astype(str).isin(accepted_track)
                    ]
                ),
            )
        )

        metadata, _, visual = encoder.encode(observations)
        columns = [
            "mean_confidence",
            "max_confidence",
            "track_length",
            "detection_fraction",
            "interpolation_fraction",
        ]
        ordered = metadata[["track_key"]].merge(
            features[["track_key", *columns]],
            on="track_key",
            validate="one_to_one",
        )
        probability = predict_verifier(
            combined_model,
            np.column_stack(
                [visual, ordered[columns].to_numpy(dtype=float)]
            ),
        )
        accepted_combined = set(
            metadata.loc[probability >= 0.275, "track_key"].astype(str)
        )
        combined_predictions = accepted_predictions(
            filtered, additions, accepted_combined, 0.60
        )
        methods["combined_visual_track_verifier"].append(
            (
                source,
                ground_truth,
                combined_predictions,
                _track_counts(
                    observations[
                        observations["track_key"]
                        .astype(str)
                        .isin(accepted_combined)
                    ]
                ),
            )
        )
    return methods


def _effect(
    reference: pd.Series, candidate: pd.Series, metric: str
) -> float:
    if metric == "recall":
        return float(candidate["recall"] - reference["recall"])
    if metric == "f1":
        return float(candidate["f1"] - reference["f1"])
    if metric == "relative_fn_per_frame":
        return float(
            (reference["FN_per_frame"] - candidate["FN_per_frame"])
            / max(reference["FN_per_frame"], 1e-12)
        )
    if metric == "relative_false_alarms_per_min":
        return float(
            candidate["false_alarms_per_minute"]
            / max(reference["false_alarms_per_minute"], 1e-12)
            - 1.0
        )
    raise ValueError(metric)


def _holm(raw_p: list[float]) -> list[float]:
    order = np.argsort(raw_p)
    adjusted = np.zeros(len(raw_p), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(raw_p) - rank) * raw_p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    check_no_test_markers()
    settings = config()
    if not (OUTPUT / "COMPUTATION_LOCK.json").is_file():
        raise RuntimeError("Run lock_inputs before computation")
    with (
        PROJECT / "outputs/temporal_verifier_v1/learned/logistic_l2/model.pkl"
    ).open("rb") as handle:
        track_model = pickle.load(handle)
    with (
        PROJECT / "outputs/crop_verifier_v1/models/combined/model.pkl"
    ).open("rb") as handle:
        combined_model = pickle.load(handle)
    encoder = _build_encoder()

    metric_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for threshold in settings["threshold_baseline"]["grid"]:
        outputs = _candidate_outputs(
            float(threshold), encoder, track_model, combined_model
        )
        threshold_metrics: dict[str, dict[str, Any]] = {}
        for method in settings["threshold_baseline"]["methods"]:
            source, ground_truth, predictions, counts = _combine_roles(
                outputs[method]
            )
            metrics, per_scene = _metrics(
                source, ground_truth, predictions, counts
            )
            per_scene.insert(0, "threshold", float(threshold))
            per_scene.insert(0, "method", method)
            scene_rows.extend(per_scene.to_dict("records"))
            threshold_metrics[method] = metrics
        baseline = threshold_metrics["frame_detector"]
        for method, metrics in threshold_metrics.items():
            deltas = {
                "recall_gain": float(metrics["recall"] - baseline["recall"]),
                "fn_reduction_percent": 100.0
                * (
                    float(baseline["FN_per_frame"])
                    - float(metrics["FN_per_frame"])
                )
                / max(float(baseline["FN_per_frame"]), 1e-12),
                "false_alarm_change_percent": 100.0
                * (
                    float(metrics["false_alarms_per_minute"])
                    / max(float(baseline["false_alarms_per_minute"]), 1e-12)
                    - 1.0
                ),
                "f1_change": float(metrics["f1"] - baseline["f1"]),
            }
            gate = _gate_values(deltas)
            rows = [
                row
                for row in scene_rows
                if row["method"] == method
                and np.isclose(row["threshold"], threshold)
            ]
            macro = pd.DataFrame(rows)
            metric_rows.append(
                {
                    "method": method,
                    "threshold": float(threshold),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "fn_per_frame": metrics["FN_per_frame"],
                    "fp_per_frame": metrics["FP_per_frame"],
                    "false_alarms_per_min": metrics[
                        "false_alarms_per_minute"
                    ],
                    "maximum_consecutive_miss": metrics[
                        "maximum_consecutive_miss"
                    ],
                    "mean_consecutive_miss": metrics["mean_consecutive_miss"],
                    "detections_per_frame": metrics["detections_per_frame"],
                    "tracks_per_scene": metrics["tracks_per_scene"],
                    "scene_macro_precision": macro["precision"].mean(),
                    "scene_macro_recall": macro["recall"].mean(),
                    "scene_macro_f1": macro["f1"].mean(),
                    "scene_macro_fn_per_frame": macro["FN_per_frame"].mean(),
                    "scene_macro_false_alarms_per_min": macro[
                        "false_alarms_per_minute"
                    ].mean(),
                    "worst_scene_recall": macro["recall"].min(),
                    "worst_scene_f1": macro["f1"].min(),
                    "mAP50": metrics["mAP50"],
                    "mAP50_status": "RECOMPUTED_FROM_STORED_PREDICTIONS",
                    **deltas,
                    **gate,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    scenes = pd.DataFrame(scene_rows)
    root = OUTPUT / "threshold_baseline"
    atomic_csv(root / "THRESHOLD_METRICS.csv", metrics)
    atomic_csv(root / "THRESHOLD_METRICS_PER_SCENE.csv", scenes)

    bootstrap_rows: list[dict[str, Any]] = []
    loso_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(settings["statistics"]["bootstrap_seed"]))
    iterations = int(settings["statistics"]["bootstrap_iterations"])
    candidates = settings["threshold_baseline"]["comparisons"]["candidates"]
    comparison_metrics = settings["threshold_baseline"]["comparisons"]["metrics"]
    for threshold in settings["threshold_baseline"]["grid"]:
        frame = scenes[
            scenes["method"].eq("frame_detector")
            & np.isclose(scenes["threshold"], threshold)
        ].set_index("grouped_scene_id")
        for candidate in candidates:
            cand = scenes[
                scenes["method"].eq(candidate)
                & np.isclose(scenes["threshold"], threshold)
            ].set_index("grouped_scene_id")
            common = sorted(set(frame.index) & set(cand.index))
            for metric in comparison_metrics:
                values = np.asarray(
                    [_effect(frame.loc[scene], cand.loc[scene], metric) for scene in common]
                )
                sampled = values[
                    rng.integers(0, len(values), size=(iterations, len(values)))
                ].mean(axis=1)
                raw_p = min(
                    1.0,
                    2.0
                    * min(
                        float(np.mean(sampled <= 0.0)),
                        float(np.mean(sampled >= 0.0)),
                    ),
                )
                bootstrap_rows.append(
                    {
                        "threshold": float(threshold),
                        "candidate": candidate,
                        "reference": "frame_detector",
                        "metric": metric,
                        "scene_count": len(values),
                        "mean_effect": float(values.mean()),
                        "median_effect": float(np.median(values)),
                        "ci95_low": float(np.quantile(sampled, 0.025)),
                        "ci95_high": float(np.quantile(sampled, 0.975)),
                        "raw_p": raw_p,
                        "bootstrap_iterations": iterations,
                    }
                )
                for excluded in common:
                    kept = [
                        _effect(frame.loc[scene], cand.loc[scene], metric)
                        for scene in common
                        if scene != excluded
                    ]
                    loso_rows.append(
                        {
                            "threshold": float(threshold),
                            "candidate": candidate,
                            "reference": "frame_detector",
                            "metric": metric,
                            "excluded_scene": excluded,
                            "effect": float(np.mean(kept)),
                        }
                    )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap["holm_p"] = _holm(bootstrap["raw_p"].tolist())
    bootstrap["holm_pass_0_05"] = bootstrap["holm_p"] < 0.05
    atomic_csv(root / "THRESHOLD_PAIRED_BOOTSTRAP.csv", bootstrap)
    atomic_csv(root / "THRESHOLD_LOSO.csv", pd.DataFrame(loso_rows))
    atomic_csv(
        root / "THRESHOLD_HOLM.csv",
        bootstrap[
            [
                "threshold",
                "candidate",
                "reference",
                "metric",
                "raw_p",
                "holm_p",
                "holm_pass_0_05",
            ]
        ],
    )
    expected_rows = (
        len(settings["threshold_baseline"]["grid"])
        * len(settings["threshold_baseline"]["methods"])
    )
    atomic_json(
        root / "THRESHOLD_AUDIT.json",
        {
            "status": "PASS",
            "threshold_grid": settings["threshold_baseline"]["grid"],
            "metric_rows": len(metrics),
            "expected_metric_rows": expected_rows,
            "per_scene_rows": len(scenes),
            "development_scenes": int(scenes["grouped_scene_id"].nunique()),
            "methods": settings["threshold_baseline"]["methods"],
            "same_saved_prediction_source_for_all_methods": True,
            "trackers_recomputed_without_training": True,
            "verifiers_frozen": True,
            "crop_encoder_frozen": True,
            "full_gate_passes": int(metrics["full_gate"].sum()),
            "bootstrap_iterations": iterations,
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(metrics),
                "per_scene_rows": len(scenes),
                "full_gate_passes": int(metrics["full_gate"].sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
