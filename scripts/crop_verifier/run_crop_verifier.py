from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scripts.crop_verifier.common import (
    OUTPUT,
    PROJECT,
    assert_locked,
    atomic_csv,
    atomic_json,
    config,
    sha256,
)
from scripts.temporal_verifier.run_verifier import (
    baseline_metrics,
    candidate_metrics,
    role_source,
)
from src.crop_verifier_v1.encoder import context_crop, select_real_crop_rows
from src.crop_verifier_v1.model import (
    FittedVerifier,
    fit_grouped_verifier,
    predict_verifier,
)
from src.temporal_safety.metrics import paired_scene_bootstrap
from src.temporal_verifier.pipeline import load_role, system_deltas


MODEL_ORDER = ("track_only", "visual_only", "combined")


@dataclass
class RoleData:
    role: str
    source: pd.DataFrame
    ground_truth: dict[str, list[dict[str, Any]]]
    raw: dict[str, list[dict[str, Any]]]
    observations: pd.DataFrame
    additions: pd.DataFrame
    track_features: pd.DataFrame
    labels: pd.DataFrame
    metadata: pd.DataFrame
    visual: np.ndarray
    matrices: dict[str, np.ndarray]
    target: np.ndarray
    groups: np.ndarray


def load_dataset(role: str) -> RoleData:
    _, source, predictions_path = role_source(role)
    ground_truth, raw = load_role(source, str(predictions_path))
    parent = PROJECT / config()["frozen_inputs"]["tracks_root"] / role
    observations = pd.read_csv(parent / "track_observations.csv")
    additions = pd.read_csv(parent / "temporal_additions.csv")
    track_features = pd.read_csv(parent / "track_features.csv")
    labels = pd.read_csv(parent / "track_labels.csv")
    embedding_root = OUTPUT / "embeddings" / role
    metadata = pd.read_csv(embedding_root / "track_metadata.csv")
    visual = np.load(embedding_root / "visual_embeddings.npz")["embeddings"]
    if len(metadata) != len(visual):
        raise RuntimeError(f"Embedding metadata mismatch for {role}")
    track = metadata[["track_key"]].merge(
        track_features,
        on="track_key",
        validate="one_to_one",
    )
    if not track["grouped_scene_id"].astype(str).equals(
        metadata["grouped_scene_id"].astype(str)
    ):
        raise RuntimeError(f"Track metadata order mismatch for {role}")
    label_map = labels.set_index("track_key")["target"].to_dict()
    target = np.asarray(
        [int(label_map[str(key)]) for key in metadata["track_key"]],
        dtype=int,
    )
    track_columns = list(config()["models"]["track_features"])
    track_matrix = track[track_columns].to_numpy(dtype=np.float64)
    matrices = {
        "track_only": track_matrix,
        "visual_only": visual.astype(np.float64),
        "combined": np.column_stack([visual, track_matrix]).astype(np.float64),
    }
    for name, matrix in matrices.items():
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"{role}/{name} contains NaN/Inf")
    return RoleData(
        role,
        source,
        ground_truth,
        raw,
        observations,
        additions,
        track_features,
        labels,
        metadata,
        visual,
        matrices,
        target,
        metadata["grouped_scene_id"].astype(str).to_numpy(),
    )


def fit_models(
    support: RoleData,
) -> tuple[dict[str, FittedVerifier], pd.DataFrame]:
    settings = config()["models"]
    fitted: dict[str, FittedVerifier] = {}
    rows: list[dict[str, Any]] = []
    labelled = support.target >= 0
    for name in MODEL_ORDER:
        model = fit_grouped_verifier(
            support.matrices[name],
            support.target,
            support.groups,
            settings["logistic"],
            int(settings["nested_scene_cv"]["maximum_inner_folds"]),
        )
        probabilities = model.oof_probabilities[labelled]
        target = support.target[labelled]
        fitted[name] = model
        rows.append({
            "model": name,
            "support_labelled_tracks": int(labelled.sum()),
            "support_positive_tracks": int((target == 1).sum()),
            "support_negative_tracks": int((target == 0).sum()),
            "support_scenes_with_tracks": int(
                len(np.unique(support.groups[labelled]))
            ),
            "OOF_AUROC": float(roc_auc_score(target, probabilities)),
            "OOF_AUPRC": float(average_precision_score(target, probabilities)),
            "OOF_Brier": float(brier_score_loss(target, probabilities)),
            "feature_dimension": int(support.matrices[name].shape[1]),
        })
    return fitted, pd.DataFrame(rows)


def accepted_keys(
    data: RoleData, probabilities: np.ndarray, threshold: float
) -> set[str]:
    return set(
        data.metadata.loc[
            probabilities >= threshold, "track_key"
        ].astype(str)
    )


def threshold_sweep(
    model_name: str,
    data: RoleData,
    probabilities: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    baseline, _ = baseline_metrics(data.source, data.ground_truth, data.raw)
    rows: list[dict[str, Any]] = []
    gate = config()["triage_gate"]
    for threshold in np.linspace(
        0.0, 1.0, int(config()["threshold_selection"]["grid_points"])
    ):
        candidate, _ = candidate_metrics(
            data.source,
            data.ground_truth,
            data.raw,
            data.additions,
            accepted_keys(data, probabilities, float(threshold)),
        )
        delta = system_deltas(baseline, candidate)
        recall_fn = (
            delta["recall"] >= float(gate["absolute_recall_improvement_min"])
            and delta["relative_FN_reduction"]
            >= float(gate["relative_FN_reduction_min"])
        )
        full = (
            recall_fn
            and delta["relative_false_alarm_increase"]
            <= float(gate["relative_false_alarms_increase_max"])
            and delta["F1"] >= -float(gate["F1_degradation_max"])
        )
        rows.append({
            "model": model_name,
            "threshold": float(threshold),
            "accepted_tracks": len(
                accepted_keys(data, probabilities, float(threshold))
            ),
            "recall_FN_constraints_pass": recall_fn,
            "fold0_triage_constraints_pass": full,
            **candidate,
            **{f"delta_{key}": value for key, value in delta.items()},
        })
    frame = pd.DataFrame(rows)
    full_pool = frame[frame["fold0_triage_constraints_pass"]]
    recall_pool = frame[frame["recall_FN_constraints_pass"]]
    if not full_pool.empty:
        pool, status = full_pool, "FOLD0_TRIAGE_CONSTRAINTS_PASS"
    elif not recall_pool.empty:
        pool, status = recall_pool, "RECALL_FN_ONLY"
    else:
        pool, status = frame, "NO_OPERATIONAL_CONSTRAINTS_PASS"
    selected = pool.sort_values(
        ["f1", "false_alarms_per_minute", "recall", "threshold"],
        ascending=[False, True, False, False],
    ).iloc[0].to_dict()
    selected["selection_status"] = status
    return selected, frame


def fold_metrics(
    model_name: str,
    data: RoleData,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    baseline, baseline_scene = baseline_metrics(
        data.source, data.ground_truth, data.raw
    )
    candidate, candidate_scene = candidate_metrics(
        data.source,
        data.ground_truth,
        data.raw,
        data.additions,
        accepted_keys(data, probabilities, threshold),
    )
    candidate["model"] = model_name
    return baseline, baseline_scene, candidate, candidate_scene


def triage_gate(
    model_name: str,
    threshold: float,
    screen_result: tuple[dict[str, Any], pd.DataFrame, dict[str, Any], pd.DataFrame],
    confirm_result: tuple[dict[str, Any], pd.DataFrame, dict[str, Any], pd.DataFrame],
) -> dict[str, Any]:
    base_metrics = [screen_result[0], confirm_result[0]]
    candidate_metrics_rows = [screen_result[2], confirm_result[2]]
    keys = ("recall", "FN_per_frame", "false_alarms_per_minute", "f1")
    baseline_macro = {
        key: float(np.mean([row[key] for row in base_metrics])) for key in keys
    }
    candidate_macro = {
        key: float(np.mean([row[key] for row in candidate_metrics_rows]))
        for key in keys
    }
    delta = system_deltas(baseline_macro, candidate_macro)
    gate = config()["triage_gate"]
    checks = {
        "recall": delta["recall"]
        >= float(gate["absolute_recall_improvement_min"]),
        "FN_per_frame": delta["relative_FN_reduction"]
        >= float(gate["relative_FN_reduction_min"]),
        "false_alarms": delta["relative_false_alarm_increase"]
        <= float(gate["relative_false_alarms_increase_max"]),
        "F1": delta["F1"] >= -float(gate["F1_degradation_max"]),
        "evaluator": all(
            row["evaluator_consistency"] == gate["evaluator_consistency"]
            for row in candidate_metrics_rows
        ),
        "lost_frames": all(
            row["lost_frames"] == int(gate["lost_frames"])
            for row in candidate_metrics_rows
        ),
        "NaN_Inf": all(
            row["NaN_Inf"] == int(gate["NaN_Inf"])
            for row in candidate_metrics_rows
        ),
    }
    baseline_scene = pd.concat(
        [
            screen_result[1].assign(fold=0),
            confirm_result[1].assign(fold=1),
        ],
        ignore_index=True,
    )
    candidate_scene = pd.concat(
        [
            screen_result[3].assign(fold=0),
            confirm_result[3].assign(fold=1),
        ],
        ignore_index=True,
    )
    scene = baseline_scene.merge(
        candidate_scene,
        on=["fold", "grouped_scene_id"],
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    bootstrap = paired_scene_bootstrap(
        dict(zip(scene["grouped_scene_id"], scene["recall_baseline"])),
        dict(zip(scene["grouped_scene_id"], scene["recall_candidate"])),
        int(config()["development_gate"]["bootstrap_iterations"]),
        int(config()["development_gate"]["bootstrap_seed"]),
    )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model": model_name,
        "threshold": threshold,
        "baseline_macro": baseline_macro,
        "candidate_macro": candidate_macro,
        "deltas": delta,
        "checks": checks,
        "failed_conditions": [key for key, value in checks.items() if not value],
        "paired_scene_bootstrap_recall": bootstrap,
        "per_fold": {
            "0": {"baseline": screen_result[0], "candidate": screen_result[2]},
            "1": {"baseline": confirm_result[0], "candidate": confirm_result[2]},
        },
    }


def select_model(
    selections: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    rows = pd.DataFrame(list(selections.values()))
    tie_order = {
        name: index
        for index, name in enumerate(config()["threshold_selection"]["model_tie_order"])
    }
    rows["tie_order"] = rows["model"].map(tie_order)
    full = rows[rows["fold0_triage_constraints_pass"]]
    recall = rows[rows["recall_FN_constraints_pass"]]
    if not full.empty:
        pool, status = full, "FOLD0_TRIAGE_CONSTRAINTS_PASS"
    elif not recall.empty:
        pool, status = recall, "RECALL_FN_ONLY"
    else:
        pool, status = rows, "NO_OPERATIONAL_CONSTRAINTS_PASS"
    selected = pool.sort_values(
        ["f1", "false_alarms_per_minute", "recall", "tie_order"],
        ascending=[False, True, False, True],
    ).iloc[0].to_dict()
    selected["model_selection_status"] = status
    return selected


def false_category(row: pd.Series) -> str:
    if row["track_length"] <= 2:
        return "short_single_noise"
    if row["interpolation_fraction"] >= 0.50:
        return "incorrect_interpolation"
    if row["duplicate_overlap_fraction"] >= 0.25:
        return "duplicate_track"
    if row["border_fraction"] >= 0.50:
        return "image_or_tile_border"
    if row["box_area_cv"] >= 0.80 or row["center_velocity_std"] >= 1.0:
        return "jumping_box"
    if row["aspect_ratio_mean"] <= 0.45 and row["center_velocity_mean"] <= 0.15:
        return "signal_or_pole"
    if row["center_velocity_mean"] <= 0.15:
        return "static_vertical_object"
    if row["track_length"] >= 5 and row["detection_fraction"] >= 0.60:
        return "persistent_hard_negative"
    return "other_false_track"


def confusion_audit(
    model_name: str,
    threshold: float,
    roles: list[tuple[RoleData, np.ndarray]],
) -> None:
    rows: list[pd.DataFrame] = []
    for data, probabilities in roles:
        labels = data.labels[["track_key", "target"]]
        table = data.metadata[["track_key", "grouped_scene_id"]].copy()
        table["probability"] = probabilities
        table["role"] = data.role
        table = table.merge(labels, on="track_key", validate="one_to_one")
        table = table.merge(
            data.track_features,
            on=["track_key", "grouped_scene_id"],
            validate="one_to_one",
        )
        table = table[table["target"] >= 0].copy()
        accepted = table["probability"] >= threshold
        table["confusion_cell"] = np.select(
            [
                table["target"].eq(1) & accepted,
                table["target"].eq(1) & ~accepted,
                table["target"].eq(0) & accepted,
                table["target"].eq(0) & ~accepted,
            ],
            [
                "true_accepted",
                "true_rejected",
                "false_accepted",
                "false_rejected",
            ],
            default="invalid",
        )
        table["false_category"] = "not_false"
        false = table["target"].eq(0)
        table.loc[false, "false_category"] = table.loc[false].apply(
            false_category, axis=1
        )
        rows.append(table)
    audit = pd.concat(rows, ignore_index=True)
    private_root = OUTPUT / "audit/confusion"
    atomic_csv(private_root / "CONFUSION_AUDIT_PRIVATE.csv", audit)
    counts = (
        audit.groupby(["role", "confusion_cell"], as_index=False)
        .size()
        .rename(columns={"size": "tracks"})
    )
    atomic_csv(OUTPUT / "audit/CONFUSION_COUNTS.csv", counts)
    false_effect = (
        audit[audit["target"].eq(0)]
        .groupby(["role", "false_category", "confusion_cell"], as_index=False)
        .size()
        .rename(columns={"size": "tracks"})
    )
    atomic_csv(OUTPUT / "audit/FALSE_CATEGORY_EFFECT.csv", false_effect)

    maximum = int(config()["confusion_audit"]["maximum_gallery_per_cell"])
    gallery = private_root / "gallery"
    for cell, candidates in audit.groupby("confusion_cell", sort=True):
        destination = gallery / str(cell)
        destination.mkdir(parents=True, exist_ok=True)
        candidates = candidates.sort_values(
            "probability", ascending=cell.endswith("rejected")
        ).head(maximum)
        for index, row in enumerate(candidates.itertuples(index=False)):
            data = next(item[0] for item in roles if item[0].role == row.role)
            group = data.observations[
                data.observations["track_key"].astype(str).eq(str(row.track_key))
            ]
            selected = select_real_crop_rows(group)
            crop_row = selected.sort_values(
                "detector_confidence", ascending=False
            ).iloc[0]
            image = cv2.imread(str(crop_row["image_path"]))
            if image is None:
                continue
            crop = context_crop(
                image,
                [
                    float(crop_row["x1"]),
                    float(crop_row["y1"]),
                    float(crop_row["x2"]),
                    float(crop_row["y2"]),
                ],
                float(config()["encoder"]["crop_context_multiplier"]),
                int(config()["encoder"]["input_size"]),
                int(config()["encoder"]["padding_value"]),
            )
            cv2.imwrite(
                str(destination / f"{index:03d}_{row.role}_{row.track_id}.jpg"),
                crop,
            )
    atomic_json(
        OUTPUT / "audit/CONFUSION_AUDIT.json",
        {
            "model": model_name,
            "threshold": threshold,
            "counts": {
                str(row.confusion_cell): int(row.tracks)
                for row in counts.groupby("confusion_cell", as_index=False)[
                    "tracks"
                ].sum().itertuples(index=False)
            },
            "gallery_public": False,
            "test_used": False,
        },
    )


def false_alarm_stages(crop_gate: dict[str, Any]) -> None:
    parent_temporal = json.loads(
        (
            PROJECT
            / "outputs/temporal_safety_v1/triage/TRIAGE_GATE.json"
        ).read_text(encoding="utf-8")
    )
    parent_verifier = json.loads(
        (
            PROJECT
            / "outputs/temporal_verifier_v1/FINAL_SUMMARY.json"
        ).read_text(encoding="utf-8")
    )
    ocsort = next(
        row for row in parent_temporal["decisions"] if row["tracker"] == "ocsort"
    )
    rows = [
        {
            "stage": "frame_detector",
            "false_alarms_per_minute": crop_gate["baseline_macro"][
                "false_alarms_per_minute"
            ],
        },
        {
            "stage": "ocsort",
            "false_alarms_per_minute": ocsort["candidate_macro"][
                "false_alarms_per_minute"
            ],
        },
        {
            "stage": "track_logistic_verifier",
            "false_alarms_per_minute": parent_verifier["final_gate"][
                "candidate_macro"
            ]["false_alarms_per_minute"],
        },
        {
            "stage": "crop_verifier",
            "false_alarms_per_minute": crop_gate["candidate_macro"][
                "false_alarms_per_minute"
            ],
        },
    ]
    atomic_csv(OUTPUT / "results/FALSE_ALARM_STAGES.csv", pd.DataFrame(rows))


def full_development_oof(
    selected_model: str,
    threshold: float,
    roles: list[RoleData],
) -> dict[str, Any]:
    all_meta = pd.concat(
        [data.metadata.assign(role=data.role) for data in roles],
        ignore_index=True,
    )
    all_labels = pd.concat(
        [data.labels.assign(role=data.role) for data in roles],
        ignore_index=True,
    )
    all_target_map = all_labels.set_index("track_key")["target"].to_dict()
    target = np.asarray(
        [int(all_target_map[str(key)]) for key in all_meta["track_key"]]
    )
    groups = all_meta["grouped_scene_id"].astype(str).to_numpy()
    matrix = np.concatenate(
        [data.matrices[selected_model] for data in roles], axis=0
    )
    probabilities = np.full(len(all_meta), np.nan, dtype=float)
    for scene in np.unique(groups):
        train = groups != scene
        heldout = groups == scene
        fitted = fit_grouped_verifier(
            matrix[train],
            target[train],
            groups[train],
            config()["models"]["logistic"],
            int(config()["models"]["nested_scene_cv"]["maximum_inner_folds"]),
        )
        probabilities[heldout] = predict_verifier(fitted, matrix[heldout])
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Full development OOF probabilities contain NaN/Inf")

    source = pd.concat([data.source for data in roles], ignore_index=True)
    ground_truth: dict[str, list[dict[str, Any]]] = {}
    raw: dict[str, list[dict[str, Any]]] = {}
    additions = []
    for data in roles:
        ground_truth.update(data.ground_truth)
        raw.update(data.raw)
        additions.append(data.additions)
    accepted = set(
        all_meta.loc[probabilities >= threshold, "track_key"].astype(str)
    )
    baseline, baseline_scene = baseline_metrics(source, ground_truth, raw)
    candidate, candidate_scene = candidate_metrics(
        source,
        ground_truth,
        raw,
        pd.concat(additions, ignore_index=True),
        accepted,
    )
    merged = baseline_scene.merge(
        candidate_scene,
        on="grouped_scene_id",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    keys = ("recall", "FN_per_frame", "false_alarms_per_minute", "f1")
    baseline_macro = {
        key: float(baseline_scene[key].mean()) for key in keys
    }
    candidate_macro = {
        key: float(candidate_scene[key].mean()) for key in keys
    }
    delta = system_deltas(baseline_macro, candidate_macro)
    recall_effect = merged["recall_candidate"] - merged["recall_baseline"]
    fn_effect = merged["FN_per_frame_candidate"] - merged["FN_per_frame_baseline"]
    improved = (recall_effect > 1e-12) | (fn_effect < -1e-12)
    settings = config()["development_gate"]
    bootstrap = paired_scene_bootstrap(
        dict(zip(merged["grouped_scene_id"], merged["recall_baseline"])),
        dict(zip(merged["grouped_scene_id"], merged["recall_candidate"])),
        int(settings["bootstrap_iterations"]),
        int(settings["bootstrap_seed"]),
    )
    checks = {
        "recall": delta["recall"]
        >= float(settings["absolute_macro_recall_improvement_min"]),
        "FN_per_frame": delta["relative_FN_reduction"]
        >= float(settings["relative_macro_FN_reduction_min"]),
        "false_alarms": delta["relative_false_alarm_increase"]
        <= float(settings["relative_false_alarms_increase_max"]),
        "F1": delta["F1"] >= -float(settings["F1_degradation_max"]),
        "worst_scene_recall": float(recall_effect.min()) >= -1e-12,
        "improved_scene_fraction": float(improved.mean())
        >= float(settings["improved_scene_fraction_min"]),
        "bootstrap_recall": float(bootstrap["ci_low"]) > 0,
        "evaluator": candidate["evaluator_consistency"]
        == settings["evaluator_consistency"],
        "lost_frames": candidate["lost_frames"] == int(settings["lost_frames"]),
        "NaN_Inf": candidate["NaN_Inf"] == int(settings["NaN_Inf"]),
    }
    atomic_csv(
        OUTPUT / "results/FULL_DEVELOPMENT_PER_SCENE.csv",
        merged,
    )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model": selected_model,
        "threshold": threshold,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_macro": baseline_macro,
        "candidate_macro": candidate_macro,
        "deltas": delta,
        "checks": checks,
        "failed_conditions": [key for key, value in checks.items() if not value],
        "improved_scene_fraction": float(improved.mean()),
        "worst_scene_recall_delta": float(recall_effect.min()),
        "paired_scene_bootstrap_recall": bootstrap,
        "cross_fitted_by_scene": True,
    }


def main() -> None:
    lock = assert_locked()
    support = load_dataset("support")
    screen = load_dataset("screening")
    fitted, support_comparison = fit_models(support)
    atomic_csv(OUTPUT / "results/SUPPORT_OOF_MODEL_COMPARISON.csv", support_comparison)

    selections: dict[str, dict[str, Any]] = {}
    screen_probabilities: dict[str, np.ndarray] = {}
    for name in MODEL_ORDER:
        probabilities = predict_verifier(fitted[name], screen.matrices[name])
        screen_probabilities[name] = probabilities
        selected, sweep = threshold_sweep(name, screen, probabilities)
        selections[name] = selected
        atomic_csv(OUTPUT / f"models/{name}/THRESHOLD_SWEEP.csv", sweep)
        model_root = OUTPUT / "models" / name
        model_root.mkdir(parents=True, exist_ok=True)
        with (model_root / "model.pkl").open("wb") as handle:
            pickle.dump(fitted[name], handle)
    selected = select_model(selections)
    atomic_csv(
        OUTPUT / "results/FOLD0_MODEL_SELECTION.csv",
        pd.DataFrame(list(selections.values())),
    )
    atomic_json(
        OUTPUT / "protocol/PRE_CONFIRMATION_FREEZE.json",
        {
            "selected_model": selected["model"],
            "selected_threshold": selected["threshold"],
            "selection": selected,
            "selected_on": "support_grouped_OOF_plus_fold0_screening",
            "confirmation_labels_or_metrics_used": False,
            "confirmation_embeddings": (
                "precomputed_by_frozen_encoder_without_model_selection"
            ),
            "global_threshold": True,
            "test_status": "SEALED",
            "test_access_count": 0,
            "model_sha256": sha256(
                OUTPUT / f"models/{selected['model']}/model.pkl"
            ),
        },
    )

    confirmation = load_dataset("confirmation")
    comparison_rows: list[dict[str, Any]] = []
    triage_results: dict[str, dict[str, Any]] = {}
    role_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in MODEL_ORDER:
        threshold = float(selections[name]["threshold"])
        confirm_probabilities = predict_verifier(
            fitted[name], confirmation.matrices[name]
        )
        role_probabilities[name] = (
            screen_probabilities[name],
            confirm_probabilities,
        )
        screen_result = fold_metrics(
            name, screen, screen_probabilities[name], threshold
        )
        confirm_result = fold_metrics(
            name, confirmation, confirm_probabilities, threshold
        )
        gate = triage_gate(
            name, threshold, screen_result, confirm_result
        )
        triage_results[name] = gate
        comparison_rows.append({
            "model": name,
            "threshold": threshold,
            "status": gate["status"],
            **gate["candidate_macro"],
            **{f"delta_{key}": value for key, value in gate["deltas"].items()},
            "failed_conditions": "|".join(gate["failed_conditions"]),
        })
        atomic_json(OUTPUT / f"models/{name}/TWO_FOLD_TRIAGE_GATE.json", gate)
    atomic_csv(
        OUTPUT / "results/TWO_FOLD_MODEL_COMPARISON.csv",
        pd.DataFrame(comparison_rows),
    )

    selected_name = str(selected["model"])
    selected_gate = triage_results[selected_name]
    selected_probabilities = role_probabilities[selected_name]
    confusion_audit(
        selected_name,
        float(selected["threshold"]),
        [
            (screen, selected_probabilities[0]),
            (confirmation, selected_probabilities[1]),
        ],
    )
    false_alarm_stages(selected_gate)

    if selected_gate["status"] == "PASS":
        full_gate = full_development_oof(
            selected_name,
            float(selected["threshold"]),
            [support, screen, confirmation],
        )
        atomic_json(OUTPUT / "results/FULL_DEVELOPMENT_GATE.json", full_gate)
        status = (
            "DEVELOPMENT_PASS"
            if full_gate["status"] == "PASS"
            else "CLOSED_NO_PRACTICAL_GATE"
        )
        next_step = (
            "TEST_RELEASE_REQUIRES_SEPARATE_ONE_TIME_RUN"
            if full_gate["status"] == "PASS"
            else "NEW_RAILWAY_SCENES_REQUIRED"
        )
    else:
        full_gate = {
            "status": "NOT_RUN",
            "reason": "TWO_FOLD_TRIAGE_FAIL",
        }
        atomic_json(OUTPUT / "results/FULL_DEVELOPMENT_GATE.json", full_gate)
        status = "CLOSED_NO_PRACTICAL_GATE"
        next_step = "NEW_RAILWAY_SCENES_REQUIRED"

    decision = {
        "protocol_id": config()["protocol_id"],
        "status": status,
        "selected_model": selected_name,
        "selected_threshold": float(selected["threshold"]),
        "two_fold_triage": selected_gate,
        "all_model_triage_status": {
            name: gate["status"] for name, gate in triage_results.items()
        },
        "full_development_gate": full_gate,
        "sensitivity_mlp": (
            "DEFERRED_AFTER_PRIMARY_PASS"
            if selected_gate["status"] == "PASS"
            else "SKIPPED_BY_FROZEN_DECISION_TREE"
        ),
        "temporal_direction": (
            "OPEN_FOR_ONE_TIME_TEST"
            if status == "DEVELOPMENT_PASS"
            else "CLOSED_NO_PRACTICAL_GATE"
        ),
        "next_step": next_step,
        "test_status": "SEALED",
        "test_access_count": 0,
        "detector_unchanged": True,
        "tracker_unchanged": True,
        "parent_verifier_unchanged": True,
        "implementation_commit": lock["implementation_commit"],
    }
    atomic_json(OUTPUT / "decision_trace.json", decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
