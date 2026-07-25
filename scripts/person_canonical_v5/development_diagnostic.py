from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.audit_evaluator import average_precision, match_dataset
from scripts.person_v3.evaluate import (
    detections_frame,
    infer_all,
    matched_indices,
    person_gt,
    reparse,
    selected_sources,
)
from scripts.person_canonical_v5.common import (
    OUTPUT,
    PROJECT,
    assert_diagnostic_locked,
    atomic_csv,
    atomic_json,
    load_protocol,
    now,
    sha256,
)


def checkpoint_config(
    protocol: dict[str, Any], state: str, fold: int
) -> dict[str, Any]:
    config = protocol["development_diagnostic"]["states"][state]
    return config if "checkpoint" in config else config[f"fold_{fold}"]


def best_epoch(results: Path, role: str) -> int:
    frame = pd.read_csv(results)
    frame.columns = [column.strip() for column in frame.columns]
    if frame.empty:
        raise RuntimeError(f"Empty training results: {results}")
    if role.endswith("_last"):
        return int(frame.iloc[-1]["epoch"])
    required = {"metrics/mAP50(B)", "metrics/mAP50-95(B)"}
    if not required.issubset(frame.columns):
        return int(frame.iloc[-1]["epoch"])
    fitness = (
        0.1 * frame["metrics/mAP50(B)"]
        + 0.9 * frame["metrics/mAP50-95(B)"]
    )
    return int(frame.loc[fitness.idxmax(), "epoch"])


def area_recall_rows(
    gt: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    threshold: float,
) -> list[dict[str, Any]]:
    bins = (
        ("lt_0.0001", 0.0, 0.0001),
        ("0.0001_0.0003", 0.0001, 0.0003),
        ("0.0003_0.001", 0.0003, 0.001),
        ("0.001_0.01", 0.001, 0.01),
        ("ge_0.01", 0.01, math.inf),
    )
    counts = {
        name: {"gt": 0, "tp": 0} for name, _, _ in bins
    }
    image_area = 4112 * 2504
    for image, targets in gt.items():
        matched = matched_indices(
            targets, predictions.get(image, []), threshold
        )
        for index, target in enumerate(targets):
            x1, y1, x2, y2 = target["box"]
            ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
            for name, lower, upper in bins:
                if lower <= ratio < upper:
                    counts[name]["gt"] += 1
                    counts[name]["tp"] += int(index in matched)
                    break
    return [
        {
            "area_ratio_bin": name,
            **values,
            "recall": values["tp"] / values["gt"] if values["gt"] else None,
        }
        for name, values in counts.items()
    ]


def confidence_summary(
    predictions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    values = np.asarray(
        [
            float(prediction["confidence"])
            for rows in predictions.values()
            for prediction in rows
        ],
        dtype=float,
    )
    if not len(values):
        return {
            "prediction_count": 0,
            "confidence_q01": None,
            "confidence_q25": None,
            "confidence_q50": None,
            "confidence_q75": None,
            "confidence_q99": None,
        }
    quantiles = np.quantile(values, [0.01, 0.25, 0.50, 0.75, 0.99])
    return {
        "prediction_count": int(len(values)),
        **{
            f"confidence_q{label}": float(value)
            for label, value in zip(("01", "25", "50", "75", "99"), quantiles)
        },
    }


def evaluate_state(
    state: str, fold: int, protocol: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = checkpoint_config(protocol, state, fold)
    checkpoint = PROJECT / config["checkpoint"]
    if sha256(checkpoint) != config["sha256"]:
        raise RuntimeError(f"Checkpoint hash changed: {state}, fold {fold}")
    source = selected_sources(fold)
    if not source["split"].isin(["train", "val"]).all():
        raise RuntimeError("Development diagnostic attempted to access test")
    gt = person_gt(source)
    destination = OUTPUT / f"development_diagnostic/fold_{fold}/{state}"
    predictions, runtime_ms = infer_all(
        fold, checkpoint, source, destination
    )
    detections = detections_frame(source, gt, predictions)
    csv_path = destination / "predictions_and_ground_truth.csv"
    atomic_csv(detections, csv_path)
    repeated_gt, repeated_predictions = reparse(pd.read_csv(csv_path))
    diagnostic = protocol["development_diagnostic"]
    threshold = float(diagnostic["confidence_threshold"])
    iou = float(diagnostic["iou_threshold"])
    map50 = average_precision(gt, predictions, 0, iou)
    map50_repeated = average_precision(
        repeated_gt, repeated_predictions, 0, iou
    )
    aps = [
        average_precision(gt, predictions, 0, float(value))
        for value in diagnostic["AP_IoU"]
    ]
    metrics = match_dataset(gt, predictions, threshold, iou)
    repeated_metrics = match_dataset(
        repeated_gt, repeated_predictions, threshold, iou
    )
    consistent = abs(map50 - map50_repeated) <= 1e-4 and all(
        abs(float(metrics[key]) - float(repeated_metrics[key])) <= 1e-4
        for key in ("precision", "recall", "f1")
    )
    scenes = {
        str(Path(row.output_image).resolve()): str(row.grouped_scene_id)
        for row in source.itertuples(index=False)
    }
    scene_rows = []
    for scene in sorted(set(scenes.values())):
        images = {image for image, value in scenes.items() if value == scene}
        scene_gt = {image: gt[image] for image in images}
        scene_predictions = {
            image: predictions.get(image, []) for image in images
        }
        scene_metrics = match_dataset(
            scene_gt, scene_predictions, threshold, iou
        )
        scene_rows.append(
            {
                "state": state,
                "fold": fold,
                "grouped_scene_id": scene,
                "frames": len(images),
                "GT": sum(len(values) for values in scene_gt.values()),
                "mAP50": average_precision(
                    scene_gt, scene_predictions, 0, iou
                ),
                **scene_metrics,
            }
        )
    finite_values = [map50, *aps, *map(float, metrics.values())]
    result = {
        "state": state,
        "role": protocol["development_diagnostic"]["states"][state]["role"],
        "fold": fold,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "best_epoch": best_epoch(PROJECT / config["results"], state),
        "frames": int(len(source)),
        "scenes": int(source["grouped_scene_id"].nunique()),
        "GT": int(sum(len(rows) for rows in gt.values())),
        "threshold": threshold,
        "IoU": iou,
        "mAP50": float(map50),
        "mAP50_95": float(np.mean(aps)),
        **metrics,
        "FN_per_frame": float(metrics["fn"]) / max(len(source), 1),
        "FP_per_frame": float(metrics["fp"]) / max(len(source), 1),
        "worst_scene_recall": min(
            (
                float(row["recall"])
                for row in scene_rows
                if int(row["GT"]) > 0
            ),
            default=0.0,
        ),
        **confidence_summary(predictions),
        "runtime_ms": float(runtime_ms),
        "evaluator_consistency": "PASS" if consistent else "FAIL",
        "NaN_Inf": sum(not np.isfinite(value) for value in finite_values),
        "lost_GT": abs(
            sum(len(values) for values in gt.values())
            - sum(len(values) for values in repeated_gt.values())
        ),
        "test_used": False,
    }
    area_rows = [
        {"state": state, "fold": fold, **row}
        for row in area_recall_rows(gt, predictions, threshold)
    ]
    atomic_json(destination / "evaluation_result.json", result)
    atomic_csv(pd.DataFrame(scene_rows), destination / "per_scene.csv")
    atomic_csv(pd.DataFrame(area_rows), destination / "per_area_recall.csv")
    return result, scene_rows, area_rows


def run() -> dict[str, Any]:
    assert_diagnostic_locked()
    protocol = load_protocol()
    states = list(protocol["development_diagnostic"]["states"])
    results = []
    scene_rows = []
    area_rows = []
    for state in states:
        for fold in protocol["development_diagnostic"]["folds"]:
            result, per_scene, per_area = evaluate_state(
                state, int(fold), protocol
            )
            results.append(result)
            scene_rows.extend(per_scene)
            area_rows.extend(per_area)
    frame = pd.DataFrame(results)
    macro = (
        frame.groupby(["state", "role"], as_index=False)
        .agg(
            macro_mAP50=("mAP50", "mean"),
            macro_mAP50_95=("mAP50_95", "mean"),
            macro_precision=("precision", "mean"),
            macro_recall=("recall", "mean"),
            macro_F1=("f1", "mean"),
            worst_fold_recall=("recall", "min"),
            worst_scene_recall=("worst_scene_recall", "min"),
            mean_FN_per_frame=("FN_per_frame", "mean"),
            mean_FP_per_frame=("FP_per_frame", "mean"),
        )
    )
    b1 = macro.loc[macro["state"].eq("B1")].iloc[0]
    d1 = macro.loc[macro["state"].eq("D1_best")].iloc[0]
    gradual_allowed = bool(
        b1["macro_recall"] > d1["macro_recall"]
        or b1["worst_fold_recall"] > d1["worst_fold_recall"]
    )
    output = OUTPUT / "development_diagnostic"
    atomic_csv(frame, output / "state_fold_results.csv")
    atomic_csv(macro, output / "state_macro_results.csv")
    atomic_csv(pd.DataFrame(scene_rows), output / "per_scene_results.csv")
    atomic_csv(pd.DataFrame(area_rows), output / "per_area_results.csv")
    summary = {
        "status": "PASS"
        if frame["evaluator_consistency"].eq("PASS").all()
        and frame["NaN_Inf"].eq(0).all()
        and frame["lost_GT"].eq(0).all()
        else "FAIL",
        "created_at": now(),
        "protocol_id": protocol["protocol_id"],
        "fixed_threshold": protocol["development_diagnostic"][
            "confidence_threshold"
        ],
        "fixed_IoU": protocol["development_diagnostic"]["iou_threshold"],
        "states": states,
        "folds": protocol["development_diagnostic"]["folds"],
        "gradual_transfer_allowed": gradual_allowed,
        "gradual_transfer_rule": (
            "B1 macro Recall or worst-fold Recall must exceed D1-best"
        ),
        "test_opened": False,
        "scientific_result": False,
    }
    atomic_json(output / "development_diagnostic_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

