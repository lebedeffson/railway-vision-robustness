from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from rescue_common import OUTPUT_ROOT, atomic_json


ROOT = OUTPUT_ROOT / "micro_overfit"


def scalar_rows(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    return frame[
        (frame["split"] == "val")
        & (frame["operating_point"] == "safety")
        & (frame["scope"] == scope)
    ].copy()


def diagnose() -> dict[str, Any]:
    result = json.loads((ROOT / "result.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(ROOT / "evaluation/clean_metrics_by_class_and_size.csv")
    classes = scalar_rows(metrics, "class")
    sizes = scalar_rows(metrics, "size")
    gradients = pd.read_csv(ROOT / "gradient_metrics.csv")

    class_rows = {
        str(row["name"]): {
            "tp": int(row["tp"]),
            "gt": int(row["gt"]),
            "fn": int(row["gt"] - row["tp"]),
            "recall": float(row["recall"]),
        }
        for _, row in classes.iterrows()
    }
    size_recall = {
        str(row["name"]): float(row["recall"])
        for _, row in sizes.iterrows()
    }
    total_fn = sum(row["fn"] for row in class_rows.values())
    signal_fn = class_rows.get("signal", {}).get("fn", 0)
    nonzero_gradient_observed = bool(
        (gradients["backbone_gradient_norm"].astype(float) > 0).any()
        and (gradients["head_gradient_norm"].astype(float) > 0).any()
    )
    initial_weight_norm = float(gradients["initial_weight_norm"].iloc[0])
    final_weight_norm = float(gradients["weight_norm"].iloc[-1])
    parameter_updates_observed = abs(final_weight_norm - initial_weight_norm) > 1e-8
    signal_share = float(signal_fn / total_fn) if total_fn else 0.0
    small_recall = size_recall.get("small", 1.0)
    dominant = (
        "small_object_signal_detection"
        if signal_share >= 0.5 and small_recall < 0.9
        else "mixed_detection_errors"
    )
    gradient_conclusion = (
        "direct_nonzero_gradients_observed"
        if nonzero_gradient_observed
        else (
            "training_updates_confirmed_but_callback_measurement_invalid"
            if parameter_updates_observed and result["loss_decreased"]
            else "gradient_flow_not_confirmed"
        )
    )
    return {
        "status": "DIAGNOSED_MICRO_OVERFIT_FAILURE",
        "scientific_gate_passed": False,
        "candidate_matrix_allowed": False,
        "test_opened": False,
        "dominant_error_source": dominant,
        "class_metrics": class_rows,
        "size_recall": size_recall,
        "total_false_negatives": total_fn,
        "signal_false_negatives": signal_fn,
        "signal_false_negative_share": signal_share,
        "nonzero_gradient_observed": nonzero_gradient_observed,
        "gradient_logging_valid_for_this_run": nonzero_gradient_observed,
        "gradient_callback_issue": (
            None if nonzero_gradient_observed
            else "ultralytics_8_4_102_does_not_dispatch_on_before_zero_grad"
        ),
        "initial_weight_norm": initial_weight_norm,
        "final_weight_norm": final_weight_norm,
        "parameter_updates_observed": parameter_updates_observed,
        "loss_decreased": bool(result["loss_decreased"]),
        "gradient_flow_conclusion": gradient_conclusion,
        "visual_evidence": [
            "evaluation/false_negative_examples/"
            "01_8_station_altona_8.2__205_1631700829.600000028.png",
            "evaluation/false_negative_examples/"
            "02_4_station_pedestrian_bridge_4.2__026_1631702292.700000027.png",
        ],
        "interpretation": (
            "The model learned medium and large objects and most mapped classes, "
            "but did not memorize the dense tiny-signal subset or the extremely "
            "rare animal/train examples. The falling loss and changed weights "
            "exclude a disconnected optimizer as the primary explanation. "
            "The zero gradient columns are an instrumentation defect, not proof "
            "of zero gradient flow."
        ),
        "required_next_action": (
            "Freeze a new protocol before any rerun; do not start R0-R4 or test "
            "under canonical-v2-rescue-v1."
        ),
    }


def main() -> None:
    payload = diagnose()
    atomic_json(ROOT / "failure_diagnosis.json", payload)
    (ROOT / "failure_diagnosis.md").write_text(
        "\n".join([
            "# Micro-overfit failure diagnosis",
            "",
            f"- dominant error source: `{payload['dominant_error_source']}`",
            f"- total FN: {payload['total_false_negatives']}",
            f"- signal FN/share: {payload['signal_false_negatives']}/"
            f"{payload['signal_false_negative_share']:.2%}",
            f"- size Recall: {payload['size_recall']}",
            f"- class metrics: {payload['class_metrics']}",
            f"- gradient-flow conclusion: `{payload['gradient_flow_conclusion']}`",
            "",
            payload["interpretation"],
            "",
            payload["required_next_action"],
            "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
