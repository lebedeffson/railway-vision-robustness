from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    assert_role_allowed,
    atomic_json,
    load_protocol,
    now,
    sha256,
)
from canonical_m4_runtime import (
    FrameInput,
    detection,
    fgsm,
    frame_labels,
    load_image,
    load_model,
    pgd,
)


SOURCE_MANIFEST = PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"
DESTINATION = OUTPUT_ROOT / "attack_calibration"


def condition_id(
    image_path: str, attack: str, adaptive: bool, epsilon_px: float
) -> str:
    value = f"{image_path}|{attack}|{adaptive}|{epsilon_px:.12g}"
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def result_metrics(result, clean: torch.Tensor) -> dict[str, float]:
    delta = result.adversarial - clean
    return {
        "actual_L1": float(delta.abs().sum()),
        "actual_L2": float(delta.norm()),
        "actual_Linf": float(delta.abs().max()),
        "attack_loss_clean": float(result.clean_attack_loss),
        "attack_loss_final": float(result.attack_loss),
        "gradient_L1": float(result.path_gradient.abs().sum()),
        "gradient_L2": float(result.path_gradient.norm()),
        "gradient_Linf": float(result.path_gradient.abs().max()),
    }


def run(checkpoint: Path) -> dict[str, Any]:
    assert_role_allowed("attack_calibration")
    protocol = load_protocol()
    gate = json.loads(
        (OUTPUT_ROOT / "validation/quality_gate.json").read_text(encoding="utf-8")
    )
    if gate["checkpoint_sha256"] != sha256(checkpoint):
        raise RuntimeError("Attack calibration checkpoint differs from frozen gate")
    threshold = float(gate["safety_threshold"])
    source = pd.read_csv(SOURCE_MANIFEST)
    source = source[source["split"].eq("val")].sort_values(
        ["grouped_scene_id", "subsequence_id", "frame_id"]
    )
    if len(source) != int(protocol["dataset"]["split_frame_counts"]["val"]):
        raise RuntimeError("Attack calibration validation frame count mismatch")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    cache = DESTINATION / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    model = load_model(checkpoint, device)
    seeds = [int(value) for value in protocol["attack_calibration"]["seeds"]]
    conditions = [
        ("fgsm", False, float(epsilon), 1)
        for epsilon in protocol["attack_calibration"]["fgsm_epsilon_px"]
    ]
    conditions += [
        (
            "pgd",
            False,
            float(epsilon),
            int(protocol["attack_calibration"]["pgd"]["steps"]),
        )
        for epsilon in protocol["attack_calibration"]["pgd"]["epsilon_px"]
    ]
    conditions += [
        (
            "pgd",
            True,
            float(epsilon),
            int(protocol["attack_calibration"]["adaptive_product_pgd"]["steps"]),
        )
        for epsilon in protocol["attack_calibration"]["adaptive_product_pgd"][
            "epsilon_px"
        ]
    ]
    rows = []
    for source_row in source.itertuples(index=False):
        frame = FrameInput(
            Path(source_row.output_image),
            Path(source_row.output_label),
            str(source_row.grouped_scene_id),
            str(source_row.subsequence_id),
        )
        clean = load_image(frame.image_path, device)
        labels = frame_labels(frame, protocol)
        for attack, adaptive, epsilon_px, steps in conditions:
            path = cache / f"{condition_id(str(frame.image_path), attack, adaptive, epsilon_px)}.json"
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload["checkpoint_sha256"] != sha256(checkpoint):
                    raise RuntimeError("Attack calibration cache checkpoint mismatch")
                rows.extend(payload["rows"])
                continue
            selected_seeds = [seeds[0]] if attack == "fgsm" else seeds
            candidates = []
            for seed in selected_seeds:
                result = (
                    fgsm(
                        model, clean, labels, protocol, epsilon_px, adaptive
                    )
                    if attack == "fgsm"
                    else pgd(
                        model,
                        clean,
                        labels,
                        protocol,
                        epsilon_px,
                        steps,
                        seed,
                        adaptive,
                    )
                )
                candidates.append((seed, result))
            best_index = max(
                range(len(candidates)),
                key=lambda index: candidates[index][1].attack_loss,
            )
            condition_rows = []
            for restart, (seed, result) in enumerate(candidates):
                defense = "Product_preprocessing" if adaptive else "none"
                detected, _predictions, _tiles = detection(
                    model,
                    result.adversarial,
                    labels,
                    protocol,
                    threshold,
                    defense,
                )
                condition_rows.append({
                    "grouped_scene_id": frame.grouped_scene_id,
                    "subsequence_id": frame.subsequence_id,
                    "image_path": str(frame.image_path.resolve()),
                    "checkpoint_sha256": sha256(checkpoint),
                    "threshold": threshold,
                    "attack": attack,
                    "adaptive": adaptive,
                    "epsilon_px": epsilon_px,
                    "epsilon": epsilon_px / 255.0,
                    "steps": steps,
                    "restart": restart,
                    "seed": seed,
                    "selected_best": restart == best_index,
                    "defense": defense,
                    **result_metrics(result, clean),
                    "precision": detected["precision"],
                    "recall": detected["recall"],
                    "f1": detected["f1"],
                    "f2": detected["f2"],
                    "fn": detected["fn"],
                    "nms_timeout": detected["nms_timeout"],
                    "nms_output_complete": detected["nms_output_complete"],
                })
            atomic_json(path, {
                "status": "PASS",
                "checkpoint_sha256": sha256(checkpoint),
                "rows": condition_rows,
            })
            rows.extend(condition_rows)
            del candidates, condition_rows
        del clean
    result = pd.DataFrame(rows)
    selected = result[result["selected_best"].astype(bool)].copy()
    summary = selected.groupby(
        ["attack", "adaptive", "epsilon_px"], as_index=False
    ).agg(
        frames=("image_path", "nunique"),
        f1_zero_fraction=("f1", lambda value: float((value <= 1e-12).mean())),
        recall_zero_fraction=(
            "recall", lambda value: float((value <= 1e-12).mean())
        ),
        actual_Linf_max=("actual_Linf", "max"),
        attack_loss_increase_mean=(
            "attack_loss_final",
            lambda value: float(
                (
                    value.to_numpy()
                    - selected.loc[value.index, "attack_loss_clean"].to_numpy()
                ).mean()
            ),
        ),
    )
    summary["floor_fraction"] = summary[
        ["f1_zero_fraction", "recall_zero_fraction"]
    ].max(axis=1)
    limit = float(protocol["attack_calibration"]["floor_fraction_max_exclusive"])
    summary["selected"] = summary["floor_fraction"] < limit
    if bool(
        (
            summary["actual_Linf_max"]
            > summary["epsilon_px"].astype(float) / 255.0 + 1e-6
        ).any()
    ):
        raise RuntimeError("Canonical M4 attack exceeds frozen L-infinity budget")
    for attack, adaptive in (("fgsm", False), ("pgd", False), ("pgd", True)):
        scope = summary[
            summary["attack"].eq(attack)
            & summary["adaptive"].eq(adaptive)
        ]
        if not bool(scope["selected"].any()):
            raise RuntimeError(
                f"All validation budgets have floor effect: {attack}/{adaptive}"
            )
    raw_path = DESTINATION / "validation_attack_calibration.csv"
    summary_path = DESTINATION / "budget_floor_audit.csv"
    atomic_csv(result, raw_path)
    atomic_csv(summary, summary_path)
    adaptive_rows = result[result["adaptive"].astype(bool)]
    adaptive_audit = {
        "status": (
            "PASS"
            if len(adaptive_rows)
            and bool((adaptive_rows["gradient_L2"] > 0).all())
            and bool(np.isfinite(adaptive_rows["gradient_L2"]).all())
            else "FAIL"
        ),
        "full_pipeline": "original_image_to_tiles_to_Product_preprocessing_to_YOLO_loss",
        "detach_in_adaptive_graph": False,
        "numpy_in_adaptive_graph": False,
        "no_grad_in_adaptive_graph": False,
        "gradient_L2_min": float(adaptive_rows["gradient_L2"].min()),
        "gradient_L2_max": float(adaptive_rows["gradient_L2"].max()),
        "attack_loss_increase_fraction": float(
            (
                adaptive_rows["attack_loss_final"]
                > adaptive_rows["attack_loss_clean"]
            ).mean()
        ),
    }
    atomic_json(DESTINATION / "adaptive_gradient_audit.json", adaptive_audit)
    if adaptive_audit["status"] != "PASS":
        raise RuntimeError("Canonical M4 adaptive gradient audit failed")
    selected_budget = {
        "fgsm_epsilon_px": summary[
            summary["attack"].eq("fgsm")
            & ~summary["adaptive"]
            & summary["selected"]
        ]["epsilon_px"].astype(float).tolist(),
        "pgd_epsilon_px": summary[
            summary["attack"].eq("pgd")
            & ~summary["adaptive"]
            & summary["selected"]
        ]["epsilon_px"].astype(float).tolist(),
        "adaptive_pgd_epsilon_px": summary[
            summary["attack"].eq("pgd")
            & summary["adaptive"]
            & summary["selected"]
        ]["epsilon_px"].astype(float).tolist(),
    }
    lock = {
        "status": "LOCKED",
        "locked_at": now(),
        "protocol_id": protocol["protocol_id"],
        "checkpoint_sha256": sha256(checkpoint),
        "threshold": threshold,
        "threshold_role": "inherited_safety_maximum_F2",
        "selection_split": "val",
        "test_used": False,
        "floor_rule": f"fraction_F1_or_Recall_zero < {limit}",
        "candidate_grid": {
            "fgsm_epsilon_px":
                protocol["attack_calibration"]["fgsm_epsilon_px"],
            "pgd_epsilon_px":
                protocol["attack_calibration"]["pgd"]["epsilon_px"],
            "adaptive_pgd_epsilon_px":
                protocol["attack_calibration"]["adaptive_product_pgd"]["epsilon_px"],
        },
        "selected": selected_budget,
        "pgd_steps": int(protocol["attack_calibration"]["pgd"]["steps"]),
        "adaptive_pgd_steps": int(
            protocol["attack_calibration"]["adaptive_product_pgd"]["steps"]
        ),
        "restarts": int(protocol["attack_calibration"]["pgd"]["restarts"]),
        "seeds": seeds,
        "raw_sha256": sha256(raw_path),
        "floor_audit_sha256": sha256(summary_path),
    }
    atomic_json(DESTINATION / "attack_protocol_lock.json", lock)
    return lock


def main() -> None:
    gate = json.loads(
        (OUTPUT_ROOT / "validation/quality_gate.json").read_text(encoding="utf-8")
    )
    print(json.dumps(run(Path(gate["checkpoint"])), indent=2))


if __name__ == "__main__":
    main()
