from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scripts.person_v4.losses import update_group_dro_weights


def classify_proxy_gate(
    paired_rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Classify three paired A3-A0 proxy splits without article claims."""
    if len(paired_rows) != int(gate["split_count"]):
        raise ValueError("Proxy gate requires exactly the frozen split count")
    deltas = {
        metric: np.asarray(
            [float(row[f"delta_{metric}"]) for row in paired_rows],
            dtype=float,
        )
        for metric in ("mAP50", "recall", "small_recall")
    }
    technical = all(
        int(row["NaN_Inf"]) == int(gate["NaN_Inf"])
        and int(row["lost_GT"]) == int(gate["lost_GT"])
        and row["evaluator_consistency"] == gate["evaluator_consistency"]
        for row in paired_rows
    )
    improved = sum(
        float(row["delta_mAP50"]) > 0
        and float(row["delta_recall"]) > 0
        and float(row["delta_small_recall"]) > 0
        for row in paired_rows
    )
    checks = {
        "median_delta_mAP50": float(np.median(deltas["mAP50"])),
        "median_delta_recall": float(np.median(deltas["recall"])),
        "median_delta_small_recall": float(
            np.median(deltas["small_recall"])
        ),
        "worst_split_delta_recall": float(deltas["recall"].min()),
        "improved_splits": int(improved),
        "technical_checks_passed": bool(technical),
    }
    passed = (
        checks["median_delta_mAP50"]
        >= float(gate["median_delta_mAP50_min"])
        and checks["median_delta_recall"]
        >= float(gate["median_delta_recall_min"])
        and checks["median_delta_small_recall"]
        >= float(gate["median_delta_small_recall_min"])
        and checks["worst_split_delta_recall"]
        >= float(gate["worst_split_delta_recall_min"])
        and checks["improved_splits"] >= int(gate["improved_splits_min"])
        and checks["worst_split_delta_recall"]
        >= -float(gate["maximum_single_split_recall_drop"])
        and technical
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "scientific_result": False,
        "article_evidence": False,
        "external_fold_allowed": bool(passed),
        "test_allowed": False,
        "checks": checks,
    }


def synthetic_four_domain_benchmark(
    seed: int = 5,
    samples_per_domain: int = 256,
    steps: int = 500,
) -> dict[str, Any]:
    """Deterministic toy check for style overfit and worst-group weighting."""
    generator = torch.Generator().manual_seed(int(seed))

    def domain(correlation: float) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.where(
            torch.arange(samples_per_domain) % 2 == 0, 1.0, -1.0
        )
        core = (
            0.55 * labels
            + 0.60
            * torch.randn(samples_per_domain, generator=generator)
        )
        style = (
            correlation * labels
            + 0.25
            * torch.randn(samples_per_domain, generator=generator)
        )
        return torch.stack((core, style), dim=1), (labels > 0).float()

    source = [domain(value) for value in (2.0, 1.5, -0.8)]
    heldout = domain(-2.0)

    def fit(use_dg: bool) -> tuple[float, list[float], list[float]]:
        weight = torch.zeros(2, requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        group_weights = torch.full((3,), 1 / 3)
        for step in range(int(steps)):
            losses = []
            for features, labels in source:
                transformed = features
                if use_dg:
                    transformed = features.clone()
                    transformed[:, 1] = torch.roll(
                        features[:, 1], shifts=(step % 17) + 1
                    )
                logits = transformed @ weight + bias
                losses.append(
                    F.binary_cross_entropy_with_logits(logits, labels)
                )
            vector = torch.stack(losses)
            if use_dg:
                for index, loss in enumerate(vector.detach()):
                    group_weights = update_group_dro_weights(
                        group_weights, index, loss, eta=0.05
                    )
                objective = (group_weights * vector).sum()
            else:
                objective = vector.mean()
            weight_grad, bias_grad = torch.autograd.grad(
                objective, (weight, bias)
            )
            with torch.no_grad():
                weight -= 0.08 * weight_grad
                bias -= 0.08 * bias_grad
            weight.requires_grad_()
            bias.requires_grad_()
        features, labels = heldout
        probabilities = torch.sigmoid(features @ weight + bias)
        accuracy = (
            (probabilities >= 0.5) == labels.bool()
        ).float().mean()
        return (
            float(accuracy),
            [float(value) for value in weight.detach()],
            [float(value) for value in group_weights],
        )

    erm_accuracy, erm_weight, uniform_weights = fit(False)
    dg_accuracy, dg_weight, group_weights = fit(True)
    improvement = dg_accuracy - erm_accuracy
    if not all(
        math.isfinite(value)
        for value in (
            erm_accuracy,
            dg_accuracy,
            improvement,
            *erm_weight,
            *dg_weight,
            *group_weights,
        )
    ):
        raise FloatingPointError("Synthetic DG benchmark is not finite")
    return {
        "status": "PASS" if improvement > 0.20 else "FAIL",
        "purpose": "integration_test_only",
        "article_evidence": False,
        "source_domain_count": 3,
        "heldout_domain_count": 1,
        "ERM_heldout_accuracy": erm_accuracy,
        "DG_heldout_accuracy": dg_accuracy,
        "accuracy_improvement": improvement,
        "ERM_feature_weights": erm_weight,
        "DG_feature_weights": dg_weight,
        "final_group_weights": group_weights,
        "style_mixing": "deterministic_cross_example_permutation",
    }
