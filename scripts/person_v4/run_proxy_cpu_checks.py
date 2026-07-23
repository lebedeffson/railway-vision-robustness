from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.person_v4.losses import (
    normalized_wasserstein_similarity,
    quality_focal_loss,
    update_group_dro_weights,
)
from scripts.person_v4.proxy import synthetic_four_domain_benchmark
from scripts.person_v4.swad import average_state_dicts, state_dict_sha256
from scripts.person_v4.trainer import MixStyleSceneBank


class _TrainingFlag(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.active_scene_ids: list[str] = []


def run_checks() -> tuple[dict, dict]:
    checks: dict[str, bool | float | str] = {}

    target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    shifted = torch.tensor(
        [[1.0, 0.0, 3.0, 2.0]], requires_grad=True
    )
    identity = normalized_wasserstein_similarity(target, target, 8.0)
    similarity = normalized_wasserstein_similarity(target, shifted, 8.0)
    (1 - similarity).sum().backward()
    checks["NWD_identity"] = bool(torch.allclose(identity, torch.ones(1)))
    checks["NWD_tiny_gradient_finite"] = bool(
        torch.isfinite(shifted.grad).all()
    )
    one_pixel_loss = float((1 - similarity).detach())
    checks["NWD_one_pixel_shift_loss"] = one_pixel_loss
    checks["NWD_smoother_than_IoU"] = one_pixel_loss < 2 / 3

    positive = torch.tensor([-2.0], requires_grad=True)
    quality_focal_loss(
        positive, torch.tensor([0.8]), beta=2.0
    ).sum().backward()
    background = torch.tensor([2.0], requires_grad=True)
    quality_focal_loss(
        background, torch.tensor([0.0]), beta=2.0
    ).sum().backward()
    checks["QFL_raises_underpredicted_quality"] = float(positive.grad) < 0
    checks["QFL_lowers_background_score"] = float(background.grad) > 0

    weights = torch.full((3,), 1 / 3)
    for index, loss in enumerate((0.2, 0.5, 1.2)):
        weights = update_group_dro_weights(
            weights, index, loss, eta=0.5
        )
    checks["GroupDRO_worst_scene_weight_largest"] = bool(
        weights[2] > weights[1] > weights[0]
    )
    checks["GroupDRO_weight_sum"] = float(weights.sum())

    torch.manual_seed(3)
    model = _TrainingFlag()
    mix = MixStyleSceneBank(
        model=model,
        probability=1.0,
        beta_alpha=0.1,
        epsilon=1.0e-6,
    )
    model.train()
    model.active_scene_ids = ["scene_a"]
    first = torch.randn(1, 4, 3, 3)
    mix(torch.nn.Identity(), (first,), first)
    model.active_scene_ids = ["scene_b"]
    second = torch.randn(1, 4, 3, 3) + 10
    mixed = mix(torch.nn.Identity(), (second,), second)
    model.eval()
    held = torch.randn(1, 4, 3, 3)
    checks["MixStyle_cross_scene_mixed"] = not torch.equal(second, mixed)
    checks["MixStyle_shape_preserved"] = mixed.shape == second.shape
    checks["MixStyle_eval_disabled"] = torch.equal(
        held, mix(torch.nn.Identity(), (held,), held)
    )

    states = [
        {
            "weight": torch.tensor([1.0, 3.0]),
            "bn.running_mean": torch.tensor([2.0]),
            "bn.num_batches_tracked": torch.tensor(4),
        },
        {
            "weight": torch.tensor([3.0, 5.0]),
            "bn.running_mean": torch.tensor([4.0]),
            "bn.num_batches_tracked": torch.tensor(7),
        },
    ]
    averaged = average_state_dicts(states)
    repeated = average_state_dicts(states)
    checks["SWAD_parameter_mean_exact"] = torch.equal(
        averaged["weight"], torch.tensor([2.0, 4.0])
    )
    checks["SWAD_BN_counter_explicit_max"] = (
        int(averaged["bn.num_batches_tracked"]) == 7
    )
    checks["SWAD_hash_deterministic"] = (
        state_dict_sha256(averaged) == state_dict_sha256(repeated)
    )
    boolean_checks = [
        value for value in checks.values() if isinstance(value, bool)
    ]
    status = "PASS" if all(boolean_checks) else "FAIL"
    return (
        {
            "status": status,
            "device": "cpu",
            "scientific_result": False,
            "checks": checks,
        },
        synthetic_four_domain_benchmark(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("protocols/person_v4_train_only_proxy_v1"),
    )
    args = parser.parse_args()
    math_report, synthetic_report = run_checks()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("math_sanity.json", math_report),
        ("synthetic_four_domain.json", synthetic_report),
    ):
        (args.output_root / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "math": math_report["status"],
                "synthetic": synthetic_report["status"],
            },
            indent=2,
        )
    )
    if (
        math_report["status"] != "PASS"
        or synthetic_report["status"] != "PASS"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
