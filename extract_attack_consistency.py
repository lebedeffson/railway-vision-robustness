from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from audit_final_practice import canonical_path, load_manifest
from checkpoint_selection import selected_checkpoint
from extract_feature_consistency import loader, to_device, tnorm_filter, yolo_loss


PROJECT_DIR = Path(__file__).resolve().parent
MODEL = selected_checkpoint()
DATA = PROJECT_DIR / "data/yolo_osdar23/data.yaml"
MANIFEST = PROJECT_DIR / "data/yolo_osdar23/manifest.csv"
OUTPUT = PROJECT_DIR / "outputs/final_practice/attack_consistency_raw.csv"
SEEDS = [42, 123, 999]
FGSM_EPS = [0.5, 1.0, 2.0, 4.0, 8.0]
PGD_EPS = [0.1, 0.25, 0.5, 1.0]
EPS = 1e-12
MAX_SPATIAL_SAMPLES = 65_536


@dataclass
class AttackResult:
    adversarial: Tensor
    clean_gradient: Tensor
    path_gradient: Tensor
    attack_loss: float


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def defended_loss(
    model: nn.Module,
    batch: dict[str, Any],
    images: Tensor,
    adaptive: bool,
) -> Tensor:
    model_input = (
        checkpoint(tnorm_filter, images, use_reentrant=False)
        if adaptive and images.requires_grad
        else tnorm_filter(images)
        if adaptive
        else images
    )
    return yolo_loss(model, batch, model_input)


def loss_gradient(
    model: nn.Module,
    batch: dict[str, Any],
    images: Tensor,
    adaptive: bool,
) -> tuple[Tensor, float]:
    differentiable = images.detach().requires_grad_(True)
    loss = defended_loss(model, batch, differentiable, adaptive)
    gradient = torch.autograd.grad(loss, differentiable)[0]
    return gradient.detach(), float(loss.detach())


def fgsm(
    model: nn.Module,
    batch: dict[str, Any],
    epsilon_px: float,
    adaptive: bool,
) -> AttackResult:
    clean = batch["img"].detach()
    gradient, _ = loss_gradient(model, batch, clean, adaptive)
    adversarial = (
        clean + (epsilon_px / 255.0) * gradient.sign()
    ).clamp(0.0, 1.0).detach()
    final_loss = float(defended_loss(model, batch, adversarial, adaptive).detach())
    return AttackResult(adversarial, gradient, gradient, final_loss)


def pgd(
    model: nn.Module,
    batch: dict[str, Any],
    epsilon_px: float,
    steps: int,
    seed: int,
    adaptive: bool,
) -> AttackResult:
    if batch["img"].shape[0] != 1:
        raise RuntimeError("PGD restart selection requires --batch 1")
    clean = batch["img"].detach()
    epsilon = epsilon_px / 255.0
    alpha = epsilon / 4.0
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    random_delta = torch.empty_like(clean).uniform_(
        -epsilon, epsilon, generator=generator
    )
    adversarial = (clean + random_delta).clamp(0.0, 1.0).detach()
    clean_gradient, _ = loss_gradient(model, batch, clean, adaptive)
    path_gradient_sum = torch.zeros_like(clean)
    best_adversarial = adversarial.clone()
    best_loss = float(defended_loss(model, batch, adversarial, adaptive).detach())

    for _ in range(steps):
        gradient, _ = loss_gradient(model, batch, adversarial, adaptive)
        path_gradient_sum.add_(gradient)
        adversarial = adversarial + alpha * gradient.sign()
        delta = (adversarial - clean).clamp(-epsilon, epsilon)
        adversarial = (clean + delta).clamp(0.0, 1.0).detach()
        candidate_loss = float(defended_loss(model, batch, adversarial, adaptive).detach())
        if candidate_loss > best_loss:
            best_loss = candidate_loss
            best_adversarial = adversarial.clone()

    path_gradient = path_gradient_sum / steps
    return AttackResult(best_adversarial, clean_gradient, path_gradient, best_loss)


def object_masks(batch: dict[str, Any], height: int, width: int) -> Tensor:
    count = batch["img"].shape[0]
    masks = torch.zeros((count, height, width), dtype=torch.bool, device=batch["img"].device)
    for box, batch_index in zip(batch["bboxes"], batch["batch_idx"], strict=True):
        x, y, box_width, box_height = (float(value) for value in box)
        left = max(0, min(width, math.floor((x - box_width / 2.0) * width)))
        right = max(0, min(width, math.ceil((x + box_width / 2.0) * width)))
        top = max(0, min(height, math.floor((y - box_height / 2.0) * height)))
        bottom = max(0, min(height, math.ceil((y + box_height / 2.0) * height)))
        if right > left and bottom > top:
            masks[int(batch_index), top:bottom, left:right] = True
    return masks


def fuzzy_tnorm(left: Tensor, right: Tensor, operator: str) -> Tensor:
    if operator == "product":
        return left * right
    if operator == "godel":
        return torch.minimum(left, right)
    if operator == "lukasiewicz":
        return torch.clamp(left + right - 1.0, min=0.0)
    raise ValueError(operator)


def masked_vectors(left: Tensor, right: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    expanded = mask.unsqueeze(0).expand_as(left)
    return left[expanded].float(), right[expanded].float()


def directional_scores(gradient: Tensor, delta: Tensor, mask: Tensor) -> tuple[float, float]:
    left, right = masked_vectors(gradient, delta, mask)
    if left.numel() == 0 or left.norm() <= EPS or right.norm() <= EPS:
        return math.nan, math.nan
    cosine = float(torch.dot(left, right) / (left.norm() * right.norm() + EPS))
    active = (left.abs() > EPS) | (right.abs() > EPS)
    sign_agreement = float((left[active].sign() == right[active].sign()).float().mean())
    return cosine, sign_agreement


def spatial_scores(
    gradient: Tensor,
    delta: Tensor,
    mask: Tensor,
    epsilon: float,
) -> dict[str, float]:
    gradient_magnitude = gradient.abs().mean(0)
    perturbation_magnitude = delta.abs().amax(0) / max(epsilon, EPS)
    gradient_membership = gradient_magnitude / gradient_magnitude.max().clamp_min(EPS)
    left = gradient_membership[mask]
    right = perturbation_magnitude[mask].clamp(0.0, 1.0)
    if left.numel() == 0:
        return {name: math.nan for name in ("product", "godel", "lukasiewicz", "spearman", "topk_overlap")}
    if left.numel() > MAX_SPATIAL_SAMPLES:
        sample_indices = torch.linspace(
            0, left.numel() - 1, MAX_SPATIAL_SAMPLES,
            device=left.device, dtype=torch.float64,
        ).long()
        left = left[sample_indices]
        right = right[sample_indices]
    scores = {
        operator: float(fuzzy_tnorm(left, right, operator).mean())
        for operator in ("product", "godel", "lukasiewicz")
    }
    if left.unique().numel() < 2 or right.unique().numel() < 2:
        scores["spearman"] = math.nan
    else:
        scores["spearman"] = float(spearmanr(
            left.detach().cpu().numpy(), right.detach().cpu().numpy()
        ).statistic)
    k = max(1, math.ceil(0.1 * left.numel()))
    left_top = set(torch.topk(left, k).indices.detach().cpu().tolist())
    right_top = set(torch.topk(right, k).indices.detach().cpu().tolist())
    scores["topk_overlap"] = len(left_top & right_top) / k
    return scores


def consistency_row(
    clean: Tensor,
    result: AttackResult,
    object_mask: Tensor,
    epsilon_px: float,
    gradient: Tensor,
    prefix: str,
) -> dict[str, float]:
    adversarial = (
        result.adversarial[0]
        if result.adversarial.ndim == clean.ndim + 1
        else result.adversarial
    )
    delta = adversarial - clean
    epsilon = epsilon_px / 255.0
    masks = {
        "global": torch.ones_like(object_mask),
        "object": object_mask,
        "background": ~object_mask,
    }
    row: dict[str, float] = {}
    for scope, mask in masks.items():
        spatial = spatial_scores(gradient, delta, mask, epsilon)
        cosine, sign_agreement = directional_scores(gradient, delta, mask)
        c_dir = (cosine + 1.0) / 2.0 if math.isfinite(cosine) else math.nan
        row[f"{prefix}c_sp_{scope}"] = spatial["product"]
        row[f"{prefix}c_sp_godel_{scope}"] = spatial["godel"]
        row[f"{prefix}c_sp_lukasiewicz_{scope}"] = spatial["lukasiewicz"]
        row[f"{prefix}c_dir_{scope}"] = c_dir
        row[f"{prefix}gradient_delta_cosine_{scope}"] = cosine
        row[f"{prefix}sign_agreement_{scope}"] = sign_agreement
        row[f"{prefix}spearman_abs_{scope}"] = spatial["spearman"]
        row[f"{prefix}topk_overlap_{scope}"] = spatial["topk_overlap"]
        if not math.isfinite(c_dir):
            row[f"{prefix}c_atk_product_{scope}"] = math.nan
            row[f"{prefix}c_atk_godel_{scope}"] = math.nan
            row[f"{prefix}c_atk_lukasiewicz_{scope}"] = math.nan
        else:
            row[f"{prefix}c_atk_product_{scope}"] = spatial["product"] * c_dir
            row[f"{prefix}c_atk_godel_{scope}"] = min(spatial["godel"], c_dir)
            row[f"{prefix}c_atk_lukasiewicz_{scope}"] = max(
                0.0, spatial["lukasiewicz"] + c_dir - 1.0
            )
    return row


def sequence_lookup(manifest: Path) -> tuple[dict[str, str], dict[str, str]]:
    rows = load_manifest(manifest)
    exact = {canonical_path(row["image_path"]): row["sequence_id"] for row in rows}
    by_name: dict[str, set[str]] = {}
    for row in rows:
        by_name.setdefault(Path(row["image_path"]).name, set()).add(row["sequence_id"])
    unique_names = {name: next(iter(values)) for name, values in by_name.items() if len(values) == 1}
    return exact, unique_names


def model_loss_weights(model: nn.Module) -> dict[str, float | None]:
    args = model.args if isinstance(model.args, dict) else vars(model.args)
    return {name: float(args[name]) if name in args else None for name in ("box", "cls", "dfl")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract attack-side consistency diagnostics")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--pgd-eps", type=parse_floats, default=PGD_EPS)
    parser.add_argument("--fgsm-eps", type=parse_floats, default=FGSM_EPS)
    parser.add_argument("--seeds", type=parse_ints, default=SEEDS)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch != 1:
        raise SystemExit("Use --batch 1: best-restart loss must be selected per image")
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    yolo = YOLO(str(args.model))
    model = yolo.model.to(device).float().eval()
    overrides = model.args if isinstance(model.args, dict) else vars(model.args)
    model.args = get_cfg(overrides=overrides)
    model.criterion = None
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    exact_sequences, named_sequences = sequence_lookup(args.manifest)
    data_loader = loader(args.data, args.split, args.imgsz, 1, args.workers, device.type == "cuda")
    conditions = [("fgsm", eps, 1) for eps in args.fgsm_eps]
    conditions += [("pgd", eps, args.pgd_steps) for eps in args.pgd_eps]
    if args.quick:
        conditions = [("fgsm", 0.5, 1), ("pgd", 0.1, 2)]

    output = args.output or (
        OUTPUT.with_name("attack_consistency_adaptive_raw.csv")
        if args.adaptive else OUTPUT
    )
    rows: list[dict[str, object]] = []
    weights = model_loss_weights(model)
    for raw_batch in tqdm(data_loader, desc="attack consistency", unit="image"):
        batch = to_device(raw_batch, device)
        clean = batch["img"].detach()
        height, width = clean.shape[-2:]
        mask = object_masks(batch, height, width)[0]
        path = str(batch["im_file"][0])
        sequence_id = exact_sequences.get(canonical_path(path), named_sequences.get(Path(path).name))
        if sequence_id is None:
            raise RuntimeError(f"No sequence_id for {path}")

        for attack_name, epsilon_px, steps in conditions:
            seeds = [args.seeds[0]] if attack_name == "fgsm" else args.seeds
            candidates: list[AttackResult] = []
            for seed in seeds:
                torch.manual_seed(seed)
                result = (
                    fgsm(model, batch, epsilon_px, args.adaptive)
                    if attack_name == "fgsm"
                    else pgd(model, batch, epsilon_px, steps, seed, args.adaptive)
                )
                candidates.append(result)
            best = max(range(len(candidates)), key=lambda index: candidates[index].attack_loss)
            for restart, (seed, result) in enumerate(zip(seeds, candidates, strict=True)):
                base: dict[str, object] = {
                    "sequence_id": sequence_id,
                    "image_path": path,
                    "split": args.split,
                    "attack": attack_name,
                    "adaptive": args.adaptive,
                    "epsilon_px": epsilon_px,
                    "epsilon": epsilon_px / 255.0,
                    "steps": steps,
                    "restart": restart,
                    "seed": seed,
                    "selected_best": restart == best,
                    "attack_loss": result.attack_loss,
                    "attack_objective": "ultralytics_box_cls_dfl",
                    "lambda_box": weights["box"],
                    "lambda_cls": weights["cls"],
                    "lambda_dfl": weights["dfl"],
                }
                base.update(consistency_row(
                    clean[0], result, mask, epsilon_px, result.clean_gradient[0], "clean_gradient_"
                ))
                base.update(consistency_row(
                    clean[0], result, mask, epsilon_px, result.path_gradient[0], "path_gradient_"
                ))
                # Canonical aliases used by the unified final table.
                base["c_sp_global"] = base["path_gradient_c_sp_global"]
                base["c_sp_object"] = base["path_gradient_c_sp_object"]
                base["c_sp_background"] = base["path_gradient_c_sp_background"]
                base["c_dir"] = base["path_gradient_c_dir_global"]
                base["c_dir_object"] = base["path_gradient_c_dir_object"]
                base["c_dir_background"] = base["path_gradient_c_dir_background"]
                base["c_atk_global"] = base["path_gradient_c_atk_product_global"]
                base["c_atk_object"] = base["path_gradient_c_atk_product_object"]
                base["c_atk_background"] = base[
                    "path_gradient_c_atk_product_background"
                ]
                rows.append(base)

    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    config = {
        "output": str(output),
        "rows": len(rows),
        "split": args.split,
        "adaptive": args.adaptive,
        "fgsm_eps_px": args.fgsm_eps,
        "pgd_eps_px": args.pgd_eps,
        "pgd_steps": args.pgd_steps,
        "seeds": args.seeds,
        "restarts": len(args.seeds),
        "best_restart_rule": "maximum attack_loss per image and condition",
        "within_restart_rule": "retain maximum-loss iterate, including random start",
        "attack_objective": "Ultralytics YOLO11 configured box + cls + DFL loss; no legacy L_obj",
        "spatial_memberships": "channel-mean abs gradient normalized by image max; channel-max abs delta normalized by epsilon",
        "c_sp": "mean Product T-norm of spatial memberships",
        "c_dir": "(cosine(gradient, delta) + 1) / 2",
        "c_atk": "Product(c_sp, c_dir)",
        "max_spatial_samples_per_scope": MAX_SPATIAL_SAMPLES,
    }
    output.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
