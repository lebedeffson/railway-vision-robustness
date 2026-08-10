from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.metrics import box_iou

from audit_final_practice import canonical_path, load_manifest
from checkpoint_selection import configured_checkpoint
from evaluate_image_level_detection import (
    DEFAULT_CONFIDENCE,
    detection_metrics,
    get_ground_truth,
    predict_batch,
)
from extract_attack_consistency import (
    FGSM_EPS,
    PGD_EPS,
    SEEDS,
    consistency_row,
    fgsm,
    model_loss_weights,
    object_masks,
    pgd,
    sequence_lookup,
)
from extract_feature_consistency import (
    FeatureHook,
    clipped_recovery,
    defend,
    defense_consistency,
    distance_recovery,
    loader,
    metrics,
    recovery,
    to_device,
)
from revision_q1.feature_metrics import (
    distance_recovery as revision_distance_recovery,
    pair_metrics as revision_pair_metrics,
    similarity_recovery as revision_similarity_recovery,
)
from revision_q1.normalization import normalized_quality_recovery


PROJECT_DIR = Path(__file__).resolve().parent
MODEL = configured_checkpoint()
DATA = PROJECT_DIR / "data/yolo_osdar23/data.yaml"
MANIFEST = PROJECT_DIR / "data/yolo_osdar23/manifest.csv"
STATS = PROJECT_DIR / "outputs/diagnostics/feature_consistency/feature_normalization_val.pt"
OUTPUT = PROJECT_DIR / "outputs/final_practice/unified_diagnostics_raw.csv"
DEFENSES = ["none", "tnorm", "bilateral", "gaussian", "median", "jpeg"]
DEADLINE_ROOT = PROJECT_DIR / "outputs/final_practice/deadline"


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_strings(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_modes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def prediction_diagnostics(
    prediction: Tensor | None,
    gt_boxes: Tensor,
    gt_classes: Tensor,
    confidence: float,
) -> dict[str, float | int]:
    base = detection_metrics(prediction, gt_boxes, gt_classes, confidence)
    if prediction is None or prediction.numel() == 0:
        return {**base, "mean_confidence": 0.0, "mean_matched_iou": 0.0}
    filtered = prediction[prediction[:, 4] >= confidence]
    if filtered.numel() == 0:
        return {**base, "mean_confidence": 0.0, "mean_matched_iou": 0.0}
    mean_confidence = float(filtered[:, 4].mean())
    matched: set[int] = set()
    matched_ious: list[float] = []
    for row in filtered[filtered[:, 4].argsort(descending=True)]:
        candidates = [
            index for index in range(len(gt_boxes))
            if index not in matched
            and int(gt_classes[index].item()) == int(row[5].item())
        ]
        if not candidates:
            continue
        candidate_tensor = torch.tensor(candidates, device=gt_boxes.device)
        ious = box_iou(row[:4].view(1, 4), gt_boxes[candidate_tensor]).view(-1)
        position = int(torch.argmax(ious))
        value = float(ious[position])
        if value >= 0.5:
            matched.add(candidates[position])
            matched_ious.append(value)
    return {
        **base,
        "mean_confidence": mean_confidence,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def detection_for_image(
    model: nn.Module,
    batch: dict[str, Any],
    images: Tensor,
    confidence: float,
    nms_max_time_img: float = 0.05,
) -> dict[str, float | int]:
    predictions, nms_diagnostics = predict_batch(
        model, images, max_time_img=nms_max_time_img, return_diagnostics=True
    )
    prediction = predictions[0]
    boxes, classes = get_ground_truth(batch, 0)
    return {
        **prediction_diagnostics(prediction, boxes, classes, confidence),
        **nms_diagnostics[0],
    }


def feature_values(
    attacked: dict[str, float],
    defended: dict[str, float],
    preservation: dict[str, float],
) -> dict[str, float]:
    gains_raw = {
        name: recovery(attacked[name], defended[name])
        for name in ("cos_norm", "product", "godel", "lukas")
    }
    gains = {
        name: clipped_recovery(attacked[name], defended[name])
        for name in ("cos_norm", "product", "godel", "lukas")
    }
    return {
        "cosine": attacked["cos_norm"],
        "mse": attacked["mse"],
        "mae": attacked["mae"],
        "relative_l2": attacked["relative_l2"],
        "mean_shift": attacked["mean_shift"],
        "entropy": attacked["entropy_change"],
        "product": attacked["product"],
        "godel": attacked["godel"],
        "lukasiewicz": attacked["lukas"],
        "cosine_recovery": recovery(attacked["cos_norm"], defended["cos_norm"]),
        "mse_recovery": distance_recovery(attacked["mse"], defended["mse"]),
        "mae_recovery": distance_recovery(attacked["mae"], defended["mae"]),
        "relative_l2_recovery": distance_recovery(
            attacked["relative_l2"], defended["relative_l2"]
        ),
        "mean_shift_recovery": distance_recovery(
            attacked["mean_shift"], defended["mean_shift"]
        ),
        "entropy_recovery": distance_recovery(
            attacked["entropy_change"], defended["entropy_change"]
        ),
        "product_recovery": recovery(attacked["product"], defended["product"]),
        "godel_recovery": recovery(attacked["godel"], defended["godel"]),
        "lukasiewicz_recovery": recovery(attacked["lukas"], defended["lukas"]),
        "p_clean_preservation": preservation["product"],
        "a_attacked_similarity": attacked["product"],
        "r_restored_similarity": defended["product"],
        "g_recovery": gains_raw["product"],
        "g_recovery_clipped": gains["product"],
        "c_def": defense_consistency(
            "product", preservation["product"], gains["product"]
        ),
        "p_godel": preservation["godel"],
        "a_godel": attacked["godel"],
        "r_godel": defended["godel"],
        "g_godel": gains_raw["godel"],
        "g_godel_clipped": gains["godel"],
        "c_def_godel": defense_consistency(
            "godel", preservation["godel"], gains["godel"]
        ),
        "p_lukasiewicz": preservation["lukas"],
        "a_lukasiewicz": attacked["lukas"],
        "r_lukasiewicz": defended["lukas"],
        "g_lukasiewicz": gains_raw["lukas"],
        "g_lukasiewicz_clipped": gains["lukas"],
        "c_def_lukasiewicz": defense_consistency(
            "lukas", preservation["lukas"], gains["lukas"]
        ),
    }


def revision_feature_values(
    clean: Tensor,
    attacked: Tensor,
    defended: Tensor,
    filtered_clean: Tensor,
    statistics: dict[str, Tensor],
    mode: str,
) -> dict[str, float]:
    attacked_values = revision_pair_metrics(clean, attacked, statistics, mode)[0]
    defended_values = revision_pair_metrics(clean, defended, statistics, mode)[0]
    preservation_values = revision_pair_metrics(clean, filtered_clean, statistics, mode)[0]
    return revision_feature_values_from_metrics(
        attacked_values, defended_values, preservation_values, mode
    )


def revision_feature_values_from_metrics(
    attacked_values: dict[str, float],
    defended_values: dict[str, float],
    preservation_values: dict[str, float],
    mode: str,
) -> dict[str, float]:
    result = {f"{mode}_{name}": value for name, value in attacked_values.items()}
    similarities = {
        "cosine_similarity", "pearson_correlation", "spearman_correlation",
        "product", "godel", "lukasiewicz",
    }
    for name in attacked_values:
        result[f"{mode}_{name}_recovery"] = (
            revision_similarity_recovery(attacked_values[name], defended_values[name])
            if name in similarities
            else revision_distance_recovery(attacked_values[name], defended_values[name])
        )
    for operator in ("product", "godel", "lukasiewicz"):
        preservation = preservation_values[operator]
        before = attacked_values[operator]
        after = defended_values[operator]
        gain = revision_similarity_recovery(before, after)
        operator_name = "lukas" if operator == "lukasiewicz" else operator
        result[f"{mode}_p_{operator}"] = preservation
        result[f"{mode}_a_{operator}"] = before
        result[f"{mode}_r_{operator}"] = after
        result[f"{mode}_g_{operator}"] = gain
        result[f"{mode}_g_{operator}_clipped"] = min(1.0, max(0.0, gain))
        result[f"{mode}_c_def_{operator}"] = defense_consistency(
            operator_name, preservation, min(1.0, max(0.0, gain))
        )
    return result


def conditions(args: argparse.Namespace) -> list[tuple[str, float, int, bool]]:
    rows = [("fgsm", epsilon, 1, False) for epsilon in args.fgsm_eps]
    for epsilon in args.pgd_eps:
        for steps in args.pgd_steps:
            rows.append(("pgd", epsilon, steps, False))
    if args.adaptive_pgd:
        adaptive_eps = args.adaptive_pgd_eps or args.pgd_eps
        for epsilon in adaptive_eps:
            for steps in args.adaptive_pgd_steps:
                rows.append(("pgd", epsilon, steps, True))
    return rows


def expected_rows_per_image(args: argparse.Namespace) -> int:
    total = 0
    for attack_name, _epsilon, _steps, adaptive in conditions(args):
        restart_count = 1 if attack_name == "fgsm" else len(args.seeds)
        defenses = (
            [name for name in args.defenses if name in {"none", "tnorm"}]
            if adaptive else args.defenses
        )
        total += restart_count * len(defenses) * 3
    return total


def prepare_partial(path: Path, expected_count: int) -> tuple[set[str], list[str] | None, int]:
    if not path.is_file() or path.stat().st_size == 0:
        return set(), None, 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "image_path" not in fieldnames:
        raise RuntimeError(f"Invalid checkpoint CSV: {path}")
    counts = Counter(row["image_path"] for row in rows)
    completed = {image for image, count in counts.items() if count == expected_count}
    retained = [row for row in rows if row["image_path"] in completed]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)
    return completed, fieldnames, len(retained)


def condition_key(
    image_path: str, attack: str, adaptive: bool | str,
    epsilon_px: float | str, steps: int | str,
) -> tuple[str, str, str, str, str]:
    return (
        image_path, str(attack), str(adaptive).lower(),
        f"{float(epsilon_px):.12g}", str(int(float(steps))),
    )


def expected_rows_per_condition(
    args: argparse.Namespace, attack: str, adaptive: bool,
) -> int:
    restarts = 1 if attack == "fgsm" else len(args.seeds)
    defenses = (
        [name for name in args.defenses if name in {"none", "tnorm"}]
        if adaptive else args.defenses
    )
    return restarts * len(defenses) * 3


def prepare_condition_partial(
    path: Path, args: argparse.Namespace,
) -> tuple[set[str], set[tuple[str, str, str, str, str]], list[str] | None, int]:
    if not path.is_file() or path.stat().st_size == 0:
        return set(), set(), None, 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    required = {"image_path", "attack", "adaptive", "epsilon_px", "steps"}
    if not fieldnames or not required <= set(fieldnames):
        raise RuntimeError(f"Invalid condition checkpoint CSV: {path}")
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = condition_key(
            row["image_path"], row["attack"], row["adaptive"],
            row["epsilon_px"], row["steps"],
        )
        grouped.setdefault(key, []).append(row)
    completed_conditions = {
        key for key, values in grouped.items()
        if len(values) == expected_rows_per_condition(
            args, key[1], key[2] in {"true", "1", "yes"},
        )
    }
    retained = [
        row for key in completed_conditions for row in grouped[key]
    ]
    counts = Counter(key[0] for key in completed_conditions)
    completed_images = {
        image for image, count in counts.items() if count == len(conditions(args))
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(retained)
    return completed_images, completed_conditions, fieldnames, len(retained)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen final experiment matrix")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--stats", type=Path, default=STATS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument(
        "--nms-max-time-img", type=float, default=0.05,
        help="Ultralytics per-image NMS allowance added to its fixed two-second limit",
    )
    parser.add_argument("--fgsm-eps", type=parse_floats, default=FGSM_EPS)
    parser.add_argument("--pgd-eps", type=parse_floats, default=PGD_EPS)
    parser.add_argument("--pgd-steps", type=parse_ints, default=[20])
    parser.add_argument(
        "--adaptive-pgd-eps", type=parse_floats,
        help="Optional adaptive-only epsilon grid; defaults to --pgd-eps",
    )
    parser.add_argument(
        "--adaptive-pgd-steps", type=parse_ints, default=[20, 40],
        help="PGD step counts through the differentiable Product filter",
    )
    parser.add_argument("--seeds", type=parse_ints, default=SEEDS)
    parser.add_argument("--defenses", type=parse_strings, default=DEFENSES)
    parser.add_argument("--adaptive-pgd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--revision-stats", type=Path,
        help="Per-layer/channel clean-validation statistics for Q1 metrics",
    )
    parser.add_argument(
        "--normalizations", type=parse_modes, default=[],
        help="Comma-separated Q1 normalization modes",
    )
    parser.add_argument("--checkpoint-name", default="stage2_best")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_metadata_lookup(
    manifest: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    rows = load_manifest(manifest)
    exact: dict[str, dict[str, str]] = {}
    names: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        metadata = {
            "grouped_scene_id": row["sequence_id"],
            "subsequence_id": row.get("subsequence_id") or row.get("source_sequence") or row["sequence_id"],
        }
        exact[row["image_path"]] = metadata
        names.setdefault(Path(row["image_path"]).name, []).append(metadata)
    unique = {name: values[0] for name, values in names.items() if len(values) == 1}
    return exact, unique


def apply_deadline_defaults(args: argparse.Namespace) -> None:
    """Apply the validation-frozen minimal matrix to the canonical test run."""
    gate_path = DEADLINE_ROOT / "pilot/pilot_gate.json"
    statistics_path = DEADLINE_ROOT / "normalization/layer_channel_statistics.pt"
    canonical_test = args.output.resolve() == OUTPUT.resolve()
    canonical_v2_block = PROJECT_DIR / "outputs/canonical_v2/LEGACY_TEST_BLOCKED"
    if args.split == "test" and canonical_test and canonical_v2_block.is_file():
        raise RuntimeError(
            "Legacy/current-split test is blocked by canonical v2 baseline-rescue protocol"
        )
    if (
        args.split != "test" or not canonical_test or args.revision_stats is not None
        or not gate_path.is_file() or not statistics_path.is_file()
    ):
        return
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("Canonical test matrix is blocked by the deadline pilot gate")
    args.revision_stats = statistics_path
    args.normalizations = ["N1_quantile"]
    budget_path = DEADLINE_ROOT / "config/canonical_budget_selection.json"
    if not budget_path.is_file():
        raise RuntimeError(
            "Canonical test matrix is blocked until validation budgets are frozen"
        )
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    if budget.get("status") != "PASS":
        raise RuntimeError("Canonical test matrix is blocked by floor-effect budget gate")
    args.fgsm_eps = [float(value) for value in budget["selected"]["fgsm_epsilon_px"]]
    args.pgd_eps = [float(value) for value in budget["selected"]["pgd_epsilon_px"]]
    args.pgd_steps = [20]
    args.adaptive_pgd_eps = [
        float(value) for value in budget["selected"]["adaptive_pgd_epsilon_px"]
    ]
    args.adaptive_pgd_steps = [20]
    args.seeds = [42, 123, 999]
    args.defenses = ["none", "tnorm", "bilateral", "median"]
    args.checkpoint_name = "stage2_best"
    args.nms_max_time_img = 10.0


def main() -> None:
    args = parse_args()
    if args.quick:
        args.fgsm_eps, args.pgd_eps, args.pgd_steps, args.seeds = [0.5], [0.1], [2], [42]
        args.adaptive_pgd_eps = [0.1]
        args.adaptive_pgd_steps = [2]
        args.defenses, args.max_images = ["none", "tnorm"], 1
    else:
        apply_deadline_defaults(args)
    if args.data.name == "pilot_data.yaml" and "canonical_v2" in str(args.data):
        selection_path = args.data.parent / "pilot_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("selection_method") != "deterministic_even_frame_order_no_model_or_difficulty":
            raise RuntimeError("Canonical pilot selection is stale or result-dependent")
    config_path = args.output.with_suffix(".json")
    if args.output.is_file() and config_path.is_file():
        print(f"Final matrix already complete: {args.output}")
        return
    print(f"Selected checkpoint: {args.model.resolve()}", flush=True)
    checkpoint_hash = sha256(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = args.output.with_suffix(args.output.suffix + ".tmp")
    completed_images, completed_conditions, fieldnames, total_rows = (
        prepare_condition_partial(partial_path, args)
    )
    if completed_images:
        print(f"Resuming final matrix after {len(completed_images)} complete images")
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    yolo = YOLO(str(args.model))
    model = yolo.model.to(device).float().eval()
    overrides = model.args if isinstance(model.args, dict) else vars(model.args)
    model.args = get_cfg(overrides=overrides)
    model.criterion = None
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    normalization = torch.load(args.stats, map_location="cpu")["stats"]
    revision_statistics = None
    if args.revision_stats is not None:
        revision_statistics = torch.load(
            args.revision_stats, map_location="cpu"
        )["statistics"]
    if args.normalizations and revision_statistics is None:
        raise RuntimeError("--normalizations requires --revision-stats")
    exact_sequences, named_sequences = sequence_lookup(args.manifest)
    exact_metadata, named_metadata = frame_metadata_lookup(args.manifest)
    hook = FeatureHook(model)
    loss_weights = model_loss_weights(model)
    data_loader = loader(args.data, args.split, args.imgsz, 1, args.workers, device.type == "cuda")

    try:
        with partial_path.open("a", encoding="utf-8", newline="") as checkpoint:
            writer: csv.DictWriter | None = None
            for image_index, raw_batch in enumerate(tqdm(data_loader, desc="final matrix")):
                if args.max_images is not None and image_index >= args.max_images:
                    break
                batch = to_device(raw_batch, device)
                clean = batch["img"].detach()
                path = str(batch["im_file"][0])
                if path in completed_images:
                    continue
                sequence_id = exact_sequences.get(
                    canonical_path(path), named_sequences.get(Path(path).name)
                )
                if sequence_id is None:
                    raise RuntimeError(f"No sequence_id for {path}")
                frame_metadata = exact_metadata.get(
                    canonical_path(path), named_metadata.get(Path(path).name)
                )
                if frame_metadata is None:
                    raise RuntimeError(f"No grouped-scene/subsequence metadata for {path}")
                clean_features = hook.extract(model, clean)
                clean_detection = detection_for_image(
                    model, batch, clean, args.confidence, args.nms_max_time_img
                )
                clean_defense_features: dict[str, list[Tensor]] = {"none": clean_features}
                for defense_name in args.defenses:
                    if defense_name != "none":
                        clean_defense_features[defense_name] = hook.extract(
                            model, defend(clean, defense_name)
                        )
                revision_preservation_cache: dict[
                    tuple[str, int, str], dict[str, float]
                ] = {}
                if revision_statistics is not None:
                    for defense_name, defense_features in clean_defense_features.items():
                        for level_index, level in enumerate(("P3", "P4", "P5")):
                            for mode in args.normalizations:
                                revision_preservation_cache[(
                                    defense_name, level_index, mode
                                )] = revision_pair_metrics(
                                    clean_features[level_index],
                                    defense_features[level_index],
                                    revision_statistics[level], mode,
                                )[0]
                mask = object_masks(batch, clean.shape[-2], clean.shape[-1])[0]

                for attack_name, epsilon_px, steps, adaptive in conditions(args):
                    current_condition = condition_key(
                        path, attack_name, adaptive, epsilon_px, steps
                    )
                    if current_condition in completed_conditions:
                        continue
                    condition_rows: list[dict[str, object]] = []
                    seeds = [args.seeds[0]] if attack_name == "fgsm" else args.seeds
                    candidates = []
                    for seed in seeds:
                        torch.manual_seed(seed)
                        with torch.enable_grad():
                            result = (
                                fgsm(model, batch, epsilon_px, adaptive)
                                if attack_name == "fgsm"
                                else pgd(model, batch, epsilon_px, steps, seed, adaptive)
                            )
                        candidates.append(result)
                    best_restart = max(
                        range(len(candidates)), key=lambda index: candidates[index].attack_loss
                    )
                    for restart, (seed, result) in enumerate(zip(seeds, candidates, strict=True)):
                        adversarial = result.adversarial
                        perturbation = adversarial - clean
                        attacked_features = hook.extract(model, adversarial)
                        revision_attacked_cache: dict[
                            tuple[int, str], dict[str, float]
                        ] = {}
                        if revision_statistics is not None:
                            for level_index, level in enumerate(("P3", "P4", "P5")):
                                for mode in args.normalizations:
                                    revision_attacked_cache[(
                                        level_index, mode
                                    )] = revision_pair_metrics(
                                        clean_features[level_index],
                                        attacked_features[level_index],
                                        revision_statistics[level], mode,
                                    )[0]
                        attacked_detection = detection_for_image(
                            model, batch, adversarial, args.confidence,
                            args.nms_max_time_img,
                        )
                        clean_attack_values = consistency_row(
                            clean[0], result, mask, epsilon_px,
                            result.clean_gradient[0], "clean_gradient_",
                        )
                        path_attack_values = consistency_row(
                            clean[0], result, mask, epsilon_px,
                            result.path_gradient[0], "path_gradient_",
                        )
                        selected_defenses = (
                            [name for name in args.defenses if name in {"none", "tnorm"}]
                            if adaptive else args.defenses
                        )
                        for defense_name in selected_defenses:
                            defended = adversarial if defense_name == "none" else defend(
                                adversarial, defense_name
                            )
                            defended_features = (
                                attacked_features if defense_name == "none"
                                else hook.extract(model, defended)
                            )
                            defended_detection = (
                                attacked_detection if defense_name == "none"
                                else detection_for_image(
                                    model, batch, defended, args.confidence,
                                    args.nms_max_time_img,
                                )
                            )
                            for level_index, level in enumerate(("P3", "P4", "P5")):
                                attacked_metric = metrics(
                                    clean_features[level_index], attacked_features[level_index],
                                    normalization[level],
                                )[0]
                                defended_metric = metrics(
                                    clean_features[level_index], defended_features[level_index],
                                    normalization[level],
                                )[0]
                                preservation_metric = metrics(
                                    clean_features[level_index],
                                    clean_defense_features[defense_name][level_index],
                                    normalization[level],
                                )[0]
                                row: dict[str, object] = {
                                    "checkpoint_name": args.checkpoint_name,
                                    "checkpoint_sha256": checkpoint_hash,
                                    "sequence_id": sequence_id,
                                    "grouped_scene_id": frame_metadata["grouped_scene_id"],
                                    "subsequence_id": frame_metadata["subsequence_id"],
                                    "image_path": path,
                                    "split": args.split,
                                    "class_name": "all",
                                    "attack": attack_name,
                                    "adaptive": adaptive,
                                    "epsilon": epsilon_px / 255.0,
                                    "epsilon_px": epsilon_px,
                                    "steps": steps,
                                    "step_size": (
                                        epsilon_px / 255.0
                                        if attack_name == "fgsm"
                                        else (epsilon_px / 255.0) / 4.0
                                    ),
                                    "random_start": attack_name == "pgd",
                                    "restarts": len(seeds),
                                    "restart": restart,
                                    "seed": seed,
                                    "selected_best": restart == best_restart,
                                    "defense": defense_name,
                                    "layer": level,
                                    "normalization": (
                                        args.normalizations[0]
                                        if len(args.normalizations) == 1 else "wide_multi_mode"
                                    ),
                                    "threshold": args.confidence,
                                    "perturbation_norm": "linf",
                                    "f1_clean": clean_detection["f1"],
                                    "f1_attack": attacked_detection["f1"],
                                    "f1_defended": defended_detection["f1"],
                                    "recall_clean": clean_detection["recall"],
                                    "recall_attack": attacked_detection["recall"],
                                    "recall_defended": defended_detection["recall"],
                                    "precision_clean": clean_detection["precision"],
                                    "precision_attack": attacked_detection["precision"],
                                    "precision_defended": defended_detection["precision"],
                                    "precision": defended_detection["precision"],
                                    "recall": defended_detection["recall"],
                                    "f1": defended_detection["f1"],
                                    "f2": (
                                        5.0 * defended_detection["precision"]
                                        * defended_detection["recall"]
                                        / max(
                                            4.0 * defended_detection["precision"]
                                            + defended_detection["recall"], 1e-12,
                                        )
                                    ),
                                    "fn": defended_detection["fn"],
                                    "map50": math.nan,
                                    "map50_95": math.nan,
                                    "mAP50": math.nan,
                                    "mAP50-95": math.nan,
                                    "false_negatives": defended_detection["fn"],
                                    "fn_clean": clean_detection["fn"],
                                    "fn_attack": attacked_detection["fn"],
                                    "fn_defended": defended_detection["fn"],
                                    "nms_timeout": bool(
                                        clean_detection["nms_timeout"]
                                        or attacked_detection["nms_timeout"]
                                        or defended_detection["nms_timeout"]
                                    ),
                                    "nms_timeout_clean": clean_detection["nms_timeout"],
                                    "nms_timeout_attack": attacked_detection["nms_timeout"],
                                    "nms_timeout_defended": defended_detection["nms_timeout"],
                                    "nms_runtime_ms": max(
                                        float(clean_detection["nms_runtime_ms"]),
                                        float(attacked_detection["nms_runtime_ms"]),
                                        float(defended_detection["nms_runtime_ms"]),
                                    ),
                                    "runtime_ms": max(
                                        float(clean_detection["nms_runtime_ms"]),
                                        float(attacked_detection["nms_runtime_ms"]),
                                        float(defended_detection["nms_runtime_ms"]),
                                    ),
                                    "nms_runtime_clean_ms": clean_detection["nms_runtime_ms"],
                                    "nms_runtime_attack_ms": attacked_detection["nms_runtime_ms"],
                                    "nms_runtime_defended_ms": defended_detection["nms_runtime_ms"],
                                    "predictions_before_or_after_timeout": "after_nms",
                                    "predictions_after_nms": defended_detection[
                                        "predictions_after_nms"
                                    ],
                                    "nms_candidates_before": defended_detection[
                                        "nms_candidates_before"
                                    ],
                                    "nms_output_complete": bool(
                                        clean_detection["nms_output_complete"]
                                        and attacked_detection["nms_output_complete"]
                                        and defended_detection["nms_output_complete"]
                                    ),
                                    "confidence_drop": (
                                        clean_detection["mean_confidence"]
                                        - attacked_detection["mean_confidence"]
                                    ),
                                    "iou_shift": (
                                        clean_detection["mean_matched_iou"]
                                        - attacked_detection["mean_matched_iou"]
                                    ),
                                    "attack_loss": result.attack_loss,
                                    "attack_loss_clean": result.clean_attack_loss,
                                    "attack_loss_final": result.attack_loss,
                                    "lambda_box": loss_weights["box"],
                                    "lambda_cls": loss_weights["cls"],
                                    "lambda_dfl": loss_weights["dfl"],
                                    "attack_objective": "lambda_box*L_box+lambda_cls*L_cls+lambda_dfl*L_dfl",
                                    "c_sp_global": path_attack_values["path_gradient_c_sp_global"],
                                    "c_sp_object": path_attack_values["path_gradient_c_sp_object"],
                                    "c_sp_background": path_attack_values["path_gradient_c_sp_background"],
                                    "c_dir": path_attack_values["path_gradient_c_dir_global"],
                                    "c_dir_object": path_attack_values[
                                        "path_gradient_c_dir_object"
                                    ],
                                    "c_dir_background": path_attack_values[
                                        "path_gradient_c_dir_background"
                                    ],
                                    "c_atk_global": path_attack_values[
                                        "path_gradient_c_atk_product_global"
                                    ],
                                    "c_atk_object": path_attack_values[
                                        "path_gradient_c_atk_product_object"
                                    ],
                                    "c_atk_background": path_attack_values[
                                        "path_gradient_c_atk_product_background"
                                    ],
                                    "c_atk_clean_gradient": clean_attack_values[
                                        "clean_gradient_c_atk_product_global"
                                    ],
                                    "c_atk_path_gradient": path_attack_values[
                                        "path_gradient_c_atk_product_global"
                                    ],
                                    "latency_ms": math.nan,
                                    "damage": clean_detection["f1"] - attacked_detection["f1"],
                                    "recovery": defended_detection["f1"] - attacked_detection["f1"],
                                    "delta_recall_damage": (
                                        clean_detection["recall"]
                                        - attacked_detection["recall"]
                                    ),
                                    "delta_recall_recovery": (
                                        defended_detection["recall"]
                                        - attacked_detection["recall"]
                                    ),
                                    "false_negatives_increase": (
                                        attacked_detection["fn"] - clean_detection["fn"]
                                    ),
                                    "false_negatives_reduction": (
                                        attacked_detection["fn"] - defended_detection["fn"]
                                    ),
                                    "normalized_quality_recovery": normalized_quality_recovery(
                                        float(clean_detection["f1"]),
                                        float(attacked_detection["f1"]),
                                        float(defended_detection["f1"]),
                                    ),
                                    "perturbation_l1": float(perturbation.abs().sum()),
                                    "perturbation_l2": float(perturbation.norm()),
                                    "perturbation_linf": float(perturbation.abs().max()),
                                    "actual_l1": float(perturbation.abs().sum()),
                                    "actual_l2": float(perturbation.norm()),
                                    "actual_linf": float(perturbation.abs().max()),
                                    "actual_L1": float(perturbation.abs().sum()),
                                    "actual_L2": float(perturbation.norm()),
                                    "actual_Linf": float(perturbation.abs().max()),
                                    "legacy_metric_family": "legacy_compatibility",
                                    "canonical_metric_family": (
                                        "canonical_tnorm"
                                        if revision_statistics is not None else "not_computed"
                                    ),
                                    "gradient_l1": float(result.path_gradient.abs().sum()),
                                    "gradient_l2": float(result.path_gradient.norm()),
                                    "gradient_linf": float(result.path_gradient.abs().max()),
                                }
                                row.update(feature_values(
                                    attacked_metric, defended_metric, preservation_metric
                                ))
                                if revision_statistics is not None:
                                    for mode in args.normalizations:
                                        defended_revision = (
                                            revision_attacked_cache[(level_index, mode)]
                                            if defense_name == "none"
                                            else revision_pair_metrics(
                                                clean_features[level_index],
                                                defended_features[level_index],
                                                revision_statistics[level],
                                                mode,
                                            )[0]
                                        )
                                        row.update(revision_feature_values_from_metrics(
                                            revision_attacked_cache[(level_index, mode)],
                                            defended_revision,
                                            revision_preservation_cache[(
                                                defense_name, level_index, mode
                                            )],
                                            mode,
                                        ))
                                        if len(args.normalizations) == 1 and mode == args.normalizations[0]:
                                            row.update({
                                                "canonical_Product": row[f"{mode}_product"],
                                                "canonical_Lukasiewicz": row[f"{mode}_lukasiewicz"],
                                                "canonical_Product_recovery": row[f"{mode}_product_recovery"],
                                                "canonical_Lukasiewicz_recovery": row[f"{mode}_lukasiewicz_recovery"],
                                                "P": row[f"{mode}_p_product"],
                                                "A": row[f"{mode}_a_product"],
                                                "R": row[f"{mode}_r_product"],
                                                "G_raw": row[f"{mode}_g_product"],
                                                "G_clipped": row[f"{mode}_g_product_clipped"],
                                                "C_def": row[f"{mode}_c_def_product"],
                                            })
                                row.update(clean_attack_values)
                                row.update(path_attack_values)
                                condition_rows.append(row)
                    expected = expected_rows_per_condition(
                        args, attack_name, adaptive
                    )
                    if len(condition_rows) != expected:
                        raise RuntimeError(
                            f"Incomplete condition checkpoint for {path}/"
                            f"{attack_name}/{epsilon_px}/{steps}/{adaptive}: "
                            f"{len(condition_rows)} != {expected}"
                        )
                    if writer is None:
                        fieldnames = fieldnames or list(condition_rows[0])
                        writer = csv.DictWriter(
                            checkpoint, fieldnames=fieldnames, extrasaction="ignore"
                        )
                        if checkpoint.tell() == 0:
                            writer.writeheader()
                    writer.writerows(condition_rows)
                    checkpoint.flush()
                    total_rows += len(condition_rows)
                expected = expected_rows_per_image(args)
                image_total = sum(
                    expected_rows_per_condition(args, attack, adaptive)
                    for attack, _epsilon, _steps, adaptive in conditions(args)
                )
                if image_total != expected:
                    raise RuntimeError(
                        f"Internal expected-row mismatch: {image_total} != {expected}"
                    )
    finally:
        hook.close()

    if not partial_path.is_file() or partial_path.stat().st_size == 0 or total_rows == 0:
        raise RuntimeError("Final matrix produced no rows")
    partial_path.replace(args.output)
    config = {
        "model": str(args.model.resolve()), "data": str(args.data), "manifest": str(args.manifest),
        "checkpoint_sha256": sha256(args.model),
        "split": args.split, "conditions": conditions(args), "seeds": args.seeds,
        "checkpoint_name": args.checkpoint_name,
        "revision_stats": (
            str(args.revision_stats.resolve()) if args.revision_stats is not None else None
        ),
        "normalizations": args.normalizations,
        "adaptive_pgd_eps": args.adaptive_pgd_eps,
        "adaptive_pgd_steps": args.adaptive_pgd_steps,
        "defenses": args.defenses, "confidence": args.confidence,
        "nms_max_time_img": args.nms_max_time_img,
        "statistical_unit": "sequence_id", "rows": total_rows,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
