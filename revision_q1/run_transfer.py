from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

from audit_final_practice import canonical_path
from evaluate_image_level_detection import DEFAULT_CONFIDENCE
from extract_attack_consistency import consistency_row, fgsm, object_masks, pgd, sequence_lookup
from extract_feature_consistency import FeatureHook, defend, loader, to_device
from revision_q1.feature_metrics import pair_metrics
from revision_q1.protocol import load_protocol, output_root
from revision_q1.statistics import cluster_mean_interval
from revision_q1.transfer import validate_transfer_pair
from run_final_matrix import detection_for_image


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    model = YOLO(str(path)).model.to(device).float().eval()
    overrides = model.args if isinstance(model.args, dict) else vars(model.args)
    model.args = get_cfg(overrides=overrides)
    model.criterion = None
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main() -> None:
    protocol = load_protocol()
    root = output_root(protocol)
    parser = argparse.ArgumentParser(description="Cross-checkpoint transfer, not cross-architecture")
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data/yolo_osdar23/data.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data/yolo_osdar23/manifest.csv")
    parser.add_argument("--stats", type=Path, default=root / "normalization/layer_channel_statistics.pt")
    parser.add_argument("--selection", type=Path, default=root / "config/normalization_selection.json")
    parser.add_argument("--output", type=Path, default=root)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    raw_path = args.output / "raw/cross_checkpoint_transfer.csv"
    if raw_path.is_file():
        print(f"Cross-checkpoint transfer already complete: {raw_path}")
        return
    checkpoints = {
        "stage2_best": PROJECT_DIR / protocol["primary_checkpoint"],
        "stage1_best": PROJECT_DIR / protocol["sensitivity_checkpoint"],
    }
    mode = json.loads(args.selection.read_text(encoding="utf-8"))["selected_normalization"]
    statistics = torch.load(args.stats, map_location="cpu")["statistics"]
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    exact, named = sequence_lookup(args.manifest)
    epsilon = float(protocol["transfer"]["epsilon_px"][0])
    rows: list[dict[str, object]] = []
    for source_name, target_name in protocol["transfer"]["directions"]:
        source_path, target_path = validate_transfer_pair(
            checkpoints[source_name], checkpoints[target_name]
        )
        source_model = load_model(source_path, device)
        target_model = load_model(target_path, device)
        hook = FeatureHook(target_model)
        try:
            for image_index, raw_batch in enumerate(tqdm(
                loader(args.data, "test", args.imgsz, 1, 0, device.type == "cuda"),
                desc=f"transfer {source_name}->{target_name}",
            )):
                if args.max_images is not None and image_index >= args.max_images:
                    break
                batch = to_device(raw_batch, device)
                path = str(batch["im_file"][0])
                sequence_id = exact.get(canonical_path(path), named.get(Path(path).name))
                clean = batch["img"].detach()
                clean_features = hook.extract(target_model, clean)
                clean_detection = detection_for_image(
                    target_model, batch, clean, DEFAULT_CONFIDENCE
                )
                mask = object_masks(batch, clean.shape[-2], clean.shape[-1])[0]
                for attack_name in protocol["transfer"]["attacks"]:
                    if attack_name == "fgsm":
                        with torch.enable_grad():
                            result = fgsm(source_model, batch, epsilon, False)
                        restart, seed = 0, int(protocol["attacks"]["seeds"][0])
                    else:
                        candidates = []
                        for candidate_seed in protocol["attacks"]["seeds"]:
                            with torch.enable_grad():
                                candidates.append(pgd(
                                    source_model, batch, epsilon, 20,
                                    int(candidate_seed), False,
                                ))
                        restart = max(
                            range(len(candidates)), key=lambda value: candidates[value].attack_loss
                        )
                        result = candidates[restart]
                        seed = int(protocol["attacks"]["seeds"][restart])
                    attack_values = consistency_row(
                        clean[0], result, mask, epsilon, result.path_gradient[0],
                        "path_gradient_",
                    )
                    for defense in ("none", "tnorm"):
                        evaluated = (
                            result.adversarial if defense == "none"
                            else defend(result.adversarial, defense)
                        )
                        features = hook.extract(target_model, evaluated)
                        detection = detection_for_image(
                            target_model, batch, evaluated, DEFAULT_CONFIDENCE
                        )
                        for layer_index, layer in enumerate(("P3", "P4", "P5")):
                            values = pair_metrics(
                                clean_features[layer_index], features[layer_index],
                                statistics[layer], mode,
                            )[0]
                            rows.append({
                                "sequence_id": sequence_id, "image_path": path,
                                "source_checkpoint": source_name,
                                "target_checkpoint": target_name,
                                "transfer_kind": "cross_checkpoint_transfer",
                                "attack": attack_name, "epsilon_px": epsilon,
                                "steps": 1 if attack_name == "fgsm" else 20,
                                "restart": restart, "seed": seed, "defense": defense,
                                "layer": layer, "normalization": mode,
                                "f1_clean_target": clean_detection["f1"],
                                "f1_transfer": detection["f1"],
                                "recall_clean_target": clean_detection["recall"],
                                "recall_transfer": detection["recall"],
                                "fn_clean_target": clean_detection["fn"],
                                "fn_transfer": detection["fn"],
                                "f1_drop": clean_detection["f1"] - detection["f1"],
                                "cosine": values["cosine_similarity"],
                                "product": values["product"], "godel": values["godel"],
                                "lukasiewicz": values["lukasiewicz"],
                                "c_atk_object": attack_values[
                                    "path_gradient_c_atk_product_object"
                                ],
                            })
        finally:
            hook.close()
            del source_model, target_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)
    per_image = raw.groupby([
        "sequence_id", "image_path", "source_checkpoint", "target_checkpoint",
        "attack", "defense",
    ], as_index=False).agg({
        "f1_clean_target": "first", "f1_transfer": "first", "f1_drop": "first",
        "recall_transfer": "first", "product": "mean", "godel": "mean",
        "lukasiewicz": "mean", "c_atk_object": "mean",
    })
    summary_rows = []
    for keys, group in per_image.groupby([
        "source_checkpoint", "target_checkpoint", "attack", "defense",
    ]):
        interval = cluster_mean_interval(
            group, "f1_drop", iterations=int(protocol["bootstrap_iterations"]),
            seed=int(protocol["random_seed"]),
        )
        summary_rows.append({
            "source_checkpoint": keys[0], "target_checkpoint": keys[1],
            "attack": keys[2], "defense": keys[3],
            "f1_clean_target": group["f1_clean_target"].mean(),
            "f1_transfer": group["f1_transfer"].mean(),
            "f1_drop": interval["estimate"],
            "f1_drop_ci_low": interval["ci_low"],
            "f1_drop_ci_high": interval["ci_high"],
            "recall_transfer": group["recall_transfer"].mean(),
            "product": group["product"].mean(), "godel": group["godel"].mean(),
            "lukasiewicz": group["lukasiewicz"].mean(),
            "c_atk_object": group["c_atk_object"].mean(),
            "frames": group["image_path"].nunique(),
            "sequences": interval["sequences"],
            "bootstrap_iterations": interval["bootstrap_iterations"],
        })
    summary = pd.DataFrame(summary_rows)
    args.output.joinpath("tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output / "tables/11_transfer_attack.csv", index=False)
    print(json.dumps({"rows": len(raw), "conditions": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
