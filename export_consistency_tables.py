from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INPUT = Path("outputs/final_practice/unified_diagnostics_raw.csv")
OUTPUT = Path("outputs/final_practice")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export attack- and defense-side raw tables")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    common = [
        "sequence_id", "image_path", "split", "attack", "adaptive", "epsilon",
        "epsilon_px", "steps", "restart", "seed", "selected_best",
    ]
    attack_columns = common + [
        column for column in data
        if column.startswith(("c_sp_", "clean_gradient_", "path_gradient_"))
    ]
    attack_columns += [
        column for column in (
            "c_dir", "c_dir_object", "c_dir_background", "c_atk_global",
            "c_atk_object", "c_atk_background", "c_atk_clean_gradient",
            "c_atk_path_gradient", "attack_loss", "step_size", "random_start",
            "restarts", "attack_objective",
            "lambda_box", "lambda_cls", "lambda_dfl",
        ) if column in data
    ]
    defense_columns = common + ["defense", "layer"] + [
        column for column in (
            "p_clean_preservation", "a_attacked_similarity", "r_restored_similarity",
            "g_recovery", "c_def", "p_godel", "a_godel", "r_godel",
            "g_godel", "c_def_godel", "p_lukasiewicz", "a_lukasiewicz",
            "r_lukasiewicz", "g_lukasiewicz", "c_def_lukasiewicz",
            "cosine_recovery", "mse_recovery", "mae_recovery",
            "relative_l2_recovery", "mean_shift_recovery", "entropy_recovery",
            "product_recovery", "godel_recovery", "lukasiewicz_recovery",
        ) if column in data
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    attack = data[attack_columns].drop_duplicates()
    defense = data[defense_columns]
    attack.to_csv(args.output / "attack_consistency_raw.csv", index=False)
    defense.to_csv(args.output / "defense_consistency_raw.csv", index=False)
    raw = args.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    attack.to_csv(raw / "attack_consistency.csv", index=False)
    defense.to_csv(raw / "defense_consistency.csv", index=False)


if __name__ == "__main__":
    main()
