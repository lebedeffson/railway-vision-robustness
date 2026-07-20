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
    attack_columns = common + [column for column in data if column.startswith("c_sp_")]
    attack_columns += [
        column for column in (
            "c_dir", "c_atk_global", "c_atk_object", "attack_loss",
            "lambda_box", "lambda_cls", "lambda_dfl",
        ) if column in data
    ]
    defense_columns = common + ["defense", "layer"] + [
        column for column in (
            "p_clean_preservation", "a_attacked_similarity", "r_restored_similarity",
            "g_recovery", "c_def", "p_godel", "g_godel", "c_def_godel",
            "p_lukasiewicz", "g_lukasiewicz", "c_def_lukasiewicz",
            "cosine_recovery", "mse_recovery", "mae_recovery",
            "relative_l2_recovery", "mean_shift_recovery", "entropy_recovery",
            "product_recovery", "godel_recovery", "lukasiewicz_recovery",
        ) if column in data
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    data[attack_columns].drop_duplicates().to_csv(
        args.output / "attack_consistency_raw.csv", index=False
    )
    data[defense_columns].to_csv(args.output / "defense_consistency_raw.csv", index=False)


if __name__ == "__main__":
    main()
