from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path("outputs/final_practice/unified_diagnostics_raw.csv")
STATS = Path("outputs/final_practice/09_statistics")
OUTPUT = Path("outputs/final_practice/figures")


def save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the eight required final-practice figures")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--statistics", type=Path, default=STATS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    data = data[data.get("selected_best", True).astype(bool)].copy()
    args.output.mkdir(parents=True, exist_ok=True)
    base = data[(data["defense"] == "none") & (~data["adaptive"].astype(bool))]

    curves = base.groupby(["attack", "epsilon_px"], as_index=False)[["recall_attack", "f1_attack"]].mean()
    for metric in ("recall_attack", "f1_attack"):
        for attack, group in curves.groupby("attack"):
            plt.plot(group["epsilon_px"], group[metric], marker="o", label=f"{attack} {metric}")
    plt.xlabel("epsilon (pixel levels / 255)"); plt.ylabel("score"); plt.legend(fontsize=8)
    save(args.output / "01_recall_f1_vs_epsilon.png")

    scatter = base.dropna(subset=["c_atk_object", "damage"])
    plt.scatter(scatter["c_atk_object"], scatter["damage"], s=7, alpha=.25)
    plt.xlabel("C_atk object"); plt.ylabel("F1 damage")
    save(args.output / "02_f1_damage_vs_c_atk_object.png")

    restored = data[data["defense"] != "none"].dropna(subset=["g_recovery", "recovery"])
    for defense, group in restored.groupby("defense"):
        plt.scatter(group["g_recovery"], group["recovery"], s=7, alpha=.2, label=defense)
    plt.xlabel("G feature recovery"); plt.ylabel("F1 recovery"); plt.legend(fontsize=7)
    save(args.output / "03_f1_recovery_vs_g.png")

    adaptive = data[(data["attack"] == "pgd") & (data["defense"] == "tnorm")]
    adaptive_curve = adaptive.groupby(["adaptive", "epsilon_px"], as_index=False)["f1_defended"].mean()
    for flag, group in adaptive_curve.groupby("adaptive"):
        plt.plot(group["epsilon_px"], group["f1_defended"], marker="o", label="adaptive" if flag else "non-adaptive")
    plt.xlabel("epsilon (pixel levels / 255)"); plt.ylabel("defended F1"); plt.legend()
    save(args.output / "04_adaptive_vs_nonadaptive_pgd.png")

    tnorms = base.groupby("epsilon_px", as_index=False)[["product", "godel", "lukasiewicz"]].mean()
    for metric in ("product", "godel", "lukasiewicz"):
        plt.plot(tnorms["epsilon_px"], tnorms[metric], marker="o", label=metric)
    plt.xlabel("epsilon (pixel levels / 255)"); plt.ylabel("similarity"); plt.legend()
    save(args.output / "05_tnorms_vs_budget.png")

    bootstrap_path = args.statistics / "sequence_bootstrap.csv"
    bootstrap = pd.read_csv(bootstrap_path)
    gains = bootstrap[(bootstrap["metric"] == "r2") & (bootstrap["task"] == "damage")]
    x = np.arange(len(gains)); values = gains["observed_gain"].to_numpy()
    errors = np.vstack((values - gains["ci_low"], gains["ci_high"] - values))
    plt.errorbar(x, values, yerr=errors, fmt="o", capsize=4); plt.axhline(0, color="black", lw=.8)
    plt.xticks(x, gains["comparison"], rotation=25, ha="right"); plt.ylabel("Delta R2 (95% CI)")
    save(args.output / "06_model_increment_ci.png")

    consistency = base[["c_sp_object", "c_sp_background"]].mean()
    plt.bar(["object", "background"], consistency.to_numpy()); plt.ylabel("spatial consistency")
    save(args.output / "07_object_vs_background.png")

    layer = restored.groupby("layer", as_index=False)["g_recovery"].mean()
    plt.bar(layer["layer"], layer["g_recovery"]); plt.ylabel("mean G recovery")
    save(args.output / "08_layerwise_recovery.png")


if __name__ == "__main__":
    main()
