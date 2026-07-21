from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT = Path("outputs/final_practice/unified_diagnostics_raw.csv")
STATS = Path("outputs/final_practice/09_statistics")
OUTPUT = Path("outputs/final_practice/figures")
CLEAN_UTILITY = Path("outputs/final_practice/03_clean_utility/clean_utility_summary.csv")
LATENCY = Path("outputs/final_practice/08_latency/latency.csv")


def save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the nine required final-practice figures")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--statistics", type=Path, default=STATS)
    parser.add_argument("--clean-utility", type=Path, default=CLEAN_UTILITY)
    parser.add_argument("--latency", type=Path, default=LATENCY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    selected = data["selected_best"].astype(str).str.lower().isin(("true", "1"))
    data = data[selected].copy()
    adaptive_flag = data["adaptive"].astype(str).str.lower().isin(("true", "1"))
    data["adaptive"] = adaptive_flag
    args.output.mkdir(parents=True, exist_ok=True)
    base = data[(data["defense"] == "none") & (~data["adaptive"])]

    curves = base.groupby(["attack", "epsilon_px"], as_index=False)[["recall_attack", "f1_attack"]].mean()
    for metric in ("recall_attack", "f1_attack"):
        for attack, group in curves.groupby("attack"):
            plt.plot(group["epsilon_px"], group[metric], marker="o", label=f"{attack} {metric.replace('_attack', '')}")
    plt.xlabel("epsilon (pixel levels / 255)")
    plt.ylabel("score")
    plt.legend(fontsize=8)
    save(args.output / "01_f1_recall_vs_epsilon.png")

    scatter = base.dropna(subset=["c_atk_object", "damage"])
    plt.scatter(scatter["c_atk_object"], scatter["damage"], s=7, alpha=.25)
    plt.xlabel("C_atk object")
    plt.ylabel("F1 drop")
    save(args.output / "02_attack_consistency_vs_f1_drop.png")

    consistency = base[["c_sp_object", "c_sp_background"]].mean()
    plt.bar(["object", "background"], consistency.to_numpy())
    plt.ylabel("spatial consistency")
    save(args.output / "03_object_vs_background_consistency.png")

    restored = data[(data["defense"] != "none") & (~data["adaptive"])].dropna(
        subset=["g_recovery", "recovery"]
    )
    for defense, group in restored.groupby("defense"):
        plt.scatter(group["g_recovery"], group["recovery"], s=7, alpha=.2, label=defense)
    plt.xlabel("G feature recovery")
    plt.ylabel("F1 recovery")
    plt.legend(fontsize=7)
    save(args.output / "04_feature_recovery_vs_f1_recovery.png")

    heatmap = restored.pivot_table(index="defense", columns="layer", values="g_recovery", aggfunc="mean")
    heatmap = heatmap.reindex(columns=[name for name in ("P3", "P4", "P5") if name in heatmap])
    image = plt.imshow(heatmap.to_numpy(), aspect="auto", cmap="viridis")
    plt.xticks(np.arange(len(heatmap.columns)), heatmap.columns)
    plt.yticks(np.arange(len(heatmap.index)), heatmap.index)
    plt.colorbar(image, label="mean G recovery")
    save(args.output / "05_layerwise_recovery_heatmap.png")

    tnorms = base.groupby("epsilon_px", as_index=False)[["product", "godel", "lukasiewicz"]].mean()
    for metric in ("product", "godel", "lukasiewicz"):
        plt.plot(tnorms["epsilon_px"], tnorms[metric], marker="o", label=metric)
    plt.xlabel("epsilon (pixel levels / 255)")
    plt.ylabel("similarity")
    plt.legend()
    save(args.output / "06_product_godel_lukasiewicz_by_budget.png")

    bootstrap = pd.read_csv(args.statistics / "sequence_bootstrap.csv")
    gains = bootstrap[(bootstrap["metric"] == "r2") & (bootstrap["task"] == "damage")]
    x = np.arange(len(gains))
    values = gains["observed_gain"].to_numpy()
    errors = np.vstack((values - gains["ci_low"], gains["ci_high"] - values))
    plt.errorbar(x, values, yerr=errors, fmt="o", capsize=4)
    plt.axhline(0, color="black", lw=.8)
    plt.xticks(x, gains["comparison"], rotation=25, ha="right")
    plt.ylabel("Delta R2 (95% sequence-bootstrap CI)")
    save(args.output / "07_model_gain_with_ci.png")

    adaptive = data[(data["attack"] == "pgd") & (data["defense"] == "tnorm")]
    adaptive_curve = adaptive.groupby(["adaptive", "steps", "epsilon_px"], as_index=False)["f1_defended"].mean()
    for (flag, steps), group in adaptive_curve.groupby(["adaptive", "steps"]):
        label = ("adaptive" if flag else "non-adaptive") + f" PGD-{steps}"
        plt.plot(group["epsilon_px"], group["f1_defended"], marker="o", label=label)
    plt.xlabel("epsilon (pixel levels / 255)")
    plt.ylabel("Product-filtered F1")
    plt.legend(fontsize=8)
    save(args.output / "08_adaptive_vs_nonadaptive_pgd.png")

    utility = pd.read_csv(args.clean_utility)
    latency = pd.read_csv(args.latency)
    latency = latency.assign(
        defense=latency["method"].str.replace("+detector", "", regex=False).replace({"detector": "none"})
    )
    tradeoff = utility.merge(latency[["defense", "mean_latency_ms"]], on="defense", how="inner")
    for _, row in tradeoff.iterrows():
        plt.scatter(row["mean_latency_ms"], row["f1"], s=35)
        plt.annotate(row["defense"], (row["mean_latency_ms"], row["f1"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.xlabel("mean latency, ms")
    plt.ylabel("clean F1")
    save(args.output / "09_clean_utility_vs_latency.png")
    # Compatibility with the already-running pre-upgrade orchestrator. The
    # canonical final figure is 05_layerwise_recovery_heatmap.png.
    shutil.copy2(
        args.output / "05_layerwise_recovery_heatmap.png",
        args.output / "08_layerwise_recovery.png",
    )


if __name__ == "__main__":
    main()
