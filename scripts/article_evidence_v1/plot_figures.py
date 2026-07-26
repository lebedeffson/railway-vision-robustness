from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.article_evidence_v1.common import OUTPUT


FIGURES = OUTPUT / "figures"
METHOD_LABELS = {
    "frame_detector": "Frame detector",
    "bytetrack": "ByteTrack",
    "ocsort": "OC-SORT",
    "track_verifier": "Track verifier",
    "combined_visual_track_verifier": "Combined verifier",
}
COLORS = {
    "frame_detector": "#000000",
    "bytetrack": "#0072B2",
    "ocsort": "#009E73",
    "track_verifier": "#D55E00",
    "combined_visual_track_verifier": "#CC79A7",
}


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def threshold_tradeoff() -> None:
    data = pd.read_csv(
        OUTPUT / "threshold_baseline/THRESHOLD_METRICS.csv"
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method, group in data.groupby("method", sort=False):
        group = group.sort_values("false_alarms_per_min")
        ax.plot(
            group["false_alarms_per_min"],
            group["recall"],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=METHOD_LABELS[method],
            color=COLORS[method],
        )
        for row in group.itertuples(index=False):
            ax.annotate(
                f"{row.threshold:g}",
                (row.false_alarms_per_min, row.recall),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
            )
    ax.set_xlabel("False alarms / min")
    ax.set_ylabel("Recall")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "FIG_01_THRESHOLD_RECALL_FALSE_ALARMS")


def verifier_pr() -> None:
    data = pd.read_csv(
        OUTPUT / "track_verifier/TRACK_VERIFIER_201_THRESHOLDS.csv"
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ordered = data.sort_values("track_recall")
    ax.plot(
        ordered["track_recall"],
        ordered["track_precision"],
        color="#0072B2",
        linewidth=1.6,
    )
    points = [
        ("is_selected_threshold", "Selected", "o"),
        ("is_max_track_f1", "Max track F1", "s"),
        ("is_closest_to_full_gate", "Closest to full gate", "^"),
    ]
    for column, label, marker in points:
        selected = data[data[column].astype(bool)]
        ax.scatter(
            selected["track_recall"],
            selected["track_precision"],
            marker=marker,
            s=42,
            label=label,
        )
    ax.set_xlabel("Track recall")
    ax.set_ylabel("Track precision")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "FIG_02_TRACK_PRECISION_RECALL")


def verifier_roc() -> None:
    data = pd.read_csv(OUTPUT / "track_verifier/TRACK_VERIFIER_ROC.csv")
    auroc = float(data["auroc"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.plot(
        data["false_positive_rate"],
        data["true_positive_rate"],
        color="#0072B2",
        linewidth=1.6,
        label=f"AUROC = {auroc:.4f}",
    )
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(frameon=False)
    _save(fig, "FIG_03_TRACK_ROC")


def system_threshold_metrics() -> None:
    data = pd.read_csv(
        OUTPUT / "track_verifier/TRACK_VERIFIER_201_THRESHOLDS.csv"
    )
    panels = [
        ("recall_gain", "Recall gain", 0.10),
        ("fn_reduction_percent", "FN reduction (%)", 15.0),
        ("false_alarm_change_percent", "False-alarm change (%)", 20.0),
        ("f1_change", "F1 change", -0.03),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), sharex=True)
    for ax, (column, label, gate) in zip(axes.flat, panels, strict=True):
        ax.plot(data["threshold"], data[column], color="#0072B2", linewidth=1.2)
        ax.axhline(gate, color="#D55E00", linestyle="--", linewidth=0.9)
        ax.set_ylabel(label)
        ax.grid(True, linewidth=0.4, alpha=0.5)
    axes[1, 0].set_xlabel("Verifier threshold")
    axes[1, 1].set_xlabel("Verifier threshold")
    fig.tight_layout()
    _save(fig, "FIG_04_TRACK_THRESHOLD_SYSTEM_METRICS")


def gate_space() -> None:
    data = pd.read_csv(
        OUTPUT / "track_verifier/TRACK_VERIFIER_201_THRESHOLDS.csv"
    )
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    passed = data["full_gate"].astype(bool)
    size = 18 + np.clip(data["fn_reduction_percent"], 0, 30) * 2
    ax.scatter(
        data.loc[~passed, "false_alarm_change_percent"],
        data.loc[~passed, "recall_gain"],
        s=size[~passed],
        facecolors="none",
        edgecolors="#666666",
        linewidths=0.7,
        label="FAIL",
    )
    if passed.any():
        ax.scatter(
            data.loc[passed, "false_alarm_change_percent"],
            data.loc[passed, "recall_gain"],
            s=size[passed],
            color="#009E73",
            label="PASS",
        )
    closest = data[data["is_closest_to_full_gate"].astype(bool)]
    ax.scatter(
        closest["false_alarm_change_percent"],
        closest["recall_gain"],
        marker="x",
        s=70,
        color="#D55E00",
        label="Closest to full gate",
    )
    ax.axvline(20.0, color="#D55E00", linestyle="--", linewidth=0.9)
    ax.axhline(0.10, color="#D55E00", linestyle="--", linewidth=0.9)
    ax.set_xlabel("False-alarm change (%)")
    ax.set_ylabel("Recall gain")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "FIG_05_GATE_SPACE")


def false_track_categories() -> None:
    data = pd.read_csv(
        OUTPUT / "false_tracks/FALSE_TRACK_CATEGORY_SUMMARY.csv"
    )
    data = data[data["count"] > 0].sort_values("count")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(data["category"], data["count"], color="#4C78A8")
    for index, row in enumerate(data.itertuples(index=False)):
        ax.text(row.count + 1, index, f"{row.count} ({row.share_percent:.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("False tracks")
    ax.set_ylabel("")
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.5)
    _save(fig, "FIG_06_FALSE_TRACK_CATEGORIES")


def main() -> None:
    threshold_tradeoff()
    verifier_pr()
    verifier_roc()
    system_threshold_metrics()
    gate_space()
    false_track_categories()
    print(f"created {len(list(FIGURES.glob('*.png')))} PNG and {len(list(FIGURES.glob('*.svg')))} SVG")


if __name__ == "__main__":
    main()
