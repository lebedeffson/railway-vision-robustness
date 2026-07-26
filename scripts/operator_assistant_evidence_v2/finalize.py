from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.operator_assistant_evidence_v2.common import (
    OUTPUT,
    PROJECT,
    assert_test_sealed,
    atomic_json,
    atomic_text,
    sha256,
)


FIGURES = OUTPUT / "figures"
PUBLIC_ZIP = OUTPUT / "operator_assistant_evidence_v2_public.zip"
LOCAL_PATH_PATTERN = re.compile(
    r"(?:/home/|/mnt/|/opt/|/root/|[A-Za-z]:\\\\)"
)


def _save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    figure.savefig(
        FIGURES / f"{stem}.png",
        bbox_inches="tight",
        dpi=320,
        metadata={"Software": "operator-assistant-evidence-v2"},
    )
    plt.close(figure)


def _plot_figures() -> None:
    sensitivity = pd.read_csv(OUTPUT / "EVENT_SENSITIVITY_FULL.csv")
    baseline = sensitivity[sensitivity["baseline_configuration"].astype(bool)].iloc[0]

    grouped = (
        sensitivity.groupby("join_time_seconds", as_index=False)
        .agg(
            unique_events=("unique_events", "median"),
            events_min=("unique_events", "min"),
            events_max=("unique_events", "max"),
        )
        .sort_values("join_time_seconds")
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(
        grouped["join_time_seconds"],
        grouped["unique_events"],
        marker="o",
        color="#1f4e79",
        label="median over spatial/reopen settings",
    )
    ax.fill_between(
        grouped["join_time_seconds"],
        grouped["events_min"],
        grouped["events_max"],
        color="#1f4e79",
        alpha=0.16,
        label="min–max",
    )
    ax.scatter(
        [baseline["join_time_seconds"]],
        [baseline["unique_events"]],
        marker="s",
        color="#a61c00",
        label="locked baseline",
        zorder=4,
    )
    ax.set(xlabel="Join time, s", ylabel="Unique events")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, "event_count_sensitivity")

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    scatter = ax.scatter(
        sensitivity["fragmentation"],
        sensitivity["false_merge_rate"],
        c=sensitivity["join_time_seconds"],
        cmap="viridis",
        s=38,
        alpha=0.8,
    )
    ax.scatter(
        [baseline["fragmentation"]],
        [baseline["false_merge_rate"]],
        marker="s",
        s=70,
        color="#a61c00",
        label="locked baseline",
        zorder=4,
    )
    ax.set(xlabel="Fragmentation", ylabel="False merge rate")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, label="Join time, s")
    _save_figure(fig, "fragmentation_false_merge")

    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    scatter = ax.scatter(
        sensitivity["false_events_per_minute"],
        sensitivity["true_episode_recall"],
        c=sensitivity["unique_events"],
        cmap="plasma",
        s=38,
        alpha=0.8,
    )
    ax.scatter(
        [baseline["false_events_per_minute"]],
        [baseline["true_episode_recall"]],
        marker="s",
        s=70,
        color="#146c2e",
        label="locked baseline",
        zorder=4,
    )
    ax.set(xlabel="False events per minute", ylabel="True episode recall")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.colorbar(scatter, ax=ax, label="Unique events")
    _save_figure(fig, "event_recall_false_events")

    per_frame = pd.read_csv(OUTPUT / "EVENT_RUNTIME_PER_FRAME.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.hist(
        per_frame["full_pipeline_ms"],
        bins=36,
        color="#1f4e79",
        edgecolor="white",
        linewidth=0.4,
    )
    p50 = float(per_frame["full_pipeline_ms"].quantile(0.50))
    p95 = float(per_frame["full_pipeline_ms"].quantile(0.95))
    ax.axvline(p50, color="#146c2e", linestyle="--", label=f"P50 {p50:.2f} ms")
    ax.axvline(p95, color="#a61c00", linestyle="--", label=f"P95 {p95:.2f} ms")
    ax.set(xlabel="Full-pipeline latency, ms", ylabel="Measured frames")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save_figure(fig, "event_latency_distribution")


def _assert_finite_csvs() -> None:
    for path in OUTPUT.glob("*.csv"):
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include="number")
        if numeric.empty:
            continue
        values = numeric.to_numpy()
        if not all(math.isfinite(float(value)) for value in values.flat):
            raise RuntimeError(f"Non-finite value in {path.name}")


def _technical_summary() -> str:
    lock = json.loads((OUTPUT / "EVENT_PROTOCOL_LOCK.json").read_text())
    replay = json.loads((OUTPUT / "EVENT_BASELINE_REPLAY.json").read_text())
    runtime = json.loads((OUTPUT / "EVENT_RUNTIME_FINAL.json").read_text())
    failsafe = json.loads((OUTPUT / "FAILSAFE_AUDIT.json").read_text())
    stage_counts = dict(
        pd.read_csv(OUTPUT / "PIPELINE_STAGE_COUNTS.csv").itertuples(
            index=False, name=None
        )
    )
    sensitivity = pd.read_csv(OUTPUT / "EVENT_SENSITIVITY_FULL.csv")
    baseline = sensitivity[sensitivity["baseline_configuration"].astype(bool)].iloc[0]
    source = lock["benchmark"]
    return f"""# Operator assistant evidence v2 — technical summary

## Frozen inputs

- Computation base commit: `{lock["git_commit_before_computation"]}`
- Benchmark source: development scene `{source["grouped_scene_id"]}`, sequence `{source["source_sequence_id"]}`
- Benchmark frames: {source["frames"]}
- Test status: `SEALED`
- Test access count: 0
- New model training: none

## Reproducible event replay

- Raw detector boxes: {replay["direct_raw_detector_boxes"]}
- Aggregator observations: {stage_counts["aggregator_input"]}
- Direct unique events: {replay["direct_unique_events"]}
- Replay unique events: {replay["replay_unique_events"]}
- Exact event signature match: `{str(replay["exact_event_signature_match"]).lower()}`

The earlier demonstration ratio `296 -> 3` is not used as v2 evidence. The
reproducible v2 benchmark is `{replay["direct_raw_detector_boxes"]} -> {replay["direct_unique_events"]}`.

## Locked event configuration

- Join/close time: {baseline["join_time_seconds"]:.0f} s
- Minimum IoU: {baseline["minimum_iou"]:.2f}
- Maximum center distance: {baseline["maximum_center_distance_ratio"]:.2f} frame diagonal
- Reopen window: {baseline["reopen_window_seconds"]:.0f} s
- True episode recall: {baseline["true_episode_recall"]:.6f}
- Fragmentation: {baseline["fragmentation"]:.6f}
- False merge rate: {baseline["false_merge_rate"]:.6f}
- Episode frame coverage: {baseline["episode_frame_coverage"]:.6f}

All {len(sensitivity)} prospectively specified sensitivity configurations were
calculated. The locked baseline was not selected from this grid.

## Single-stream runtime

- Warm-up frames: {runtime["warmup_frames"]}
- Measured frames: {runtime["measured_frames"]}
- End-to-end FPS: {runtime["end_to_end_fps"]:.6f}
- Raw detections: {runtime["raw_detections"]}
- Tracker observations: {runtime["tracker_observations"]}
- Verifier accepted/rejected: {runtime["verifier_accepted"]}/{runtime["verifier_rejected"]}
- Aggregator observations: {runtime["aggregator_observations"]}
- Unique events: {runtime["unique_events"]}
- Multi-stream execution: `DEFERRED`

## Fail-safe audit

- Checks: {failsafe["checks"]}
- Passed: {failsafe["passed"]}
- Failed: {failsafe["failed"]}
- All pass: `{str(failsafe["all_pass"]).lower()}`

Verifier failure preserves the candidate in the general review queue with
`VERIFIER_UNAVAILABLE`. Recoverable processing failure preserves a technical
review event with `PROCESSING_FALLBACK`. Storage-integrity failure remains a
hard stop.

## Blocked or deferred calculations

- Two- and four-stream scaling: `DEFERRED_HARDWARE_CONSTRAINT`
- Railway closed test: `SEALED_NOT_ACCESSED`
- New training: `FORBIDDEN_NOT_RUN`
- Manual operator workload study: `OUT_OF_SCOPE`
"""


def _write_root_manifest() -> None:
    excluded = {"MANIFEST.sha256", PUBLIC_ZIP.name}
    paths = sorted(
        path
        for path in OUTPUT.rglob("*")
        if path.is_file() and path.name not in excluded
    )
    lines = [f"{sha256(path)}  {path.relative_to(OUTPUT).as_posix()}" for path in paths]
    atomic_text(OUTPUT / "MANIFEST.sha256", "\n".join(lines) + "\n")


def _public_files() -> dict[str, bytes]:
    replay = json.loads((OUTPUT / "EVENT_BASELINE_REPLAY.json").read_text())
    replay_public = {
        key: value
        for key, value in replay.items()
        if key not in {"direct_signature", "replay_signature"}
    }
    lock = json.loads((OUTPUT / "EVENT_PROTOCOL_LOCK.json").read_text())
    lock_public = {
        "protocol_id": lock["protocol_id"],
        "git_commit_before_computation": lock["git_commit_before_computation"],
        "test_status": lock["test_status"],
        "test_access_count": lock["test_access_count"],
        "training": lock["training"],
        "multistream_benchmark": lock["multistream_benchmark"],
        "benchmark": {
            "frames": lock["benchmark"]["frames"],
            "fps": lock["benchmark"]["fps"],
            "width": lock["benchmark"]["width"],
            "height": lock["benchmark"]["height"],
        },
        "baseline_event_parameters": lock["baseline_event_parameters"],
        "sensitivity_grid": lock["sensitivity_grid"],
        "expected_configurations": 108,
    }
    payloads: dict[str, bytes] = {
        "EVENT_PROTOCOL_LOCK_PUBLIC.json": (
            json.dumps(lock_public, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "EVENT_BASELINE_REPLAY_PUBLIC.json": (
            json.dumps(replay_public, indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    names = [
        "PIPELINE_STAGE_COUNTS.csv",
        "EVENT_SENSITIVITY_FULL.csv",
        "EVENT_RUNTIME_SUMMARY.csv",
        "EVENT_RUNTIME_FINAL.json",
        "FAILSAFE_CHECKS.csv",
        "FAILSAFE_AUDIT.json",
        "FINAL_TECHNICAL_SUMMARY.md",
        "TEST_RESULTS.txt",
    ]
    for name in names:
        payloads[name] = (OUTPUT / name).read_bytes()
    for path in sorted(FIGURES.glob("*")):
        payloads[f"figures/{path.name}"] = path.read_bytes()
    protocol = (
        PROJECT
        / "protocol/operator_assistant_evidence_v2/GT_EPISODE_ANNOTATION_PROTOCOL.md"
    )
    payloads["GT_EPISODE_ANNOTATION_PROTOCOL.md"] = protocol.read_bytes()
    return payloads


def _write_public_zip() -> None:
    payloads = _public_files()
    for name, data in payloads.items():
        if Path(name).suffix.lower() in {".json", ".csv", ".md", ".txt"}:
            text = data.decode("utf-8")
            if LOCAL_PATH_PATTERN.search(text):
                raise RuntimeError(f"Local absolute path in public file {name}")
            if "TEST_OPENED" in text:
                raise RuntimeError(f"Test marker string in public file {name}")
    manifest = "\n".join(
        f"{__import__('hashlib').sha256(data).hexdigest()}  {name}"
        for name, data in sorted(payloads.items())
    ) + "\n"
    payloads["MANIFEST.sha256"] = manifest.encode()
    with zipfile.ZipFile(PUBLIC_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            archive.writestr(name, data)


def main() -> None:
    assert_test_sealed()
    _assert_finite_csvs()
    _plot_figures()
    atomic_text(OUTPUT / "FINAL_TECHNICAL_SUMMARY.md", _technical_summary())
    if not (OUTPUT / "TEST_RESULTS.txt").exists():
        atomic_text(
            OUTPUT / "TEST_RESULTS.txt",
            "PENDING_FINAL_TEST_RUN\n",
        )
    _write_root_manifest()
    _write_public_zip()
    atomic_json(
        OUTPUT / "FINALIZATION_AUDIT.json",
        {
            "figures_png": len(list(FIGURES.glob("*.png"))),
            "figures_svg": len(list(FIGURES.glob("*.svg"))),
            "public_zip": PUBLIC_ZIP.name,
            "public_zip_sha256": sha256(PUBLIC_ZIP),
            "sensitivity_configurations": 108,
            "test_access_count": 0,
            "test_status": "SEALED",
        },
    )
    _write_root_manifest()
    _write_public_zip()
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "public_zip": str(PUBLIC_ZIP),
                "public_zip_sha256": sha256(PUBLIC_ZIP),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
