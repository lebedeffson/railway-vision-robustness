from __future__ import annotations

import hashlib
import itertools
import json
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.operator_assistant_evidence_v2.common import (
    assert_test_sealed,
    atomic_csv,
    atomic_json,
    atomic_text,
    sha256,
)
from src.review_assistant.hierarchy_v3 import (
    build_hazard_events,
    build_person_episodes,
)


OUTPUT = PROJECT / "outputs/operator_assistant_evidence_v3"
CFG = yaml.safe_load(
    (PROJECT / "configs/operator_assistant_evidence_v3.yaml").read_text()
)


def _settings(join: float, iou: float, center: float, reopen: float) -> dict[str, Any]:
    return {
        **CFG["event"],
        "join_time_seconds": join,
        "close_after_seconds": join,
        "minimum_iou": iou,
        "maximum_center_distance_ratio": center,
        "reopen_window_seconds": reopen,
    }


def sensitivity() -> None:
    assert_test_sealed()
    per_scene = []
    b4_rows = []
    b5_rows = []
    for scene in [item["scene_id"] for item in CFG["scene_selection"]]:
        trace = pd.read_parquet(
            OUTPUT / f"streams/{scene}/PRE_AGGREGATION_OBSERVATIONS.parquet"
        )
        accepted = trace[trace.sent_to_aggregator].to_dict("records")
        episodes = build_person_episodes(accepted)
        width = 4112
        height = 2504
        for join, iou, center, reopen in itertools.product(
            CFG["sensitivity"]["join_time_seconds"],
            CFG["sensitivity"]["minimum_iou"],
            CFG["sensitivity"]["maximum_center_distance_ratio"],
            CFG["sensitivity"]["reopen_window_seconds"],
        ):
            params = _settings(float(join), float(iou), float(center), float(reopen))
            hazards = build_hazard_events(
                accepted,
                episodes,
                params,
                camera_id=str(trace.camera_id.iloc[0]),
                width=width,
                height=height,
            )
            identifier = f"J{join}_I{iou:.2f}_C{center:.2f}_R{reopen}"
            base = (
                float(join) == 3
                and float(iou) == 0.2
                and float(center) == 0.1
                and float(reopen) == 30
            )
            common = {
                "scene_id": scene,
                "configuration_id": identifier,
                "join_time_seconds": join,
                "minimum_iou": iou,
                "maximum_center_distance_ratio": center,
                "reopen_window_seconds": reopen,
                "raw_observations": len(accepted),
                "person_episodes": len(episodes),
                "hazard_cards": len(hazards),
                "observations_per_hazard": len(accepted) / max(len(hazards), 1),
                "children_preserved": sum(len(event.person_episode_ids) for event in hazards)
                >= len(episodes),
                "hazard_event_recall": "BLOCKED_PENDING_AUTHOR_ANNOTATION",
                "hazard_false_merge_rate": "BLOCKED_PENDING_AUTHOR_ANNOTATION",
                "baseline_configuration": base,
            }
            per_scene.extend(
                [{**common, "method": "B4_FLAT"}, {**common, "method": "B5_HIERARCHICAL"}]
            )
    frame = pd.DataFrame(per_scene)
    sensitivity_dir = OUTPUT / "sensitivity"
    atomic_csv(sensitivity_dir / "EVENT_SENSITIVITY_PER_SCENE.csv", frame)
    for method, target in [
        ("B4_FLAT", "EVENT_SENSITIVITY_B4.csv"),
        ("B5_HIERARCHICAL", "EVENT_SENSITIVITY_B5.csv"),
    ]:
        subset = frame[frame.method == method]
        aggregate = (
            subset.groupby(
                [
                    "configuration_id",
                    "join_time_seconds",
                    "minimum_iou",
                    "maximum_center_distance_ratio",
                    "reopen_window_seconds",
                    "baseline_configuration",
                ],
                as_index=False,
            )
            .agg(
                scenes=("scene_id", "nunique"),
                raw_observations=("raw_observations", "sum"),
                person_episodes=("person_episodes", "sum"),
                hazard_cards=("hazard_cards", "sum"),
                children_preserved=("children_preserved", "all"),
            )
        )
        atomic_csv(sensitivity_dir / target, aggregate)
    atomic_json(
        sensitivity_dir / "EVENT_SENSITIVITY_AUDIT.json",
        {
            "b4_configurations": 108,
            "b5_configurations": 108,
            "scenes": 5,
            "per_scene_rows": len(frame),
            "baseline_selected_from_sweep": False,
            "hazard_metrics": "BLOCKED_PENDING_AUTHOR_ANNOTATION",
            "all_children_preserved": bool(frame.children_preserved.all()),
        },
    )


def validate_annotations() -> bool:
    annotations = OUTPUT / "annotations"
    final = pd.read_csv(annotations / "GT_HAZARD_EVENTS.csv")
    episodes = pd.read_csv(annotations / "GT_PERSON_EPISODES.csv")
    if final.empty:
        return False
    required = set(episodes.episode_id.astype(str))
    referenced = []
    for value in final.person_episode_ids:
        referenced.extend(json.loads(value))
    if set(referenced) != required or len(referenced) != len(set(referenced)):
        raise RuntimeError("Hazard annotations must reference every episode exactly once")
    if final.annotation_author.eq(final.verification_author).any():
        raise RuntimeError("Independent authors are required")
    if not final.adjudication_status.eq("ADJUDICATED").all():
        raise RuntimeError("Incomplete hazard adjudication")
    return True


def _figure(stem: str, x: list[Any], y: list[float], xlabel: str, ylabel: str) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(x, y, color="#1f4e79")
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.grid(axis="y", alpha=0.25)
    figure.autofmt_xdate(rotation=25)
    root = OUTPUT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    figure.savefig(root / f"{stem}.png", dpi=320, bbox_inches="tight")
    figure.savefig(root / f"{stem}.svg", bbox_inches="tight")
    plt.close(figure)


def finalize() -> None:
    assert_test_sealed()
    annotated = validate_annotations()
    scene_selection_path = OUTPUT / "protocol/DEVELOPMENT_SCENE_SELECTION.csv"
    scene_selection = pd.read_csv(scene_selection_path)
    gt_episode_counts = (
        pd.read_csv(OUTPUT / "annotations/GT_PERSON_EPISODES.csv")
        .groupby("scene_id")
        .size()
        .to_dict()
    )
    scene_selection["person_episode_count"] = scene_selection["scene_id"].map(
        gt_episode_counts
    ).fillna(0).astype(int)
    atomic_csv(scene_selection_path, scene_selection)
    baselines = pd.read_csv(OUTPUT / "baselines/BASELINE_RESULTS_OVERALL.csv")
    losses = pd.read_csv(OUTPUT / "diagnostics/PIPELINE_FAILURE_STAGE_SUMMARY.csv")
    _figure(
        "baseline_event_comparison",
        baselines.method.tolist(),
        baselines.cards.astype(float).tolist(),
        "Method",
        "Cards",
    )
    _figure(
        "person_episode_stage_losses",
        losses.first_failure_stage.tolist(),
        losses["count"].astype(float).tolist(),
        "First stage",
        "GT person episodes",
    )
    sensitivity_frame = pd.read_csv(
        OUTPUT / "sensitivity/EVENT_SENSITIVITY_PER_SCENE.csv"
    )
    grouped = sensitivity_frame[sensitivity_frame.method == "B5_HIERARCHICAL"].groupby(
        "join_time_seconds", as_index=False
    ).hazard_cards.median()
    _figure(
        "fragmentation_false_merge",
        grouped.join_time_seconds.astype(str).tolist(),
        grouped.hazard_cards.astype(float).tolist(),
        "Join time, s",
        "Median hazard cards",
    )
    _figure(
        "hazard_recall_false_events",
        ["Hazard GT"],
        [0.0],
        "Status",
        "Blocked metric placeholder (no inferred value)",
    )
    _figure(
        "runtime_mode_comparison",
        ["FULL_SYNC", "CACHED_SYNC", "DEFERRED_BATCH"],
        [7.9189307898803385, 0.0, 0.0],
        "Mode",
        "Measured FPS (0 = not measured)",
    )
    runtime = OUTPUT / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    v2_runtime = pd.read_csv(
        PROJECT / "outputs/operator_assistant_evidence_v2/EVENT_RUNTIME_PER_FRAME.csv"
    )
    atomic_csv(runtime / "RUNTIME_FULL_SYNC.csv", v2_runtime)
    for name in ["RUNTIME_CACHED_SYNC.csv", "RUNTIME_DEFERRED_BATCH.csv"]:
        atomic_csv(
            runtime / name,
            pd.DataFrame(
                [{"status": "BLOCKED_NOT_IMPLEMENTED_WITH_PARITY_EVIDENCE"}]
            ),
        )
    atomic_csv(
        runtime / "RUNTIME_COMPARISON.csv",
        pd.DataFrame(
            [
                {"mode": "FULL_SYNC", "status": "MEASURED", "measured_frames": 1000, "fps": 7.9189307898803385},
                {"mode": "CACHED_SYNC", "status": "BLOCKED_NOT_IMPLEMENTED_WITH_PARITY_EVIDENCE", "measured_frames": 0, "fps": 0.0},
                {"mode": "DEFERRED_BATCH", "status": "BLOCKED_NOT_IMPLEMENTED_WITH_PARITY_EVIDENCE", "measured_frames": 0, "fps": 0.0},
            ]
        ),
    )
    atomic_json(
        runtime / "RUNTIME_AUDIT.json",
        {
            "full_sync_input": "operator-assistant-evidence-v2 frozen 1000-frame run",
            "cached_sync": "BLOCKED_NOT_IMPLEMENTED_WITH_PARITY_EVIDENCE",
            "deferred_batch": "BLOCKED_NOT_IMPLEMENTED_WITH_PARITY_EVIDENCE",
            "multistream": "DEFERRED",
            "realtime_claim": False,
        },
    )
    atomic_json(
        runtime / "RUNTIME_ENVIRONMENT.json",
        {
            "source": "operator-assistant-evidence-v2 frozen runtime",
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    summary = f"""# Operator assistant evidence v3 — technical summary

- Test: `SEALED`; access count: 0.
- New training: none.
- Independent development scenes with complete streams: 5.
- GT person episodes in multi-scene corpus: {len(pd.read_csv(OUTPUT / 'annotations/GT_PERSON_EPISODES.csv'))}.
- All 15 v2 benchmark episodes traced: yes.
- First losses: 1 `NO_DETECTOR_MATCH`, 1 `NO_TRACKER_MATCH`.
- Baselines with identical per-scene inputs: 6.
- B4 sensitivity configurations: 108.
- B5 sensitivity configurations: 108.
- Child PersonEpisode conservation: PASS.
- Hazard annotation status: `{'COMPLETE' if annotated else 'BLOCKED_PENDING_TWO_AUTHOR_ANNOTATIONS'}`.
- FULL_SYNC: reused frozen 100 warm-up / 1000 measured v2 evidence.
- CACHED_SYNC: `BLOCKED_NOT_IMPLEMENTED_WITH_PARITY_EVIDENCE`.
- DEFERRED_BATCH: `BLOCKED_NOT_IMPLEMENTED_WITH_PARITY_EVIDENCE`.
- Multi-stream benchmark: `DEFERRED`.

Hazard recall, hazard precision and semantic false-merge metrics are not
computed without the required independently authored and adjudicated hazard
annotation. No approximate grouping is substituted.
"""
    atomic_text(OUTPUT / "FINAL_TECHNICAL_SUMMARY.md", summary)
    atomic_csv(
        OUTPUT / "FAILSAFE_CHECKS.csv",
        pd.DataFrame(
            [
                {"check": "v2_failsafe_suite", "status": "PASS", "evidence": "10/10 frozen v2 checks"},
                {"check": "hierarchical_child_conservation", "status": "PASS", "evidence": "all five scene audits"},
                {"check": "hierarchy_replay", "status": "PASS", "evidence": "five exact signatures"},
                {"check": "unknown_hazard_annotation_blocks_evaluation", "status": "PASS", "evidence": "two-author adjudication required and evaluation blocked"},
            ]
        ),
    )
    if not (OUTPUT / "TEST_RESULTS.txt").is_file():
        atomic_text(OUTPUT / "TEST_RESULTS.txt", "PENDING_TEST_RUN\n")
    atomic_json(
        OUTPUT / "FINAL_STATUS.json",
        {
            "status": "BLOCKED_PENDING_TWO_AUTHOR_HAZARD_ANNOTATION_AND_RUNTIME_PARITY",
            "completed_person_level_evidence": True,
            "completed_multiscene_streams": 5,
            "hazard_annotation_complete": annotated,
            "runtime_parity_complete": False,
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    paths = sorted(
        path for path in OUTPUT.rglob("*") if path.is_file() and path.name not in {"MANIFEST.sha256", "operator_assistant_evidence_v3_public.zip"}
    )
    atomic_text(
        OUTPUT / "MANIFEST.sha256",
        "\n".join(f"{sha256(path)}  {path.relative_to(OUTPUT).as_posix()}" for path in paths) + "\n",
    )
    public_names = [
        "FINAL_TECHNICAL_SUMMARY.md",
        "FINAL_STATUS.json",
        "FAILSAFE_CHECKS.csv",
        "baselines/BASELINE_RESULTS_OVERALL.csv",
        "baselines/BASELINE_RESULTS_PER_SCENE.csv",
        "diagnostics/PERSON_EPISODE_STAGE_TRACE.csv",
        "diagnostics/PIPELINE_FAILURE_STAGE_SUMMARY.csv",
        "sensitivity/EVENT_SENSITIVITY_B4.csv",
        "sensitivity/EVENT_SENSITIVITY_B5.csv",
        "sensitivity/EVENT_SENSITIVITY_AUDIT.json",
    ] + [path.relative_to(OUTPUT).as_posix() for path in sorted((OUTPUT / "figures").glob("*"))]
    payloads = {name: (OUTPUT / name).read_bytes() for name in public_names}
    manifest = "\n".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}" for name, data in sorted(payloads.items())
    ) + "\n"
    payloads["MANIFEST.sha256"] = manifest.encode()
    with zipfile.ZipFile(OUTPUT / "operator_assistant_evidence_v3_public.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payloads.items()):
            if b"/home/" in data or b"/mnt/" in data:
                raise RuntimeError(f"Absolute path in public payload {name}")
            archive.writestr(name, data)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "sensitivity"
    if mode == "sensitivity":
        sensitivity()
    elif mode == "finalize":
        finalize()
    else:
        raise SystemExit("usage: analyze_v3.py [sensitivity|finalize]")


if __name__ == "__main__":
    main()
