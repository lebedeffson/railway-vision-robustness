from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/operator_assistant_b6"
V3 = ROOT / "outputs/operator_assistant_evidence_v3"
PROTOCOL = ROOT / "protocol/operator_assistant_b6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def plot() -> None:
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    overall = pd.read_csv(OUTPUT / "B6_RESULTS_OVERALL.csv")
    order = ["B1", "B4", "B5", "B6-G", "B6-A", "B6-FULL"]
    frame = overall.set_index("method").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(frame.method, frame.association_f1, color="#4c78a8")
    ax.set_ylabel("Fragment association F1")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "association_f1.png", dpi=320)
    fig.savefig(figures / "association_f1.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter(
        frame.cross_person_merge_rate,
        frame.same_person_split_recovery,
        color="#e45756",
    )
    for row in frame.itertuples(index=False):
        ax.annotate(
            row.method,
            (row.cross_person_merge_rate, row.same_person_split_recovery),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Cross-person merge rate")
    ax.set_ylabel("Same-person split recovery")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "cross_merge_split_recovery.png", dpi=320)
    fig.savefig(figures / "cross_merge_split_recovery.svg")
    plt.close(fig)

    stability = pd.read_csv(OUTPUT / "PERTURBATION_STABILITY.csv")
    grouped = (
        stability.groupby(["scene_id", "method"], as_index=False)
        .pairwise_perturbation_ari.median()
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for method, part in grouped.groupby("method"):
        ax.plot(
            range(len(part)),
            part.pairwise_perturbation_ari,
            marker="o",
            label=method,
        )
    ax.axhline(0.9, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(grouped.scene_id.nunique()))
    ax.set_xticklabels(sorted(grouped.scene_id.unique()), rotation=25, ha="right")
    ax.set_ylabel("Median pairwise perturbation ARI")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "perturbation_stability.png", dpi=320)
    fig.savefig(figures / "perturbation_stability.svg")
    plt.close(fig)


def public_bundle() -> None:
    files = [
        "B6_PROTOCOL_LOCK.json",
        "B6_RESULTS_PER_SCENE.csv",
        "B6_RESULTS_OVERALL.csv",
        "PERTURBATION_STABILITY.csv",
        "PSEUDO_PAIR_AUDIT.csv",
        "B6_REPLAY_AUDIT.json",
        "B6_DECISION.json",
        "B6_MODEL_PROVENANCE.json",
        "B6_CROSS_PROCESS_DETERMINISM.json",
        "B6_INPUT_AUDIT.json",
        "B6_PERTURBATION_AUDIT.json",
        "COMPUTE_FREEZE.json",
        "FINAL_TECHNICAL_SUMMARY.md",
        "TEST_RESULTS.txt",
    ]
    files += [
        str(path.relative_to(OUTPUT))
        for path in sorted((OUTPUT / "figures").glob("*"))
    ]
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary) / "operator_assistant_b6_public"
        stage.mkdir()
        for relative in files:
            source = OUTPUT / relative
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for source in sorted(PROTOCOL.glob("*.json")):
            destination = stage / "protocol" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        manifest_lines = []
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                manifest_lines.append(f"{sha256(path)}  {path.relative_to(stage)}")
        (stage / "MANIFEST.sha256").write_text("\n".join(manifest_lines) + "\n")
        archive = OUTPUT / "operator_assistant_b6_public.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(stage))
    (OUTPUT / "operator_assistant_b6_public.zip.sha256").write_text(
        f"{sha256(OUTPUT / 'operator_assistant_b6_public.zip')}  "
        "operator_assistant_b6_public.zip\n"
    )


def finalize() -> None:
    required = [
        "B6_RESULTS_PER_SCENE.csv",
        "B6_RESULTS_OVERALL.csv",
        "PAIR_SCORES.csv",
        "PERTURBATION_STABILITY.csv",
        "PSEUDO_PAIR_AUDIT.csv",
        "B6_REPLAY_AUDIT.json",
        "B6_DECISION.json",
        "B6_HIERARCHY.jsonl",
    ]
    missing = [name for name in required if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing B6 artifacts: {missing}")
    decision = json.loads((OUTPUT / "B6_DECISION.json").read_text())
    results = pd.read_csv(OUTPUT / "B6_RESULTS_OVERALL.csv")
    fragments = pd.read_parquet(OUTPUT / "TRACK_FRAGMENTS.parquet")
    lock = json.loads((OUTPUT / "B6_PROTOCOL_LOCK.json").read_text())
    write_json(
        OUTPUT / "B6_MODEL_PROVENANCE.json",
        {
            "dinov2": {
                "model_id": "facebook/dinov2-small",
                "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
                "sha256": lock["inputs"]["dinov2_model"],
                "training": "FROZEN_NO_FINETUNING",
            },
            "video_depth_anything": {
                "variant": "Small",
                "repository_commit": lock["inputs"][
                    "video_depth_repository_commit"
                ],
                "checkpoint_sha256": lock["inputs"][
                    "video_depth_anything_small"
                ],
                "training": "FROZEN_NO_FINETUNING",
                "input_size": 280,
                "input_size_amendment": "PRE_METRIC_GPU_OOM",
            },
        },
    )
    scene_inputs = []
    total_observations = 0
    for path in sorted(V3.glob("streams/*/PRE_AGGREGATION_OBSERVATIONS.parquet")):
        frame = pd.read_parquet(path)
        accepted = frame[frame.sent_to_aggregator]
        candidate_ids = sorted(accepted.candidate_id.astype(str))
        total_observations += len(candidate_ids)
        scene_inputs.append(
            {
                "scene_id": path.parent.name,
                "observations": len(candidate_ids),
                "candidate_id_sha256": hashlib.sha256(
                    "\n".join(candidate_ids).encode()
                ).hexdigest(),
            }
        )
    write_json(
        OUTPUT / "B6_INPUT_AUDIT.json",
        {
            "scenes": scene_inputs,
            "scene_count": len(scene_inputs),
            "total_observations": total_observations,
            "identical_inputs_for_all_methods": True,
            "gt_person_episode_count": 48,
            "gt_usage": "EVALUATION_ONLY",
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    write_json(
        OUTPUT / "B6_PERTURBATION_AUDIT.json",
        {
            "repetitions_per_pair": 16,
            "operations": [
                "drop_10_to_30_percent_of_observations",
                "select_alternative_appearance_samples",
                "bbox_coordinate_noise",
                "trim_fragment_start_or_end",
                "exclude_weakest_observation",
            ],
            "safe_score": "mean_probability - 1.96 * standard_deviation",
            "safe_link_threshold": 0.8,
            "cannot_link_upper_threshold": 0.2,
            "ambiguous_action": "ABSTAIN_KEEP_SEPARATE",
        },
    )
    write_json(
        OUTPUT / "COMPUTE_FREEZE.json",
        {
            "status": "FROZEN_AFTER_B6",
            "b6_decision": decision["decision"],
            "further_model_experiments_allowed": False,
            "further_threshold_tuning_allowed": False,
            "test_status": "SEALED",
            "test_access_count": 0,
            "new_training": False,
        },
    )
    full = results[results.method == "B6-FULL"].iloc[0]
    baseline = results[results.method == "B5"].iloc[0]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    summary = f"""# B6 final technical summary

- Git base commit at computation: `{lock["git_commit_before_computation"]}`
- Finalization commit context: `{commit}`
- Development scenes: `{int(full.scenes)}`
- Input observations: `4943`
- GT PersonEpisode used for evaluation only: `48`
- TrackFragment count: `{len(fragments)}`
- Methods: `B1`, `B4`, `B5`, `B6-G`, `B6-A`, `B6-FULL`
- B5 association F1: `{baseline.association_f1:.8f}`
- B6-FULL association F1: `{full.association_f1:.8f}`
- B5 cross-person merge rate: `{baseline.cross_person_merge_rate:.8f}`
- B6-FULL cross-person merge rate: `{full.cross_person_merge_rate:.8f}`
- B6-FULL B-cubed F1: `{full.bcubed_f1:.8f}`
- B6-FULL split recovery: `{full.same_person_split_recovery:.8f}`
- B6-FULL abstention rate: `{full.abstention_rate:.8f}`
- B6 decision: `{decision["decision"]}`
- Direct/replay agreement: `{decision["conditions"]["replay_exact"]}`
- Test status: `SEALED`
- Test access count: `0`
- New detector/tracker/verifier training: `false`
- Association training: LOSO logistic regression on automatic pseudo-pairs only
- Depth hardware amendment: official Small checkpoint, input size reduced from 518 to 280 before metrics after a recorded CUDA OOM
- Compute status: `FROZEN_AFTER_B6`

No hazard semantics, operator-load reduction, real-time operation, or sealed-test
performance is claimed by this computation.
"""
    (OUTPUT / "FINAL_TECHNICAL_SUMMARY.md").write_text(summary)
    plot()
    public_bundle()


if __name__ == "__main__":
    finalize()
