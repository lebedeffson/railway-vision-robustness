from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/operator_assistant_b7"
PROTOCOL = ROOT / "protocol/operator_assistant_b7"
PUBLIC_ARTIFACTS = ROOT / "artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def figures() -> None:
    directory = OUTPUT / "figures"
    directory.mkdir(exist_ok=True)
    overall = pd.read_csv(OUTPUT / "B7_RESULTS_OVERALL.csv")
    order = ["B6-FULL", "B7-FLOW", "B7-HARD", "B7-CONSENSUS"]
    overall = overall.set_index("method").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    fields = [
        ("micro_association_f1", "Micro association F1"),
        ("micro_cross_person_merge_rate", "Cross-person merge rate"),
        ("median_perturbation_ari", "Median perturbation ARI"),
    ]
    for axis, (field, label) in zip(axes, fields, strict=True):
        axis.bar(overall.method, overall[field], color="#4c78a8")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "b7_primary_comparison.png", dpi=320)
    fig.savefig(directory / "b7_primary_comparison.svg")
    plt.close(fig)

    per_scene = pd.read_csv(OUTPUT / "B7_RESULTS_PER_SCENE.csv")
    pivot = per_scene.pivot(
        index="scene_id", columns="method", values="association_f1"
    )[order]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Association F1")
    ax.set_xlabel("Development scene")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "b7_per_scene_f1.png", dpi=320)
    fig.savefig(directory / "b7_per_scene_f1.svg")
    plt.close(fig)

    stability = pd.read_csv(OUTPUT / "B7_STABILITY_PER_SCENE.csv")
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for method, group in stability.groupby("method", sort=False):
        group = group.sort_values("scene_id")
        ax.plot(
            group.scene_id,
            group.median_perturbation_ari,
            marker="o",
            label=f"{method} median",
        )
        ax.plot(
            group.scene_id,
            group.p10_perturbation_ari,
            marker=".",
            linestyle="--",
            alpha=0.7,
            label=f"{method} p10",
        )
    ax.axhline(0.9, color="black", linewidth=1, linestyle=":")
    ax.set_ylabel("Perturbation ARI")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "b7_stability.png", dpi=320)
    fig.savefig(directory / "b7_stability.svg")
    plt.close(fig)


def public_bundle() -> None:
    include = [
        "B7_PROTOCOL_LOCK.json",
        "B7_DECISION.json",
        "B7_RESULTS_PER_SCENE.csv",
        "B7_RESULTS_OVERALL.csv",
        "B7_STABILITY_PER_SCENE.csv",
        "B7_SCENE_NORMALIZATION.csv",
        "B7_CANDIDATE_EDGES.csv",
        "B7_FLOW_EDGES.csv",
        "B7_HARD_PSEUDO_PAIR_INDEX.csv",
        "B7_MODEL_COEFFICIENTS.csv",
        "B7_REPLAY_AUDIT.json",
        "B7_CROSS_PROCESS_DETERMINISM.json",
        "B7_HIERARCHY.jsonl",
        "B7_INPUT_PROVENANCE.json",
        "COMPUTE_FREEZE.json",
        "FINAL_TECHNICAL_SUMMARY.md",
        "TEST_RESULTS.txt",
        "b6_audit/B6_ASSOCIATION_PER_SCENE.csv",
        "b6_audit/B6_ASSOCIATION_MICRO.json",
        "b6_audit/B6_MODEL_COEFFICIENTS.csv",
        "b6_audit/B6_NORMALIZATION_PARAMETERS.csv",
        "b6_audit/B6_PSEUDO_PAIR_INDEX.csv",
        "b6_audit/B6_STABILITY_PER_SCENE.csv",
        "b6_audit/B6_AUDIT.json",
    ]
    include.extend(
        str(path.relative_to(OUTPUT))
        for path in sorted((OUTPUT / "figures").glob("*"))
    )
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary) / "operator_assistant_b7_public"
        stage.mkdir()
        for relative in include:
            source = OUTPUT / relative
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if PROTOCOL.is_dir():
            for source in sorted(PROTOCOL.glob("*.json")):
                destination = stage / "protocol" / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        manifest = []
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                manifest.append(
                    f"{sha256(path)}  {path.relative_to(stage)}"
                )
        (stage / "MANIFEST.sha256").write_text("\n".join(manifest) + "\n")
        archive = OUTPUT / "operator_assistant_b7_public.zip"
        with zipfile.ZipFile(
            archive, "w", compression=zipfile.ZIP_DEFLATED
        ) as handle:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(stage))
    sidecar = OUTPUT / "operator_assistant_b7_public.zip.sha256"
    sidecar.write_text(
        f"{sha256(OUTPUT / 'operator_assistant_b7_public.zip')}  "
        "operator_assistant_b7_public.zip\n"
    )
    PUBLIC_ARTIFACTS.mkdir(exist_ok=True)
    shutil.copy2(archive, PUBLIC_ARTIFACTS / archive.name)
    shutil.copy2(sidecar, PUBLIC_ARTIFACTS / sidecar.name)


def finalize() -> None:
    required = [
        "B7_PROTOCOL_LOCK.json",
        "B7_DECISION.json",
        "B7_RESULTS_PER_SCENE.csv",
        "B7_RESULTS_OVERALL.csv",
        "B7_STABILITY_PER_SCENE.csv",
        "B7_FLOW_EDGES.csv",
        "B7_REPLAY_AUDIT.json",
    ]
    missing = [name for name in required if not (OUTPUT / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing B7 outputs: {missing}")
    decision = json.loads((OUTPUT / "B7_DECISION.json").read_text())
    lock = json.loads((OUTPUT / "B7_PROTOCOL_LOCK.json").read_text())
    overall = pd.read_csv(OUTPUT / "B7_RESULTS_OVERALL.csv").set_index(
        "method"
    )
    provenance = {
        "git_commit_at_computation_lock": lock[
            "git_commit_before_b7_computation"
        ],
        "git_commit_at_finalization": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "input_hashes": lock["input_hashes"],
        "development_scenes": 5,
        "frames": 479,
        "observations": 4943,
        "gt_person_episodes_evaluation_only": 48,
        "test_status": "SEALED",
        "test_access_count": 0,
    }
    write_json(OUTPUT / "B7_INPUT_PROVENANCE.json", provenance)
    write_json(
        OUTPUT / "COMPUTE_FREEZE.json",
        {
            "status": "FROZEN_AFTER_B7",
            "scientific_decision": decision["scientific_decision"],
            "operational_decision": decision["operational_decision"],
            "b8_allowed": False,
            "further_model_experiments_allowed": False,
            "further_threshold_tuning_allowed": False,
            "test_status": "SEALED",
            "test_access_count": 0,
        },
    )
    b6 = overall.loc["B6-FULL"]
    flow = overall.loc["B7-FLOW"]
    consensus = overall.loc["B7-CONSENSUS"]
    summary = f"""# B7-SCF final technical summary

- Development scenes: `5`
- Frames: `479`
- Frozen observations: `4943`
- Frozen TrackFragment: `646`
- GT PersonEpisode: `48`, evaluation only
- B6 micro association F1: `{b6.micro_association_f1:.8f}`
- B7-FLOW micro association F1: `{flow.micro_association_f1:.8f}`
- B7-CONSENSUS micro association F1: `{consensus.micro_association_f1:.8f}`
- B6 micro cross-person merge rate: `{b6.micro_cross_person_merge_rate:.8f}`
- B7-FLOW micro cross-person merge rate: `{flow.micro_cross_person_merge_rate:.8f}`
- B7-CONSENSUS micro cross-person merge rate: `{consensus.micro_cross_person_merge_rate:.8f}`
- B6 median perturbation ARI: `{b6.median_perturbation_ari:.8f}`
- B7-CONSENSUS median perturbation ARI: `{consensus.median_perturbation_ari:.8f}`
- B7-CONSENSUS p10 perturbation ARI: `{consensus.p10_perturbation_ari:.8f}`
- Improved scenes for primary B7-CONSENSUS: `{decision["improved_scenes"]}/5`
- Scientific decision: `{decision["scientific_decision"]}`
- Operational decision: `{decision["operational_decision"]}`
- Direct/replay exact: `{decision["scientific_conditions"]["direct_replay_exact"]}`
- Test status: `SEALED`
- Test access count: `0`
- Detector/tracker/verifier/encoder training: `FORBIDDEN`
- Association training: `LOSO logistic regression on automatic pseudopairs only`
- Compute status: `FROZEN_AFTER_B7`
- B8 allowed: `false`

The calculation does not claim hazard understanding, operator-load reduction,
real-time operation, product maturity, or sealed-test performance.
"""
    (OUTPUT / "FINAL_TECHNICAL_SUMMARY.md").write_text(summary)
    figures()
    public_bundle()


if __name__ == "__main__":
    finalize()
