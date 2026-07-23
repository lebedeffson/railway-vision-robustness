from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from revision_q1.protocol import load_protocol, output_root


PROJECT_DIR = Path(__file__).resolve().parents[1]
REQUIRED_TABLES = [
    "01_hypotheses_and_endpoints.csv", "02_normalization_ablation.csv",
    "03_baseline_metric_correlations.csv", "04_tnorm_vs_baseline_bootstrap.csv",
    "05_damage_models_D0_D4.csv", "06_recovery_models_R0_R4.csv",
    "07_checkpoint_paired_comparison.csv", "08_checkpoint_sensitivity.csv",
    "09_scene_difficulty_analysis.csv", "10_spatial_stress_test.csv",
    "11_transfer_attack.csv", "12_multiple_comparison_corrections.csv",
]
REQUIRED_FIGURES = [
    "01_feature_normalization_by_layer.png", "02_metric_correlations_with_ci.png",
    "03_tnorm_minus_baseline_delta_rho.png", "04_damage_model_incremental_gain.png",
    "05_recovery_model_incremental_gain.png", "06_checkpoint_sensitivity.png",
    "07_f1_recovery_vs_feature_G.png", "08_object_count_stratification.png",
    "09_small_object_stratification.png", "10_spatial_stress_test.png",
    "11_adaptive_vs_nonadaptive.png",
]
ENDPOINTS_FOR_HYPOTHESIS = {
    "H1": "delta_f1_damage",
    "H2": "delta_f1_recovery",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_DIR, text=True).strip()


def hypothesis_results(root: Path, protocol: dict) -> dict[str, dict[str, object]]:
    gains = pd.read_csv(root / "tables/12_multiple_comparison_corrections.csv")
    thresholds = protocol["statistics"]["practical_effect"]
    result: dict[str, dict[str, object]] = {}
    for hypothesis, task, comparison in (
        ("H1", "damage", "D3_vs_D2"), ("H2", "recovery", "R3_vs_R2")
    ):
        scope = gains[
            (gains["task"] == task) & (gains["comparison"] == comparison)
            & (gains["algorithm"] == "ridge")
            & (gains["endpoint"] == ENDPOINTS_FOR_HYPOTHESIS[hypothesis])
        ]
        improvement_direction = (
            ((scope["metric"] == "delta_mae") & (scope["ci_high"] < 0))
            | ((scope["metric"] != "delta_mae") & (scope["ci_low"] > 0))
        )
        significant = scope[(scope["holm_corrected_p"] < .05) & improvement_direction]
        relative_mae = float(scope["relative_mae_reduction"].dropna().max())
        delta_r2 = float(scope.loc[scope["metric"] == "delta_r2", "estimate"].iloc[0])
        delta_spearman = float(
            scope.loc[scope["metric"] == "delta_spearman", "estimate"].iloc[0]
        )
        practical = (
            relative_mae >= float(thresholds["relative_mae_reduction"])
            or delta_r2 >= float(thresholds["delta_r2"])
            or abs(delta_spearman) >= float(thresholds["abs_delta_spearman"])
        )
        sequence_count = int(scope["sequences"].dropna().min())
        generalization_confirmed = not significant.empty and sequence_count >= 5
        result[hypothesis] = {
            "confirmed": generalization_confirmed,
            "within_observed_scenes_corrected_signal": not significant.empty,
            "independent_scenes": sequence_count,
            "practically_noticeable": practical,
            "allowed_claim": (
                "exploratory direction only; fewer than five independent scenes"
                if sequence_count < 5
                else
                "practically noticeable incremental diagnostic value"
                if generalization_confirmed and practical
                else "statistically reproducible but small incremental diagnostic signal"
                if generalization_confirmed
                else "no sequence-level confirmed advantage over standard metrics"
            ),
        }
    sensitivity = pd.read_csv(root / "tables/08_checkpoint_sensitivity.csv")
    result["H3"] = {
        "confirmed": bool(
            sensitivity["same_direction"].fillna(False).all()
            and sensitivity["ci_overlap"].fillna(False).all()
        ),
        "checks": len(sensitivity),
    }
    difficulty = pd.read_csv(root / "tables/09_scene_difficulty_analysis.csv")
    confirmatory = difficulty[difficulty["status"] == "confirmatory"]
    result["H4"] = {
        "confirmed": bool(len(confirmatory) and (confirmatory["delta_r2_ci_low"] > 0).all()),
        "status": "confirmatory" if len(confirmatory) else "exploratory",
        "reason": None if len(confirmatory) else "fewer than five independent scenes per stratum",
    }
    return result


def main() -> None:
    protocol = load_protocol()
    root = output_root(protocol)
    parser = argparse.ArgumentParser(description="Validate and bundle Q1 revision")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--zip", type=Path,
        default=PROJECT_DIR / "outputs/bundles/TNormFilter_revision_q1.zip",
    )
    args = parser.parse_args()
    missing = [
        f"tables/{name}" for name in REQUIRED_TABLES
        if not (args.root / "tables" / name).is_file()
    ] + [
        f"figures/{name}" for name in REQUIRED_FIGURES
        if not (args.root / "figures" / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"Q1 bundle gate failed; missing {missing}")
    shutil.copy2(PROJECT_DIR / "config/revision_q1_protocol.yaml", args.root / "revision_protocol.yaml")
    shutil.copy2(PROJECT_DIR / "revision_q1/hypotheses.md", args.root / "hypotheses.md")
    hypotheses = hypothesis_results(args.root, protocol)
    selection = json.loads(
        (args.root / "config/normalization_selection.json").read_text(encoding="utf-8")
    )
    summary = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "primary_checkpoint": protocol["primary_checkpoint"],
        "sensitivity_checkpoint": protocol["sensitivity_checkpoint"],
        "selected_normalization": selection["selected_normalization"],
        "selection_split": "val",
        "test_used_for_selection": False,
        "hypotheses": hypotheses,
        "known_limitations": [
            "Only two checkpoints of the same YOLO11m architecture were compared.",
            "Fixed rotations/scales are a controlled stress test, not an adversarial spatial attack.",
            "Scene-stratified results are exploratory when fewer than five sequences contribute.",
            "JPEG and median are excluded from adaptive robustness ranking without BPDA.",
        ],
    }
    (args.root / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    readme = (
        "# TNormFilter Q1 revision\n\n"
        "Sequence-level, validation-frozen evaluation of incremental T-norm diagnostic value.\n"
        "See `run_summary.json` for automatically derived claims and limitations.\n"
    )
    (args.root / "README.md").write_text(readme, encoding="utf-8")
    git_info = f"commit={git('rev-parse', 'HEAD')}\nstatus={git('status', '--short')}\n"
    (args.root / "git_info.txt").write_text(git_info, encoding="utf-8")
    files = sorted(
        path for path in args.root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "checksums.sha256"}
    )
    checksums = {str(path.relative_to(args.root)): sha256(path) for path in files}
    (args.root / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )
    manifest = {
        "protocol_id": protocol["protocol_id"], "commit": git("rev-parse", "HEAD"),
        "python": platform.python_version(), "pytorch": torch.__version__,
        "cuda": torch.version.cuda, "files": checksums,
        "dataset_included": False, "weights_included": False,
    }
    (args.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    files = sorted(path for path in args.root.rglob("*") if path.is_file())
    args.zip.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.zip.with_suffix(args.zip.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path("TNormFilter_revision_q1") / path.relative_to(args.root))
    with zipfile.ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
    temporary.replace(args.zip)
    digest = sha256(args.zip)
    args.zip.with_suffix(args.zip.suffix + ".sha256").write_text(
        f"{digest}  {args.zip.name}\n", encoding="utf-8"
    )
    print(json.dumps({"zip": str(args.zip), "sha256": digest, "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
