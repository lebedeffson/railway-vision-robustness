from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from revision_q1.statistics import (
    cluster_mean_interval, holm_bonferroni, paired_cluster_delta_correlation,
)


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/canonical_v2"
ANALYSIS = ROOT / "analysis_test"
TEST = ROOT / "raw/canonical_test.csv"
OUTPUT = PROJECT_DIR / "outputs/bundles/TNormFilter_canonical_final.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def split_manifest_hash() -> str:
    return (ROOT / "split/split_v2_hash.txt").read_text(encoding="utf-8").split()[0]


def boolean(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def primary_gain(task: str) -> dict:
    frame = pd.read_csv(ANALYSIS / "tables/12_multiple_comparison_corrections.csv")
    comparison = "D3_vs_D2" if task == "damage" else "R3_vs_R2"
    endpoint = "delta_f1_damage" if task == "damage" else "delta_f1_recovery"
    scope = frame[(frame.task == task) & (frame.comparison == comparison) & (frame.endpoint == endpoint) & (frame.algorithm == "ridge")]
    metrics = {row.metric: row for row in scope.itertuples(index=False)}
    delta_mae = metrics["delta_mae"]; delta_r2 = metrics["delta_r2"]; delta_s = metrics["delta_spearman"]
    significant = any([
        delta_mae.holm_corrected_p < .05 and delta_mae.ci_high < 0,
        delta_r2.holm_corrected_p < .05 and delta_r2.ci_low > 0,
        delta_s.holm_corrected_p < .05 and delta_s.ci_low > 0,
    ])
    practical = (
        float(delta_mae.relative_mae_reduction) >= 5
        or float(delta_r2.estimate) >= .05 or abs(float(delta_s.estimate)) >= .05
    )
    return {
        "comparison": comparison, "independent_scenes": int(delta_mae.sequences),
        "statistically_confirmed": significant, "practically_meaningful": practical,
        "delta_mae": float(delta_mae.estimate),
        "relative_mae_reduction_percent": float(delta_mae.relative_mae_reduction),
        "delta_r2": float(delta_r2.estimate), "delta_spearman": float(delta_s.estimate),
        "allowed_claim": (
            "practically noticeable canonical increment" if significant and practical
            else "statistically reproducible but small complementary signal" if significant
            else "no confirmed canonical advantage over standard metrics"
        ),
    }


def h3_object_vs_global(data: pd.DataFrame) -> list[dict]:
    scope = data[boolean(data.selected_best) & ~boolean(data.adaptive) & data.defense.eq("none")]
    frame = scope.groupby(["sequence_id", "image_path", "attack", "epsilon_px", "seed"], as_index=False).agg(
        delta_recall_damage=("delta_recall_damage", "mean"),
        false_negatives_increase=("false_negatives_increase", "mean"),
        object=("c_atk_object", "mean"), global_score=("c_atk_global", "mean"),
    )
    rows = []
    for endpoint in ("delta_recall_damage", "false_negatives_increase"):
        result = paired_cluster_delta_correlation(
            frame, endpoint, "object", "global_score", iterations=5000, seed=20260720
        )
        rows.append({"hypothesis": "H3", "endpoint": endpoint, **result})
    corrected = holm_bonferroni([row["p_value"] for row in rows])
    for row, value in zip(rows, corrected, strict=True):
        row["holm_corrected_p"] = float(value)
    return rows


def h4_adaptive(data: pd.DataFrame) -> dict:
    scope = data[boolean(data.selected_best) & data.attack.eq("pgd") & data.defense.eq("tnorm")].copy()
    scope["adaptive_bool"] = boolean(scope.adaptive)
    per_image = scope.groupby(["sequence_id", "image_path", "epsilon_px", "adaptive_bool"], as_index=False).f1_defended.mean()
    rows = []
    for epsilon, group in per_image.groupby("epsilon_px"):
        pivot = group.pivot(index=["sequence_id", "image_path"], columns="adaptive_bool", values="f1_defended").dropna()
        if False in pivot and True in pivot:
            frame = pivot.reset_index(); frame["adaptive_minus_nonadaptive_f1"] = frame[True] - frame[False]
            interval = cluster_mean_interval(
                frame, "adaptive_minus_nonadaptive_f1",
                iterations=5000, seed=20260720,
            )
            rows.append({"epsilon_px": float(epsilon), **interval})
    return {"hypothesis": "H4", "comparisons": rows, "status": "available" if rows else "no_shared_frozen_epsilon"}


def copy_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)


def required_artifacts() -> list[Path]:
    return [
        ROOT / "tables" / f"{index:02d}_{name}"
        for index, name in enumerate((
            "split_v2_summary.csv", "training_and_threshold.csv",
            "clean_test_metrics.csv", "attack_parameters.csv",
            "canonical_robustness.csv", "attack_consistency.csv",
            "defense_consistency.csv", "damage_D2_D3.csv",
            "recovery_R2_R3.csv", "object_global_comparison.csv",
            "adaptive_comparison.csv", "per_scene_results.csv",
            "loso_results.csv", "latency_summary.csv", "nms_audit.csv",
        ), 1)
    ] + [
        ROOT / "figures" / f"{index:02d}_{name}"
        for index, name in enumerate((
            "training_curves.png", "precision_recall_curve.png",
            "f1_recall_vs_epsilon.png", "damage_D3_vs_D2.png",
            "recovery_R3_vs_R2.png", "tnorm_vs_baselines.png",
            "object_vs_global.png", "adaptive_vs_nonadaptive.png",
            "per_scene_effects.png", "latency_tradeoff.png",
        ), 1)
    ] + [
        PROJECT_DIR / "outputs/article/TNormFilter_canonical_final.docx",
        PROJECT_DIR / "outputs/article/TNormFilter_canonical_final.pdf",
        PROJECT_DIR / "outputs/article/article_validation.json",
        ROOT / "config/provenance_final.json",
        ROOT / "pilot/pilot_gate.json",
        ROOT / "baseline_rescue_v2/baseline_rescue_summary.json",
    ]


def main() -> None:
    missing = [str(path) for path in required_artifacts() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Canonical evidence is incomplete: {missing}")
    article_validation = json.loads(
        (PROJECT_DIR / "outputs/article/article_validation.json").read_text(encoding="utf-8")
    )
    pilot_gate = json.loads((ROOT / "pilot/pilot_gate.json").read_text(encoding="utf-8"))
    quality_gate = json.loads(
        (ROOT / "baseline_rescue_v2/baseline_rescue_summary.json").read_text(encoding="utf-8")
    )["quality_gate"]
    if article_validation.get("status") != "PASS":
        raise RuntimeError("Final article validation did not pass")
    if not pilot_gate.get("pilot_gate_passed") or not quality_gate.get("passed"):
        raise RuntimeError("Canonical gates did not pass")
    data = pd.read_csv(TEST, low_memory=False)
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "H1": primary_gain("damage"), "H2": primary_gain("recovery"),
        "H3": h3_object_vs_global(data), "H4": h4_adaptive(data),
        "legacy_role": "pilot_comparison_only", "canonical_test_tuned": False,
        "independent_test_scenes": int(data.sequence_id.nunique()),
    }
    tables = ROOT / "tables"; tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results["H1"], results["H2"]]).to_csv(
        tables / "canonical_D2_D3_R2_R3.csv", index=False
    )
    pd.DataFrame(results["H3"]).to_csv(tables / "object_vs_global_H3.csv", index=False)
    pd.DataFrame(results["H4"]["comparisons"]).to_csv(
        tables / "adaptive_vs_nonadaptive_H4.csv", index=False
    )
    ROOT.joinpath("report").mkdir(parents=True, exist_ok=True)
    (ROOT / "report/canonical_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (ROOT / "report/canonical_report.md").write_text(
        "# TNormFilter canonical v2\n\n"
        f"H1: {results['H1']['allowed_claim']}\n\n"
        f"H2: {results['H2']['allowed_claim']}\n\n"
        "H3 compares object-region and global attack consistency with paired scene bootstrap.\n\n"
        "H4 compares adaptive and non-adaptive PGD only at shared frozen validation budgets.\n",
        encoding="utf-8",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    status = json.loads((ROOT / "pipeline_status.json").read_text(encoding="utf-8"))
    failed = [name for name, row in status.get("stages", {}).items() if row.get("status") == "failed"]
    pending = [name for name, row in status.get("stages", {}).items() if row.get("status") == "pending"]
    running = [
        name for name, row in status.get("stages", {}).items()
        if row.get("status") == "running" and name != "canonical_bundle"
    ]
    if failed or pending or running:
        raise RuntimeError(
            f"Pipeline contains unresolved stages: failed={failed}, "
            f"pending={pending}, running={running}"
        )
    packaged_status = json.loads(json.dumps(status))
    packaged_status["status"] = "success"
    packaged_status.setdefault("stages", {}).setdefault("canonical_bundle", {}).update({
        "status": "success", "finished_at": datetime.now(timezone.utc).isoformat(),
        "error": None,
    })
    with tempfile.TemporaryDirectory(prefix="tnorm_canonical_") as directory:
        staging = Path(directory) / "TNormFilter_canonical_final"
        directories = {
            "audit": PROJECT_DIR / "outputs/final_practice/audit",
            "calibration": ROOT / "calibration",
            "normalization": ROOT / "normalization",
            "raw": ROOT / "raw",
            "tables": ROOT / "tables",
            "figures": ROOT / "figures",
            "statistics": ROOT / "analysis_test",
            "article": PROJECT_DIR / "outputs/article",
            "tests": ROOT / "tests",
            "report": ROOT / "report",
            "supplementary": ROOT / "supplementary",
        }
        for name, source in directories.items():
            if source.is_dir():
                copy_tree(source, staging / name)
            else:
                (staging / name).mkdir(parents=True, exist_ok=True)
        # Training evidence is included without the large checkpoint payload.
        training = ROOT / "training"
        for name in (
            "training_config.yaml", "results.csv", "training_curves.png",
            "checkpoint_selection.csv", "checkpoint_provenance.json",
        ):
            source = training / name
            if source.is_file():
                target = staging / "training" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        (staging / "configs").mkdir(parents=True, exist_ok=True)
        for source in (
            PROJECT_DIR / "config/canonical_v2_protocol.yaml",
            PROJECT_DIR / "config/canonical_v2_analysis.yaml",
            ROOT / "config/frozen_attack_budgets.yaml",
        ):
            shutil.copy2(source, staging / "configs" / source.name)
        for source, relative in (
            (ROOT / "split/split_v2_manifest.csv", "configs/split_v2_manifest.csv"),
            (ROOT / "split/split_v2_summary.json", "configs/split_v2_summary.json"),
            (ROOT / "split/split_v2_hash.txt", "configs/split_v2_hash.txt"),
        ):
            if source.is_file():
                target = staging / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        (staging / "pipeline_status.json").write_text(
            json.dumps(packaged_status, indent=2) + "\n", encoding="utf-8"
        )
        log_dir = staging / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
        log = subprocess.run(
            ["journalctl", "--user", "-u", "tnorm-canonical-v2.service", "--no-pager"],
            text=True, capture_output=True, check=False,
        )
        (log_dir / "tnorm-canonical-v2.service.log").write_text(log.stdout, encoding="utf-8")
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True).strip()
        git_status = subprocess.check_output(["git", "status", "--short"], cwd=PROJECT_DIR, text=True)
        (staging / "git_info.txt").write_text(
            f"commit={git_commit}\nworktree_status_at_bundle_build=\n{git_status}", encoding="utf-8"
        )
        (staging / "README.md").write_text(
            "# TNormFilter canonical v2 evidence\n\n"
            "Grouped-scene canonical experiment. Legacy compatibility metrics are excluded "
            "from primary claims. Dataset and full checkpoint weights are excluded; hashes "
            "and provenance are included.\n",
            encoding="utf-8",
        )
        stage_counts = Counter(
            row.get("status", "unknown")
            for row in packaged_status.get("stages", {}).values()
        )
        run_summary = {
            "status": "PASS", "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit, "manifest_sha256": split_manifest_hash(),
            "checkpoint_sha256": json.loads((ROOT / "calibration/threshold_selection.json").read_text())["checkpoint_sha256"],
            "validation_threshold": json.loads((ROOT / "calibration/threshold_selection.json").read_text())["safety"]["confidence"],
            "quality_gate": quality_gate, "pilot_gate": pilot_gate.get("status"),
            "hypotheses": {key: results[key] for key in ("H1", "H2", "H3", "H4")},
            "independent_test_scenes": results["independent_test_scenes"],
            "stage_counts": dict(stage_counts), "failed_stages": failed, "pending_stages": pending,
            "article_validation": article_validation,
        }
        (staging / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "git_commit": git_commit,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_manifest_sha256": split_manifest_hash(),
            "checkpoint_sha256": run_summary["checkpoint_sha256"],
            "dataset_included": False, "checkpoint_weights_included": False,
            "files": {path.relative_to(staging).as_posix(): sha256(path) for path in staging.rglob("*") if path.is_file()},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        (staging / "checksums.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(staging).as_posix()}\n" for path in files), encoding="utf-8")
        forbidden = []
        for path in staging.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml", ".csv"}:
                try:
                    if "[TBD" in path.read_text(encoding="utf-8"):
                        forbidden.append(str(path.relative_to(staging)))
                except UnicodeDecodeError:
                    pass
        if forbidden:
            raise RuntimeError(f"Unresolved placeholders in bundle: {forbidden}")
        temporary = OUTPUT.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in staging.rglob("*"):
                if path.is_file(): archive.write(path, path.relative_to(staging.parent).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None: raise RuntimeError("Canonical ZIP corrupt")
        temporary.replace(OUTPUT)
    OUTPUT.with_suffix(".zip.sha256").write_text(f"{sha256(OUTPUT)}  {OUTPUT.name}\n", encoding="utf-8")
    print(json.dumps({"zip": str(OUTPUT), "sha256": sha256(OUTPUT), **results}, indent=2))


if __name__ == "__main__":
    main()
