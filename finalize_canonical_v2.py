from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import zipfile
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


def main() -> None:
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
    with tempfile.TemporaryDirectory(prefix="tnorm_canonical_") as directory:
        staging = Path(directory) / "TNormFilter_canonical_final"
        for name in ("split", "baseline_rescue_v2", "normalization_v2", "analysis_test", "tables", "report", "config"):
            source = ROOT / name
            if source.is_dir(): copy_tree(source, staging / name)
        for source, relative in (
            (TEST, "raw/canonical_test.csv"),
            (TEST.with_suffix(".json"), "raw/canonical_test.json"),
            (ROOT / "raw/canonical_validation.csv", "raw/canonical_validation.csv"),
            (ROOT / "config/canonical_budget_selection.json", "config/canonical_budget_selection.json"),
            (PROJECT_DIR / "config/canonical_v2_protocol.yaml", "config/canonical_v2_protocol.yaml"),
            (ROOT / "pipeline_status.json", "pipeline_status.json"),
        ):
            if source.is_file():
                target = staging / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        (staging / "README.md").write_text("Canonical scene-level TNormFilter experiment; weights and source dataset are excluded.\n", encoding="utf-8")
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        (staging / "checksums.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(staging).as_posix()}\n" for path in files), encoding="utf-8")
        manifest = {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True).strip(),
            "files": {path.relative_to(staging).as_posix(): sha256(path) for path in staging.rglob("*") if path.is_file()},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
