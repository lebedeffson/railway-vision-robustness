from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT = PROJECT_DIR / "outputs/canonical_v2"
OUTPUT = PROJECT_DIR / "outputs/article"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def primary(frame: pd.DataFrame, task: str) -> dict[str, float]:
    endpoint = "delta_f1_damage" if task == "damage" else "delta_f1_recovery"
    scope = frame[frame["endpoint"].eq(endpoint) & frame["algorithm"].eq("ridge")]
    rows = {row.metric: row for row in scope.itertuples(index=False)}
    if not {"delta_mae", "delta_r2", "delta_spearman"} <= set(rows):
        raise RuntimeError(f"Missing primary {task} gain rows")
    mae, r2, rank = rows["delta_mae"], rows["delta_r2"], rows["delta_spearman"]
    significant = bool(
        (mae.holm_corrected_p < .05 and mae.ci_high < 0)
        or (r2.holm_corrected_p < .05 and r2.ci_low > 0)
        or (rank.holm_corrected_p < .05 and rank.ci_low > 0)
    )
    practical = bool(
        mae.relative_mae_reduction >= 5
        or r2.estimate >= .05 or abs(rank.estimate) >= .05
    )
    return {
        "delta_mae": float(mae.estimate),
        "mae_reduction": float(mae.relative_mae_reduction),
        "delta_r2": float(r2.estimate),
        "delta_spearman": float(rank.estimate),
        "ci_low": float(mae.ci_low), "ci_high": float(mae.ci_high),
        "holm_p": float(min(mae.holm_corrected_p, r2.holm_corrected_p, rank.holm_corrected_p)),
        "significant": significant, "practical": practical,
        "scenario": "positive" if significant and practical else "small_effect" if significant else "negative",
    }


def sentence(label: str, values: dict[str, float]) -> str:
    return (
        f"{label}: ΔMAE={values['delta_mae']:.4f}, снижение MAE="
        f"{values['mae_reduction']:.2f}%, ΔR²={values['delta_r2']:.4f}, "
        f"ΔSpearman={values['delta_spearman']:.4f}, 95% CI ΔMAE "
        f"[{values['ci_low']:.4f}; {values['ci_high']:.4f}], "
        f"Holm p={values['holm_p']:.4g}."
    )


def main() -> None:
    mapping = yaml.safe_load((PROJECT_DIR / "article/result_mapping.yaml").read_text(encoding="utf-8"))
    template = PROJECT_DIR / mapping["template"]
    source_hash = sha256(template)
    reference_docx = PROJECT_DIR / mapping["reference_docx"]
    reference_hash = sha256(reference_docx)
    quality = json.loads((ROOT / "baseline_rescue_v2/baseline_rescue_summary.json").read_text(encoding="utf-8"))
    if not quality["quality_gate"]["passed"]:
        raise RuntimeError("Article generation is blocked by the baseline quality gate")
    thresholds = json.loads((ROOT / "calibration/threshold_selection.json").read_text(encoding="utf-8"))
    clean = pd.read_csv(ROOT / "tables/03_clean_test_metrics.csv")
    clean = clean[clean["split"].eq("test") & clean["operating_point"].eq("safety")].iloc[0]
    split = pd.read_csv(ROOT / "tables/01_split_v2_summary.csv")
    damage = primary(pd.read_csv(ROOT / "tables/08_damage_D2_D3.csv"), "damage")
    recovery = primary(pd.read_csv(ROOT / "tables/09_recovery_R2_R3.csv"), "recovery")
    h3 = pd.read_csv(ROOT / "tables/10_object_global_comparison.csv")
    h4 = pd.read_csv(ROOT / "tables/11_adaptive_comparison.csv")
    latency = pd.read_csv(ROOT / "tables/14_latency_summary.csv")
    split_text = ", ".join(
        f"{row.split}: {int(row.grouped_scenes)} сцен/{int(row.frames)} кадров"
        for row in split.itertuples(index=False)
    )
    checkpoint = thresholds["checkpoint_sha256"]
    threshold = float(thresholds["safety"]["confidence"])
    clean_text = (
        f"На однократно открытом test: mAP50={clean['mAP50']:.4f}, "
        f"mAP50-95={clean['mAP50-95']:.4f}, Precision={clean['precision']:.4f}, "
        f"Recall={clean['recall']:.4f}, F1={clean['f1']:.4f}, "
        f"FN/кадр={clean['fn_per_frame']:.3f}."
    )
    h3_best = h3.iloc[0] if len(h3) else None
    h4_best = h4.iloc[0] if len(h4) else None
    h34 = (
        (f"Object-global Δ|ρ|={h3_best.delta_rho:.4f}, Holm p={h3_best.holm_corrected_p:.4g}. " if h3_best is not None else "Object/global comparison unavailable. ")
        + (f"Adaptive minus non-adaptive defended F1={h4_best.estimate:.4f}, 95% CI [{h4_best.ci_low:.4f}; {h4_best.ci_high:.4f}]." if h4_best is not None else "No shared adaptive comparison budget.")
    )
    detector = latency[latency["method"].eq("YOLO only")].iloc[0]
    full = latency[latency["method"].eq("full diagnostic pipeline")].iloc[0]
    latency_text = (
        f"YOLO only: mean={detector.mean_latency_ms:.2f} ms, p95={detector.p95_latency_ms:.2f} ms, "
        f"FPS={detector.fps:.2f}; полный диагностический pipeline: mean={full.mean_latency_ms:.2f} ms, "
        f"p95={full.p95_latency_ms:.2f} ms, FPS={full.fps:.2f}."
    )
    scenarios = {damage["scenario"], recovery["scenario"]}
    overall = "positive" if "positive" in scenarios else "small_effect" if "small_effect" in scenarios else "negative"
    conclusions = {
        "positive": "По крайней мере одна заранее заданная гипотеза показала статистически подтверждённый и практически заметный прирост; вывод ограничен пятью test-сценами и одной архитектурой.",
        "small_effect": "Канонические T-нормы дали статистически воспроизводимый, но небольшой дополнительный диагностический сигнал; заранее заданный порог практической значимости достигнут не для всех основных сравнений.",
        "negative": "После учёта зависимости кадров внутри железнодорожных сцен подтверждённого преимущества канонических T-норм над стандартными метриками не обнаружено.",
    }
    abstract = clean_text + " " + sentence("D3 против D2", damage) + " " + sentence("R3 против R2", recovery) + " " + conclusions[overall]
    replacements = {
        "[TBD_RU_ABSTRACT]": abstract,
        "[TBD_EN_ABSTRACT]": (
            f"On five independent test scenes, the clean detector reached mAP50={clean['mAP50']:.4f}, "
            f"Recall={clean['recall']:.4f}, and F1={clean['f1']:.4f}. "
            f"D3-vs-D2 ΔR²={damage['delta_r2']:.4f}; R3-vs-R2 ΔR²={recovery['delta_r2']:.4f}. "
            "Canonical T-norm claims were limited by scene-cluster inference and prespecified practical thresholds."
        ),
        "[TBD_SPLIT_TEXT]": f"Замороженный grouped-scene split: {split_text}.",
        "[TBD_MODEL_TEXT]": f"Checkpoint SHA-256 `{checkpoint}`. Safety operating point был выбран после обучения только на validation: confidence={threshold:.4f}. Quality gate Recall≥0.35 и mAP50≥0.25 пройден.",
        "[TBD_CLEAN_TEXT]": clean_text,
        "[TBD_DAMAGE_TEXT]": sentence("D3 против D2", damage),
        "[TBD_RECOVERY_TEXT]": sentence("R3 против R2", recovery),
        "[TBD_H3_H4_TEXT]": h34,
        "[TBD_LATENCY_TEXT]": latency_text,
        "[TBD_DISCUSSION]": conclusions[overall],
        "[TBD_CONCLUSION]": conclusions[overall] + " Product preprocessing не интерпретируется как универсальная white-box защита; вывод об устойчивости определяется сравнением с adaptive PGD.",
    }
    text = template.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace(key, value)
    if "[TBD" in text:
        raise RuntimeError("Unresolved article placeholder")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    markdown = OUTPUT / "TNormFilter_canonical_final.md"
    docx = OUTPUT / "TNormFilter_canonical_final.docx"
    markdown.write_text(text, encoding="utf-8")
    subprocess.run([
        "pandoc", str(markdown), "--reference-doc", str(reference_docx),
        "-o", str(docx),
    ], check=True, cwd=PROJECT_DIR)
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUTPUT), str(docx)
    ], check=True, cwd=PROJECT_DIR)
    pdf = OUTPUT / "TNormFilter_canonical_final.pdf"
    if not docx.is_file() or not pdf.is_file():
        raise RuntimeError("DOCX/PDF conversion failed")
    resolved = {
        "status": "PASS", "scenario": overall, "template_sha256_before": source_hash,
        "template_sha256_after": sha256(template), "checkpoint_sha256": checkpoint,
        "reference_docx_sha256_before": reference_hash,
        "reference_docx_sha256_after": sha256(reference_docx),
        "validation_threshold": threshold, "damage": damage, "recovery": recovery,
        "clean": {name: float(clean[name]) for name in ("mAP50", "mAP50-95", "precision", "recall", "f1", "fn_per_frame")},
        "docx": str(docx), "docx_sha256": sha256(docx),
        "pdf": str(pdf), "pdf_sha256": sha256(pdf),
    }
    (OUTPUT / "result_mapping_resolved.json").write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
