from __future__ import annotations

import hashlib
import json
import re
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


def primary(frame: pd.DataFrame, task: str) -> dict[str, float | bool | str]:
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
        "holm_p": float(min(
            mae.holm_corrected_p, r2.holm_corrected_p, rank.holm_corrected_p
        )),
        "significant": significant, "practical": practical,
        "scenario": (
            "positive" if significant and practical
            else "small_effect" if significant else "negative"
        ),
    }


def sentence(label: str, values: dict[str, float | bool | str]) -> str:
    return (
        f"{label}: ΔMAE={values['delta_mae']:.4f}, снижение MAE="
        f"{values['mae_reduction']:.2f}%, ΔR²={values['delta_r2']:.4f}, "
        f"ΔSpearman={values['delta_spearman']:.4f}, 95% CI ΔMAE "
        f"[{values['ci_low']:.4f}; {values['ci_high']:.4f}], "
        f"Holm p={values['holm_p']:.4g}."
    )


def markdown_table(path: Path, limit: int = 30) -> str:
    frame = pd.read_csv(path)
    if len(frame) > limit:
        frame = frame.head(limit)
    return frame.to_markdown(index=False)


def convert_to_pdf(docx: Path) -> Path:
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf",
        "--outdir", str(docx.parent), str(docx),
    ], check=True, cwd=PROJECT_DIR)
    pdf = docx.with_suffix(".pdf")
    if not pdf.is_file():
        raise RuntimeError(f"PDF conversion failed: {pdf}")
    return pdf


def main() -> None:
    mapping = yaml.safe_load(
        (PROJECT_DIR / "article/result_mapping.yaml").read_text(encoding="utf-8")
    )
    source_docx = Path(mapping["source_docx"])
    if not source_docx.is_absolute():
        source_docx = PROJECT_DIR / source_docx
    source_hash = sha256(source_docx)
    if source_hash != mapping["source_docx_sha256"]:
        raise RuntimeError("Expanded article source DOCX hash does not match frozen protocol")
    quality = json.loads(
        (ROOT / "baseline_rescue_v2/quality_gate.json").read_text(encoding="utf-8")
    )
    if not quality["passed"]:
        raise RuntimeError("Article generation is blocked by the baseline quality gate")
    thresholds = json.loads(
        (ROOT / "calibration/threshold_selection.json").read_text(encoding="utf-8")
    )
    clean = pd.read_csv(ROOT / "tables/03_clean_test_metrics.csv")
    clean = clean[
        clean["split"].eq("test") & clean["operating_point"].eq("safety")
    ].iloc[0]
    validation = pd.read_csv(ROOT / "tables/02_training_and_threshold.csv").iloc[0]
    damage = primary(pd.read_csv(ROOT / "tables/08_damage_D2_D3.csv"), "damage")
    recovery = primary(pd.read_csv(ROOT / "tables/09_recovery_R2_R3.csv"), "recovery")
    h3 = pd.read_csv(ROOT / "tables/10_object_global_comparison.csv")
    h4 = pd.read_csv(ROOT / "tables/11_adaptive_comparison.csv")
    latency = pd.read_csv(ROOT / "tables/14_latency_summary.csv")
    h3_text = (
        f"Δ|rho|={h3.iloc[0].delta_rho:.4f}, Holm p="
        f"{h3.iloc[0].holm_corrected_p:.4g}" if len(h3) else "нет сопоставимых условий"
    )
    h4_text = (
        f"adaptive−non-adaptive F1={h4.iloc[0].estimate:.4f}, 95% CI "
        f"[{h4.iloc[0].ci_low:.4f}; {h4.iloc[0].ci_high:.4f}]"
        if len(h4) else "нет общего замороженного budget"
    )
    detector = latency[latency["method"].eq("YOLO only")].iloc[0]
    full = latency[latency["method"].eq("full diagnostic pipeline")].iloc[0]
    latency_text = (
        f"YOLO only {detector.mean_latency_ms:.2f} ms; полный pipeline "
        f"{full.mean_latency_ms:.2f} ms (p95={full.p95_latency_ms:.2f} ms)."
    )
    clean_text = (
        f"На test модель достигла mAP50={clean['mAP50']:.4f}, "
        f"mAP50-95={clean['mAP50-95']:.4f}, Precision={clean['precision']:.4f}, "
        f"Recall={clean['recall']:.4f}, F1={clean['f1']:.4f}, "
        f"FN/кадр={clean['fn_per_frame']:.3f}."
    )
    scenarios = {damage["scenario"], recovery["scenario"]}
    scenario = (
        "positive" if "positive" in scenarios
        else "small_effect" if "small_effect" in scenarios else "negative"
    )
    conclusions = {
        "positive": "По крайней мере одна заранее заданная гипотеза дала статистически подтверждённый и практически заметный прирост; вывод ограничен пятью test-сценами и одной архитектурой.",
        "small_effect": "Канонические T-нормы дали статистически воспроизводимый, но небольшой дополнительный диагностический сигнал; заранее заданный порог практической значимости достигнут не для всех основных сравнений.",
        "negative": "После учёта зависимости кадров внутри железнодорожных сцен подтверждённого преимущества канонических T-норм над стандартными метриками не обнаружено.",
    }
    damage_text = sentence("D3 против D2", damage)
    recovery_text = sentence("R3 против R2", recovery)
    ru_abstract = (
        "Предложена воспроизводимая схема оценки диагностической ценности "
        "Product- и Łukasiewicz-показателей при FGSM, PGD и adaptive PGD для YOLO11m "
        "на OSDaR23. Использован scene-disjoint split 10/5/5. "
        f"{clean_text} {damage_text} {recovery_text} {conclusions[scenario]}"
    )
    en_abstract = (
        "A reproducible evaluation of canonical Product and Lukasiewicz diagnostics under "
        "FGSM, PGD, and adaptive PGD was conducted for YOLO11m on a 10/5/5 scene-disjoint "
        f"OSDaR23 split. Clean test mAP50 was {clean['mAP50']:.4f}, Recall "
        f"{clean['recall']:.4f}, and F1 {clean['f1']:.4f}. D3-vs-D2 delta R2 was "
        f"{damage['delta_r2']:.4f}; R3-vs-R2 delta R2 was {recovery['delta_r2']:.4f}. "
        "Claims are bounded by paired grouped-scene inference and prespecified practical thresholds."
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    media = OUTPUT / "source_media"
    source_markdown = OUTPUT / "TNorm_RZD_article_source_converted.md"
    subprocess.run([
        "pandoc", str(source_docx), "--extract-media", str(media),
        "-t", "gfm", "-o", str(source_markdown),
    ], check=True, cwd=PROJECT_DIR)
    text = source_markdown.read_text(encoding="utf-8")
    text = re.sub(
        r"(\*\*\u0410\u041d\u041d\u041e\u0422\u0410\u0426\u0418\u042f\*\*\n\n).*?(?=\n\*\*\u041a\u043b\u044e\u0447\u0435\u0432\u044b\u0435 \u0441\u043b\u043e\u0432\u0430:\*\*)",
        rf"\1{ru_abstract}\n", text, flags=re.S,
    )
    text = re.sub(
        r"(\*\*ABSTRACT\*\*\n\n).*?(?=\n\*\*Keywords:\*\*)",
        rf"\1{en_abstract}\n", text, flags=re.S,
    )
    text = text.replace(
        r"\[TBD\_\*\]", "служебных маркеров результата"
    )
    text = re.sub(r"\| \*\*\u0421\u0442\u0430тус \u0432\u044bч\u0438\u0441\u043b\u0435\u043dий\..*?\n\|----\|\n", "", text, flags=re.S)
    text = re.sub(
        r"\| \*\*[^\n]*Статус[^\n]*\*\*.*?\n\|----\|\n",
        "", text, flags=re.S,
    )
    results_block = (
        "## **5.2. Итоговые результаты canonical v2**\n\n"
        f"Validation quality gate: Recall={validation.validation_recall_standard:.4f}, "
        f"mAP50={validation.validation_mAP50:.4f}; frozen safety threshold="
        f"{float(thresholds['safety']['confidence']):.4f}.\n\n"
        f"{clean_text}\n\n{damage_text}\n\n{recovery_text}\n\n"
        f"Object vs global: {h3_text}. Adaptive vs non-adaptive: {h4_text}. "
        f"{latency_text}\n\n"
        "### Основные модельные сравнения\n\n"
        + markdown_table(ROOT / "tables/08_damage_D2_D3.csv", 12) + "\n\n"
        + markdown_table(ROOT / "tables/09_recovery_R2_R3.csv", 12) + "\n\n"
        f"![F1 и Recall по бюджетам]({(ROOT / 'figures/03_f1_recall_vs_epsilon.png').resolve()})\n\n"
        f"![D3 против D2]({(ROOT / 'figures/04_damage_D3_vs_D2.png').resolve()})\n\n"
        f"![R3 против R2]({(ROOT / 'figures/05_recovery_R3_vs_R2.png').resolve()})\n\n"
        f"![Adaptive и non-adaptive PGD]({(ROOT / 'figures/08_adaptive_vs_nonadaptive.png').resolve()})\n\n"
    )
    text = re.sub(
        r"## \*\*5\.2\..*?(?=## \*\*5\.3\.)", results_block, text, flags=re.S
    )
    discussion = (
        "## **6.1. Интерпретация полученного исхода**\n\n"
        f"{conclusions[scenario]} {damage_text} {recovery_text} "
        "Product preprocessing не интерпретируется как универсальная white-box защита.\n\n"
    )
    text = re.sub(
        r"## \*\*6\.1\..*?(?=## \*\*6\.2\.)", discussion, text, flags=re.S
    )
    conclusion = (
        "# **Заключение**\n\n"
        "Разработана canonical v2 схема T-нормовой диагностики состязательного "
        "повреждения и восстановления признаков. "
        f"{clean_text} {damage_text} {recovery_text} {conclusions[scenario]} "
        f"Object/global: {h3_text}. Adaptive PGD: {h4_text}.\n\n"
    )
    text = re.sub(
        r"# \*\*Заключение\*\*.*?(?=# \*\*Благодарности\*\*)",
        conclusion, text, flags=re.S,
    )
    if "[TBD" in text or "\\[TBD" in text:
        raise RuntimeError("Unresolved article placeholder")
    final_markdown = OUTPUT / "TNorm_RZD_article_final.md"
    final_docx = OUTPUT / "TNorm_RZD_article_final.docx"
    final_markdown.write_text(text, encoding="utf-8")
    subprocess.run([
        "pandoc", str(final_markdown), "--reference-doc", str(source_docx),
        "-o", str(final_docx),
    ], check=True, cwd=PROJECT_DIR)
    final_pdf = convert_to_pdf(final_docx)

    supplementary_md = OUTPUT / "TNorm_RZD_supplementary.md"
    supplementary_docx = OUTPUT / "TNorm_RZD_supplementary.docx"
    supplementary_parts = [
        "# Supplementary materials: TNormFilter canonical v2\n",
        f"Checkpoint SHA-256: `{thresholds['checkpoint_sha256']}`.\n",
        f"Frozen safety threshold: {float(thresholds['safety']['confidence']):.6f}.\n",
    ]
    for index in range(1, 16):
        matches = sorted((ROOT / "tables").glob(f"{index:02d}_*.csv"))
        if not matches:
            raise RuntimeError(f"Missing supplementary table {index:02d}")
        path = matches[0]
        supplementary_parts.append(f"## {path.stem}\n\n{markdown_table(path)}\n")
    for index in range(1, 11):
        matches = sorted((ROOT / "figures").glob(f"{index:02d}_*.png"))
        if not matches:
            raise RuntimeError(f"Missing supplementary figure {index:02d}")
        path = matches[0]
        supplementary_parts.append(f"## {path.stem}\n\n![{path.stem}]({path.resolve()})\n")
    supplementary_md.write_text("\n".join(supplementary_parts), encoding="utf-8")
    subprocess.run([
        "pandoc", str(supplementary_md), "--reference-doc", str(source_docx),
        "-o", str(supplementary_docx),
    ], check=True, cwd=PROJECT_DIR)
    supplementary_pdf = convert_to_pdf(supplementary_docx)
    if sha256(source_docx) != source_hash:
        raise RuntimeError("Read-only source DOCX changed during finalization")
    resolved = {
        "status": "PASS", "scenario": scenario,
        "source_docx": str(source_docx),
        "source_docx_sha256_before": source_hash,
        "source_docx_sha256_after": sha256(source_docx),
        "checkpoint_sha256": thresholds["checkpoint_sha256"],
        "validation_threshold": float(thresholds["safety"]["confidence"]),
        "damage": damage, "recovery": recovery,
        "clean": {name: float(clean[name]) for name in (
            "mAP50", "mAP50-95", "precision", "recall", "f1", "fn_per_frame"
        )},
        "docx": str(final_docx), "docx_sha256": sha256(final_docx),
        "pdf": str(final_pdf), "pdf_sha256": sha256(final_pdf),
        "supplementary_pdf": str(supplementary_pdf),
        "supplementary_pdf_sha256": sha256(supplementary_pdf),
    }
    (OUTPUT / "result_mapping_resolved.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
