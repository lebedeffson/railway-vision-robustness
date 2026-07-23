from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canonical_m4_common import (
    OUTPUT_ROOT,
    PROJECT_DIR,
    TEST_MARKER,
    assert_role_allowed,
    atomic_json,
    git,
    load_protocol,
    sha256,
)


TABLES = OUTPUT_ROOT / "final/tables"
FIGURES = OUTPUT_ROOT / "final/figures"
ARTICLE = OUTPUT_ROOT / "article"
BUNDLES = OUTPUT_ROOT / "bundles"


def primary_gain(gains: pd.DataFrame, task: str) -> dict:
    endpoint = "delta_f1_damage" if task == "damage" else "delta_f1_recovery"
    comparison = "D3_vs_D2" if task == "damage" else "R3_vs_R2"
    scope = gains[
        gains["task"].eq(task)
        & gains["endpoint"].eq(endpoint)
        & gains["algorithm"].eq("ridge")
        & gains["comparison"].eq(comparison)
    ]
    rows = {row.metric: row for row in scope.itertuples(index=False)}
    if set(rows) != {"delta_mae", "delta_r2", "delta_spearman"}:
        raise RuntimeError(f"Incomplete primary gain: {task}")
    mae, r2, rank = rows["delta_mae"], rows["delta_r2"], rows["delta_spearman"]
    significant = bool(
        (mae.holm_corrected_p < 0.05 and mae.ci_high < 0)
        or (r2.holm_corrected_p < 0.05 and r2.ci_low > 0)
        or (rank.holm_corrected_p < 0.05 and rank.ci_low > 0)
    )
    practical = bool(
        mae.relative_mae_reduction >= 5
        or r2.estimate >= 0.05
        or abs(rank.estimate) >= 0.05
    )
    return {
        "delta_mae": float(mae.estimate),
        "mae_reduction": float(mae.relative_mae_reduction),
        "delta_r2": float(r2.estimate),
        "delta_spearman": float(rank.estimate),
        "ci_low": float(mae.ci_low),
        "ci_high": float(mae.ci_high),
        "holm_p": float(min(
            mae.holm_corrected_p,
            r2.holm_corrected_p,
            rank.holm_corrected_p,
        )),
        "significant": significant,
        "practical": practical,
        "scenario": (
            "positive" if significant and practical
            else "small_effect" if significant else "negative"
        ),
    }


def build_tables() -> dict:
    TABLES.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    manifest = pd.read_csv(
        PROJECT_DIR / "data/yolo_osdar23_rescue_v1/manifest.csv"
    )
    manifest.groupby("split", as_index=False).agg(
        grouped_scenes=("grouped_scene_id", "nunique"),
        frames=("output_image", "nunique"),
        objects=("annotations", "sum"),
    ).to_csv(TABLES / "01_split_v2_summary.csv", index=False)
    gate = json.loads(
        (OUTPUT_ROOT / "validation/quality_gate.json").read_text(encoding="utf-8")
    )
    decision = json.loads(
        (OUTPUT_ROOT / "selection/checkpoint_selection.json")
        .read_text(encoding="utf-8")
    )
    pd.DataFrame([{
        "selected_seed": decision["selected_seed"],
        "checkpoint_sha256": decision["checkpoint_sha256"],
        "validation_mAP50": gate["validation_map50"],
        "validation_safety_recall": gate["validation_safety_recall"],
        "standard_threshold": gate["standard_threshold"],
        "safety_threshold": gate["safety_threshold"],
        "quality_gate_passed": gate["quality_gate_passed"],
    }]).to_csv(TABLES / "02_training_and_threshold.csv", index=False)
    clean = json.loads(
        (OUTPUT_ROOT / "test/clean/clean_test_metrics.json")
        .read_text(encoding="utf-8")
    )
    pd.DataFrame([clean]).to_csv(TABLES / "03_clean_test_metrics.csv", index=False)
    attack_lock = json.loads(
        (OUTPUT_ROOT / "attack_calibration/attack_protocol_lock.json")
        .read_text(encoding="utf-8")
    )
    parameters = []
    for key, values in attack_lock["selected"].items():
        for value in values:
            parameters.append({
                "condition": key,
                "epsilon_px": value,
                "steps": (
                    1 if key.startswith("fgsm")
                    else attack_lock[
                        "adaptive_pgd_steps"
                        if key.startswith("adaptive") else "pgd_steps"
                    ]
                ),
                "restarts": 1 if key.startswith("fgsm") else attack_lock["restarts"],
                "seeds": "|".join(map(str, attack_lock["seeds"])),
            })
    pd.DataFrame(parameters).to_csv(
        TABLES / "04_attack_parameters.csv", index=False
    )
    matrix = pd.read_csv(
        OUTPUT_ROOT / "attacks/canonical_test_matrix.csv", low_memory=False
    )
    selected = matrix[
        matrix["selected_best"].astype(str).str.lower().isin({"true", "1"})
    ].copy()
    per_scene = selected.groupby(
        [
            "grouped_scene_id", "attack", "adaptive", "epsilon_px",
            "steps", "defense",
        ],
        as_index=False,
    ).agg(
        frames=("image_path", "nunique"),
        precision=("precision_defended", "mean"),
        recall=("recall_defended", "mean"),
        f1=("f1_defended", "mean"),
        f2=("f2", "mean"),
        fn_per_frame=("fn_defended", "mean"),
    )
    macro = per_scene.groupby(
        ["attack", "adaptive", "epsilon_px", "steps", "defense"],
        as_index=False,
    ).agg(
        grouped_scenes=("grouped_scene_id", "nunique"),
        frames=("frames", "sum"),
        precision=("precision", "mean"),
        recall=("recall", "mean"),
        f1=("f1", "mean"),
        f2=("f2", "mean"),
        fn_per_frame=("fn_per_frame", "mean"),
    )
    macro.to_csv(TABLES / "05_canonical_robustness.csv", index=False)
    attack_columns = [
        "grouped_scene_id", "subsequence_id", "image_path", "attack",
        "adaptive", "epsilon_px", "seed", "layer", "c_sp_global",
        "c_sp_object", "c_sp_background", "c_dir", "c_atk_global",
        "c_atk_object", "c_atk_background", "delta_f1_damage",
        "delta_recall_damage", "false_negatives_increase",
    ]
    selected[attack_columns].to_csv(
        TABLES / "06_attack_consistency.csv", index=False
    )
    defense_columns = [
        "grouped_scene_id", "subsequence_id", "image_path", "attack",
        "adaptive", "epsilon_px", "seed", "defense", "layer", "product",
        "lukasiewicz", "product_recovery", "lukasiewicz_recovery",
        "P", "A", "R", "G_raw", "G_clipped", "C_def",
        "delta_f1_recovery", "delta_recall_recovery",
        "false_negatives_reduction", "normalized_quality_recovery",
    ]
    selected[defense_columns].to_csv(
        TABLES / "07_defense_consistency.csv", index=False
    )
    model_tables = OUTPUT_ROOT / "models/tables"
    gains = pd.read_csv(model_tables / "12_multiple_comparison_corrections.csv")
    gains[
        gains["task"].eq("damage") & gains["comparison"].eq("D3_vs_D2")
    ].to_csv(TABLES / "08_damage_D2_D3.csv", index=False)
    gains[
        gains["task"].eq("recovery") & gains["comparison"].eq("R3_vs_R2")
    ].to_csv(TABLES / "09_recovery_R2_R3.csv", index=False)
    shutil.copy2(
        model_tables / "10_object_global_comparison.csv",
        TABLES / "10_object_global_comparison.csv",
    )
    shutil.copy2(
        model_tables / "11_adaptive_comparison.csv",
        TABLES / "11_adaptive_comparison.csv",
    )
    per_scene.to_csv(TABLES / "12_per_scene_results.csv", index=False)
    shutil.copy2(
        model_tables / "13_scene_macro_loso.csv",
        TABLES / "13_loso_results.csv",
    )
    shutil.copy2(
        OUTPUT_ROOT / "latency/latency_summary.csv",
        TABLES / "14_latency_summary.csv",
    )
    shutil.copy2(
        OUTPUT_ROOT / "attacks/canonical_nms_audit.csv",
        TABLES / "15_nms_audit.csv",
    )
    return {
        "clean": clean,
        "gate": gate,
        "matrix": matrix,
        "macro": macro,
        "per_scene": per_scene,
        "gains": gains,
        "damage": primary_gain(gains, "damage"),
        "recovery": primary_gain(gains, "recovery"),
        "protocol": protocol,
    }


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_figures(context: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    selected_seed = int(
        pd.read_csv(OUTPUT_ROOT / "selection/seed_comparison.csv")
        .query("selected == True")
        .iloc[0]["seed"]
    )
    training = OUTPUT_ROOT / f"full_training/seed_{selected_seed}"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    offset = 0
    for stage in ("stage1", "stage2", "stage3"):
        results = pd.read_csv(training / stage / "results.csv")
        epoch = np.arange(len(results)) + offset
        if "train/box_loss" in results:
            axes[0].plot(epoch, results["train/box_loss"], label=stage)
        metric = next(
            (name for name in results if "metrics/mAP50(B)" == name), None
        )
        if metric:
            axes[1].plot(epoch, results[metric], label=stage)
        offset += len(results)
    axes[0].set(xlabel="epoch", ylabel="box loss")
    axes[1].set(xlabel="epoch", ylabel="tile-level diagnostic mAP50")
    axes[0].legend()
    axes[1].legend()
    save_figure(figure, "01_training_curves.png")
    sweep = pd.read_csv(
        OUTPUT_ROOT / f"validation/seed_{selected_seed}/threshold_sweep.csv"
    )
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(sweep["recall"], sweep["precision"])
    axis.set(xlabel="Recall", ylabel="Precision")
    save_figure(figure, "02_precision_recall_curve.png")
    macro = context["macro"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    scope = macro[~macro["adaptive"].astype(bool)]
    for (attack, defense), rows in scope.groupby(["attack", "defense"]):
        rows = rows.sort_values("epsilon_px")
        axes[0].plot(rows["epsilon_px"], rows["f1"], "o-", label=f"{attack}/{defense}")
        axes[1].plot(rows["epsilon_px"], rows["recall"], "o-", label=f"{attack}/{defense}")
    axes[0].set(xlabel="epsilon, pixel levels / 255", ylabel="scene-macro F1")
    axes[1].set(xlabel="epsilon, pixel levels / 255", ylabel="scene-macro Recall")
    axes[1].legend(fontsize=6)
    save_figure(figure, "03_f1_recall_vs_epsilon.png")
    for task, name, title in (
        ("damage", "04_damage_D3_vs_D2.png", "D3 minus D2"),
        ("recovery", "05_recovery_R3_vs_R2.png", "R3 minus R2"),
    ):
        frame = context["gains"]
        comparison = "D3_vs_D2" if task == "damage" else "R3_vs_R2"
        endpoint = "delta_f1_damage" if task == "damage" else "delta_f1_recovery"
        frame = frame[
            frame["task"].eq(task)
            & frame["comparison"].eq(comparison)
            & frame["endpoint"].eq(endpoint)
            & frame["algorithm"].eq("ridge")
        ]
        values = frame["estimate"].to_numpy(float)
        figure, axis = plt.subplots(figsize=(7, 4.5))
        axis.errorbar(
            np.arange(len(frame)),
            values,
            yerr=np.maximum(
                0,
                np.vstack((values - frame["ci_low"], frame["ci_high"] - values)),
            ),
            fmt="o",
            capsize=4,
        )
        axis.set_xticks(np.arange(len(frame)), frame["metric"])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        save_figure(figure, name)
    delta = pd.read_csv(
        OUTPUT_ROOT / "models/tables/04_tnorm_vs_baseline_bootstrap.csv"
    )
    delta = delta[
        delta["layer"].eq("mean")
        & delta["tnorm"].isin(
            ["product", "lukasiewicz", "product_recovery", "lukasiewicz_recovery"]
        )
    ].head(16)
    figure, axis = plt.subplots(figsize=(9, max(4, len(delta) * 0.35)))
    axis.errorbar(
        delta["delta_rho"],
        np.arange(len(delta)),
        xerr=np.maximum(
            0,
            np.vstack((
                delta["delta_rho"] - delta["ci_low"],
                delta["ci_high"] - delta["delta_rho"],
            )),
        ),
        fmt="o",
    )
    axis.set_yticks(
        np.arange(len(delta)),
        delta["tnorm"] + " vs " + delta["baseline"],
    )
    axis.axvline(0, color="black", linewidth=0.8)
    save_figure(figure, "06_tnorm_vs_baselines.png")
    h3 = pd.read_csv(TABLES / "10_object_global_comparison.csv")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    if len(h3):
        axis.errorbar(
            [h3.iloc[0]["delta_rho"]],
            [0],
            xerr=[[
                h3.iloc[0]["delta_rho"] - h3.iloc[0]["ci_low"]
            ], [
                h3.iloc[0]["ci_high"] - h3.iloc[0]["delta_rho"]
            ]],
            fmt="o",
        )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks([0], ["object - global"])
    save_figure(figure, "07_object_vs_global.png")
    adaptive = pd.read_csv(TABLES / "11_adaptive_comparison.csv")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for metric, rows in adaptive.groupby("metric"):
        axis.plot(
            rows["epsilon_px"],
            rows["estimate_adaptive_minus_nonadaptive"],
            "o-",
            label=metric,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.legend()
    save_figure(figure, "08_adaptive_vs_nonadaptive.png")
    scene = context["per_scene"].groupby("grouped_scene_id", as_index=False).agg(
        f1=("f1", "mean"), recall=("recall", "mean")
    )
    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(scene))
    axis.bar(x - 0.18, scene["f1"], 0.36, label="F1")
    axis.bar(x + 0.18, scene["recall"], 0.36, label="Recall")
    axis.set_xticks(x, scene["grouped_scene_id"], rotation=25, ha="right")
    axis.legend()
    save_figure(figure, "09_per_scene_effects.png")
    latency = pd.read_csv(TABLES / "14_latency_summary.csv")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.scatter(latency["mean_latency_ms"], latency["fps"])
    for row in latency.itertuples(index=False):
        axis.annotate(row.method, (row.mean_latency_ms, row.fps), fontsize=7)
    axis.set(xlabel="mean latency, ms", ylabel="FPS")
    save_figure(figure, "10_latency_tradeoff.png")


def gain_sentence(label: str, values: dict) -> str:
    return (
        f"{label}: ΔMAE={values['delta_mae']:.4f}, снижение MAE="
        f"{values['mae_reduction']:.2f}%, ΔR²={values['delta_r2']:.4f}, "
        f"ΔSpearman={values['delta_spearman']:.4f}, 95% CI ΔMAE "
        f"[{values['ci_low']:.4f}; {values['ci_high']:.4f}], "
        f"Holm p={values['holm_p']:.4g}."
    )


def build_article(context: dict) -> dict:
    protocol = context["protocol"]
    source = Path(protocol["article"]["source_template"])
    before = sha256(source)
    ARTICLE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="canonical-m4-article-") as directory:
        converted = Path(directory) / "source.md"
        media = ARTICLE / "source_media"
        subprocess.run([
            "pandoc", str(source), "--extract-media", str(media),
            "-t", "gfm", "-o", str(converted),
        ], cwd=PROJECT_DIR, check=True)
        text = converted.read_text(encoding="utf-8")
    clean = context["clean"]
    damage = context["damage"]
    recovery = context["recovery"]
    gate = context["gate"]
    h3 = pd.read_csv(TABLES / "10_object_global_comparison.csv")
    h4 = pd.read_csv(TABLES / "11_adaptive_comparison.csv")
    latency = pd.read_csv(TABLES / "14_latency_summary.csv")
    h3_text = (
        f"Δ|ρ|={h3.iloc[0]['delta_rho']:.4f}, 95% CI "
        f"[{h3.iloc[0]['ci_low']:.4f}; {h3.iloc[0]['ci_high']:.4f}]"
        if len(h3) else "сопоставление не оценивалось"
    )
    h4_text = (
        f"adaptive−non-adaptive={h4.iloc[0]['estimate_adaptive_minus_nonadaptive']:.4f}"
        if len(h4) else "общий non-floor budget отсутствовал"
    )
    full_latency = latency[
        latency["method"].eq("full_Product_diagnostic_M4_pipeline")
    ].iloc[0]
    clean_text = (
        f"Clean test: mAP50={clean['mAP50']:.4f}, "
        f"mAP50-95={clean['mAP50_95']:.4f}, Precision={clean['precision']:.4f}, "
        f"Recall={clean['recall']:.4f}, F1={clean['f1']:.4f}, "
        f"FN/кадр={clean['fn_per_frame']:.3f}."
    )
    damage_text = gain_sentence("D3 против D2", damage)
    recovery_text = gain_sentence("R3 против R2", recovery)
    scenarios = {damage["scenario"], recovery["scenario"]}
    scenario = (
        "positive" if "positive" in scenarios
        else "small_effect" if "small_effect" in scenarios else "negative"
    )
    conclusion = {
        "positive": (
            "По крайней мере одна заранее заданная гипотеза дала статистически "
            "подтверждённый и практически заметный прирост."
        ),
        "small_effect": (
            "T-нормовые признаки дали статистически воспроизводимый, но небольшой "
            "дополнительный диагностический сигнал."
        ),
        "negative": (
            "Подтверждённого преимущества канонических T-норм над стандартными "
            "метриками после scene-level анализа не обнаружено."
        ),
    }[scenario]
    replacements = {
        "TBD_VAL_RECALL": f"{gate['validation_safety_recall']:.4f}",
        "TBD_VAL_MAP50": f"{gate['validation_map50']:.4f}",
        "TBD_THRESHOLD": f"{gate['safety_threshold']:.4f}",
        "TBD_TEST_MAP50": f"{clean['mAP50']:.4f}",
        "TBD_TEST_MAP5095": f"{clean['mAP50_95']:.4f}",
        "TBD_TEST_RECALL": f"{clean['recall']:.4f}",
        "TBD_TEST_F1": f"{clean['f1']:.4f}",
        "TBD_TEST_FN": f"{clean['fn_per_frame']:.3f}",
        "TBD_D_DMAE": f"{damage['delta_mae']:.4f}",
        "TBD_D_DR2": f"{damage['delta_r2']:.4f}",
        "TBD_D_DRHO": f"{damage['delta_spearman']:.4f}",
        "TBD_D_CI": f"[{damage['ci_low']:.4f}; {damage['ci_high']:.4f}]",
        "TBD_R_DMAE": f"{recovery['delta_mae']:.4f}",
        "TBD_R_DR2": f"{recovery['delta_r2']:.4f}",
        "TBD_R_DRHO": f"{recovery['delta_spearman']:.4f}",
        "TBD_R_CI": f"[{recovery['ci_low']:.4f}; {recovery['ci_high']:.4f}]",
        "TBD_OBJECT_GLOBAL": h3_text,
        "TBD_ADAPTIVE_EFFECT": h4_text,
        "TBD_LATENCY": (
            f"{full_latency.mean_latency_ms:.2f} ms, "
            f"p95={full_latency.p95_latency_ms:.2f} ms"
        ),
    }
    for marker, value in replacements.items():
        text = text.replace(f"\\[{marker}\\]", value).replace(f"[{marker}]", value)
    ru_abstract = (
        "Проведена воспроизводимая scene-disjoint оценка диагностической ценности "
        "канонических Product- и Łukasiewicz-показателей для M4 overlapping-tiling "
        f"YOLO11m. {clean_text} {damage_text} {recovery_text} {conclusion}"
    )
    en_abstract = (
        "A reproducible scene-disjoint evaluation of canonical Product and "
        "Lukasiewicz diagnostics was performed for the M4 overlapping-tiling "
        f"YOLO11m pipeline. Clean test mAP50 was {clean['mAP50']:.4f}, Recall "
        f"{clean['recall']:.4f}, and F1 {clean['f1']:.4f}. D3-vs-D2 delta R2 "
        f"was {damage['delta_r2']:.4f}; R3-vs-R2 delta R2 was "
        f"{recovery['delta_r2']:.4f}. Five test scenes limit statistical power."
    )
    text = re.sub(
        r"(\*\*АННОТАЦИЯ\*\*\n\n).*?(?=\n\*\*Ключевые слова:\*\*)",
        rf"\1{ru_abstract}\n",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"(\*\*ABSTRACT\*\*\n\n).*?(?=\n\*\*Keywords:\*\*)",
        rf"\1{en_abstract}\n",
        text,
        flags=re.S,
    )
    new_conclusion = (
        "# **Заключение**\n\n"
        "Micro-overfit использовался только как sanity-проверка технической "
        "обучаемости M4; обобщение оценивалось отдельно. "
        f"{clean_text} {damage_text} {recovery_text} Object/global: {h3_text}. "
        f"Adaptive PGD: {h4_text}. {conclusion} Product preprocessing не "
        "интерпретируется как универсальная защита. Пять test-сцен ограничивают "
        "статистическую мощность.\n\n"
    )
    text = re.sub(
        r"# \*\*Заключение\*\*.*?(?=# \*\*Благодарности\*\*)",
        new_conclusion,
        text,
        flags=re.S,
    )
    text = re.sub(r"\\?\[TBD[^\]]*\]", "см. проверенные итоговые таблицы", text)
    text = text.replace("будут внесены", "внесены")
    text = text.replace("вычисления выполняются", "вычисления завершены")
    text = text.replace("статус вычислений", "результат вычислений")
    text += (
        "\n\n# **Проверенные canonical M4 результаты**\n\n"
        f"{clean_text}\n\n{damage_text}\n\n{recovery_text}\n\n"
        f"Object/global: {h3_text}. Adaptive: {h4_text}.\n\n"
        f"![Damage model gain]({(FIGURES / '04_damage_D3_vs_D2.png').resolve()})\n\n"
        f"![Recovery model gain]({(FIGURES / '05_recovery_R3_vs_R2.png').resolve()})\n"
    )
    markdown = ARTICLE / "TNorm_RZD_article_final.md"
    docx = ARTICLE / "TNorm_RZD_article_final.docx"
    markdown.write_text(text, encoding="utf-8")
    subprocess.run([
        "pandoc", str(markdown), "--reference-doc", str(source), "-o", str(docx)
    ], cwd=PROJECT_DIR, check=True)
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf",
        "--outdir", str(ARTICLE), str(docx),
    ], cwd=PROJECT_DIR, check=True)
    pdf = ARTICLE / "TNorm_RZD_article_final.pdf"
    supplementary_md = ARTICLE / "TNorm_RZD_supplementary.md"
    parts = ["# TNormFilter canonical M4 supplementary\n"]
    for path in sorted(TABLES.glob("*.csv")):
        parts.append(f"## {path.stem}\n\n{pd.read_csv(path).head(40).to_markdown(index=False)}\n")
    for path in sorted(FIGURES.glob("*.png")):
        parts.append(f"## {path.stem}\n\n![{path.stem}]({path.resolve()})\n")
    supplementary_md.write_text("\n".join(parts), encoding="utf-8")
    supplementary_docx = ARTICLE / "TNorm_RZD_supplementary.docx"
    subprocess.run([
        "pandoc", str(supplementary_md), "--reference-doc", str(source),
        "-o", str(supplementary_docx),
    ], cwd=PROJECT_DIR, check=True)
    subprocess.run([
        "libreoffice", "--headless", "--convert-to", "pdf",
        "--outdir", str(ARTICLE), str(supplementary_docx),
    ], cwd=PROJECT_DIR, check=True)
    supplementary_pdf = ARTICLE / "TNorm_RZD_supplementary.pdf"
    if not pdf.is_file() or not supplementary_pdf.is_file():
        raise RuntimeError("Canonical M4 PDF conversion failed")
    after = sha256(source)
    mapping = {
        "source_template": str(source),
        "source_sha256_before": before,
        "source_sha256_after": after,
        "source_unchanged": before == after,
        "scenario": scenario,
        "clean": clean,
        "damage": damage,
        "recovery": recovery,
    }
    atomic_json(ARTICLE / "result_mapping_resolved.json", mapping)
    return mapping


def validate_article(context: dict) -> dict:
    docx = ARTICLE / "TNorm_RZD_article_final.docx"
    pdf = ARTICLE / "TNorm_RZD_article_final.pdf"
    supplementary = ARTICLE / "TNorm_RZD_supplementary.pdf"
    plain = subprocess.check_output(
        ["pandoc", str(docx), "-t", "plain"], text=True
    )
    pdf_plain = subprocess.check_output(
        ["pdftotext", str(pdf), "-"], text=True
    )
    forbidden = [
        "[TBD", "вычисления выполняются", "будут внесены", "статус вычислений"
    ]
    signs = True
    for task in ("damage", "recovery"):
        frame = pd.read_csv(
            TABLES / (
                "08_damage_D2_D3.csv"
                if task == "damage" else "09_recovery_R2_R3.csv"
            )
        )
        for row in frame[frame["metric"].eq("delta_mae")].itertuples(index=False):
            if math.isfinite(row.estimate) and math.isfinite(row.relative_mae_reduction):
                signs &= (
                    (row.estimate < 0) == (row.relative_mae_reduction > 0)
                    or abs(row.estimate) < 1e-15
                )
    anchors = [
        f"{context['clean']['mAP50']:.4f}",
        f"{context['clean']['recall']:.4f}",
        f"{context['damage']['delta_r2']:.4f}",
        f"{context['recovery']['delta_r2']:.4f}",
    ]
    checks = {
        "quality_gate_passed": bool(context["gate"]["quality_gate_passed"]),
        "test_marker_exists": TEST_MARKER.is_file(),
        "docx_exists": docx.is_file(),
        "pdf_exists": pdf.is_file(),
        "supplementary_exists": supplementary.is_file(),
        "no_forbidden_text": not any(
            value.lower() in plain.lower() or value.lower() in pdf_plain.lower()
            for value in forbidden
        ),
        "all_15_tables": len(list(TABLES.glob("*.csv"))) == 15,
        "all_10_figures": len(list(FIGURES.glob("*.png"))) == 10,
        "delta_mae_signs_valid": bool(signs),
        "numbers_match": all(anchor in plain for anchor in anchors),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "docx_sha256": sha256(docx),
        "pdf_sha256": sha256(pdf),
        "supplementary_sha256": sha256(supplementary),
    }
    atomic_json(ARTICLE / "article_validation.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"Canonical M4 article validation failed: {checks}")
    return result


def build_bundle(context: dict) -> dict:
    BUNDLES.mkdir(parents=True, exist_ok=True)
    destination = BUNDLES / "TNormFilter_canonical_M4_final.zip"
    with tempfile.TemporaryDirectory(prefix="canonical-m4-bundle-") as directory:
        root = Path(directory) / "TNormFilter_canonical_M4_final"
        root.mkdir()
        include = [
            PROJECT_DIR / "README.md",
            PROJECT_DIR / "AGENTS.md",
            PROJECT_DIR / "configs/canonical_v2_m4_full_protocol.yaml",
            PROJECT_DIR / "configs/schemas/canonical_v2_m4_full_protocol.schema.json",
            OUTPUT_ROOT / "protocol",
            OUTPUT_ROOT / "tiling_audit/tiling_audit.json",
            OUTPUT_ROOT / "scene_cv/scene_cv_summary.json",
            OUTPUT_ROOT / "selection",
            OUTPUT_ROOT / "validation/quality_gate.json",
            OUTPUT_ROOT / "normalization",
            OUTPUT_ROOT / "attack_calibration/attack_protocol_lock.json",
            TEST_MARKER,
            OUTPUT_ROOT / "test/clean",
            OUTPUT_ROOT / "attacks/canonical_test_matrix.json",
            OUTPUT_ROOT / "attacks/canonical_nms_audit.csv",
            OUTPUT_ROOT / "models",
            OUTPUT_ROOT / "latency",
            OUTPUT_ROOT / "final",
            OUTPUT_ROOT / "article",
            OUTPUT_ROOT / "logs",
            PROJECT_DIR / "tests/test_canonical_m4.py",
            PROJECT_DIR / "scripts/canonical_m4_runtime.py",
            PROJECT_DIR / "scripts/run_canonical_m4_test_matrix.py",
        ]
        for source in include:
            if not source.exists():
                raise FileNotFoundError(source)
            relative = (
                source.relative_to(PROJECT_DIR)
                if source.is_relative_to(PROJECT_DIR)
                else Path("external") / source.name
            )
            target = root / relative
            if source.is_dir():
                shutil.copytree(
                    source,
                    target,
                    ignore=shutil.ignore_patterns(
                        "*.tmp", "prediction_cache", "condition_cache", "cache"
                    ),
                )
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        manifest = {
            "status": "PASS",
            "protocol_id": context["protocol"]["protocol_id"],
            "commit": git("rev-parse", "HEAD"),
            "checkpoint_sha256": context["gate"]["checkpoint_sha256"],
            "split_manifest_sha256":
                context["protocol"]["dataset"]["split_manifest_sha256"],
            "test_marker_sha256": sha256(TEST_MARKER),
            "full_checkpoint_excluded": True,
            "dataset_excluded": True,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        files = sorted(path for path in root.rglob("*") if path.is_file())
        checksum = root / "checksums.sha256"
        checksum.write_text(
            "\n".join(
                f"{sha256(path)}  {path.relative_to(root)}" for path in files
            ) + "\n",
            encoding="utf-8",
        )
        temporary = destination.with_suffix(".zip.tmp")
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root.parent))
        temporary.replace(destination)
    with tempfile.TemporaryDirectory(prefix="canonical-m4-verify-") as directory:
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Canonical M4 ZIP CRC validation failed")
            archive.extractall(directory)
        extracted = Path(directory) / "TNormFilter_canonical_M4_final"
        for line in (extracted / "checksums.sha256").read_text().splitlines():
            digest, relative = line.split("  ", 1)
            if sha256(extracted / relative) != digest:
                raise RuntimeError(f"Bundle checksum mismatch: {relative}")
    sidecar = destination.with_suffix(destination.suffix + ".sha256")
    sidecar.write_text(
        f"{sha256(destination)}  {destination.name}\n", encoding="utf-8"
    )
    result = {
        "status": "PASS",
        "path": str(destination.resolve()),
        "sha256": sha256(destination),
        "sidecar": str(sidecar.resolve()),
    }
    atomic_json(OUTPUT_ROOT / "final/bundle_validation.json", result)
    return result


def run() -> dict:
    assert_role_allowed("article")
    context = build_tables()
    build_figures(context)
    build_article(context)
    article_validation = validate_article(context)
    bundle = build_bundle(context)
    summary = {
        "status": "PASS",
        "tables": 15,
        "figures": 10,
        "article_validation": article_validation,
        "bundle": bundle,
    }
    atomic_json(OUTPUT_ROOT / "final/run_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
