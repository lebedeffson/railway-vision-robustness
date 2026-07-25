from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "outputs/person_v8b/final"
ARTICLE = FINAL / "article"
REFERENCE_DOCX = ROOT / "article/internal_review_template.docx"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_table(frame: pd.DataFrame, digits: int = 5) -> str:
    copy = frame.copy()
    for column in copy.select_dtypes(include="number").columns:
        copy[column] = copy[column].map(
            lambda value: f"{value:.{digits}f}" if pd.notna(value) else "—"
        )
    columns = [str(column) for column in copy.columns]
    rows = [
        [
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        for row in copy.itertuples(index=False, name=None)
    ]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(values) + " |"
        for values in rows
    ]
    return "\n".join([header, divider, *body])


def model_row(comparison: pd.DataFrame, model: str, estimator: str = "Ridge") -> Any:
    return comparison[
        comparison["model"].eq(model)
        & comparison["estimator"].eq(estimator)
    ].iloc[0]


def hierarchy_row(
    hierarchy: pd.DataFrame, candidate: str, reference: str, metric: str
) -> Any:
    return hierarchy[
        hierarchy["candidate"].eq(candidate)
        & hierarchy["reference"].eq(reference)
        & hierarchy["metric"].eq(metric)
    ].iloc[0]


def build_markdown() -> str:
    metrics = json.loads((FINAL / "FINAL_METRICS.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(FINAL / "MODEL_COMPARISON.csv")
    hierarchy = pd.read_csv(FINAL / "BOOTSTRAP_FINAL.csv")
    scene = pd.read_csv(FINAL / "tables/TABLE_4_PER_SCENE.csv")
    calibration = pd.read_csv(FINAL / "CALIBRATION_FINAL.csv")
    coverage = pd.read_csv(FINAL / "RISK_COVERAGE_FINAL.csv")
    loso = pd.read_csv(FINAL / "LOSO_FINAL.csv")

    u0 = model_row(comparison, "U0")
    u1 = model_row(comparison, "U1")
    u2 = model_row(comparison, "U2")
    u3 = model_row(comparison, "U3")
    u2_elastic = model_row(comparison, "U2", "ElasticNet")
    u3_elastic = model_row(comparison, "U3", "ElasticNet")
    primary = hierarchy_row(hierarchy, "U3", "U2", "mae_fn")
    u1_step = hierarchy_row(hierarchy, "U1", "U0", "mae_fn")
    u2_step = hierarchy_row(hierarchy, "U2", "U1", "mae_fn")
    recall_step = hierarchy_row(hierarchy, "U3", "U2", "mae_recall")
    elastic_reduction = (
        (u2_elastic.scene_macro_mae_fn - u3_elastic.scene_macro_mae_fn)
        / u2_elastic.scene_macro_mae_fn
        * 100.0
    )

    primary_table = comparison[
        comparison["model"].isin(["U0", "U1", "U2", "U3"])
        & comparison["estimator"].eq("Ridge")
    ][
        [
            "model",
            "scene_macro_mae_fn",
            "scene_macro_r2_fn",
            "scene_macro_abs_spearman_fn",
            "scene_macro_mae_recall",
        ]
    ].rename(
        columns={
            "scene_macro_mae_fn": "MAE FN/frame",
            "scene_macro_r2_fn": "R² FN",
            "scene_macro_abs_spearman_fn": "|Spearman| FN",
            "scene_macro_mae_recall": "MAE Recall",
        }
    )
    sensitivity_table = comparison[
        comparison["model"].isin(["U2", "U3"])
    ][
        [
            "model",
            "estimator",
            "scene_macro_mae_fn",
            "frame_weighted_mae_fn",
            "median_scene_mae_fn",
            "trimmed_mean_scene_mae_fn",
        ]
    ].rename(
        columns={
            "scene_macro_mae_fn": "Macro MAE",
            "frame_weighted_mae_fn": "Frame-weighted MAE",
            "median_scene_mae_fn": "Median scene MAE",
            "trimmed_mean_scene_mae_fn": "10% trimmed mean MAE",
        }
    )
    calibration_table = calibration.rename(
        columns={
            "scene_macro_brier": "Brier",
            "scene_macro_ece": "ECE",
            "scene_macro_auroc": "AUROC",
            "scene_macro_auprc": "AUPRC",
        }
    )

    content = f"""---
title: "T-Norm Representation Consistency Does Not Improve Frozen-Detector Failure-Risk Estimation in Railway Person Detection"
author:
  - "Yuri V. Trofimov"
  - "Alexey N. Averkin"
  - "Mikhail D. Lebedev"
  - "Alexander D. Lebedev"
date: "2026"
lang: en
---

# Аннотация

В работе представлен воспроизводимый протокол проверки дополнительной
диагностической ценности Product- и Łukasiewicz-признаков для прогноза
пропусков людей железнодорожным детектором. Замороженный B0-детектор
оценивался на 1 085 кадрах из 15 сгруппированных development-сцен.
Risk-модели сравнивались иерархически: от confidence-only U0 до U3,
добавляющей T-нормовую согласованность к detector statistics и стандартным
representation distances. Все статистики нормализации, прототипы и
гиперпараметры переоценивались только на train-сценах nested LOSO.
Первичным endpoint была scene-macro MAE прогноза FN/frame. U2 получила
MAE {u2.scene_macro_mae_fn:.5f}, U3 — {u3.scene_macro_mae_fn:.5f};
относительное изменение составило {metrics['relative_MAE_reduction_percent']:.2f}%,
а U3 выиграла только на {metrics['scene_wins']}/15 сценах. Парный
scene-bootstrap 95% CI для MAE(U3)−MAE(U2) равен
[{primary.ci95_low:.5f}; {primary.ci95_high:.5f}]. Следовательно,
дополнительная ценность T-норм не подтверждена. Test не открывался.

**Ключевые слова:** железнодорожное компьютерное зрение, обнаружение людей,
T-нормы, оценка риска отказа, selective prediction, отрицательный результат,
scene-level validation.

# Abstract

We present a reproducible protocol for testing whether Product and
Łukasiewicz representation-consistency features add failure-risk information
beyond detector outputs and conventional representation distances. A frozen
person detector was evaluated on 1,085 frames from 15 grouped development
scenes. Risk estimators were compared hierarchically from confidence-only U0
to U3, which augments U2 with T-norm features. Normalization, reference
prototypes, calibration, and hyperparameter selection were refit exclusively
inside each nested leave-one-scene-out training portion. The prespecified
primary endpoint was equal-weight scene-macro mean absolute error (MAE) for
FN/frame. U2 achieved {u2.scene_macro_mae_fn:.5f}, whereas U3 achieved
{u3.scene_macro_mae_fn:.5f}, corresponding to a
{abs(metrics['relative_MAE_reduction_percent']):.2f}% deterioration rather
than improvement. U3 won on {metrics['scene_wins']} of 15 scenes, and the
paired scene-bootstrap 95% confidence interval for MAE(U3)−MAE(U2) was
[{primary.ci95_low:.5f}, {primary.ci95_high:.5f}]. Small descriptive
calibration gains did not rescue the failed primary endpoint. The sealed test
set was not opened. The result supports hierarchical, scene-level validation
before novel feature metrics are used in safety monitoring.

**Keywords:** railway vision, person detection, T-norm, failure-risk
estimation, selective prediction, negative result, scene-level validation.

# 1. Introduction

Railway perception systems operate under strong scene shifts caused by camera
placement, station geometry, illumination, vegetation, infrastructure, and
the small apparent size of distant people. A detector can therefore perform
well on familiar sequences while failing on a new grouped scene. In a
safety-oriented pipeline, a complementary risk estimator may be useful even
when the detector itself cannot be improved: frames estimated to be unsafe
can be referred to an operator or trigger a conservative fallback.

This study asks a narrow incremental-value question. After detector outputs
and conventional representation distances are available, do Product and
Łukasiewicz consistency features improve the prediction of detector false
negatives? We do not treat T-norms as a defense, and we do not infer benefit
from a single correlation. Instead, we compare nested feature sets under
scene-disjoint validation and a prespecified primary endpoint.

The work contributes (i) a frozen-detector, nested scene-LOSO protocol;
(ii) a hierarchical U0–U3 comparison that isolates incremental feature value;
(iii) paired scene-level bootstrap, leave-one-scene-out sensitivity, and
risk–coverage analysis; and (iv) a fully reported negative result. This
emphasis is important because domain-generalization studies have repeatedly
shown that additional machinery does not reliably outperform carefully
controlled empirical-risk baselines [3,4].

# 2. Related work

OSDaR23 is a synchronized multi-sensor railway dataset comprising RGB, IR,
LiDAR, radar, and vehicle-state measurements across 45 subsequences [1].
Subsequences derived from the same operational setting are not independent;
our statistical unit is therefore the grouped scene.

Selective prediction pairs a predictive model with a rejection or referral
rule and studies risk as a function of coverage [2]. In our setting, the
detector remains unchanged. U0–U3 estimate the risk that its frame-level
output contains missed people. Risk–coverage curves are descriptive because
the confirmatory endpoint concerns FN/frame regression.

A T-norm is a monotone associative operation on membership values in [0,1].
We use the Product and Łukasiewicz forms only after train-only quantile
normalization. Their role is to summarize agreement between current
P3/P4/P5 activations and train-reference prototypes. They are tested as
additional covariates, not assumed to be intrinsically superior to cosine,
L1/L2, MAE/MSE, correlation, or entropy-shift features.

# 3. Materials and methods

## 3.1 Data, detector, and outcome

The development pool contains 1,085 RGB frames from 15 independent grouped
scenes and targets the `person` class. The frozen B0 YOLO11m checkpoint
produces detections and P3/P4/P5 representations. The primary target is
FN/frame. Recall is secondary. An unsafe frame has at least one false negative
or, when ground truth is present, Recall below 0.50.

The detector itself had been trained on 12 of these 15 development scenes.
Consequently, outer LOSO evaluates generalization of the risk estimator, not
uniformly out-of-fold detector training. This limitation was disclosed and
frozen before feature extraction.

## 3.2 Feature hierarchy

The hierarchy was fixed before evaluation:

| Model | Feature set |
|---|---|
| U0 | mean and maximum detector confidence |
| U1 | U0 plus prediction-count, entropy, low-confidence, area, and small-proposal statistics |
| U2 | U1 plus cosine, L1/L2, MSE/MAE, Pearson, entropy shift, and activation norms |
| U3 | U2 plus Product and Łukasiewicz consistency for global, proposal, and background regions |

For membership vectors μ and reference r, Product and Łukasiewicz fuzzy
Jaccard scores were calculated as:

$$
J_{{Product}}(\\mu,r)=
\\frac{{\\sum_i \\mu_i r_i}}
{{\\sum_i(\\mu_i+r_i-\\mu_i r_i)+\\tau}},
$$

$$
J_{{Luk}}(\\mu,r)=
\\frac{{\\sum_i \\max(0,\\mu_i+r_i-1)}}
{{\\sum_i \\min(1,\\mu_i+r_i)+\\tau}}.
$$

Proposal regions are deployable B0 proposal regions. Ground-truth person
regions were reserved as oracle-only diagnostics and excluded from U0–U3.

## 3.3 Nested scene-level validation

Each of 15 outer folds held out one grouped scene. Quantile normalization,
scene-balanced prototypes, Ridge regularization, logistic hyperparameters,
and isotonic calibration were recomputed using only the remaining 14 scenes.
Hyperparameters were selected by an inner scene-LOSO loop. Frame rows were
used for fitting, but all confirmatory metrics were first calculated by scene
and then equally averaged.

Ridge regression was the primary model; ElasticNet was a frozen sensitivity
analysis. Unsafe-frame classification used logistic regression followed by
train-only isotonic calibration. Pairwise differences were resampled at the
scene level for 10,000 paired bootstrap iterations. Holm correction covered
the fixed U0→U1, U1→U2, and U2→U3 comparison family.

## 3.4 Confirmatory rule

U3 could pass only if all of the following held: at least 5% scene-macro MAE
reduction versus U2; an upper paired-bootstrap confidence bound below zero;
wins on at least 10 of 15 scenes; no negative leave-one-scene-out reduction;
and noninferior Brier score and ECE. Secondary outcomes could not rescue a
primary failure. The test set remained physically sealed unless this
development gate passed.

![Canonical v8b pipeline](figures/FIGURE_1_PIPELINE.png)

# 4. Results

## 4.1 Hierarchical feature comparison

{markdown_table(primary_table)}

U1 reduced scene-macro FN MAE from {u0.scene_macro_mae_fn:.5f} to
{u1.scene_macro_mae_fn:.5f} (Δ={u1_step.delta:.5f};
95% CI [{u1_step.ci95_low:.5f}, {u1_step.ci95_high:.5f}]).
Adding conventional representation distances further reduced MAE to
{u2.scene_macro_mae_fn:.5f} (Δ={u2_step.delta:.5f};
95% CI [{u2_step.ci95_low:.5f}, {u2_step.ci95_high:.5f}]).
After multiplicity correction, neither transition was interpreted as a
confirmed T-norm result; importantly, U2 established the baseline against
which U3 had to demonstrate incremental value.

## 4.2 Primary U3-versus-U2 result

U3 increased scene-macro MAE by {primary.delta:.5f}. Expressed using the
frozen sign convention, relative MAE reduction was
{metrics['relative_MAE_reduction_percent']:.2f}%; negative values denote
worsening. U3 won on {int(primary.scene_wins)} scenes and lost on
{int(primary.scene_losses)}. The bootstrap interval included zero and its
upper bound was positive. No primary requirement was met.

![Per-scene U2 and U3 MAE](figures/FIGURE_2_PER_SCENE_MAE.png)

![Paired scene-bootstrap distribution](figures/FIGURE_3_BOOTSTRAP_DELTA.png)

## 4.3 Sensitivity analyses

{markdown_table(sensitivity_table)}

ElasticNet reproduced the direction of the primary result: U2 MAE was
{u2_elastic.scene_macro_mae_fn:.5f} and U3 MAE was
{u3_elastic.scene_macro_mae_fn:.5f}, a relative reduction of
{elastic_reduction:.2f}%. Thus, choosing the sensitivity estimator would not
reverse the conclusion.

For the secondary Recall endpoint, U3−U2 MAE was {recall_step.delta:.5f}
with 95% CI [{recall_step.ci95_low:.5f}, {recall_step.ci95_high:.5f}].
This secondary endpoint was not used to redefine the FN/frame decision.

Deleting each scene in turn left the relative MAE reduction negative in every
case. The least favorable value was
{loso.relative_MAE_reduction_percent.min():.2f}% and the largest was
{loso.relative_MAE_reduction_percent.max():.2f}%. Scene-size sensitivity also
preserved the direction: frame-weighted, median-scene, and trimmed-mean MAE
were all higher for U3 than U2.

![Leave-one-scene-out sensitivity](figures/FIGURE_4_LOSO_SENSITIVITY.png)

## 4.4 Calibration and selective operation

{markdown_table(calibration_table)}

U3 improved scene-macro Brier by
{u3.scene_macro_brier-u2.scene_macro_brier:.5f} and ECE by
{u3.scene_macro_ece-u2.scene_macro_ece:.5f}. Their paired intervals included
zero, and no endpoint survived Holm correction. These descriptive calibration
changes cannot compensate for worse FN/frame prediction.

Risk–coverage curves did not show consistent U3 dominance. At 50% coverage,
accepted FN/frame was
{coverage[(coverage.model == 'U2') & (coverage.coverage == 0.5)].iloc[0].scene_macro_accepted_fn_per_frame:.5f}
for U2 and
{coverage[(coverage.model == 'U3') & (coverage.coverage == 0.5)].iloc[0].scene_macro_accepted_fn_per_frame:.5f}
for U3.

![Risk–coverage curves](figures/FIGURE_5_RISK_COVERAGE.png)

## 4.5 Per-scene outcomes

{markdown_table(scene, digits=4)}

# 5. Discussion

The experiment provides no evidence that raw Product/Łukasiewicz consistency
features add failure-risk information after detector outputs and conventional
representation distances are available. The direction was unfavorable under
Ridge and ElasticNet, under macro and frame-weighted aggregation, and after
each leave-one-scene-out deletion. This pattern is more informative than a
single nonsignificant mean: it indicates redundancy or instability of the
current U3 representation rather than a hidden broadly consistent benefit.

The result does not imply that T-norms are universally uninformative. It
rejects a specific incremental claim under a frozen detector, feature
construction, dataset, and endpoint. A future confirmatory study must avoid
post-hoc reuse of these scenes. Two defensible routes are new independent
railway scenes or detector-out-of-fold representations for every scene.
Residualized T-norm features and causal temporal changes may then test unique
signal rather than reintroducing correlated raw features.

From a safety-engineering perspective, selective operation remains a useful
analysis framework, but the current U3 score should not control deployment.
The detector is not production-ready, and the risk model did not pass its
development gate.

# 6. Limitations

First, B0 was trained on 12 of 15 development scenes, so the risk-model LOSO
does not provide fully detector-OOF evidence. Second, the sealed test set was
not opened because the development gate failed. Third, adversarial attacks
were outside V8b and no robustness claim is made. Fourth, the study covers one
YOLO11m detector, one dataset, and the person class only. Fifth, 15 independent
scenes provide limited power, particularly for scene-level AUROC/AUPRC.
Sixth, U3 may be redundant with U2. Seventh, descriptive calibration improved
slightly while the primary endpoint worsened; interpreting calibration alone
would violate the prespecified hierarchy.

# 7. Conclusion

A reproducible scene-level protocol was developed to test the incremental
diagnostic value of T-norm representation consistency. Product and
Łukasiewicz features did not improve frozen-detector FN/frame risk estimation
over detector outputs and standard representation metrics. The negative
result was stable to estimator choice, scene-size aggregation, and deletion of
individual scenes. The test remained sealed. This outcome supports strict
hierarchical baselines and scene-level confirmation before new representation
metrics are incorporated into safety monitoring.

# Data and code availability

The public repository contains protocol files, analysis code, redacted OOF
rows, tables, figures, and checksums. Dataset images, checkpoints, raw feature
tensors, local paths, and test data are excluded. OSDaR23 is obtained from its
official distribution under its own terms.

# References

1. Tagiew R, Köppel M, Schwalbe K, et al. OSDaR23: Open Sensor Data for Rail
   2023. arXiv:2305.03001, 2023. https://arxiv.org/abs/2305.03001
2. Geifman Y, El-Yaniv R. SelectiveNet: A Deep Neural Network with an
   Integrated Reject Option. Proceedings of Machine Learning Research
   97:2151–2159, 2019. https://proceedings.mlr.press/v97/geifman19a.html
3. Gulrajani I, Lopez-Paz D. In Search of Lost Domain Generalization.
   arXiv:2007.01434, 2020. https://arxiv.org/abs/2007.01434
4. Gouk H, Bohdal O, Li D, Hospedales T. On the Limitations of General
   Purpose Domain Generalisation Methods. arXiv:2202.00563, 2022.
   https://arxiv.org/abs/2202.00563
5. Klement EP, Mesiar R, Pap E. Triangular Norms. Springer, 2000.
"""
    return content


def convert_to_pdf(docx: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="v8b-libreoffice-") as profile:
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                f"-env:UserInstallation=file://{profile}",
                "--convert-to",
                "pdf",
                "--outdir",
                str(docx.parent),
                str(docx),
            ],
            check=True,
            cwd=ROOT,
            timeout=180,
        )
    pdf = docx.with_suffix(".pdf")
    if not pdf.is_file():
        raise RuntimeError("LibreOffice did not create the V8b PDF")
    return pdf


def validate_article(markdown: Path, docx: Path, pdf: Path) -> dict[str, Any]:
    docx_plain = subprocess.check_output(
        ["pandoc", str(docx), "-t", "plain"], text=True, cwd=ROOT
    )
    pdf_plain = subprocess.check_output(
        ["pdftotext", str(pdf), "-"], text=True, cwd=ROOT
    )
    combined = "\n".join(
        [markdown.read_text(encoding="utf-8"), docx_plain, pdf_plain]
    )
    forbidden = [
        "[TBD",
        "T-norms improve accuracy",
        "adversarial robustness was obtained",
        "ready for deployment",
    ]
    checks = {
        "markdown_exists": markdown.is_file(),
        "docx_exists": docx.is_file(),
        "pdf_exists": pdf.is_file(),
        "five_figures_present": len(list((FINAL / "figures").glob("FIGURE_*.png")))
        >= 5,
        "primary_numbers_present": all(
            value in combined for value in ("1.59549", "1.76798", "-10.81")
        ),
        "negative_conclusion_present": "did not improve" in combined,
        "sealed_test_disclosed": "test remained sealed" in combined.lower(),
        "detector_overlap_disclosed": "trained on 12 of these 15" in combined,
        "no_placeholder_or_forbidden_claim": not any(
            value.lower() in combined.lower() for value in forbidden
        ),
        "no_email": re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", combined
        )
        is None,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "article_md_sha256": sha256_file(markdown),
        "article_docx_sha256": sha256_file(docx),
        "article_pdf_sha256": sha256_file(pdf),
    }


def write_reproducibility_files() -> None:
    reproduce = """# Reproduce canonical v8b finalization

The finalizer reads only the saved, development-only OOF table. It does not
read images, a detector checkpoint, raw feature tensors, or the sealed test.

```bash
PYTHONPATH="$PWD/scripts" python \
  scripts/v8b/finalize_v8b.py \
  --input outputs/person_v8b/final/OOF_INPUT_REDACTED.csv \
  --output /tmp/person_v8b_reproduction
```

The reproduced `FINAL_METRICS.json` must contain U2 MAE 1.5954889302,
U3 MAE 1.7679771025, relative reduction -10.81099148%, five U3 scene
wins, and zero test access.

Generate the manuscript after finalization:

```bash
PYTHONPATH="$PWD/scripts" python \
  scripts/v8b/build_article_v8b.py
```
"""
    (FINAL / "REPRODUCE.md").write_text(reproduce, encoding="utf-8")
    citation = """cff-version: 1.2.0
message: "If you use this reproducibility package, please cite the software."
title: "Railway Vision Robustness: Canonical V8b Failure-Risk Study"
type: software
version: "0.8b"
date-released: "2026-07-26"
authors:
  - family-names: "Trofimov"
    given-names: "Yuri V."
  - family-names: "Averkin"
    given-names: "Alexey N."
  - family-names: "Lebedev"
    given-names: "Mikhail D."
  - family-names: "Lebedev"
    given-names: "Alexander D."
repository-code: "https://github.com/lebedeffson/railway-vision-robustness"
"""
    (FINAL / "CITATION.cff").write_text(citation, encoding="utf-8")


def main() -> int:
    audit = json.loads(
        (FINAL / "FINALIZATION_AUDIT.json").read_text(encoding="utf-8")
    )
    if audit["status"] != "PASS" or audit["test_access_count"] != 0:
        raise RuntimeError("Article build is blocked by finalization audit")
    ARTICLE.mkdir(parents=True, exist_ok=True)
    markdown = ARTICLE / "article_v8b_final.md"
    docx = ARTICLE / "article_v8b_final.docx"
    markdown.write_text(build_markdown(), encoding="utf-8")
    subprocess.run(
        [
            "pandoc",
            str(markdown),
            "--from",
            "gfm",
            "--reference-doc",
            str(REFERENCE_DOCX),
            "--resource-path",
            str(FINAL),
            "-o",
            str(docx),
        ],
        check=True,
        cwd=ROOT,
        timeout=180,
    )
    pdf = convert_to_pdf(docx)
    write_reproducibility_files()
    validation = validate_article(markdown, docx, pdf)
    (ARTICLE / "ARTICLE_VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    if validation["status"] != "PASS":
        raise RuntimeError("V8b article validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
