from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/final_practice"
OUTPUT = ROOT / "TNormFilter_final_practice_report.pdf"


def paragraph_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(.08, .94, title, fontsize=17, weight="bold", va="top")
    y = .89
    for paragraph in paragraphs:
        wrapped = textwrap.fill(paragraph, width=100)
        figure.text(.08, y, wrapped, fontsize=10, va="top", linespacing=1.45)
        y -= .035 * (wrapped.count("\n") + 1) + .04
    plt.axis("off")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def image_page(pdf: PdfPages, path: Path) -> None:
    image = plt.imread(path)
    figure = plt.figure(figsize=(11.69, 8.27))
    axis = figure.add_subplot(111)
    axis.imshow(image)
    axis.axis("off")
    axis.set_title(path.stem.replace("_", " "), fontsize=14)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def dataset_source_page(pdf: PdfPages) -> None:
    links = [
        ("OSDaR23 dataset DOI", "https://doi.org/10.57806/9mv146r0"),
        ("DZSF research report", "https://doi.org/10.48755/dzsf.230012.01"),
        ("OSDaR23 labeling guide", "https://doi.org/10.48755/dzsf.230012.05"),
        ("DZSF project page", "https://www.dzsf.bund.de/SharedDocs/Standardartikel/DZSF/Projekte/Projekt_70_Reale_Datensaetze.html"),
        ("ASAM OpenLABEL specification", "https://www.asam.net/standards/detail/openlabel/"),
        ("RailLabel tools", "https://github.com/DSD-DBS/raillabel"),
    ]
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(.08, .94, "Dataset provenance and links", fontsize=17, weight="bold", va="top")
    figure.text(
        .08, .88,
        "OSDaR23 version 1.1.0 was published by DZSF with DB Netz / Digitale Schiene Deutschland and FusionSystems. The study uses only the rgb_highres_center stream and ASAM OpenLABEL annotations. JSON annotations are CC0 1.0; sensor files are CC BY-SA 3.0 DE.",
        fontsize=10, va="top", wrap=True,
    )
    y = .79
    for label, url in links:
        figure.text(.09, y, label + ":", fontsize=10, weight="bold", va="top")
        figure.text(.09, y - .025, url, fontsize=9, color="blue", va="top", url=url)
        y -= .09
    plt.axis("off")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def fmt(value: object) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final claim-safe PDF report")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.root
    required = [
        root / "audit/split_audit.json",
        root / "audit/checkpoint_selection.json",
        root / "tables/01_dataset_and_clean_model.csv",
        root / "tables/07_model_comparison_statistics.csv",
        root / "tables/08_adaptive_robustness.csv",
        root / "09_statistics/interpretation.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads((root / "audit/split_audit.json").read_text(encoding="utf-8"))
    checkpoint_selection = json.loads(
        (root / "audit/checkpoint_selection.json").read_text(encoding="utf-8")
    )
    clean = pd.read_csv(root / "tables/clean_model_metrics.csv")
    overall = clean[clean["scope"] == "overall"].iloc[0]
    interpretation = json.loads((root / "09_statistics/interpretation.json").read_text(encoding="utf-8"))
    models = pd.read_csv(root / "09_statistics/model_comparison_m0_m4.csv")
    bootstrap = pd.read_csv(root / "09_statistics/sequence_bootstrap.csv")
    adaptive = pd.read_csv(root / "tables/08_adaptive_robustness.csv")
    latency = pd.read_csv(root / "08_latency/latency.csv")
    scenes = pd.read_csv(root / "audit/scene_manifest.csv")
    attack_consistency = pd.read_csv(root / "tables/05_attack_consistency.csv")

    model_lines = []
    for task in ("damage", "recovery"):
        comparison = interpretation["comparisons"][task]
        model_lines.append(
            f"{task}: M3-M2 delta R2={comparison['delta_r2']:.4f}, MAE reduction="
            f"{100 * comparison['mae_reduction_fraction']:.2f}%; claim: {comparison['allowed_claim']}."
        )
    adaptive_text = (
        "Adaptive comparison table is empty because no matched adaptive/non-adaptive pairs were available."
        if adaptive.empty
        else f"Matched adaptive table contains {len(adaptive)} sequence-condition metric rows; "
             f"mean adaptive-minus-nonadaptive={adaptive['adaptive_minus_nonadaptive'].mean():.4f}."
    )
    object_columns = [name for name in ("c_sp_object", "c_sp_background", "c_atk_object", "c_atk_background") if name in attack_consistency]
    object_text = ", ".join(f"{name}={attack_consistency[name].mean():.4f}" for name in object_columns)
    failed = scenes[scenes["validation_status"] == "FAIL"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        paragraph_page(pdf, "TNormFilter: final practical study", [
            "Purpose: evaluate T-norm quantities as complementary diagnostic features for damage and recovery of YOLO11m internal representations under FGSM and PGD. The Product filter is not treated as a universal defense.",
            "All final inferential statistics use sequence_id as the independent unit. The previously produced image-level analysis is retained only as a baseline and is not used for the final claims.",
        ])
        paragraph_page(pdf, "1. Data and split audit", [
            f"Checked {audit['scenes_checked']} of {audit['scenes_expected']} raw scenes. Expected OpenLABEL frames: {audit['frames_expected']}; readable retained frames: {audit['frames_found_and_readable']}; configured pre-split exclusions: {audit['configured_exclusions']}.",
            f"Sequence groups by split: {audit['split_sequence_counts']}. Intersections: {audit['split_intersections']}. Failed scenes: {failed['scene_name'].tolist()}.",
        ])
        dataset_source_page(pdf)
        paragraph_page(pdf, "2. Clean baseline", [
            f"Checkpoint selection used validation only: {checkpoint_selection['selected_checkpoint']} (epoch {checkpoint_selection['selected_epoch']}). Primary criterion: {checkpoint_selection['selection_metric']}.",
            f"Test metrics from best.pt: mAP50={fmt(overall['mAP50'])}, mAP50-95={fmt(overall['mAP50-95'])}, precision={fmt(overall['precision'])}, recall={fmt(overall['recall'])}, F1={fmt(overall['f1'])}, false negatives/frame={fmt(overall['false_negatives_per_frame'])}.",
            "Confidence, IoU, NMS, image size, batch size, completed epochs, best epoch, seeds and loss history are frozen in configs/training_config.yaml. Test thresholds were not tuned to hide low clean quality.",
        ])
        paragraph_page(pdf, "3. Threat model and attacks", [
            "FGSM uses epsilon={0.5,1,2,4,8}/255. PGD pilot uses epsilon={0.1,0.25,0.5,1}/255, random start, three restarts and seeds 42,123,999. The maximum-loss restart is retained.",
            "The YOLO11 attack objective records its actual box, classification and DFL weights. At mAP50 <= 0.01 the primary outcomes are recall, image-level F1, false negatives/frame, confidence drop and IoU shift.",
        ])
        paragraph_page(pdf, "4. Attack-side T-norm diagnostics", [
            "C_sp measures spatial overlap of normalized gradient magnitude and perturbation magnitude; C_dir measures directional agreement; C_atk combines them. Global, object-mask and background-mask variants are reported.",
            f"Final object/background means: {object_text}. PGD clean-gradient and path-gradient variants, cosine, sign agreement, Spearman and top-k overlap are present in raw/attack_consistency.csv.",
        ])
        paragraph_page(pdf, "5. Defense-side diagnostics", [
            "For P3/P4/P5, P=sim(clean, filtered-clean), A=sim(clean, attacked), R=sim(clean, filtered-attacked), G=(R-A)/(1-A+tau), and C_def=T(P,clip(G,0,1)).",
            "Legacy Product/Goedel/Lukasiewicz columns are compatibility similarities and remain a historical baseline. The canonical analysis uses normalized pointwise Product and Lukasiewicz; Goedel is supplementary because it is rank-redundant with Product.",
            "G_raw is used in statistical models; G_clipped is used only inside C_def and for visualization. Cosine, MSE, MAE, relative L2, mean activation shift and feature entropy remain baselines.",
        ])
        paragraph_page(pdf, "6. Sequence-level statistics", [
            *model_lines,
            f"The model table has {len(models)} M0-M4 rows. Paired gains use {int(bootstrap['iterations'].max())} sequence bootstrap repetitions with 95% intervals; GroupKFold groups are sequence_id. FDR-adjusted layer/class correlations are stored with the statistics.",
            "Validation and test each contain only three independent scenes. Bootstrap repetitions do not create additional independent evidence; per-scene, equal-weight macro and leave-one-scene-out results are required and generalization claims remain exploratory.",
        ])
        paragraph_page(pdf, "7. Adaptive PGD", [
            "Adaptive PGD differentiates through image -> Product T-norm preprocessing -> YOLO11m -> detection loss. PGD-20 and PGD-40 are evaluated with random starts, three restarts and three fixed seeds.",
            adaptive_text,
            "No white-box robustness claim is made for JPEG or median because BPDA is not implemented.",
        ])
        detector = latency[latency["method"] == "detector"].iloc[0]
        paragraph_page(pdf, "8. Computational cost", [
            f"Detector-only latency: mean={fmt(detector['mean_latency_ms'])} ms, median={fmt(detector['median_latency_ms'])} ms, p95={fmt(detector['p95_latency_ms'])} ms, FPS={fmt(detector['fps'])}.",
            "All operations use the same GPU, image size and fixed batch. The protocol uses 30 warm-up and 100 measured runs; filter+detector and diagnostic+detector rows include incremental latency and peak GPU memory.",
        ])
        paragraph_page(pdf, "9. Limitations and conclusion", [
            "The simple threshold policy is retained as a negative result: it collapsed to constant JPEG on validation and did not generalize to test. It is not repaired by post-hoc test tuning.",
            "The permissible conclusion is diagnostic: T-norm quantities are additional features for internal-representation damage/recovery only when their sequence-level gain over cosine and standard distances is confirmed. Product preprocessing is not a universal defense, especially when adaptive PGD reduces its apparent benefit.",
        ])
        for figure_path in sorted((root / "figures").glob("0[1-9]_*.png")):
            image_page(pdf, figure_path)
        metadata = pdf.infodict()
        metadata["Title"] = "TNormFilter final practical study"
        metadata["Subject"] = "Sequence-level T-norm diagnostics and adaptive PGD"
        metadata["Author"] = "Reproducible TNormFilter pipeline"

    summary = {
        "status": "PASS",
        "report": str(args.output),
        "pages_expected_minimum": 10,
        "claim_boundary": "diagnostic features; no universal defense claim",
    }
    (root / "report_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
