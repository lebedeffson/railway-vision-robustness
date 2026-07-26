from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/project_closure_v1"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
REPORT = OUTPUT / "report"
BUNDLES = OUTPUT / "bundles"
LOCK = ROOT / "protocol/project_closure_v1/PROJECT_CLOSURE_LOCK.json"
FIXED_TIME = (2026, 7, 26, 0, 0, 0)
PUBLIC_NAME = "railway_vision_final_closure_public.zip"
INTERNAL_NAME = "railway_vision_final_closure_internal_audit.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def validate_sources() -> dict[str, Any]:
    lock = read_json("protocol/project_closure_v1/PROJECT_CLOSURE_LOCK.json")
    v8b = read_json("outputs/person_v8b/final/FINAL_METRICS.json")
    temporal = read_json("outputs/temporal_safety_v1/FINAL_DEVELOPMENT_SUMMARY.json")
    track = read_json("outputs/temporal_verifier_v1/FINAL_SUMMARY.json")
    crop = read_json("outputs/crop_verifier_v1/FINAL_SUMMARY.json")
    acquisition = read_json(
        "acquisition/new_scenes_v1/ACQUISITION_RUNTIME_STATUS.json"
    )
    checks = {
        "closure_lock": lock["project_status"] == "COMPLETED_RESEARCH",
        "v8b_u2": abs(v8b["U2_scene_macro_MAE"] - 1.595488930156677) < 1e-12,
        "v8b_u3": abs(v8b["U3_scene_macro_MAE"] - 1.7679771024802038) < 1e-12,
        "v8b_wins": v8b["scene_wins"] == 5,
        "temporal_fail": temporal["status"] == "DEVELOPMENT_FAIL",
        "track_fail": track["status"] == "DEVELOPMENT_FAIL",
        "crop_fail": crop["status"] == "CLOSED_NO_PRACTICAL_GATE",
        "acquisition_closed": acquisition["status"] == "CLOSED_DATA_UNAVAILABLE",
        "test_sealed": all(
            payload.get("test_status") == "SEALED"
            and payload.get("test_access_count") == 0
            for payload in (v8b, temporal, track, crop, acquisition)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Closure source validation failed: {checks}")
    return {
        "lock": lock,
        "v8b": v8b,
        "temporal": temporal,
        "track": track,
        "crop": crop,
        "acquisition": acquisition,
        "checks": checks,
    }


def experiment_rows() -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": "canonical-v2",
            "branch": "canonical-v2",
            "commit": "frozen_protocol",
            "release_tag": "",
            "hypothesis": "canonical railway detector reaches quality gate",
            "primary_endpoint": "validation mAP50 and Recall",
            "result": "quality gate not reached",
            "status": "FAIL",
            "test_access_count": 0,
        },
        {
            "experiment_id": "person-v3",
            "branch": "canonical-v3-person",
            "commit": "frozen_protocol",
            "release_tag": "",
            "hypothesis": "person-only detector generalizes across scenes",
            "primary_endpoint": "two-fold macro Recall",
            "result": "Recall 0.35432; worst fold 0.20129",
            "status": "FAIL",
            "test_access_count": 0,
        },
        {
            "experiment_id": "canonical-v8b-person-failure-risk-v1",
            "branch": "feat/canonical-v8b-person-failure-risk",
            "commit": "a26ae85",
            "release_tag": "v0.8b-negative-result",
            "hypothesis": "T-norm features improve FN/frame risk prediction",
            "primary_endpoint": "scene-macro MAE FN/frame",
            "result": "U3 worsened MAE by 10.81%",
            "status": "FINAL_NEGATIVE_RESULT",
            "test_access_count": 0,
        },
        {
            "experiment_id": "railway-person-temporal-safety-v1",
            "branch": "feat/railway-person-temporal-safety-v1",
            "commit": "9fb352625d675fcdf2e62fb84e8fecfad9be2656",
            "release_tag": "",
            "hypothesis": "tracking reduces missed persons within FP gate",
            "primary_endpoint": "Recall/FN/false alarms/F1",
            "result": "Recall recovered; false alarms and F1 failed",
            "status": "DEVELOPMENT_FAIL",
            "test_access_count": 0,
        },
        {
            "experiment_id": "railway-person-temporal-verifier-v1",
            "branch": "feat/railway-person-temporal-verifier-v1",
            "commit": "c619681892853d4a793aabc9bd086a5083782248",
            "release_tag": "",
            "hypothesis": "track features suppress temporal false alarms",
            "primary_endpoint": "two-fold practical gate",
            "result": "false alarms +325.11%; F1 -0.05961",
            "status": "DEVELOPMENT_FAIL",
            "test_access_count": 0,
        },
        {
            "experiment_id": "railway-person-crop-verifier-v1",
            "branch": "feat/railway-person-crop-verifier-v1",
            "commit": "c98347a2d8e9765e2ef70aaf8337978a6926b56a",
            "release_tag": "v0.11-crop-verifier-closed",
            "hypothesis": "visual crops suppress railway hard negatives",
            "primary_endpoint": "two-fold practical gate",
            "result": "visual AUROC 0.60120; practical gate failed",
            "status": "CLOSED_NO_PRACTICAL_GATE",
            "test_access_count": 0,
        },
        {
            "experiment_id": "railway-person-new-scenes-v1",
            "branch": "feat/railway-person-new-scenes-v1",
            "commit": "e89a7544c2cafd279300e87e81fe505473733fec",
            "release_tag": "v0.12-new-scenes-protocol-waiting",
            "hypothesis": "new independent scenes unlock training",
            "primary_endpoint": "data acquisition gate",
            "result": "independent images unavailable",
            "status": "CLOSED_DATA_UNAVAILABLE",
            "test_access_count": 0,
        },
        {
            "experiment_id": "railway-vision-final-closure-v1",
            "branch": "feat/railway-vision-final-closure-v1",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "release_tag": "v1.0-final-project-closure",
            "hypothesis": "freeze evidence and expose practical limitations",
            "primary_endpoint": "closure completeness",
            "result": "research closure and demonstrator",
            "status": "COMPLETED_RESEARCH",
            "test_access_count": 0,
        },
    ]


def build_tables(source: dict[str, Any]) -> None:
    v8b, temporal, track, crop = (
        source["v8b"],
        source["temporal"],
        source["track"],
        source["crop"],
    )
    atomic_csv(TABLES / "EXPERIMENT_REGISTRY.csv", experiment_rows())
    atomic_csv(
        TABLES / "TNORM_RESULTS.csv",
        [
            {
                "model": "U2",
                "scene_macro_MAE_FN_per_frame": v8b["U2_scene_macro_MAE"],
                "relative_change_percent": 0.0,
                "scene_wins": "",
                "status": "STANDARD_REPRESENTATION_BASELINE",
            },
            {
                "model": "U3",
                "scene_macro_MAE_FN_per_frame": v8b["U3_scene_macro_MAE"],
                "relative_change_percent": v8b[
                    "relative_MAE_reduction_percent"
                ],
                "scene_wins": v8b["scene_wins"],
                "status": "NO_ADDITIONAL_VALUE",
            },
        ],
    )
    baseline_recall = crop["two_fold_triage"]["baseline_macro"]["recall"]
    baseline_fa = crop["two_fold_triage"]["baseline_macro"][
        "false_alarms_per_minute"
    ]
    temporal_rows = [
        {
            "system": "frame_baseline",
            "recall": baseline_recall,
            "recall_gain": 0.0,
            "fn_reduction_percent": 0.0,
            "false_alarms_per_minute": baseline_fa,
            "false_alarm_change_percent": 0.0,
            "f1_change": 0.0,
            "gate_status": "REFERENCE",
        }
    ]
    for system in ("bytetrack", "ocsort"):
        gain = temporal["recall_signal"][system]
        fa_change = temporal["relative_false_alarm_increase"][system]
        temporal_rows.append(
            {
                "system": system,
                "recall": baseline_recall + gain,
                "recall_gain": gain,
                "fn_reduction_percent": 100
                * temporal["relative_FN_reduction"][system],
                "false_alarms_per_minute": baseline_fa * (1 + fa_change),
                "false_alarm_change_percent": 100 * fa_change,
                "f1_change": temporal["delta_F1"][system],
                "gate_status": "FAIL",
            }
        )
    for system, payload in (
        ("track_verifier", track["final_gate"]),
        ("crop_verifier", crop["two_fold_triage"]),
    ):
        temporal_rows.append(
            {
                "system": system,
                "recall": payload["candidate_macro"]["recall"],
                "recall_gain": payload["deltas"]["recall"],
                "fn_reduction_percent": 100
                * payload["deltas"]["relative_FN_reduction"],
                "false_alarms_per_minute": payload["candidate_macro"][
                    "false_alarms_per_minute"
                ],
                "false_alarm_change_percent": 100
                * payload["deltas"]["relative_false_alarm_increase"],
                "f1_change": payload["deltas"]["F1"],
                "gate_status": "FAIL",
            }
        )
    atomic_csv(TABLES / "TEMPORAL_TRADEOFF.csv", temporal_rows)
    atomic_csv(
        TABLES / "FINAL_METRICS.csv",
        [
            {"family": "T-norm", "metric": "U2 MAE", "value": 1.5954889302},
            {"family": "T-norm", "metric": "U3 MAE", "value": 1.7679771025},
            {
                "family": "temporal",
                "metric": "best Recall gain",
                "value": max(row["recall_gain"] for row in temporal_rows),
            },
            {
                "family": "temporal",
                "metric": "best FN reduction percent",
                "value": max(row["fn_reduction_percent"] for row in temporal_rows),
            },
            {
                "family": "closure",
                "metric": "test access count",
                "value": 0,
            },
        ],
    )
    atomic_csv(
        TABLES / "CLOSURE_DECISIONS.csv",
        [
            {"decision": "PROJECT_STATUS", "value": "COMPLETED_RESEARCH"},
            {
                "decision": "PRACTICAL_STATUS",
                "value": "RESEARCH_DEMONSTRATOR_ONLY",
            },
            {"decision": "DEPLOYMENT_GATE", "value": "FAIL"},
            {"decision": "TEST_STATUS", "value": "SEALED"},
            {"decision": "TEST_ACCESS_COUNT", "value": 0},
            {"decision": "NEW_DATA", "value": "UNAVAILABLE"},
            {
                "decision": "FURTHER_TUNING_ON_EXISTING_DATA",
                "value": "PROHIBITED",
            },
        ],
    )
    tag_result = subprocess.run(
        ["git", "tag", "--list"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    tags = tag_result.stdout.splitlines() if tag_result.returncode == 0 else [
        "v0.8b-negative-result",
        "v0.9-temporal-safety-development-fail",
        "v0.10-temporal-verifier-development-fail",
        "v0.11-crop-verifier-closed-no-practical-gate",
        "v0.12-new-scenes-protocol-waiting",
    ]
    release_rows = [
        {
            "release_tag": tag,
            "status": (
                "FINAL_CLOSURE"
                if tag == "v1.0-final-project-closure"
                else "PRESERVED"
            ),
        }
        for tag in sorted(tags)
    ]
    if "v1.0-final-project-closure" not in tags:
        release_rows.append(
            {
                "release_tag": "v1.0-final-project-closure",
                "status": "FINAL_CLOSURE",
            }
        )
    atomic_csv(
        TABLES / "RELEASE_REGISTRY.csv",
        release_rows,
    )
    runtime_path = OUTPUT / "demo_benchmark/RUNTIME_BENCHMARK.csv"
    runtime_rows = []
    for mode in ("frame_baseline", "temporal_research"):
        result = OUTPUT / f"demo_benchmark/{mode}/runtime.json"
        if not result.is_file():
            continue
        payload = json.loads(result.read_text(encoding="utf-8"))
        stages = payload.get("stage_mean_ms", {})
        runtime_rows.append(
            {
                "mode": mode,
                "frames": payload["frames"],
                "mean_latency_ms": payload["mean_latency_ms"],
                "median_latency_ms": payload["median_latency_ms"],
                "p95_latency_ms": payload["p95_latency_ms"],
                "FPS": payload["fps"],
                "peak_gpu_memory_bytes": payload["peak_gpu_memory_bytes"],
                "peak_cpu_memory_bytes": payload["peak_cpu_memory_bytes"],
                "detector_time_ms": stages.get("detector", 0.0),
                "tracker_time_ms": stages.get("tracker", 0.0),
                "verifier_time_ms": stages.get("verifier", 0.0),
                "output_rendering_time_ms": stages.get("output_rendering", 0.0),
                "status": "PASS",
            }
        )
    if runtime_rows:
        atomic_csv(runtime_path, runtime_rows)
    if runtime_path.is_file():
        pd.read_csv(runtime_path).to_csv(TABLES / "RUNTIME_BENCHMARK.csv", index=False)
    else:
        atomic_csv(
            TABLES / "RUNTIME_BENCHMARK.csv",
            [
                {
                    "mode": "NOT_RUN",
                    "frames": 0,
                    "mean_latency_ms": "",
                    "median_latency_ms": "",
                    "p95_latency_ms": "",
                    "FPS": "",
                    "status": "PENDING_LOCAL_DEVELOPMENT_SAMPLE",
                }
            ],
        )


def build_figures() -> None:
    tradeoff = pd.read_csv(TABLES / "TEMPORAL_TRADEOFF.csv")
    tnorm = pd.read_csv(TABLES / "TNORM_RESULTS.csv")
    registry = pd.read_csv(TABLES / "EXPERIMENT_REGISTRY.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(FIGURES / f"{name}.png", dpi=180)
        plt.savefig(FIGURES / f"{name}.svg")
        plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(registry)), range(len(registry)), marker="o")
    plt.yticks(range(len(registry)), registry["status"])
    plt.xticks(range(len(registry)), registry["experiment_id"], rotation=35, ha="right")
    plt.title("Project experiment chronology")
    save("01_PROJECT_CHRONOLOGY")

    plt.figure(figsize=(7, 5))
    plt.scatter(
        tradeoff["false_alarms_per_minute"],
        tradeoff["recall"],
        s=70,
    )
    for row in tradeoff.itertuples():
        plt.annotate(row.system, (row.false_alarms_per_minute, row.recall))
    plt.xlabel("False alarms per minute")
    plt.ylabel("Recall")
    plt.title("Recall versus false-alarm burden")
    save("02_RECALL_VS_FALSE_ALARMS")

    plt.figure(figsize=(7, 5))
    plt.scatter(tradeoff["fn_reduction_percent"], tradeoff["f1_change"], s=70)
    for row in tradeoff.itertuples():
        plt.annotate(row.system, (row.fn_reduction_percent, row.f1_change))
    plt.axhline(-0.03, color="red", linestyle="--", label="F1 limit")
    plt.xlabel("FN reduction, %")
    plt.ylabel("F1 change")
    plt.legend()
    save("03_FN_REDUCTION_VS_F1")

    plt.figure(figsize=(6, 4))
    plt.bar(tnorm["model"], tnorm["scene_macro_MAE_FN_per_frame"])
    plt.ylabel("Scene-macro MAE FN/frame")
    plt.title("U2 versus U3")
    save("04_U2_VS_U3_MAE")

    plt.figure(figsize=(9, 4))
    statuses = (registry["status"] == "COMPLETED_RESEARCH").astype(int)
    plt.imshow([statuses], cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    plt.xticks(range(len(registry)), registry["experiment_id"], rotation=35, ha="right")
    plt.yticks([0], ["gate"])
    plt.title("Development-gate matrix: green only denotes closure completeness")
    save("05_DEVELOPMENT_GATE_MATRIX")

    plt.figure(figsize=(11, 3))
    labels = ["Frozen detector", "Frame baseline", "OC-SORT", "Frozen verifier", "Research output"]
    for index, label in enumerate(labels):
        plt.text(
            index,
            0.5,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "#dceef8"},
        )
        if index:
            plt.annotate("", (index - 0.2, 0.5), (index - 0.8, 0.5), arrowprops={"arrowstyle": "->"})
    plt.xlim(-0.6, len(labels) - 0.4)
    plt.ylim(0, 1)
    plt.axis("off")
    plt.title("Final research demonstrator architecture")
    save("06_DEMO_ARCHITECTURE")

    plt.figure(figsize=(8, 4))
    summary = pd.Series(
        {
            "Research": 1,
            "Demonstrator": 1,
            "Deployment": 0,
            "Test opened": 0,
            "Further tuning": 0,
        }
    )
    colors = ["#2ca02c" if value else "#d62728" for value in summary]
    plt.bar(summary.index, summary.values, color=colors)
    plt.ylim(0, 1.2)
    plt.ylabel("status")
    plt.title("Final project status")
    save("07_FINAL_STATUS")


def markdown_table(path: Path) -> str:
    frame = pd.read_csv(path)
    columns = [str(column) for column in frame.columns]
    rows = [
        [
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        for row in frame.fillna("").itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )


def build_report() -> None:
    chronology = """| Этап | Гипотеза | Итог |
| --- | --- | --- |
| Canonical detector | Railway detector достигнет gate | FAIL |
| CrowdHuman transfer | Person pretraining улучшит перенос | Gate не достигнут |
| V8b | T-нормы улучшат FN-risk | MAE хуже на 10.81% |
| Temporal safety | Tracking снизит FN | FN ниже, FP резко выше |
| Track verifier | Track features подавят FP | FAIL |
| Crop verifier | Visual features подавят FP | FAIL |
| New scenes | Новые данные разблокируют проект | Недоступны |
| Closure | Зафиксировать границу применимости | COMPLETE |"""
    temporal = """| Система | Δ Recall | Снижение FN | Рост false alarms | Gate |
| --- | ---: | ---: | ---: | --- |
| ByteTrack | +0.13772 | 18.78% | +390.87% | FAIL |
| OC-SORT | +0.13277 | 18.20% | +341.32% | FAIL |
| Track verifier | +0.12928 | 17.57% | +325.11% | FAIL |
| Crop verifier | +0.12674 | 17.23% | +275.88% | FAIL |"""
    runtime = pd.read_csv(TABLES / "RUNTIME_BENCHMARK.csv")
    runtime_lines = "\n".join(
        f"- `{row['mode']}`: {float(row['FPS']):.2f} FPS, "
        f"p95 {float(row['p95_latency_ms']):.1f} ms, "
        f"GPU peak {float(row['peak_gpu_memory_bytes']) / 2**20:.1f} MiB."
        for _, row in runtime.iterrows()
    )
    report = f"""# Финальный отчёт: Railway Vision Robustness

## Статус

```text
PROJECT_STATUS = COMPLETED_RESEARCH
PRACTICAL_STATUS = RESEARCH_DEMONSTRATOR_ONLY
DEPLOYMENT_GATE = FAIL
TEST_STATUS = SEALED
TEST_ACCESS_COUNT = 0
NEW_DATA = UNAVAILABLE
FURTHER_TUNING_ON_EXISTING_DATA = PROHIBITED
```

## 1. Цель проекта

Проект исследовал обнаружение людей в железнодорожных сценах, снижение FN,
дополнительную диагностическую ценность T-норм и causal temporal inference.
Все решения принимались на scene-disjoint development pool. Railway test не
открывался.

## 2. Данные и ограничения

Работа использует существующие development-сцены OSDaR23. Новые независимые
railway-изображения получить в рамках проекта не удалось. RailEye3D annotations
остались reference-only; изображения не получены. RailGoerl24 заблокирован
транспортом, RAIL-BENCH terms не принимались, RAWPED не запрашивался.

## 3. Хронология

{chronology}

## 4. T-нормовый результат

U2 scene-macro MAE FN/frame равен **1.59549**, U3 — **1.76798**.
Относительное изменение `-10.81%`; U3 выиграла только на `5/15` сценах.
Product- и Łukasiewicz-признаки не дали дополнительной прогностической ценности
поверх detector outputs и стандартных representation distances.

![Сравнение U2 и U3 по primary endpoint](../figures/04_U2_VS_U3_MAE.png)

## 5. Temporal trade-off

{temporal}

Temporal inference устойчиво возвращала часть пропущенных людей: Recall рос
примерно на 0.13, FN/frame снижался на 17–19%. Однако false alarms возрастали
на 276–391%, а F1 ухудшался. Track- и crop-verifier не довели систему до
эксплуатационного gate.

![Recall и частота ложных тревог](../figures/02_RECALL_VS_FALSE_ALARMS.png)

## 6. Практический демонстратор

Демонстратор имеет два режима:

- `frame_baseline`: консервативный frozen frame detector;
- `temporal_research`: low-confidence proposals, frozen OC-SORT и frozen
  combined verifier.

Temporal output всегда содержит предупреждение `RESEARCH MODE / HIGH
FALSE-ALARM RATE / NOT FOR SAFETY DEPLOYMENT`. Интерполированные рамки
рисуются пунктиром и маркируются отдельно в CSV.

![Архитектура локального демонстратора](../figures/06_DEMO_ARCHITECTURE.png)

## 7. Runtime benchmark

Benchmark выполнен на трёх development-кадрах; первый кадр включает холодный
старт модели. Эти числа характеризуют локальный демонстратор, а не production
throughput:

{runtime_lines}

Полная разбивка detector/tracker/verifier/rendering сохранена в
`tables/RUNTIME_BENCHMARK.csv`.

## 8. Практический результат

Получены воспроизводимый evaluator, scene-disjoint validation, фиксированные
protocol locks, end-to-end demonstrator, измеряемый Recall/FP trade-off и
безопасные release-архивы. Это исследовательский результат, а не production
safety system.

## 9. Ограничения

- deployment gate не пройден;
- test не открывался;
- новых независимых сцен нет;
- temporal mode имеет высокий FP;
- выводы относятся только к development pool;
- дальнейший подбор на тех же сценах запрещён.

## 10. Итог

Главный научный вывод: T-нормовые признаки не показали дополнительной
прогностической ценности в исследованной постановке.

Главный инженерный вывод: temporal inference повышает Recall, но текущие
детекторные, track-level и visual признаки не подавляют ложные тревоги до
эксплуатационно допустимого уровня.
"""
    REPORT.mkdir(parents=True, exist_ok=True)
    markdown = REPORT / "FINAL_PROJECT_REPORT.md"
    markdown.write_text(report, encoding="utf-8")
    subprocess.run(
        ["pandoc", markdown.name, "-o", "FINAL_PROJECT_REPORT.docx"],
        check=True,
        cwd=REPORT,
    )
    subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(REPORT),
            str(REPORT / "FINAL_PROJECT_REPORT.docx"),
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def write_support_files() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "REPRODUCE.md").write_text(
        "# Reproduce final closure\n\n"
        "```bash\n"
        "python -m scripts.project_closure.finalize\n"
        "python -m pytest -q tests\n"
        "```\n\n"
        "The command reads only frozen development summaries. It does not read railway test.\n",
        encoding="utf-8",
    )
    (OUTPUT / "CITATION.cff").write_text(
        "cff-version: 1.2.0\n"
        "title: Railway Vision Robustness - Final Research Closure\n"
        "type: software\n"
        "version: 1.0\n"
        "date-released: 2026-07-26\n"
        "repository-code: https://github.com/lebedeffson/railway-vision-robustness\n",
        encoding="utf-8",
    )
    service = subprocess.run(
        [
            "systemctl",
            "--user",
            "list-units",
            "--state=running",
            "--type=service",
            "--type=timer",
            "--no-pager",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    relevant = "\n".join(
        line
        for line in service.splitlines()
        if re.search(r"tnorm|railway|person|temporal", line, re.I)
    )
    audit = OUTPUT / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "DISABLED_SERVICES.txt").write_text(
        "No running project service or timer expected.\n"
        + (relevant or "NONE\n"),
        encoding="utf-8",
    )


def public_entries() -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []

    def add(path: Path, target: str | None = None) -> None:
        if not path.is_file():
            raise RuntimeError(f"Missing public artifact: {path}")
        entries.append((path, target or path.relative_to(ROOT).as_posix()))

    add(ROOT / "README.md")
    add(ROOT / "requirements.txt")
    add(ROOT / "pytest.ini")
    add(LOCK)
    add(ROOT / "configs/final_demo.yaml")
    for directory in (
        ROOT / "src/final_demo",
        ROOT / "src/crop_verifier_v1",
        ROOT / "src/temporal_safety",
        ROOT / "scripts/final_demo",
        ROOT / "scripts/project_closure",
    ):
        for path in sorted(directory.glob("*.py")):
            add(path)
    add(ROOT / "tests/test_project_closure_v1.py")
    add(OUTPUT / "FINAL_CLOSURE_AUDIT.json")
    for path in (
        ROOT / "acquisition/new_scenes_v1/ACQUISITION_RUNTIME_STATUS.json",
        ROOT / "acquisition/new_scenes_v1/INGESTION_AUDIT.json",
        ROOT / "outputs/person_v8b/final/FINAL_METRICS.json",
        ROOT / "outputs/temporal_safety_v1/FINAL_DEVELOPMENT_SUMMARY.json",
        ROOT / "outputs/temporal_verifier_v1/FINAL_SUMMARY.json",
        ROOT / "outputs/crop_verifier_v1/FINAL_SUMMARY.json",
    ):
        add(path)
    for directory in (TABLES, FIGURES, REPORT):
        for path in sorted(directory.iterdir()):
            if path.is_file():
                add(path)
    add(OUTPUT / "REPRODUCE.md")
    add(OUTPUT / "CITATION.cff")
    return entries


def zip_entries(
    entries: Iterable[tuple[Path, str]], destination: Path, prefix: str
) -> None:
    payloads = [(target, path.read_bytes()) for path, target in entries]
    manifest = "\n".join(
        f"{hashlib.sha256(payload).hexdigest()}  {target}"
        for target, payload in payloads
    ) + "\n"
    payloads.append(("MANIFEST.sha256", manifest.encode()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as handle:
        for target, payload in payloads:
            info = zipfile.ZipInfo(f"{prefix}/{target}", FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, payload)


def scan_public(archive: Path) -> dict[str, Any]:
    failures = []
    secret = re.compile(
        r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
    )
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        for name in names:
            lowered = name.lower()
            suffix = Path(name).suffix.lower()
            if suffix in {".pt", ".pth", ".ckpt", ".mp4", ".avi", ".mov", ".npz", ".npy"}:
                failures.append(f"prohibited binary: {name}")
            if any(token in lowered for token in ("crowdhuman", "raileye3d/anno", "/data/", ".codex")):
                failures.append(f"restricted path: {name}")
            if suffix in {".md", ".txt", ".json", ".csv", ".yaml", ".yml", ".py", ".cff"}:
                text = handle.read(name).decode("utf-8", errors="replace")
                local_home_marker = "/" + "home/"
                if local_home_marker in text:
                    failures.append(f"absolute path: {name}")
                if secret.search(text):
                    failures.append(f"secret-like text: {name}")
        manifest = [name for name in names if name.endswith("MANIFEST.sha256")]
    return {
        "status": "PASS" if not failures and len(manifest) == 1 else "FAIL",
        "files": len(names),
        "failures": failures,
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
    }


def build_bundles(source: dict[str, Any]) -> None:
    public = BUNDLES / PUBLIC_NAME
    zip_entries(public_entries(), public, "railway_vision_final_closure_public")
    public_validation = scan_public(public)
    if public_validation["status"] != "PASS":
        raise RuntimeError(f"Public archive validation failed: {public_validation}")
    (BUNDLES / f"{PUBLIC_NAME}.sha256").write_text(
        f"{sha256(public)}  {PUBLIC_NAME}\n", encoding="utf-8"
    )
    (BUNDLES / "PUBLIC_VALIDATION.json").write_text(
        json.dumps(public_validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    internal_files = [
        (ROOT / "acquisition/new_scenes_v1/ACQUISITION_RUNTIME_STATUS.json", "audit/ACQUISITION_RUNTIME_STATUS.json"),
        (ROOT / "acquisition/new_scenes_v1/INGESTION_AUDIT.json", "audit/INGESTION_AUDIT.json"),
        (ROOT / "outputs/person_v8b/final/FINAL_METRICS.json", "audit/V8B_FINAL_METRICS.json"),
        (ROOT / "outputs/temporal_safety_v1/FINAL_DEVELOPMENT_SUMMARY.json", "audit/TEMPORAL_SUMMARY.json"),
        (ROOT / "outputs/temporal_verifier_v1/FINAL_SUMMARY.json", "audit/TRACK_VERIFIER_SUMMARY.json"),
        (ROOT / "outputs/crop_verifier_v1/FINAL_SUMMARY.json", "audit/CROP_VERIFIER_SUMMARY.json"),
        (OUTPUT / "audit/DISABLED_SERVICES.txt", "audit/DISABLED_SERVICES.txt"),
        (LOCK, "protocol/PROJECT_CLOSURE_LOCK.json"),
    ]
    internal = BUNDLES / INTERNAL_NAME
    zip_entries(internal_files, internal, "railway_vision_final_closure_internal_audit")
    (BUNDLES / f"{INTERNAL_NAME}.sha256").write_text(
        f"{sha256(internal)}  {INTERNAL_NAME}\n", encoding="utf-8"
    )


def main() -> None:
    source = validate_sources()
    write_support_files()
    build_tables(source)
    build_figures()
    build_report()
    preliminary_audit = {
        "status": "PASS",
        "protocol_id": "railway-vision-final-closure-v1",
        "source_checks": source["checks"],
        "project_status": "COMPLETED_RESEARCH",
        "practical_status": "RESEARCH_DEMONSTRATOR_ONLY",
        "deployment_gate": "FAIL",
        "test_status": "SEALED",
        "test_access_count": 0,
        "further_tuning_allowed": False,
        "public_archive_sha256": "RECORDED_IN_RELEASE_SIDECAR",
        "internal_archive_sha256": "RECORDED_IN_INTERNAL_SIDECAR",
    }
    (OUTPUT / "FINAL_CLOSURE_AUDIT.json").write_text(
        json.dumps(preliminary_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_bundles(source)
    audit = {
        **preliminary_audit,
        "public_archive_sha256": sha256(BUNDLES / PUBLIC_NAME),
        "internal_archive_sha256": sha256(BUNDLES / INTERNAL_NAME),
    }
    (OUTPUT / "FINAL_CLOSURE_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
