# Railway Vision Robustness

> Reproducible railway vision experiments from scene-level quality gates to
> adversarial and T-norm diagnostics.

[GitHub repository](https://github.com/lebedeffson/railway-vision-robustness)

Исследовательский pipeline для проверки дополнительной диагностической
ценности канонических T-норм при состязательном повреждении детектора людей в
железнодорожной среде OSDaR23.

`TNormFilter` сохраняется как внутреннее название диагностического модуля;
репозиторий охватывает более широкую задачу межсценовой переносимости,
small-object detection и adversarial robustness.

Проект не предполагает обязательного положительного результата. Основной
критерий — leakage-free сравнение T-нормовых признаков со стандартными
feature-distance baseline на независимых `grouped_scene_id`.

## Текущий научный статус

Основная задача сейчас — получить переносимый person-only baseline до открытия
test и запуска атак.

| Ветка | Статус | Test |
|---|---|---|
| Legacy six-class | исторический baseline; не является canonical evidence | закрыт для новых настроек |
| Canonical v2 | validation quality gate FAIL | не открывался |
| Multiclass v3 | blocked: недостаточная class-by-scene support | не открывался |
| Person v3 | two-fold hard FAIL | не открывался |
| Person v4 DG/NWD | A1/A3 expedited selection FAIL | не открывался |
| Person v4 train-only proxy | FAIL; подтвердил отказ от сложного DG-стека | не открывался |
| Person v5 data-first | CrowdHuman pretraining завершён; railway D1 gate FAIL | не открывался |
| Person v5 range-aware | V5-A и V5-B: two-fold FAIL; официальный V5-C остановлен до первой эпохи из-за label-integrity defect | запечатан |
| Person v5 expedited screening | C0/C1/C2 successive halving, только compute screening | запечатан |
| Person v6 temporal T-norm | T0 fold-0 FAIL; fold 1 skipped, T1 blocked | запечатан |
| Person v7 tracklet verifier | V0 и frozen-crop V1/V2: fold-0 FAIL; fold 1 skipped | запечатан |
| Person v8 active data | BLOCKED_NO_NEW_DATA; GPU_NOT_STARTED | запечатан |
| Person v8b failure risk | FINALIZED_NEGATIVE_RESULT; U3 worsened primary FN/frame MAE | не открывался |
| Person v9 residual-temporal | BLOCKED_PREREQUISITES; design frozen, execution not started | запечатан |

Зафиксированный person-v3 baseline на folds 0/1:

- macro mAP50: `0.35883`;
- macro Recall: `0.35432`;
- macro small Recall: `0.21130`;
- worst-fold Recall: `0.20129`.

Эти числа — development evidence, а не итоговый test-результат.

## Текущий протокол: person v8 active data

`canonical-v8-person-active-data-v1` не добавлял новую архитектурную или
математическую надстройку. Он разрешает продолжение только после добавления
минимум трёх действительно новых railway-сцен, 300 вручную проверенных кадров
и 500 person boxes, из которых не менее 40% относятся к small/distant people.

Протокол замораживается в два этапа:

1. acquisition-lock фиксирует критерии отбора, схемы manifest/correction log,
   CPU-аудит и правила split до сбора данных;
2. execution-lock создаётся только после `CPU_GATE: PASS` и хеширует реальные
   данные, аудит и новые scene-disjoint folds.

Новые независимые сцены не были доступны. Протокол закрыт как
`BLOCKED_NO_NEW_DATA`: это не quality-gate FAIL, обучение не запускалось,
execution-lock не создавался, test не открывался. Acquisition-lock сохранён
без изменений.

```bash
PYTHONPATH="$PWD:$PWD/scripts" python \
  scripts/person_v8/audit_active_data.py
```

Шаблоны:

- [`acquisition_manifest.csv`](protocol/v8/templates/acquisition_manifest.csv);
- [`correction_log.csv`](protocol/v8/templates/correction_log.csv).

После поступления новых сцен порядок фиксирован:

```text
manual annotation -> CPU audit -> frozen splits -> execution lock
-> B0 5/10/20 screening -> untouched confirmation -> five-fold OOF
```

Test, attacks и H1-H4 остаются закрытыми до полного OOF PASS. Для v7
операционные TP/FP/Recall/F1 при повторной проверке совпали, однако frozen
AP-parity формально остался `FAIL` из-за различия порядка ties после
сериализации (`<1e-5`); этот статус не переписан задним числом.

## Текущий протокол: person v8b failure risk

`canonical-v8b-person-failure-risk-v1` не улучшает и не переобучает детектор.
Он проверяет, позволяют ли Product/Łukasiewicz consistency-признаки P3/P4/P5
предсказывать `FN/frame` замороженного B0 лучше стандартных representation
distances.

```text
U0: confidence
U1: U0 + detector outputs
U2: U1 + standard representation distances
U3: U2 + Product/Lukasiewicz consistency
```

Первичный endpoint зафиксирован заранее: scene-macro MAE прогноза `FN/frame`
для `U3 − U2`. Вторичные AUROC/AUPRC/R²/Spearman не могут заменить провал
первичного endpoint. Normalization, prototypes, hyperparameters и isotonic
calibration пересчитываются только на train-сценах каждого внешнего и
внутреннего LOSO.

Deployable object-region features строятся по B0 proposals. GT-person regions
разрешены только как oracle supplement и не входят в U0–U3. Замороженный B0
был обучен на 12 из 15 development-сцен; это явно раскрыто, поэтому LOSO
проверяет переносимость risk estimator, а не единообразный detector-OOF
результат.

Порядок:

```text
F0 audit -> F1 frozen-B0 features -> F2 nested 15-scene LOSO
-> paired scene bootstrap/Holm -> development gate
-> только при PASS один risk-only test
```

FGSM/PGD и заявления об adversarial robustness исключены из v8b.

Итог development-проверки:

```text
U2 scene-macro MAE FN/frame: 1.59549
U3 scene-macro MAE FN/frame: 1.76798
relative MAE reduction:      -10.81%
paired bootstrap 95% CI:     [-0.00850, 0.40791]
scene wins:                  5/15
status:                      DEVELOPMENT_FAIL
test_access_count:           0
```

U3 немного улучшил Brier и ECE, но вторичные показатели не могут заменить
провал заранее зафиксированного первичного endpoint. Положительный вклад
T-нормовых признаков на текущем development pool не подтверждён.

Финальный V8b-пакет воспроизводится только из сохранённой development OOF
таблицы:

```bash
PYTHONPATH="$PWD/scripts" python scripts/v8b/finalize_v8b.py
PYTHONPATH="$PWD/scripts" python scripts/v8b/build_article_v8b.py
PYTHONPATH="$PWD/scripts" python scripts/v8b/build_release_v8b.py
```

Выходы находятся в `outputs/person_v8b/final/`: шесть таблиц статьи, пять
рисунков, MD/DOCX/PDF, независимый audit и публичный ZIP. Датасет,
изображения, checkpoint, сырые признаки, test и локальные пути в архив не
включаются.

## Prospective protocol: person v9

`canonical-v9-person-residual-temporal-v1` проверяет уникальный остаточный
T-нормовый сигнал после train-only residualization и причинные временные
признаки. Primary endpoint остаётся `FN/frame`.

V9 не запущен. Его активация разрешена только после одного из условий:

- не менее пяти новых независимых railway development-сцен; либо
- detector-OOF predictions/features для всех 15 сцен, где held-out сцена
  исключена из обучения соответствующего детектора.

До этого `V9_PREREQUISITES.json` сохраняет
`BLOCKED_PREREQUISITES`, `execution_status=NOT_STARTED` и
`test_access_count=0`.

## Данные и статистическая единица

Эксперименты используют открытый набор
[OSDaR23 — Open Sensor Data for Rail 2023](https://data.fid-move.de/dataset/osdar23).
Описание состава данных и процедуры сбора приведено в
[статье OSDaR23](https://arxiv.org/abs/2305.03001) и на странице
[Data Factory — Digitale Schiene Deutschland](https://digitale-schiene-deutschland.de/en/projects/DataFactory).

- Development pool: прежние train + validation, 15 независимых grouped scenes.
- Sealed test: 5 grouped scenes.
- Целевой класс canonical v3/v4: `person`.
- Группа для CV, bootstrap и анализа: `grouped_scene_id`.
- Tiles и соседние кадры не считаются независимыми наблюдениями.

Исходные аннотации не перезаписываются. Person-only dataset является
производным представлением: люди — целевые объекты, кадры без людей —
отрицательный фон. Сам датасет не включён в Git-репозиторий; перед загрузкой
необходимо ознакомиться с условиями распространения на официальной странице.

## Датасет OSDaR23

В экспериментах используется **Open Sensor Data for Rail 2023 (OSDaR23)** —
открытый мультимодальный железнодорожный набор с синхронизированными RGB/IR
камерами, LiDAR, radar, IMU/GNSS и аннотациями ASAM OpenLABEL.

- Официальный каталог и загрузка:
  [data.fid-move.de/dataset/osdar23](https://data.fid-move.de/dataset/osdar23)
- Страница проекта Digitale Schiene Deutschland:
  [Data Factory / OSDaR23](https://digitale-schiene-deutschland.de/en/projects/DataFactory)
- Описание набора:
  [OSDaR23: Open Sensor Data for Rail 2023](https://arxiv.org/abs/2305.03001)

Датасет, преобразованные изображения и локальные manifests **не включаются в
Git-репозиторий**. Для воспроизведения необходимо отдельно получить OSDaR23,
принять условия его распространения и построить локальное представление
скриптами проекта. Каталог `data/` исключён через `.gitignore`.

## Текущий протокол: person v5 range-aware

`person-canonical-v5-range-aware-v1` проверяет узкую абляцию поверх YOLO11m:

```text
B0: исходный person baseline
V5-A: B0 + P2
V5-B: B0 + P2 + Coordinate Attention
V5-C: B0 + perspective-aware masked person pasting
V5-D: B0 + P2 + person pasting
```

Предварительная диагностика B0/B1/D1-best/D1-last выполнена одним evaluator
при одинаковых tiling, fusion, threshold, IoU и folds. Она прошла проверку
воспроизводимости, но CrowdHuman zero-shot уступил D1-best по macro Recall
(`0.24242` против `0.33264`) и worst-fold Recall (`0.09098` против
`0.18196`). Поэтому gradual transfer V5-E исключён заранее зафиксированным
правилом.

Для train-only instance bank RGB bbox и LiDAR cuboid связываются только по
одному OpenLABEL object UUID. Если такой связи нет, объект получает явную
метку `bbox_area_scale_fallback`; размер рамки не называется дальностью.
Railway inference остаётся RGB-only.

Официальная матрица остановлена после полного V5-B. V5-A и V5-B получили
two-fold `FAIL`. До первой завершённой эпохи V5-C строгий аудит выявил
multiclass label-файлы в person-only dataset; частичный запуск изолирован и не
считается результатом.

Оставшийся вычислительный отбор вынесен в prospective amendment
`canonical-v5-expedited-screening-v1`:

```text
C0: B0 control
C1: B0 + audited 25% person pasting
C2: C1 + train-only hard-negative sampler

3 candidates × 5 epochs
→ at most 2 × 10 epochs
→ at most 1 × 20 epochs on fold 0
→ one confirmation fold
```

Screening не является материалом статьи и не заменяет полный OOF. Его сервис:

```bash
systemctl --user status tnorm-person-v5-screening.service --no-pager
journalctl --user -u tnorm-person-v5-screening.service -f
```

Amendment защищён отдельным hash-lock. Все кандидаты используют один
checkpoint, seed, fold, размер входа, tiling и evaluator. Решения записываются
в append-only `decision_trace.json`. До отдельного полного OOF PASS test и
атаки физически заблокированы.

## Текущий протокол: person v6 temporal T-norm

После отрицательного frame-only цикла `canonical-v6-person-temporal-tnorm-v1`
проверяет дополнительную информацию уже существующих видеопоследовательностей,
не меняя B0:

```text
T0-A: raw B0
T0-B: ByteTrack-style causal association
T0-C: Bayesian temporal existence fusion
T0-D: Bayesian fusion + Product T-norm reliability gate
```

Окно содержит текущий и не более четырёх прошлых кадров и никогда не пересекает
`subsequence_id`. Камера компенсируется ORB/RANSAC homography; не прошедшее
quality gate преобразование получает нулевой temporal weight. Fold 0 является
development-решением. Fold 1 физически не читается runner-ом при fold-0 FAIL.
Test и атаки остаются закрытыми.

Фактический T0 завершён с `FAIL`. ByteTrack-style вариант поднял Recall, но
увеличил FP/frame примерно в пять раз. Bayesian + Product дал только
`ΔRecall=+0.0032` и `Δsmall Recall=+0.0036`, что существенно ниже frozen gate.
Полный отчёт: [`reports/v6/V6_T0_FINAL_REPORT.md`](reports/v6/V6_T0_FINAL_REPORT.md).

Запуск после создания frozen lock:

```bash
systemctl --user status tnorm-person-v6-t0.service --no-pager
journalctl --user -u tnorm-person-v6-t0.service -f
```

## Текущий протокол: person v7 tracklet verifier

`canonical-v7-person-tracklet-verifier-v1` проверяет, можно ли отделить
дополнительные TP высокорекольного ByteTrack-потока от FP без переобучения B0.
Два монотонных scorer-а обучались только по train-сценам fold 0; модель,
калибровка и standard/safety thresholds выбирались по grouped train-scene OOF
до чтения held-out fold 0.

V0 завершён с `FAIL`. Выбранный monotonic gradient boosting получил на
held-out standard point `mAP50=0.22048`, `Recall=0.08776`,
`small Recall=0.02681`, то есть уступил B0 (`0.25114/0.20129/0.14656`).
Отдельно зафиксированный crop-amendment также завершён с `FAIL`: train-OOF
выбрал V2 fusion, но held-out standard дал `0.22571/0.09098/0.02949`.
Fold 1 не читался. Дополнительный подбор verifier-а на тех же сценах запрещён.

Отчёты:
[`V0`](reports/v7/V7_V0_FINAL_REPORT.md) и
[`V1/V2 crop`](reports/v7/V7_CROP_FINAL_REPORT.md).

Официальное состояние исполнения:

```text
runtime/v5/RUN_STATE.json
runtime/v5/heartbeat.json
runtime/v5/completed/
```

Итог двух folds автоматически формирует scene-level результаты, 10 000
парных cluster-bootstrap выборок, Holm correction и решение gate в
`results/v5/`. Все pre-result amendments сохранены в `protocol/v5/` и
`protocols/person_canonical_v5_range_aware_v1/`.

## Предыдущий протокол: person v5 data-first

`canonical-v5-person-data-first-v1` возвращается к простому ERM и меняет
источник предобучения, а не формулу loss:

```text
COCO YOLO11m
→ CrowdHuman person-only pretraining
→ railway fold fine-tuning с train-only hard mining
→ folds 0/1
→ только при gate — folds 2–4
```

CrowdHuman использован только в рамках его условий для некоммерческих
исследований и образования. Выбраны `vbox` (видимая область человека);
`mask` и записи с `extra.ignore=1` исключаются. Изображения CrowdHuman,
архивы и производное YOLO-представление не входят в Git или release bundle.

- Официальная страница:
  [CrowdHuman](https://www.crowdhuman.org/)
- Формат, загрузка и условия:
  [CrowdHuman download](https://www.crowdhuman.org/download.html)
- Статья:
  [CrowdHuman: A Benchmark for Detecting Human in a Crowd](https://arxiv.org/abs/1805.00123)

На момент заморозки официальные Google Drive URL возвращали HTTP 404.
Протокол допускает только транспортный mirror с теми же именами файлов и
обязательной фиксацией SHA-256 после получения. Test CrowdHuman не нужен и не
загружается.

Загрузка требует явного подтверждения условий и сохраняет resume-файлы:

```bash
PYTHONPATH="$PWD:$PWD/scripts" .venv/bin/python \
  scripts/person_v5/download_crowdhuman.py \
  --accept-noncommercial-research-terms
```

Без этого флага скрипт завершится до первого сетевого запроса. После загрузки
конвертация `vbox` выполняется отдельно и создаёт audit:

```bash
PYTHONPATH="$PWD:$PWD/scripts" .venv/bin/python \
  scripts/person_v5/prepare_crowdhuman.py \
  --source data/crowdhuman_downloads \
  --accept-noncommercial-research-terms
```

На experiment host обе стадии запускаются одним resumable user-service:

```bash
systemctl --user link "$PWD/systemd/tnorm-person-v5-data.service"
systemctl --user daemon-reload
systemctl --user enable --now tnorm-person-v5-data.service
```

Сервис выполняет только download и conversion/audit. Он не запускает GPU
training и сохраняет runtime provenance под игнорируемым
`outputs/person_v5/`.

После `crowdhuman_data_audit=PASS` data-service сам запускает runtime lock и
передаёт управление training-service. Те же действия можно выполнить вручную:

```bash
PYTHONPATH="$PWD:$PWD/scripts" .venv/bin/python \
  scripts/person_v5/lock_runtime.py
systemctl --user link "$PWD/systemd/tnorm-person-v5-training.service"
systemctl --user daemon-reload
systemctl --user enable --now tnorm-person-v5-training.service
```

Training-service выполнил CrowdHuman pretraining, затем ровно D1 folds 0/1 с
train-only hard mining и независимым global evaluator. Двухфолдовый D1 gate
завершился FAIL; test и атаки остались заблокированными. Эти результаты не
перезаписываются новой range-aware веткой.

Первичный v5-кандидат только один: YOLO11m с CrowdHuman pretraining.
RT-DETR отложен до отдельного prospective amendment; NWD/QFL, GroupDRO,
MixStyle и SWAD не используются.

Gate folds 0/1:

- macro mAP50 ≥ `0.45`;
- macro Recall ≥ `0.45`;
- macro small Recall ≥ `0.30`;
- worst-fold Recall ≥ `0.30`;
- улучшены оба fold относительно D0;
- evaluator consistency PASS, lost GT/NaN/Inf = 0.

До прохождения полного OOF gate test и атаки физически закрыты.

## Предыдущие замороженные протоколы

- Полный DG/NWD: `configs/canonical_v4_person_dg_nwd.yaml`
- Expedited amendment:
  `configs/canonical_v4_person_dg_nwd_expedited.yaml`
- Protocol locks:
  `outputs/person_v4*/protocol/protocol_lock.json`

Person v4 использует:

- A1: NWD assignment, гибридный CIoU/NWD box loss и QFL;
- A2: A1 + scene-wise GroupDRO;
- A3: A2 + different-scene MixStyle, ограниченные scale-aware augmentation и
  SWAD.

Expedited amendment не меняет initialization, seed, folds, эпохи, tiling,
evaluator или коэффициенты loss. Он сокращает только число запускаемых
кандидатов.

### Expedited decision tree

1. Завершить `A1/fold0` и выполнить независимый evaluator.
2. Запустить `A1/fold1` только при одновременном улучшении fold 0:
   `mAP50`, Recall и small Recall минимум на `0.05`.
3. Если A1 исключён, пропустить A2 и проверить полный A3 на fold 0.
4. Fold 1 разрешён только кандидату, прошедшему fold-0 gate.
5. Победитель должен иметь на folds 0/1:
   macro mAP50 ≥ `0.45`, macro Recall ≥ `0.45`,
   прирост small Recall ≥ `0.05`, worst-fold Recall ≥ `0.25`.
6. Только первый прошедший кандидат выполняет folds 2–4.
7. Test остаётся закрытым до полного пятифолдового OOF PASS.

Все решения сохраняются в
`outputs/person_v4_expedited/decision_trace.json`.

## Быстрая проверка состояния

```bash
cat outputs/person_v4_proxy/results/proxy_gate.json
cat protocols/canonical_v5_person_data_first_v1/protocol_lock.json
test ! -e outputs/person_v3/test/TEST_OPENED.json
```

Во время перехода с полного протокола также используется transient handoff
unit. Наличие `activating` у `Type=oneshot` означает выполняющийся pipeline, а
не зависание.

## Установка и проверки

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Для тестов нужен `pytest`. Проверки frozen protocol и decision gates не требуют
локального датасета или весов:

```bash
.venv/bin/pip install pytest
PYTHONPATH="$PWD:$PWD/scripts" \
  .venv/bin/python -m pytest -q --import-mode=importlib \
  tests/test_person_v4_protocol.py \
  tests/test_person_v4_expedited.py \
  tests/test_person_v5_protocol.py
```

Полный acceptance suite дополнительно требует локальные OSDaR23-derived
manifests, frozen checkpoints и evidence outputs, которые намеренно не
публикуются в Git:

```bash
PYTHONPATH="$PWD:$PWD/scripts" \
  .venv/bin/python -m pytest -q --import-mode=importlib tests
```

## Структура

```text
configs/                 frozen scientific protocols
scripts/person_v4/       training, evaluator and expedited decision pipeline
scripts/person_v5/       data-first protocol lock and CrowdHuman conversion
scripts/person_canonical_v5/
                         P2, range linkage, person pasting and candidate gates
scripts/person_v6/       causal temporal T0 runner and protocol lock
src/temporal/            camera compensation and temporal evidence aggregation
systemd/                 resumable user services
tests/                   protocol, math and leakage checks
data/                    local datasets and manifests; not a release artifact
outputs/person_v3/       immutable failed person baseline
outputs/person_v4/       full-protocol checkpoints and fold evidence
outputs/person_v4_expedited/
                         expedited decisions, OOF gate and bundle
outputs/person_v5/       local v5 evidence; excluded from Git
outputs/person_canonical_v5/screening/
                         local successive-halving evidence; excluded from Git
article/                  read-only template tooling and article validation
```

## Claim boundaries

- Micro-overfit доказывает только техническую обучаемость.
- Двухфолдовый triage используется только для экономии вычислений.
- H1–H4 нельзя считать проверенными до полного OOF gate, однократного test и
  canonical attack matrix.
- Test не выбирает checkpoint, threshold, attack budget, признаки или модель.
- T-нормы рассматриваются как диагностические признаки. Product preprocessing,
  median и bilateral — отдельные входные преобразования.
- Отрицательные и skipped результаты сохраняются в decision trace.

Историческое описание старого эксперимента находится в `HANDOFF.md`.
Актуальные ограничения и порядок практики поддерживаются в `AGENTS.md` и
`.codex/notes/final-practice.md`.

## Что нельзя очищать во время вычислений

Не удалять:

- `outputs/person_v4*`;
- checkpoints, prediction CSV, protocol locks и completion markers;
- `outputs/rejected_runs`;
- `data/`;
- активное `.venv/`;
- логи systemd до формирования итогового bundle.

Безопасно пересоздаются только Python/pytest caches и пустые локальные `runs/`.
