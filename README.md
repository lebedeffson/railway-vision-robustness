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
| Person v4 DG/NWD | development-only expedited selection выполняется | запечатан |

Зафиксированный person-v3 baseline на folds 0/1:

- macro mAP50: `0.35883`;
- macro Recall: `0.35432`;
- macro small Recall: `0.21130`;
- worst-fold Recall: `0.20129`.

Эти числа — development evidence, а не итоговый test-результат.

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

## Замороженные протоколы

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
systemctl --user status tnorm-person-v4-expedited.service --no-pager
journalctl --user -u tnorm-person-v4-expedited.service -n 100 --no-pager
cat outputs/person_v4_expedited/pipeline_status.json
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
  tests/test_person_v4_expedited.py
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
systemd/                 resumable user services
tests/                   protocol, math and leakage checks
data/                    local datasets and manifests; not a release artifact
outputs/person_v3/       immutable failed person baseline
outputs/person_v4/       full-protocol checkpoints and fold evidence
outputs/person_v4_expedited/
                         expedited decisions, OOF gate and bundle
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
