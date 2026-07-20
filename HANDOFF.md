# TNormFilter handoff

> The numerical results below are the immutable v1 baseline only. They used
> image-level grouping and are not final article evidence. The corrected
> sequence-level protocol is in `FINAL_PRACTICE_PROTOCOL.md`; baseline source is
> tagged `diagnostics_v1_current`.

## Окружение

Python-проект для анализа устойчивости YOLO11m на OSDaR23.

Модель:

outputs/training/yolo11m_baseline_stage2/weights/best.pt

Конфигурация датасета:

data/yolo_osdar23/data.yaml

## Основные скрипты

- extract_feature_consistency.py
- evaluate_image_level_detection.py
- analyze_feature_diagnostics.py
- final_diagnostics.py
- make_final_figures.py

## Готовые результаты

Validation:

- outputs/diagnostics/feature_consistency/feature_consistency_val.csv
- outputs/diagnostics/image_detection/image_detection_val.csv
- outputs/diagnostics/analysis_val/

Test:

- outputs/diagnostics/feature_consistency/feature_consistency_test.csv
- outputs/diagnostics/image_detection/image_detection_test.csv
- outputs/diagnostics/analysis/

Финальная статистика:

- outputs/diagnostics/final_analysis/bootstrap_model_comparison.csv
- outputs/diagnostics/final_analysis/bootstrap_correlation_comparison.csv
- outputs/diagnostics/final_analysis/defense_policy_results.csv
- outputs/diagnostics/final_analysis/defense_policy.json
- outputs/diagnostics/final_analysis/final_summary.md
- outputs/diagnostics/final_analysis/figures/

## Запуск

Создание окружения:

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Повторный анализ test:

python analyze_feature_diagnostics.py

Повторный анализ validation:

python analyze_feature_diagnostics.py --features outputs/diagnostics/feature_consistency/feature_consistency_val.csv --detections outputs/diagnostics/image_detection/image_detection_val.csv --output outputs/diagnostics/analysis_val

Финальная статистика:

python final_diagnostics.py

Графики:

python make_final_figures.py

## Главные результаты

На test добавление T-норм к cosine, MSE, MAE, relative L2 и mean shift:

Damage:
- MAE gain: 0.000529, p=0.1059
- R2 gain: 0.006798, p=0.2438
- Spearman gain: 0.007154, p=0.2577

Recovery:
- MAE gain: 0.000943, p=0.0020
- R2 gain: 0.012800, p=0.0020
- Spearman gain: 0.006993, p=0.0040

Парные корреляции на test:

- Product сильнее cosine для damage:
  absolute rho gain = 0.007763, p=0.0020

- Lukasiewicz сильнее cosine для damage:
  absolute rho gain = 0.056232, p=0.0020

- Product не отличается от cosine для recovery:
  p=0.8232

- Lukasiewicz слабее cosine для recovery:
  absolute rho gain = -0.014697, p=0.0020

Политика выбора защиты по одному Lukasiewicz-показателю не обобщилась:

- validation выбрал JPEG
- test adaptive F1 = 0.039979
- лучший фиксированный test-метод: T-norm, F1 = 0.048566

## Текущий вывод

T-нормы полезны как дополнительные диагностические признаки.

Для damage отдельные Product и Lukasiewicz сильнее cosine.

Для recovery преимущество появляется при совместном использовании T-норм
со стандартными метриками в многомерной модели.

Простая пороговая политика выбора защиты не работает стабильно.

## Что осталось

- финально оформить Overleaf;
- добавить второй датасет или вторую модель для более сильной статьи;
- проверить естественные искажения;
- измерить скорость диагностического контура;
- добавить энтропию признаков и специализированные baseline-детекторы атак.
