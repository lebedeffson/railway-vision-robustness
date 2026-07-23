---
title: "T-нормы как диагностические признаки повреждения и восстановления представлений железнодорожного детектора"
author: "TNormFilter canonical v2"
lang: ru-RU
---

# Аннотация

[TBD_RU_ABSTRACT]

**Ключевые слова:** обнаружение объектов, железнодорожные сцены, T-нормы, FGSM, PGD, диагностические признаки, YOLO11m.

# Abstract

[TBD_EN_ABSTRACT]

**Keywords:** object detection, railway scenes, T-norms, FGSM, PGD, diagnostic features, YOLO11m.

# 1. Цель и гипотезы

Исследование проверяет дополнительную диагностическую ценность канонических Product и Łukasiewicz T-норм сверх cosine similarity и стандартных расстояний. H1 относится к повреждению признаков (D3 против D2), H2 — к восстановлению (R3 против R2), H3 — к объектной и глобальной согласованности атаки, H4 — к adaptive PGD против Product preprocessing. Положительный результат не задавался как условие завершения.

# 2. Данные и протокол

[TBD_SPLIT_TEXT]

Разбиение выполнено по `grouped_scene_id`; соседние кадры одной независимой сцены не пересекают train, validation и test. Checkpoint, confidence threshold, бюджеты атак и нормализация фиксировались до анализа test. Единицей cross-validation и bootstrap была сгруппированная сцена.

# 3. Модель и рабочая точка

[TBD_MODEL_TEXT]

# 4. Атаки и диагностические показатели

Использованы FGSM, PGD-20 с random start и тремя restart, а также adaptive PGD-20 через полный дифференцируемый конвейер Product preprocessing → YOLO11m → detection loss. Для каждого возмущения сохранены фактические L1, L2 и L∞, начальный и конечный attack loss. Нормализация P3/P4/P5 была обучена отдельно по слоям и каналам только на чистой validation-выборке.

Для восстановления использовались $P$, $A$, $R$, необрезанное $G=(R-A)/(1-A+\tau)$ и $C_{def}=T(P,clip(G,0,1))$. Legacy compatibility-показатели не смешивались с каноническими T-нормами.

# 5. Результаты

## 5.1. Clean test

[TBD_CLEAN_TEXT]

## 5.2. Повреждение D3 против D2

[TBD_DAMAGE_TEXT]

## 5.3. Восстановление R3 против R2

[TBD_RECOVERY_TEXT]

## 5.4. Object/global и adaptive/non-adaptive

[TBD_H3_H4_TEXT]

## 5.5. Вычислительная стоимость

[TBD_LATENCY_TEXT]

# 6. Обсуждение

[TBD_DISCUSSION]

Основные ограничения: только пять независимых test-сцен, сложность и неоднородность OSDaR23, зависимость выводов от одной архитектуры YOLO11m и от validation-замороженной рабочей точки. Cluster bootstrap характеризует неопределённость для этих сцен, но не создаёт дополнительные независимые сцены. Отрицательные и малые эффекты сохранены без подбора условий по test.

# 7. Заключение

[TBD_CONCLUSION]

# Воспроизводимость

Split manifest, checkpoint hash, frozen threshold, budgets, raw matrices, scene-level statistics, LOSO, Holm/BH corrections, NMS audit, figures and checksums are included in the canonical evidence bundle.

