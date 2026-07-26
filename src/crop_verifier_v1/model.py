from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedVerifier:
    model: Pipeline
    calibrator: LogisticRegression
    oof_probabilities: np.ndarray
    labelled_mask: np.ndarray


def _factory(settings: dict[str, Any]) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                penalty="l2",
                C=float(settings["C"]),
                class_weight=settings["class_weight"],
                max_iter=int(settings["max_iter"]),
                random_state=int(settings["seed"]),
            ),
        ),
    ])


def fit_grouped_verifier(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    settings: dict[str, Any],
    maximum_folds: int,
) -> FittedVerifier:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    labelled = y >= 0
    if not np.isfinite(x).all():
        raise RuntimeError("Verifier matrix contains NaN/Inf")
    if set(np.unique(y[labelled])) != {0, 1}:
        raise RuntimeError("Verifier requires both labelled classes")
    unique_groups = np.unique(groups[labelled])
    folds = min(int(maximum_folds), len(unique_groups))
    if folds < 2:
        raise RuntimeError("Verifier requires at least two scene groups")
    labelled_index = np.where(labelled)[0]
    raw = np.full(len(y), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=folds)
    for local_train, local_validation in splitter.split(
        x[labelled], y[labelled], groups[labelled]
    ):
        train = labelled_index[local_train]
        validation = labelled_index[local_validation]
        if set(np.unique(y[train])) != {0, 1}:
            raise RuntimeError("Grouped inner-train split lacks a class")
        model = _factory(settings)
        model.fit(x[train], y[train])
        raw[validation] = model.decision_function(x[validation])
    if not np.isfinite(raw[labelled]).all():
        raise RuntimeError("Grouped OOF scores contain NaN/Inf")
    calibrator = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        random_state=int(settings["seed"]),
    )
    calibrator.fit(raw[labelled].reshape(-1, 1), y[labelled])
    probabilities = np.full(len(y), np.nan, dtype=float)
    probabilities[labelled] = calibrator.predict_proba(
        raw[labelled].reshape(-1, 1)
    )[:, 1]
    final = _factory(settings)
    final.fit(x[labelled], y[labelled])
    return FittedVerifier(final, calibrator, probabilities, labelled)


def predict_verifier(fitted: FittedVerifier, x: np.ndarray) -> np.ndarray:
    raw = fitted.model.decision_function(np.asarray(x, dtype=np.float64))
    return fitted.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

