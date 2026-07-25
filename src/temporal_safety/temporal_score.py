from __future__ import annotations


def accumulated_score(previous: float, confidence: float, decay: float) -> float:
    return float(decay * previous + confidence)


def missing_score(previous: float, decay: float) -> float:
    return float(decay * previous)

