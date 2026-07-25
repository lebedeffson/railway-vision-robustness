"""Leakage-safe tracklet verification for canonical person v7."""

from .verifier import (
    FEATURE_NAMES,
    MONOTONIC_DIRECTIONS,
    MonotoneRankLogistic,
    build_tracklet_features,
    label_tracklets,
)

__all__ = [
    "FEATURE_NAMES",
    "MONOTONIC_DIRECTIONS",
    "MonotoneRankLogistic",
    "build_tracklet_features",
    "label_tracklets",
]

