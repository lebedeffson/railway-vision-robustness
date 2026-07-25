from __future__ import annotations

import numpy as np


def constant_velocity_box(
    previous_box: list[float] | None,
    current_box: list[float],
    gap: int = 1,
) -> list[float]:
    if previous_box is None:
        return list(current_box)
    previous = np.asarray(previous_box, dtype=float)
    current = np.asarray(current_box, dtype=float)
    return (current + gap * (current - previous)).tolist()

