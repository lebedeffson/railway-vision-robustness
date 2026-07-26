from __future__ import annotations

import cv2
import numpy as np


WARNING_LINES = (
    "RESEARCH MODE",
    "HIGH FALSE-ALARM RATE",
    "NOT FOR SAFETY DEPLOYMENT",
)


def draw_research_warning(
    frame: np.ndarray, lines: tuple[str, ...] = WARNING_LINES
) -> np.ndarray:
    output = frame.copy()
    height = 28 * len(lines) + 10
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], height), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (12, 26 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output
