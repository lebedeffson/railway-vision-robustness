from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/final_practice/interim_audit"
SIZE = 1280


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    index = max(i for i, value in enumerate(parts) if value == "images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) == 5:
            rows.append((int(float(values[0])), *(float(value) for value in values[1:])))
    return rows


def letterbox_boxes(
    image: np.ndarray, boxes: list[tuple[int, float, float, float, float]]
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    height, width = image.shape[:2]
    scale = min(SIZE / width, SIZE / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    left = round((SIZE - resized_width) / 2 - .1)
    top = round((SIZE - resized_height) / 2 - .1)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    transformed = []
    for class_id, x, y, box_width, box_height in boxes:
        center_x = (x * width * scale + left) / SIZE
        center_y = (y * height * scale + top) / SIZE
        transformed.append((
            class_id, center_x, center_y,
            box_width * width * scale / SIZE,
            box_height * height * scale / SIZE,
        ))
    return canvas, transformed


def mask(boxes: list[tuple[int, float, float, float, float]], size: int) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.uint8)
    for _, x, y, width, height in boxes:
        left = max(0, min(size, int(np.floor((x - width / 2) * size))))
        right = max(0, min(size, int(np.ceil((x + width / 2) * size))))
        top = max(0, min(size, int(np.floor((y - height / 2) * size))))
        bottom = max(0, min(size, int(np.ceil((y + height / 2) * size))))
        result[top:bottom, left:right] = 1
    return result


def draw_boxes(image: np.ndarray, boxes: list[tuple[int, float, float, float, float]]) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for class_id, x, y, box_width, box_height in boxes:
        left, right = int((x - box_width / 2) * width), int((x + box_width / 2) * width)
        top, bottom = int((y - box_height / 2) * height), int((y + box_height / 2) * height)
        cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(output, str(class_id), (left, max(12, top)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    return output


def main() -> None:
    selected = pd.read_csv(ROOT / "independent_recheck_frames.csv")
    examples = selected.groupby("sequence_id", group_keys=False).head(2).head(5)
    destination = ROOT / "visualizations"
    destination.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(examples.itertuples(index=False), 1):
        image_path = Path(row.image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        boxes = read_boxes(label_path(image_path))
        boxed = draw_boxes(rgb, boxes)
        letterboxed, transformed = letterbox_boxes(rgb, boxes)
        figure, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes[0, 0].imshow(boxed); axes[0, 0].set_title("input + ground truth")
        axes[0, 1].imshow(letterboxed); axes[0, 1].set_title("1280 letterbox")
        axes[0, 2].text(
            .5, .5,
            "Raw predictions, gradients and perturbation maps\nare not stored in the legacy CSV.\nScheduled for isolated pilot.",
            ha="center", va="center", wrap=True,
        ); axes[0, 2].set_title("unavailable legacy tensors")
        for axis, level, resolution in zip(axes[1], ("P3", "P4", "P5"), (160, 80, 40), strict=True):
            level_mask = mask(transformed, resolution)
            axis.imshow(level_mask, cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"object mask {level} ({resolution}x{resolution})")
        for axis in axes.flat:
            axis.axis("off")
        figure.suptitle(f"{row.sequence_id}: {image_path.name}")
        figure.tight_layout()
        figure.savefig(destination / f"{index:02d}_{image_path.stem}_masks.png", dpi=140)
        plt.close(figure)
    (destination / "README.md").write_text(
        "Ground-truth and independently scaled P3/P4/P5 masks. Raw predictions, gradients, "
        "perturbations and overlap tensors were not stored by the active legacy process and "
        "must be produced by the isolated pilot before canonical test.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
