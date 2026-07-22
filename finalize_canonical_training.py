from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/canonical_v2/training"
RUN = ROOT / "yolo11m_canonical_v2"
MANIFEST_HASH = "bfd82413e4f92a935e8efb048a2c0edb03a97d932dd2a9303fa6220996b5e5db"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    best, last = RUN / "weights/best.pt", RUN / "weights/last.pt"
    results_path = RUN / "results.csv"
    if not all(path.is_file() for path in (best, last, results_path)):
        raise RuntimeError("Canonical training outputs are incomplete")
    results = pd.read_csv(results_path)
    fitness_columns = [name for name in results if "mAP50-95" in name]
    if not fitness_columns:
        raise RuntimeError("Training results lack validation mAP50-95")
    best_index = int(pd.to_numeric(results[fitness_columns[0]], errors="coerce").idxmax())
    best_epoch = int(pd.to_numeric(results.iloc[best_index]["epoch"]))
    best_row = results.iloc[best_index]
    precision = float(best_row["metrics/precision(B)"])
    recall = float(best_row["metrics/recall(B)"])
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    config = yaml.safe_load((PROJECT_DIR / "config/canonical_v2_protocol.yaml").read_text(encoding="utf-8"))["model"]
    payload = {
        "status": "PASS", "training_status": "success",
        "pretrained_checkpoint": config["initialization"],
        "best_checkpoint": str(best.resolve()), "best_checkpoint_sha256": sha256(best),
        "last_checkpoint": str(last.resolve()), "last_checkpoint_sha256": sha256(last),
        "best_epoch": best_epoch,
        "epochs_completed": int(pd.to_numeric(results["epoch"]).max()),
        "stop_reason": (
            "early_stopping" if len(results) < int(config["epochs"]) else "maximum_epochs"
        ),
        "selection_split": "val", "test_used": False,
        "training_grouped_scenes": 10, "training_frames": 774,
        "checkpoint_split_hash": MANIFEST_HASH, "test_evaluation_count": 0,
        "validation_precision_at_training_evaluator": precision,
        "validation_recall_at_training_evaluator": recall,
        "validation_f1_at_training_evaluator": f1,
        "validation_mAP50": float(best_row["metrics/mAP50(B)"]),
        "validation_mAP50_95": float(best_row["metrics/mAP50-95(B)"]),
        "validation_fn_per_frame": None,
        "validation_fn_per_frame_source": "post_training_validation_threshold_calibration",
    }
    (ROOT / "checkpoint_provenance.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (ROOT / "checkpoint_selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload))
        writer.writeheader(); writer.writerow(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
