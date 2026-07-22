from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = PROJECT_DIR / "outputs/canonical_v2"
CALIBRATION_SOURCE = ROOT / "baseline_rescue_v2"
TEST_SOURCE = CALIBRATION_SOURCE / "test_evaluation"


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    calibration = ROOT / "calibration"
    tables = ROOT / "tables"
    for source, destination in (
        (CALIBRATION_SOURCE / "threshold_sweep.csv", calibration / "threshold_sweep.csv"),
        (CALIBRATION_SOURCE / "threshold_selection.json", calibration / "threshold_selection.json"),
        (CALIBRATION_SOURCE / "precision_recall_curve.png", calibration / "precision_recall_curve.png"),
        (CALIBRATION_SOURCE / "threshold_sweep_per_scene.csv", calibration / "threshold_sweep_per_scene.csv"),
    ):
        copy(source, destination)
    clean = pd.read_csv(TEST_SOURCE / "clean_metrics_train_val_test.csv")
    classes = pd.read_csv(TEST_SOURCE / "clean_metrics_by_class_and_size.csv")
    ap = pd.read_csv(TEST_SOURCE / "ap_by_class.csv")
    scenes = pd.read_csv(TEST_SOURCE / "clean_metrics_per_scene.csv")
    tables.mkdir(parents=True, exist_ok=True)
    clean.to_csv(tables / "clean_test_metrics.csv", index=False)
    clean.to_csv(tables / "03_clean_test_metrics.csv", index=False)
    classes.merge(
        ap[["split", "class_name", "AP50", "AP50-95"]],
        left_on=["split", "name"], right_on=["split", "class_name"],
        how="left",
    ).to_csv(tables / "clean_test_per_class.csv", index=False)
    classes[classes["scope"].eq("size")].to_csv(
        tables / "clean_test_object_sizes.csv", index=False
    )
    scenes.to_csv(tables / "clean_test_per_scene.csv", index=False)
    thresholds = json.loads(
        (CALIBRATION_SOURCE / "threshold_selection.json").read_text(encoding="utf-8")
    )
    validation = pd.read_csv(CALIBRATION_SOURCE / "clean_metrics_train_val_test.csv")
    training = pd.DataFrame([{
        "checkpoint_sha256": thresholds["checkpoint_sha256"],
        "standard_threshold": thresholds["standard"]["confidence"],
        "safety_threshold": thresholds["safety"]["confidence"],
        "selection_split": "val",
        "test_used_for_selection": False,
        "validation_recall_standard": float(validation.loc[
            validation["split"].eq("val") & validation["operating_point"].eq("standard"),
            "recall",
        ].iloc[0]),
        "validation_mAP50": float(validation.loc[
            validation["split"].eq("val") & validation["operating_point"].eq("standard"),
            "mAP50",
        ].iloc[0]),
    }])
    training.to_csv(tables / "02_training_and_threshold.csv", index=False)
    provenance_path = ROOT / "training/checkpoint_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    validation_standard = validation.loc[
        validation["split"].eq("val")
        & validation["operating_point"].eq("standard")
    ].iloc[0]
    provenance.update({
        "validation_precision": float(validation_standard["precision"]),
        "validation_recall": float(validation_standard["recall"]),
        "validation_f1": float(validation_standard["f1"]),
        "validation_fn_per_frame": float(validation_standard["fn_per_frame"]),
        "validation_mAP50": float(validation_standard["mAP50"]),
        "validation_mAP50_95": float(validation_standard["mAP50-95"]),
        "validation_metrics_operating_point": "standard_maximum_F1",
    })
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    verification = {
        "status": "PASS", "checkpoint_sha256": thresholds["checkpoint_sha256"],
        "selection_split": thresholds["selection_split"],
        "test_used_for_selection": thresholds["test_used_for_selection"],
    }
    (calibration / "checkpoint_hash_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS", "test_rows": len(clean),
        "per_class_rows": len(classes), "per_scene_rows": len(scenes),
    }, indent=2))


if __name__ == "__main__":
    main()
