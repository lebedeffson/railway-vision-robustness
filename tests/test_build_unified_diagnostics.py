from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

try:
    import pandas as pd
    from build_unified_diagnostics import build
except ModuleNotFoundError:
    pd = None


@unittest.skipIf(pd is None, "optional numerical test dependencies are not installed")
class BuildUnifiedDiagnosticsTest(unittest.TestCase):
    def test_legacy_columns_are_mapped_to_final_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images/test/scene.1__frame.png"
            image.parent.mkdir(parents=True)
            image.touch()
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "split", "group", "sequence", "frame_id", "output_image", "annotations"
                ])
                writer.writeheader()
                writer.writerow({
                    "split": "test", "group": "scene", "sequence": "scene.1",
                    "frame_id": "1", "output_image": str(image), "annotations": "0",
                })

            base = {
                "image_path": str(image), "split": "test", "attack": "pgd",
                "epsilon_px": 1.0, "defense": "tnorm", "level": "P3",
                "cos_norm_attack": 0.7, "mse_attack": 0.1, "mae_attack": 0.2,
                "relative_l2_attack": 0.3, "mean_shift_attack": 0.4,
                "entropy_change_attack": 0.05, "product_attack": 0.6,
                "godel_attack": 0.5, "lukas_attack": 0.4,
                "recovery_cos_norm": 0.5, "recovery_mse": 0.5,
                "recovery_mae": 0.5, "recovery_relative_l2": 0.5,
                "recovery_mean_shift": 0.5, "recovery_entropy": 0.5,
                "recovery_product": 0.5, "recovery_godel": 0.5,
                "recovery_lukas": 0.5, "p_product": 0.9, "a_product": 0.6,
                "r_product": 0.8, "g_product": 0.5, "c_def_product": 0.45,
            }
            features = root / "features.csv"
            pd.DataFrame([base]).to_csv(features, index=False)

            detections = root / "detections.csv"
            detection_rows = [
                {"image_path": str(image), "split": "test", "attack": "clean", "epsilon_px": 0,
                 "defense": "none", "f1": 0.8, "recall": 0.7, "fn": 2},
                {"image_path": str(image), "split": "test", "attack": "pgd", "epsilon_px": 1.0,
                 "defense": "none", "f1": 0.4, "recall": 0.3, "fn": 5},
                {"image_path": str(image), "split": "test", "attack": "pgd", "epsilon_px": 1.0,
                 "defense": "tnorm", "f1": 0.5, "recall": 0.4, "fn": 4},
            ]
            pd.DataFrame(detection_rows).to_csv(detections, index=False)
            result = build(features, detections, manifest, [])
            row = result.iloc[0]
            self.assertEqual(row["sequence_id"], "scene")
            self.assertEqual(row["steps"], 20)
            self.assertAlmostEqual(row["damage"], 0.4)
            self.assertAlmostEqual(row["recovery"], 0.1)
            self.assertAlmostEqual(row["mse"], 0.1)
            self.assertAlmostEqual(row["c_def"], 0.45)


if __name__ == "__main__":
    unittest.main()
