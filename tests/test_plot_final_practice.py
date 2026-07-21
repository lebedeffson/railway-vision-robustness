import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import plot_final_practice


class PlotFinalPracticeTest(unittest.TestCase):
    def test_all_required_figures_are_created(self) -> None:
        rows = []
        for adaptive in (False, True):
            for defense in ("none", "tnorm"):
                for layer in ("P3", "P4", "P5"):
                    rows.append({
                        "selected_best": True, "defense": defense,
                        "adaptive": adaptive, "attack": "pgd", "epsilon_px": 0.1, "steps": 20,
                        "recall_attack": 0.4, "f1_attack": 0.3,
                        "f1_defended": 0.35 if defense == "tnorm" else 0.3,
                        "c_atk_object": 0.5, "damage": 0.2,
                        "g_recovery": 0.1, "recovery": 0.05,
                        "product": 0.8, "godel": 0.82, "lukasiewicz": 0.75,
                        "c_sp_object": 0.6, "c_sp_background": 0.3, "layer": layer,
                    })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw.csv"
            stats = root / "stats"
            output = root / "figures"
            stats.mkdir()
            pd.DataFrame(rows).to_csv(input_path, index=False)
            pd.DataFrame([{
                "task": "damage", "comparison": "M1_minus_M0", "metric": "r2",
                "observed_gain": 0.02, "ci_low": -0.01, "ci_high": 0.04,
            }]).to_csv(stats / "sequence_bootstrap.csv", index=False)
            utility = root / "utility.csv"
            latency = root / "latency.csv"
            pd.DataFrame([
                {"defense": "none", "f1": 0.3},
                {"defense": "tnorm", "f1": 0.35},
            ]).to_csv(utility, index=False)
            pd.DataFrame([
                {"method": "detector", "mean_latency_ms": 10.0},
                {"method": "tnorm+detector", "mean_latency_ms": 12.0},
            ]).to_csv(latency, index=False)
            argv = [
                "plot_final_practice.py", "--input", str(input_path),
                "--statistics", str(stats), "--output", str(output),
                "--clean-utility", str(utility), "--latency", str(latency),
            ]
            with patch.object(sys, "argv", argv):
                plot_final_practice.main()
            required = {
                "01_f1_recall_vs_epsilon.png",
                "02_attack_consistency_vs_f1_drop.png",
                "03_object_vs_background_consistency.png",
                "04_feature_recovery_vs_f1_recovery.png",
                "05_layerwise_recovery_heatmap.png",
                "06_product_godel_lukasiewicz_by_budget.png",
                "07_model_gain_with_ci.png",
                "08_adaptive_vs_nonadaptive_pgd.png",
                "09_clean_utility_vs_latency.png",
            }
            self.assertTrue(required.issubset({path.name for path in output.glob("*.png")}))


if __name__ == "__main__":
    unittest.main()
