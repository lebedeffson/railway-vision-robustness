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
                        "adaptive": adaptive, "attack": "pgd", "epsilon_px": 0.1,
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
            argv = [
                "plot_final_practice.py", "--input", str(input_path),
                "--statistics", str(stats), "--output", str(output),
            ]
            with patch.object(sys, "argv", argv):
                plot_final_practice.main()
            self.assertEqual(len(list(output.glob("*.png"))), 8)


if __name__ == "__main__":
    unittest.main()
