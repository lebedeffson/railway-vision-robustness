from __future__ import annotations

import unittest

try:
    import pandas as pd
    from analyze_final_practice import bootstrap_predictions, evaluate_task
except ModuleNotFoundError:
    pd = None


@unittest.skipIf(pd is None, "optional numerical test dependencies are not installed")
class AnalyzeFinalPracticeTest(unittest.TestCase):
    def data(self) -> "pd.DataFrame":
        rows = []
        for sequence_index in range(8):
            for attack_index, attack in enumerate(("fgsm", "pgd")):
                epsilon = (attack_index + 1) / 255.0
                base = 0.05 + 0.01 * sequence_index + 0.02 * attack_index
                common = {
                    "sequence_id": f"sequence_{sequence_index}",
                    "image_path": f"image_{sequence_index}_{attack}.png",
                    "split": "test",
                    "attack": attack,
                    "adaptive": False,
                    "epsilon": epsilon,
                    "steps": 20 if attack == "pgd" else 1,
                    "restart": 0,
                    "seed": 42,
                    "layer": "P3",
                    "perturbation_norm": "linf",
                    "f1_clean": 0.8,
                    "f1_attack": 0.8 - base,
                    "cosine": 1.0 - base,
                    "mse": base,
                    "mae": base,
                    "relative_l2": base,
                    "mean_shift": base,
                    "entropy": base,
                    "product": 1.0 - base,
                    "godel": 1.0 - base,
                    "lukasiewicz": 1.0 - base,
                    "c_sp_object": base,
                    "c_dir": 0.9,
                    "c_atk_object": base * 0.9,
                    "cosine_recovery": 0.5,
                    "mse_recovery": 0.5,
                    "mae_recovery": 0.5,
                    "relative_l2_recovery": 0.5,
                    "mean_shift_recovery": 0.5,
                    "entropy_recovery": 0.5,
                    "product_recovery": 0.5,
                    "godel_recovery": 0.5,
                    "lukasiewicz_recovery": 0.5,
                    "p_clean_preservation": 0.95,
                    "a_attacked_similarity": 1.0 - base,
                    "r_restored_similarity": 1.0 - base / 2,
                    "g_recovery": 0.5,
                    "c_def": 0.475,
                }
                rows.append({**common, "defense": "none", "f1_defended": 0.8 - base})
                rows.append({**common, "defense": "tnorm", "f1_defended": 0.8 - base / 2})
        return pd.DataFrame(rows)

    def test_models_and_sequence_bootstrap(self) -> None:
        data = self.data()
        for task in ("damage", "recovery"):
            models, predictions = evaluate_task(data, task)
            self.assertEqual(models["model"].tolist(), ["M0", "M1", "M2", "M3", "M4"])
            self.assertEqual(models["sequences"].unique().tolist(), [8])
            predictions.insert(0, "task", task)
            bootstrap = bootstrap_predictions(predictions, task, iterations=20, seed=1)
            self.assertEqual(len(bootstrap), 12)
            self.assertTrue((bootstrap["sequences"] == 8).all())


if __name__ == "__main__":
    unittest.main()
