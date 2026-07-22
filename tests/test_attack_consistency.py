from __future__ import annotations

import unittest

try:
    import torch
    from extract_attack_consistency import AttackResult, consistency_row
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class AttackConsistencyTest(unittest.TestCase):
    def test_batched_adversarial_matches_unbatched_mask(self) -> None:
        clean = torch.zeros(3, 8, 8)
        adversarial = torch.full((1, 3, 8, 8), 0.1 / 255.0)
        gradient = torch.ones(1, 3, 8, 8)
        result = AttackResult(adversarial, gradient, gradient, 0.5, 1.0)
        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[2:6, 2:6] = True
        row = consistency_row(
            clean, result, mask, 0.1, gradient[0], "path_gradient_"
        )
        self.assertIn("path_gradient_c_sp_object", row)
        self.assertGreater(row["path_gradient_c_sp_object"], 0.0)
        self.assertGreater(row["path_gradient_c_atk_product_object"], 0.0)


if __name__ == "__main__":
    unittest.main()
