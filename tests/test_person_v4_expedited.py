from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from scripts.person_v4.run_expedited import (
    fold0_decision,
    two_fold_decision,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT / "configs/canonical_v4_person_dg_nwd_expedited.yaml"
)


class PersonV4ExpeditedProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = yaml.safe_load(
            PROTOCOL.read_text(encoding="utf-8")
        )

    def test_amendment_was_prospective(self) -> None:
        self.assertTrue(
            self.protocol[
                "created_before_A1_fold0_final_evaluation"
            ]
        )
        self.assertEqual(
            self.protocol["parent_commit"],
            "cd45bfe3a83da20025df4f283d2dd18037c9b6b1",
        )

    def test_test_and_attacks_remain_blocked(self) -> None:
        sealed = self.protocol["sealed_scope"]
        self.assertTrue(sealed["test_sealed"])
        self.assertTrue(
            sealed["attacks_blocked_until_full_OOF_gate"]
        )
        self.assertFalse(
            (ROOT / sealed["test_marker"]).exists()
        )

    def test_A2_is_skipped_prospectively(self) -> None:
        self.assertEqual(
            self.protocol["execution_tree"]["A2"],
            "skipped_by_expedited_amendment",
        )
        self.assertEqual(
            self.protocol["execution_tree"]["candidates_in_order"],
            ["A1", "A3"],
        )

    def test_fold0_gate_requires_all_three_gains(self) -> None:
        gate = self.protocol["execution_tree"]["fold0"][
            "require_all"
        ]
        self.assertEqual(gate["delta_mAP50_min"], 0.05)
        self.assertEqual(gate["delta_recall_min"], 0.05)
        self.assertEqual(gate["delta_small_recall_min"], 0.05)
        self.assertEqual(gate["lost_GT"], 0)
        self.assertEqual(gate["NaN_Inf"], 0)

    def test_two_fold_gate_matches_amendment(self) -> None:
        gate = self.protocol["execution_tree"]["two_fold"][
            "require_all"
        ]
        self.assertEqual(gate["macro_mAP50_min"], 0.45)
        self.assertEqual(gate["macro_recall_min"], 0.45)
        self.assertEqual(
            gate["delta_macro_small_recall_min"], 0.05
        )
        self.assertEqual(gate["worst_fold_recall_min"], 0.25)

    def test_training_recipe_is_unchanged(self) -> None:
        training = self.protocol["unchanged_training"]
        self.assertEqual(training["seed"], 20260723)
        self.assertEqual(training["epochs"]["stage1"], 5)
        self.assertEqual(training["epochs"]["stage2"], 15)
        self.assertTrue(training["loss_coefficients_unchanged"])

    def test_fold0_gate_is_conjunctive(self) -> None:
        baseline = {
            "macro_mAP50": 0.20,
            "macro_recall": 0.20,
            "macro_small_recall": 0.20,
        }
        candidate = {
            "variant": "A1",
            "macro_mAP50": 0.25,
            "macro_recall": 0.25,
            "macro_small_recall": 0.25,
            "technical_checks_passed": True,
        }
        self.assertTrue(fold0_decision(baseline, candidate)["passed"])
        candidate["macro_small_recall"] = 0.249
        self.assertFalse(
            fold0_decision(baseline, candidate)["passed"]
        )

    def test_two_fold_gate_is_conjunctive(self) -> None:
        baseline = {"macro_small_recall": 0.20}
        candidate = {
            "variant": "A3",
            "macro_mAP50": 0.45,
            "macro_recall": 0.45,
            "macro_small_recall": 0.25,
            "worst_fold_recall": 0.25,
            "technical_checks_passed": True,
        }
        self.assertTrue(
            two_fold_decision(baseline, candidate)["passed"]
        )
        candidate["worst_fold_recall"] = 0.249
        self.assertFalse(
            two_fold_decision(baseline, candidate)["passed"]
        )


if __name__ == "__main__":
    unittest.main()
