from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_canonical_m4_expedited_triage import classify


class CanonicalM4ExpeditedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = yaml.safe_load(
            Path("configs/canonical_v2_m4_expedited_protocol.yaml").read_text(
                encoding="utf-8"
            )
        )

    @staticmethod
    def evaluation(map50: float, recall: float) -> dict:
        return {
            "mAP50": map50,
            "mAP50_95": 0.1,
            "safety_precision": 0.5,
            "safety_recall": recall,
            "safety_f1": 0.4,
            "evaluator_consistency_passed": True,
            "no_missing_scene": True,
            "no_nan_or_inf": True,
        }

    def test_promising_requires_both_frozen_thresholds(self) -> None:
        scenes = pd.DataFrame({
            "recall": [0.4, 0.5],
            "tp": [4, 5],
            "fn": [6, 5],
        })
        status, reasons = classify(
            self.evaluation(0.25, 0.35), scenes, 0, self.protocol
        )
        self.assertEqual(status, "promising")
        self.assertEqual(reasons, [])

    def test_low_map_is_hard_fail(self) -> None:
        status, reasons = classify(
            self.evaluation(0.249, 0.8),
            pd.DataFrame({
                "recall": [0.8, 0.7],
                "tp": [8, 7],
                "fn": [2, 3],
            }),
            0,
            self.protocol,
        )
        self.assertEqual(status, "hard_fail")
        self.assertIn("map50_below_triage_threshold", reasons)

    def test_zero_scene_recall_is_catastrophic(self) -> None:
        status, reasons = classify(
            self.evaluation(0.5, 0.5),
            pd.DataFrame({
                "recall": [0.5, 0.0],
                "tp": [5, 0],
                "fn": [5, 3],
            }),
            0,
            self.protocol,
        )
        self.assertEqual(status, "hard_fail")
        self.assertIn("catastrophic_scene_failure", reasons)

    def test_empty_gt_scene_is_not_catastrophic(self) -> None:
        status, reasons = classify(
            self.evaluation(0.5, 0.5),
            pd.DataFrame({
                "recall": [0.5, 0.0],
                "tp": [5, 0],
                "fn": [5, 0],
            }),
            0,
            self.protocol,
        )
        self.assertEqual(status, "promising")
        self.assertNotIn("catastrophic_scene_failure", reasons)

    def test_expedited_protocol_keeps_test_sealed(self) -> None:
        self.assertTrue(self.protocol["test_sealed"])
        self.assertTrue(
            self.protocol["attacks_blocked_until_official_validation_pass"]
        )
        self.assertEqual(
            self.protocol["expedited_full_training"]["seed"], 20260722
        )


if __name__ == "__main__":
    unittest.main()
