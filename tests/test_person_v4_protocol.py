from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/canonical_v4_person_dg_nwd.yaml"


class PersonV4ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))

    def test_test_is_sealed(self) -> None:
        marker = ROOT / self.protocol["data"]["test_marker"]
        self.assertTrue(self.protocol["data"]["test_sealed"])
        self.assertFalse(marker.exists())

    def test_matrix_is_sequential_and_frozen(self) -> None:
        self.assertEqual(
            self.protocol["matrix"]["execution_order"],
            ["A0", "A1", "A2", "A3"],
        )
        self.assertEqual(self.protocol["training"]["triage_folds"], [0, 1])
        self.assertEqual(self.protocol["training"]["seed"], 20260723)

    def test_group_is_scene_and_accumulation_spans_scenes(self) -> None:
        dro = self.protocol["group_dro"]
        self.assertEqual(dro["group"], "grouped_scene_id")
        self.assertGreaterEqual(
            dro["minimum_distinct_scenes_per_optimizer_step"], 2
        )
        self.assertGreaterEqual(
            dro["nominal_batch_size_for_accumulation"], 2
        )

    def test_nwd_and_qfl_are_prospectively_defined(self) -> None:
        self.assertEqual(
            self.protocol["nwd"]["constant_rule"],
            "median_sqrt_area_of_person_boxes_after_letterbox_on_fold_train_only",
        )
        weights = self.protocol["quality_focal_loss"]["quality_target"]
        self.assertAlmostEqual(
            weights["IoU_weight"] + weights["NWD_weight"], 1.0
        )

    def test_selection_requires_all_primary_improvements(self) -> None:
        requirements = self.protocol["selection"]["require_all"]
        self.assertEqual(
            requirements["absolute_macro_mAP50_gain_min"], 0.05
        )
        self.assertEqual(
            requirements["absolute_macro_recall_gain_min"], 0.05
        )
        self.assertEqual(
            requirements["absolute_macro_small_recall_gain_min"], 0.05
        )
        self.assertTrue(
            requirements["worst_fold_recall_must_not_decrease"]
        )
        self.assertTrue(requirements["false_positives_must_decrease"])


if __name__ == "__main__":
    unittest.main()
