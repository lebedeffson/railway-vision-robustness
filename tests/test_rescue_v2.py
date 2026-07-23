from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rescue_v2_common import (  # noqa: E402
    PROTOCOL_PATH,
    assert_frozen_inputs,
    assert_role_allowed,
    load_protocol,
)


class SmallSignalRescueV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()

    def test_protocol_identity_and_parent_are_frozen(self) -> None:
        self.assertEqual(
            self.protocol["protocol_id"],
            "canonical-v2-small-signal-rescue-v2",
        )
        self.assertEqual(
            self.protocol["parent_commit"],
            "4001c169c9dd368b8aef70abdcf004b885f237c2",
        )
        self.assertTrue(self.protocol["frozen_before_computation"])
        assert_frozen_inputs()

    def test_test_attacks_and_article_are_physically_blocked(self) -> None:
        for role in ("test", "attack", "article"):
            with self.assertRaisesRegex(RuntimeError, "quality gate missing"):
                assert_role_allowed(role)

    def test_failed_quality_gate_cannot_open_downstream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "quality_gate.json"
            gate.write_text(json.dumps({
                "quality_gate_passed": False,
                "checkpoint_frozen": False,
                "thresholds_frozen": False,
            }))
            for role in ("test", "attack", "article"):
                with self.assertRaisesRegex(RuntimeError, "did not pass"):
                    assert_role_allowed(role, gate)

    def test_micro_set_and_gates_are_not_relaxed(self) -> None:
        gate = self.protocol["micro_gate"]
        self.assertEqual(self.protocol["micro_dataset"]["frames"], 32)
        self.assertFalse(
            self.protocol["scientific_boundaries"]["micro_set_may_change"]
        )
        self.assertGreaterEqual(gate["map50_min"], 0.90)
        self.assertGreaterEqual(gate["recall_min"], 0.90)
        self.assertGreaterEqual(gate["small_recall_min"], 0.90)
        self.assertGreaterEqual(gate["medium_recall_min"], 0.95)
        self.assertGreaterEqual(gate["large_recall_min"], 0.95)
        self.assertTrue(gate["map_uses_full_confidence_ranking"])

    def test_micro_candidate_order_is_prospective_and_bounded(self) -> None:
        selection = self.protocol["micro_selection"]
        self.assertEqual(selection["order"], ["M1", "M2", "M3", "M4", "M5"])
        self.assertEqual(selection["maximum_winners"], 2)
        self.assertEqual(selection["rule"], "simplest_passing_candidate")
        self.assertTrue(selection["single_seed_cannot_open_test"])
        self.assertEqual(
            self.protocol["micro_candidates"]["M5"]["enabled_if"],
            "no_candidate_in_M1_M2_M3_M4_passes_micro_gate",
        )

    def test_fn_audit_contract_covers_all_errors_and_dynamic_strides(self) -> None:
        audit = self.protocol["fn_audit"]
        self.assertEqual(audit["expected_false_negatives"], 80)
        self.assertEqual(
            set(audit["error_types"]), {"A", "B", "C", "D", "E", "F", "G"}
        )
        self.assertEqual(audit["feature_strides_source"], "loaded_model")
        self.assertTrue(audit["hardcoded_feature_strides_forbidden"])
        self.assertEqual(audit["pixel_size_bins"][:5], [0, 4, 8, 16, 32])

    def test_full_training_is_limited_and_validation_only(self) -> None:
        full = self.protocol["full_training"]
        self.assertEqual(full["maximum_candidates"], 2)
        self.assertEqual(full["seeds"], [20260722, 20260723, 20260724])
        self.assertEqual(full["inner_validation"]["groups"], "grouped_scene_id")
        self.assertFalse(
            self.protocol["quality_gate"]["checkpoint_selection_uses_test"]
        )
        self.assertLessEqual(
            full["balanced_sampling"]["maximum_frame_weight"], 4.0
        )


if __name__ == "__main__":
    unittest.main()
