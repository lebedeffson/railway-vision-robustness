from __future__ import annotations

import unittest
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from deadline_select_budgets import budget_table, select_budgets
import run_final_matrix
from verify_canonical_provenance import verify_hash


ROOT = Path(__file__).resolve().parents[1]


class CanonicalV2Test(unittest.TestCase):
    def test_split_v2_has_five_independent_validation_and_test_scenes(self) -> None:
        frame = pd.read_csv(ROOT / "outputs/canonical_v2/split/split_v2_manifest.csv")
        self.assertTrue({"subsequence_id", "grouped_scene_id", "sequence_id"}.issubset(frame.columns))
        self.assertTrue(frame["sequence_id"].astype(str).equals(frame["grouped_scene_id"].astype(str)))
        groups = {
            split: set(scope.grouped_scene_id.astype(str))
            for split, scope in frame.groupby("split")
        }
        self.assertEqual({name: len(value) for name, value in groups.items()}, {
            "train": 10, "val": 5, "test": 5,
        })
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])

    def test_canonical_protocol_freezes_hypotheses_and_test_role(self) -> None:
        protocol = yaml.safe_load(
            (ROOT / "config/canonical_v2_protocol.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(protocol["statistical_unit"], "sequence_id")
        self.assertEqual(set(protocol["hypotheses"]), {"H1", "H2", "H3", "H4"})
        self.assertEqual(protocol["split_scenes"], {"train": 10, "val": 5, "test": 5})
        self.assertTrue(protocol["frozen_before_test"])
        self.assertFalse(protocol["legacy_thresholds_reused_by_canonical_v2"])
        order = protocol["canonical_stage_order"]
        self.assertLess(order.index("train_canonical_v2"), order.index("threshold_sweep_validation_v2"))
        self.assertLess(order.index("baseline_quality_gate"), order.index("clean_test_v2_once"))
        self.assertLess(order.index("clean_test_v2_once"), order.index("canonical_validation_attacks"))

    def test_canonical_analysis_uses_required_D2_D3_and_R2_R3_contract(self) -> None:
        protocol = yaml.safe_load(
            (ROOT / "config/canonical_v2_analysis.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(protocol["statistical_unit"], "sequence_id")
        self.assertEqual(protocol["bootstrap_iterations"], 5000)
        self.assertTrue({"product", "lukasiewicz"}.issubset(protocol["damage_models"]["D3"]))
        self.assertTrue({"product_recovery", "lukasiewicz_recovery", "g_recovery", "c_def"}.issubset(protocol["recovery_models"]["R3"]))
        self.assertNotIn("godel", protocol["damage_models"]["D3"])

    def test_canonical_matrix_schema_carries_group_and_subsequence(self) -> None:
        exact, _ = run_final_matrix.frame_metadata_lookup(
            ROOT / "data/yolo_osdar23_v2/manifest.csv"
        )
        self.assertTrue(exact)
        self.assertTrue(all(
            {"grouped_scene_id", "subsequence_id"} <= set(value)
            for value in exact.values()
        ))

    def test_article_pipeline_preserves_internal_template(self) -> None:
        mapping = yaml.safe_load(
            (ROOT / "article/result_mapping.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(mapping["template"], "article/internal_review_template.md")
        fill = (ROOT / "article/fill_article_results.py").read_text(encoding="utf-8")
        self.assertIn("template_sha256_before", fill)
        self.assertIn("template_sha256_after", fill)
        validation = (ROOT / "article/validate_final_article.py").read_text(encoding="utf-8")
        self.assertIn("delta_mae_signs_valid", validation)

    def test_legacy_smoke_failure_cannot_relax_canonical_gate(self) -> None:
        source = (ROOT / "finalize_legacy_smoke.py").read_text(encoding="utf-8")
        self.assertIn('ALLOWED_LEGACY_FAILURES = {"membership_saturation_below_20_percent"}', source)
        self.assertIn('"canonical_pilot_gate_affected": False', source)
        canonical = (ROOT / "canonical_v2_pilot_gate.py").read_text(encoding="utf-8")
        self.assertIn('"membership_saturation_below_20_percent": saturation_max < 0.20', canonical)

    def test_split_summary_reports_class_scene_size_and_scene_contributions(self) -> None:
        summary = json.loads(
            (ROOT / "outputs/canonical_v2/split/split_v2_summary.json").read_text()
        )
        self.assertEqual(summary["grouping_contract"]["independent_unit"], "grouped_scene_id")
        for split in ("train", "val", "test"):
            self.assertEqual(set(summary["class_grouped_scene_counts"][split]), set("012345"))
            self.assertEqual(set(summary["size_object_counts"][split]), {"small", "medium", "large"})
            self.assertEqual(len(summary["scene_contributions"][split]), {"train": 10, "val": 5, "test": 5}[split])

    def test_canonical_runtime_order_does_not_open_test_before_gate(self) -> None:
        source = (ROOT / "run_canonical_v2_pipeline.py").read_text(encoding="utf-8")
        calibration = source.index('stage("canonical_v2_threshold_calibration"')
        gate = source.index("quality_gate(v2_audit")
        clean_test = source.index('stage("canonical_v2_clean_test"')
        attacks = source.index('stage("canonical_validation"')
        self.assertLess(calibration, gate)
        self.assertLess(gate, clean_test)
        self.assertLess(clean_test, attacks)
        calibration_command = source[calibration:gate]
        self.assertIn('"--splits", "train,val"', calibration_command)
        self.assertNotIn('"--splits", "test"', calibration_command)

    def test_checkpoint_hash_must_match_threshold_clean_test_and_attacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"canonical checkpoint")
            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            for role in ("threshold", "clean_test", "validation_attack", "test_attack"):
                payload = {"checkpoint_sha256": expected}
                verify_hash(payload, expected, Path(role))
            with self.assertRaisesRegex(RuntimeError, "provenance mismatch"):
                verify_hash({"checkpoint_sha256": "wrong"}, expected, Path("attack"))

    def test_strict_budget_gate_does_not_fallback_to_clean_detectable_subset(self) -> None:
        rows = []
        for attack, adaptive, epsilon in (("fgsm", False, .1), ("pgd", False, .1), ("pgd", True, .25)):
            for index in range(6):
                rows.append({
                    "sequence_id": f"s{index % 5}", "image_path": f"{attack}-{adaptive}-{index}",
                    "attack": attack, "adaptive": adaptive, "epsilon_px": epsilon,
                    "steps": 1 if attack == "fgsm" else 20, "defense": "none",
                    "layer": "P3", "selected_best": True,
                    "f1_clean": 1.0 if index >= 3 else 0.0,
                    "recall_clean": 1.0 if index >= 3 else 0.0,
                    "f1_attack": 1.0 if index >= 3 else 0.0,
                    "recall_attack": 1.0 if index >= 3 else 0.0,
                })
        result = select_budgets(budget_table(pd.DataFrame(rows)), allow_conditional=False)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["failed_families"])

    def test_legacy_three_scene_test_is_hard_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "legacy_test.csv"
            blocker = root / "outputs/canonical_v2/LEGACY_TEST_BLOCKED"
            blocker.parent.mkdir(parents=True); blocker.write_text("blocked")
            args = argparse.Namespace(
                split="test", output=output, revision_stats=None,
                normalizations=[],
            )
            with patch.object(run_final_matrix, "PROJECT_DIR", root), patch.object(
                run_final_matrix, "OUTPUT", output
            ):
                with self.assertRaisesRegex(RuntimeError, "blocked by canonical v2"):
                    run_final_matrix.apply_deadline_defaults(args)


if __name__ == "__main__":
    unittest.main()
