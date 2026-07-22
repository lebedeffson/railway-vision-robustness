from __future__ import annotations

import unittest
import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from deadline_select_budgets import budget_table, select_budgets
import run_final_matrix


ROOT = Path(__file__).resolve().parents[1]


class CanonicalV2Test(unittest.TestCase):
    def test_split_v2_has_five_independent_validation_and_test_scenes(self) -> None:
        frame = pd.read_csv(ROOT / "outputs/canonical_v2/split/split_v2_manifest.csv")
        groups = {
            split: set(scope.sequence_id.astype(str))
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
