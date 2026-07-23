from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
import yaml

from scripts.canonical_v3.audit_generalization_failure import (
    fold_audit,
    partition_score,
)


class CanonicalV3ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = yaml.safe_load(
            Path(
                "configs/canonical_v3_development_pool_small_object.yaml"
            ).read_text(encoding="utf-8")
        )

    def test_test_is_sealed_and_cpu_audit_blocks_training(self) -> None:
        self.assertTrue(self.protocol["data"]["test_sealed"])
        self.assertTrue(self.protocol["cpu_audit"]["required_before_training"])
        self.assertFalse(self.protocol["cpu_audit"]["may_read_test_labels"])
        self.assertEqual(
            self.protocol["data"]["development_source_splits"],
            ["train", "val"],
        )

    def test_only_two_candidates_are_frozen(self) -> None:
        self.assertEqual(
            self.protocol["development_triage"]["maximum_candidates"], 2
        )
        self.assertIn("candidate_C1", self.protocol)
        self.assertIn("candidate_C2", self.protocol)

    def test_fold_support_rejects_unseen_heldout_class(self) -> None:
        matrix = pd.DataFrame([
            {
                "grouped_scene_id": "a",
                "frames": 1,
                "small": 1,
                "medium": 0,
                "large": 0,
                "class_0_person": 5,
            },
            {
                "grouped_scene_id": "b",
                "frames": 1,
                "small": 1,
                "medium": 0,
                "large": 0,
                "class_0_person": 0,
            },
            {
                "grouped_scene_id": "c",
                "frames": 1,
                "small": 1,
                "medium": 0,
                "large": 0,
                "class_0_person": 0,
            },
        ])
        protocol = {
            "data": {"class_names": {0: "person"}},
            "cpu_audit": {
                "constrained_split": {
                    "minimum_train_scenes_per_heldout_class": 2
                }
            },
        }
        rows, valid = fold_audit([["a"], ["b"], ["c"]], matrix, protocol)
        self.assertFalse(valid)
        self.assertFalse(
            rows[(rows["fold"] == 0) & (rows["class_id"] == 0)][
                "valid"
            ].iloc[0]
        )

    def test_partition_score_counts_support_violations(self) -> None:
        matrix = pd.DataFrame([
            {
                "grouped_scene_id": name,
                "frames": 1,
                "small": 0,
                "medium": 0,
                "large": 1,
                "class_0_person": int(name in {"a", "b"}),
            }
            for name in ("a", "b", "c", "d")
        ])
        violations, _ = partition_score(
            [["a", "b"], ["c", "d"]],
            matrix,
            ["class_0_person"],
            2,
        )
        self.assertGreater(violations, 0)


if __name__ == "__main__":
    unittest.main()
