from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from scripts.person_v3.prepare import filtered_person_label


class PersonV3ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = yaml.safe_load(
            Path("configs/canonical_v3_person_safety.yaml").read_text(
                encoding="utf-8"
            )
        )

    def test_scope_is_person_only_and_test_is_sealed(self) -> None:
        self.assertEqual(self.protocol["target"]["original_class_id"], 0)
        self.assertEqual(self.protocol["target"]["class_name"], "person")
        self.assertTrue(self.protocol["data"]["test_sealed"])
        self.assertFalse(self.protocol["data"]["test_labels_read_before_gate"])

    def test_two_fold_triage_is_frozen(self) -> None:
        self.assertEqual(self.protocol["triage"]["folds"], [0, 1])
        self.assertEqual(self.protocol["pipeline"]["seed"], 20260723)
        self.assertEqual(
            self.protocol["evaluation"]["triage_threshold_rule"],
            "maximum_F1_on_pooled_first_two_OOF_folds",
        )

    def test_person_label_view_keeps_only_class_zero(self) -> None:
        temporary = Path("/tmp/person_v3_test_label.txt")
        temporary.write_text(
            "0 0.5 0.5 0.1 0.1\n1 0.4 0.4 0.2 0.2\n",
            encoding="utf-8",
        )
        try:
            self.assertEqual(
                filtered_person_label(temporary),
                "0 0.5 0.5 0.1 0.1\n",
            )
        finally:
            temporary.unlink(missing_ok=True)

    def test_no_architecture_search_reintroduced(self) -> None:
        self.assertFalse(self.protocol["pipeline"]["P2"])
        self.assertFalse(self.protocol["pipeline"]["halo"])
        self.assertEqual(
            self.protocol["pipeline"]["tiling"],
            "frozen_M4_overlapping_2x2",
        )


if __name__ == "__main__":
    unittest.main()
