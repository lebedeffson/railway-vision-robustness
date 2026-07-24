from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from scripts.person_v5.hard_mining import iou
from scripts.person_v5.lock_runtime import CODE


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "configs/canonical_v5_person_data_first_runtime.yaml"
SERVICE = ROOT / "systemd/tnorm-person-v5-training.service"


class PersonV5RuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = yaml.safe_load(RUNTIME.read_text(encoding="utf-8"))

    def test_execution_is_data_first_and_two_fold_only(self) -> None:
        order = self.runtime["execution"]["order"]
        self.assertEqual(order[0], "external_pretraining")
        self.assertEqual(self.runtime["railway"]["folds"], [0, 1])
        self.assertEqual(order[-1], "two_fold_gate")
        self.assertNotIn("RT-DETR", " ".join(order))

    def test_runtime_freezes_hard_mining_before_GPU(self) -> None:
        mining = self.runtime["hard_mining"]
        self.assertEqual(mining["hard_positive_weight"], 4)
        self.assertEqual(mining["target_background_fraction"], 0.25)
        self.assertEqual(mining["missed_match_IoU"], 0.50)
        self.assertEqual(mining["hard_negative_confidence"], 0.05)

    def test_runtime_lock_covers_config_and_training_code(self) -> None:
        self.assertIn(
            "configs/canonical_v5_person_data_first_runtime.yaml", CODE
        )
        self.assertIn("scripts/person_v5/train.py", CODE)
        self.assertIn("scripts/person_v5/hard_mining.py", CODE)
        self.assertIn("scripts/person_v5/evaluate.py", CODE)
        self.assertIn("scripts/person_v5/run_pipeline.py", CODE)

    def test_training_service_requires_lock_and_test_seal(self) -> None:
        text = SERVICE.read_text(encoding="utf-8")
        self.assertIn("ConditionPathExists=!", text)
        self.assertIn("runtime_lock.json", text)
        self.assertIn(".venv/bin/python", text)
        self.assertNotIn("run_test", text)
        self.assertNotIn("attack", text.lower())

    def test_iou_reference_cases(self) -> None:
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)
        self.assertEqual(iou([0, 0, 1, 1], [2, 2, 3, 3]), 0)
        self.assertAlmostEqual(
            iou([0, 0, 2, 2], [1, 1, 3, 3]), 1 / 7
        )


if __name__ == "__main__":
    unittest.main()

