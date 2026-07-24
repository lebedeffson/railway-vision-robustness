from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.person_v5.lock_protocol import validate
from scripts.person_v5.prepare_crowdhuman import (
    iter_odgt,
    visible_person_boxes,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/canonical_v5_person_data_first.yaml"


class PersonV5ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_test_is_sealed_and_inputs_are_unchanged(self) -> None:
        self.assertTrue(self.protocol["claim_boundary"]["test_sealed"])
        marker = ROOT / self.protocol["claim_boundary"]["test_marker"]
        self.assertFalse(marker.exists())
        self.assertEqual(set(validate(self.protocol)), {
            "development_manifest",
            "folds",
            "railway_tile_manifest",
            "coco_initialization",
        })

    def test_v1_runs_only_crowdhuman_yolo_candidate(self) -> None:
        self.assertEqual(
            self.protocol["matrix"]["execution_order"],
            ["D1_fold0", "D1_fold1"],
        )
        self.assertEqual(
            self.protocol["matrix"]["D2"]["role"],
            "deferred_not_part_of_v1",
        )

    def test_data_first_excludes_failed_custom_stack(self) -> None:
        self.assertEqual(
            set(self.protocol["external_pretraining"]["forbidden"]),
            {"NWD", "QFL", "GroupDRO", "MixStyle", "SWAD"},
        )
        self.assertEqual(
            self.protocol["railway_finetuning"]["loss"],
            "standard_ultralytics_box_cls_dfl",
        )

    def test_hard_mining_is_train_only_and_bounded(self) -> None:
        mining = self.protocol["railway_finetuning"]["hard_mining"]
        self.assertEqual(
            mining["data_scope"], "current_fold_train_scenes_only"
        )
        self.assertEqual(mining["heldout_fold_mining"], "prohibited")
        self.assertEqual(mining["test_mining"], "prohibited")
        self.assertLessEqual(mining["maximum_frame_weight"], 4.0)

    def test_two_fold_gate_is_not_weakened(self) -> None:
        gate = self.protocol["two_fold_gate"]["require_all"]
        self.assertEqual(gate["macro_mAP50_min"], 0.45)
        self.assertEqual(gate["macro_recall_min"], 0.45)
        self.assertEqual(gate["macro_small_recall_min"], 0.30)
        self.assertEqual(gate["worst_fold_recall_min"], 0.30)
        self.assertEqual(gate["folds_improved_over_D0_min"], 2)


class CrowdHumanConversionTest(unittest.TestCase):
    def test_visible_box_filter_and_clipping(self) -> None:
        record = {
            "ID": "sample",
            "gtboxes": [
                {
                    "tag": "person",
                    "vbox": [-2, 1, 6, 5],
                    "extra": {"ignore": 0},
                },
                {
                    "tag": "person",
                    "vbox": [2, 2, 3, 3],
                    "extra": {"ignore": 1},
                },
                {"tag": "mask", "vbox": [1, 1, 2, 2]},
            ],
        }
        boxes, ignored = visible_person_boxes(record, 10, 10)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(ignored, 2)
        self.assertEqual(boxes[0], (0.2, 0.35, 0.4, 0.5))

    def test_odgt_is_line_delimited_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.odgt"
            path.write_text(
                json.dumps({"ID": "a", "gtboxes": []}) + "\n"
                + json.dumps({"ID": "b", "gtboxes": []}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual([row["ID"] for row in iter_odgt(path)], ["a", "b"])


if __name__ == "__main__":
    unittest.main()

