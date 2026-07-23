from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import torch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_m4_common import (  # noqa: E402
    PROTOCOL_LOCK,
    TEST_MARKER,
    assert_test_sealed,
    expected_protocol_lock,
    load_protocol,
    validate_protocol_schema,
    verify_frozen_inputs,
)
from canonical_m4_tiling import (  # noqa: E402
    assign_ground_truth_to_tiles,
    frozen_tiles,
    fuse_predictions,
    local_box,
    restore_global_box,
    validate_scene_folds,
)
from canonical_m4_trainer import DifferentialLRDetectionTrainer  # noqa: E402
from canonical_m4_runtime import (  # noqa: E402
    image_to_tile_batch,
    prediction_tiles_to_global,
    tile_geometry,
)
from evaluate_canonical_m4 import select_thresholds  # noqa: E402
from prepare_canonical_m4_tiles import fold_mapping, yolo_lines  # noqa: E402


class CanonicalM4ProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()

    def test_m4_protocol_schema_and_inputs(self) -> None:
        self.assertEqual(
            validate_protocol_schema()["protocol_id"],
            "canonical-v2-m4-full-v1",
        )
        verify_frozen_inputs()

    def test_test_is_sealed_before_gate(self) -> None:
        self.assertFalse(TEST_MARKER.exists())
        assert_test_sealed()

    def test_micro_metrics_are_not_labeled_as_validation(self) -> None:
        snapshot = json.loads(
            (ROOT / "configs/canonical_v2_m4_micro_selection_snapshot.json")
            .read_text(encoding="utf-8")
        )
        self.assertTrue(snapshot["micro_metrics_are_not_validation_metrics"])
        self.assertEqual(snapshot["selection_role"], "technical_learnability_sanity_check_only")

    def test_full_image_coverage_and_determinism(self) -> None:
        first = frozen_tiles(self.protocol)
        second = frozen_tiles(self.protocol)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(min(tile.left for tile in first), 0)
        self.assertEqual(min(tile.top for tile in first), 0)
        self.assertEqual(max(tile.right for tile in first), 4112)
        self.assertEqual(max(tile.bottom for tile in first), 2504)
        self.assertLessEqual(first[0].right, first[1].right)

    def test_border_object_is_not_lost_and_class_survives(self) -> None:
        label = {"class_id": 5, "box": [1740.0, 1050.0, 1800.0, 1100.0]}
        assigned = assign_ground_truth_to_tiles([label], self.protocol)
        retained = [row for rows in assigned.values() for row in rows]
        self.assertTrue(retained)
        self.assertEqual({row["class_id"] for row in retained}, {5})
        self.assertTrue(all(row["visible_fraction"] >= 0.50 for row in retained))

    def test_empty_tile_is_supported(self) -> None:
        assigned = assign_ground_truth_to_tiles([], self.protocol)
        self.assertEqual(set(assigned), {tile.tile_id for tile in frozen_tiles(self.protocol)})
        self.assertTrue(all(not rows for rows in assigned.values()))

    def test_global_coordinate_roundtrip(self) -> None:
        tile = frozen_tiles(self.protocol)[3]
        global_box = [1800.0, 1200.0, 2000.0, 1400.0]
        self.assertEqual(restore_global_box(local_box(global_box, tile), tile), global_box)

    def test_overlap_duplicates_are_fused_deterministically(self) -> None:
        duplicate = [
            {"class_id": 1, "confidence": 0.9, "box": [100.0, 100.0, 120.0, 120.0]},
            {"class_id": 1, "confidence": 0.8, "box": [101.0, 101.0, 121.0, 121.0]},
        ]
        first = fuse_predictions(duplicate, self.protocol)
        second = fuse_predictions(list(reversed(duplicate)), self.protocol)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)

    def test_same_scene_tiles_remain_in_same_fold(self) -> None:
        scene_by_image = {"image_a": "scene_a", "image_b": "scene_b"}
        tile_source = {
            "image_a__tile_0": "image_a",
            "image_a__tile_1": "image_a",
            "image_b__tile_0": "image_b",
        }
        validate_scene_folds(
            scene_by_image, tile_source, {"scene_a": 0, "scene_b": 1}
        )

    def test_differential_lr_groups_preserve_protocol_rates(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.ModuleList([
                    torch.nn.Sequential(
                        torch.nn.Conv2d(3, 4, 3),
                        torch.nn.BatchNorm2d(4),
                    ),
                    torch.nn.Conv2d(4, 6, 1),
                ])

            def forward(self, value):
                for layer in self.model:
                    value = layer(value)
                return value

        trainer = object.__new__(DifferentialLRDetectionTrainer)
        trainer.backbone_lr = 1e-4
        trainer.head_lr = 3e-4
        optimizer = trainer.build_optimizer(
            TinyModel(), name="AdamW", momentum=0.9, decay=5e-4
        )
        by_name = {
            group["param_group"]: group["lr"]
            for group in optimizer.param_groups
        }
        self.assertEqual(by_name["backbone_weight"], 1e-4)
        self.assertEqual(by_name["head_weight"], 3e-4)

    def test_threshold_rules_are_frozen_and_deterministic(self) -> None:
        sweep = pd.DataFrame([
            {"threshold": 0.1, "f1": 0.8, "f2": 0.7, "recall": 0.7},
            {"threshold": 0.2, "f1": 0.8, "f2": 0.9, "recall": 0.8},
            {"threshold": 0.3, "f1": 0.7, "f2": 0.9, "recall": 0.7},
        ])
        standard, safety = select_thresholds(sweep)
        self.assertEqual(float(standard["threshold"]), 0.2)
        self.assertEqual(float(safety["threshold"]), 0.2)

    def test_deployable_tile_letterbox_and_inverse_geometry(self) -> None:
        image = torch.zeros((1, 3, 2504, 4112))
        batch = image_to_tile_batch(image, self.protocol)
        self.assertEqual(tuple(batch.shape), (4, 3, 640, 640))
        scale, _, _, left, top = tile_geometry(self.protocol)
        local = [100.0, 200.0, 300.0, 400.0]
        encoded = torch.tensor([[
            local[0] * scale + left,
            local[1] * scale + top,
            local[2] * scale + left,
            local[3] * scale + top,
            0.9,
            1.0,
        ]])
        empty = torch.empty((0, 6))
        restored = prediction_tiles_to_global(
            [encoded, empty, empty, empty], self.protocol
        )
        self.assertEqual(len(restored), 1)
        self.assertTrue(all(
            abs(observed - expected) < 1e-4
            for observed, expected in zip(restored[0]["box"], local)
        ))

    def test_frozen_scene_folds_cover_each_train_scene_once(self) -> None:
        mapping = fold_mapping(self.protocol)
        self.assertEqual(len(mapping), 10)
        self.assertEqual(set(mapping.values()), set(range(5)))
        self.assertTrue(all(list(mapping.values()).count(fold) == 2 for fold in range(5)))

    def test_empty_yolo_label_is_supported(self) -> None:
        self.assertEqual(yolo_lines([], 2350, 1431), "")

    def test_lock_payload_is_hash_bound(self) -> None:
        expected = expected_protocol_lock()
        self.assertEqual(expected["status"], "LOCKED")
        self.assertTrue(expected["test_sealed"])
        if PROTOCOL_LOCK.exists():
            current = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
            self.assertEqual(current["protocol_sha256"], expected["protocol_sha256"])

    def test_service_restart_routes_open_test_directly_to_post_gate(self) -> None:
        source = (ROOT / "scripts/run_canonical_m4_pre_gate.py").read_text(
            encoding="utf-8"
        )
        marker_check = source.index("if TEST_MARKER.is_file():")
        sealed_check = source.index('assert_role_allowed("scene_cv")', marker_check)
        self.assertLess(marker_check, sealed_check)
        self.assertIn('run_script("run_canonical_m4_post_gate.py")', source)

    def test_three_full_training_seeds_are_frozen(self) -> None:
        self.assertEqual(
            self.protocol["full_training"]["seeds"],
            [20260722, 20260723, 20260724],
        )


if __name__ == "__main__":
    unittest.main()
