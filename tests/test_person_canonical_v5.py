from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn

from scripts.person_canonical_v5.lock_protocol import validate
from src.augmentation.person_pasting import (
    AuditFlag,
    Box,
    InstanceRecord,
    PerspectiveModel,
    mask_quality,
    paste_person,
    validate_instance_bank,
)
from src.data.range_assignment import (
    assign_range_or_scale,
    preferred_levels,
    small_object_weight,
)
from src.models.coordinate_attention import CoordinateAttention
from src.models.p2_head import build_p2_model, detection_strides
from src.training.gradual_transfer import (
    L2SPAnchor,
    configure_phase,
    parameter_groups,
    replay_source,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/person_v5/protocol.yaml"


class PersonCanonicalV5ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_protocol_is_development_only_and_test_is_sealed(self) -> None:
        self.assertTrue(self.protocol["claim_boundary"]["test_sealed"])
        self.assertTrue(
            self.protocol["claim_boundary"][
                "development_only_until_full_OOF_PASS"
            ]
        )
        self.assertFalse(
            (
                ROOT
                / self.protocol["claim_boundary"]["test_marker"]
            ).exists()
        )

    def test_protocol_inputs_and_checkpoint_hashes_match(self) -> None:
        self.assertGreater(len(validate(self.protocol)), 10)

    def test_diagnostic_uses_one_evaluator_contract(self) -> None:
        diagnostic = self.protocol["development_diagnostic"]
        self.assertEqual(diagnostic["folds"], [0, 1])
        self.assertEqual(diagnostic["confidence_threshold"], 0.07)
        self.assertEqual(diagnostic["iou_threshold"], 0.50)
        self.assertEqual(
            set(diagnostic["states"]),
            {"B0", "B1", "D1_best", "D1_last"},
        )

    def test_candidate_ablation_does_not_mix_all_changes(self) -> None:
        candidates = self.protocol["architecture"]["candidates"]
        self.assertTrue(candidates["V5-A"]["P2"])
        self.assertFalse(candidates["V5-A"]["person_pasting"])
        self.assertFalse(candidates["V5-C"]["P2"])
        self.assertTrue(candidates["V5-C"]["person_pasting"])
        self.assertTrue(candidates["V5-D"]["P2"])
        self.assertTrue(candidates["V5-D"]["person_pasting"])
        self.assertEqual(
            candidates["V5-E"]["gradual_transfer"],
            "diagnostic_gate_only",
        )

    def test_gate_is_not_weakened(self) -> None:
        gate = self.protocol["two_fold_gate"]["require_all"]
        self.assertEqual(gate["macro_mAP50_min"], 0.45)
        self.assertEqual(gate["macro_recall_min"], 0.45)
        self.assertEqual(gate["macro_small_recall_min"], 0.30)
        self.assertEqual(gate["worst_fold_recall_min"], 0.30)


class P2AndCoordinateAttentionTest(unittest.TestCase):
    def test_coordinate_attention_preserves_shape_and_gradient(self) -> None:
        layer = CoordinateAttention(32)
        value = torch.randn(2, 32, 24, 40, requires_grad=True)
        output = layer(value)
        self.assertEqual(output.shape, value.shape)
        output.mean().backward()
        self.assertIsNotNone(value.grad)
        self.assertGreater(float(value.grad.abs().sum()), 0.0)

    def test_p2_models_expose_four_detection_strides(self) -> None:
        for name in ("yolo11m-p2.yaml", "yolo11m-p2-ca.yaml"):
            model = build_p2_model(
                ROOT / "configs/person_v5/models" / name
            )
            self.assertEqual(detection_strides(model), [4, 8, 16, 32])


class RangeAssignmentTest(unittest.TestCase):
    def test_verified_lidar_and_scale_fallback_are_not_conflated(self) -> None:
        lidar = assign_range_or_scale(
            area_ratio=0.0001,
            lidar_distance_m=70,
            verified_rgb_object_link=True,
        )
        fallback = assign_range_or_scale(
            area_ratio=0.0001,
            lidar_distance_m=70,
            verified_rgb_object_link=False,
        )
        self.assertEqual(lidar.source, "verified_lidar_geometry")
        self.assertEqual(lidar.group, "far")
        self.assertEqual(fallback.source, "bbox_area_scale_fallback")
        self.assertEqual(fallback.group, "small")
        self.assertEqual(preferred_levels(lidar), ("P2", "P3"))

    def test_small_object_weight_is_monotonic(self) -> None:
        small = small_object_weight(16, lambda_s=1.0, tau_s=32)
        large = small_object_weight(4096, lambda_s=1.0, tau_s=32)
        self.assertGreater(small, large)
        self.assertEqual(large, 1.0)


class PersonPastingTest(unittest.TestCase):
    def test_mask_quality_rejects_rectangular_foreground(self) -> None:
        full = np.ones((20, 10), dtype=np.uint8) * 255
        passed, metrics = mask_quality(
            full,
            minimum_foreground_fraction=0.10,
            maximum_foreground_fraction=0.85,
            maximum_border_fraction=0.25,
        )
        self.assertFalse(passed)
        self.assertEqual(metrics["foreground_fraction"], 1.0)

    def test_train_only_bank_rejects_scene_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = InstanceRecord(
                instance_id="instance",
                source_scene_id="heldout",
                source_frame_id="frame",
                image_path=root / "image.png",
                mask_path=root / "mask.png",
                box=Box(1, 1, 5, 10),
                original_width=20,
                original_height=20,
                estimated_range_m=None,
                visibility=1.0,
                occlusion=0.0,
                quality_status="PASS",
            )
            with self.assertRaisesRegex(ValueError, "outside fold train"):
                validate_instance_bank(
                    [record],
                    train_scene_ids={"train"},
                    heldout_scene_ids={"heldout"},
                    test_scene_ids={"test"},
                )

    def test_paste_requires_surface_and_updates_geometry(self) -> None:
        target = Image.new("RGB", (100, 100), "gray")
        instance = Image.new("RGB", (10, 20), "red")
        instance.info["instance_id"] = "one"
        instance.info["source_scene_id"] = "train"
        mask_array = np.zeros((20, 10), dtype=np.uint8)
        mask_array[2:18, 1:9] = 255
        mask = Image.fromarray(mask_array)
        surface = np.zeros((100, 100), dtype=bool)
        surface[50:, :] = True
        result = paste_person(
            target,
            instance,
            mask,
            target_scene_id="train",
            train_scene_ids={"train"},
            bottom_center=(50, 80),
            surface_mask=surface,
            perspective=PerspectiveModel(float(np.log(20)), 0.0),
            existing_boxes=[],
            used_instance_ids=set(),
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.box, Box(45, 60, 55, 80))

        blocked = paste_person(
            target,
            instance,
            mask,
            target_scene_id="train",
            train_scene_ids={"train"},
            bottom_center=(50, 30),
            surface_mask=surface,
            perspective=PerspectiveModel(float(np.log(20)), 0.0),
            existing_boxes=[],
            used_instance_ids=set(),
        )
        self.assertFalse(blocked.accepted)
        self.assertIn(AuditFlag.INVALID_SURFACE, blocked.flags)


class GradualTransferTest(unittest.TestCase):
    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.ModuleList(
                [nn.Linear(2, 2) for _ in range(14)]
            )

    def test_phases_and_differential_learning_rates(self) -> None:
        model = self.ToyModel()
        counts = configure_phase(model, "T1")
        self.assertEqual(counts["backbone"], 0)
        self.assertGreater(counts["neck"], 0)
        self.assertGreater(counts["head"], 0)
        groups = parameter_groups(model, base_lr=0.001, phase="T1")
        rates = {str(group["name"]): float(group["lr"]) for group in groups}
        self.assertEqual(rates["neck"], 0.0005)
        self.assertEqual(rates["head"], 0.001)

    def test_l2sp_penalty_and_replay_schedule(self) -> None:
        model = self.ToyModel()
        configure_phase(model, "T2")
        anchor = L2SPAnchor(model)
        first = next(
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        with torch.no_grad():
            first.add_(1.0)
        self.assertGreater(
            float(anchor.penalty(model, 0.1).detach()), 0.0
        )
        self.assertEqual(
            [replay_source(index) for index in range(5)],
            ["OSDaR23", "OSDaR23", "OSDaR23", "OSDaR23", "CrowdHuman"],
        )


if __name__ == "__main__":
    unittest.main()
