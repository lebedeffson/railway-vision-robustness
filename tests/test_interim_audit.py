from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

import extract_attack_consistency as attacks
from audit_final_practice import load_manifest, split_leakage
from evaluate_image_level_detection import detection_metrics, non_max_suppression
from extract_attack_consistency import fuzzy_tnorm, object_masks
from extract_feature_consistency import clipped_recovery, recovery
from revision_q1.feature_metrics import similarity_recovery, tnorm
from revision_q1.normalization import LAYERS, fit_channel_statistics
from revision_q1.protocol import assert_split_action, load_protocol
from revision_q1.statistics import cluster_sample_plan, group_fold_assignments


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "outputs/final_practice/interim_audit/validation_snapshot_129.csv"


def simple_loss(_model, _batch, images: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(.1, 1.0, images.numel(), device=images.device).view_as(images)
    return (images * weights).sum()


class InterimAuditTest(unittest.TestCase):
    def test_split_count_matches_manifest(self) -> None:
        rows = load_manifest(ROOT / "data/yolo_osdar23/manifest.csv")
        counts = pd.Series([row["split"] for row in rows]).value_counts().to_dict()
        self.assertEqual(counts, {"train": 1057, "val": 198, "test": 150})
        self.assertFalse(any(split_leakage(rows).values()))

    def test_stage_name_matches_actual_split(self) -> None:
        if not SNAPSHOT.is_file():
            self.skipTest("interim snapshot not generated")
        values = pd.read_csv(SNAPSHOT, usecols=["split"])["split"].unique().tolist()
        self.assertEqual(values, ["val"])

    def test_checkpoint_hash_matches_config(self) -> None:
        selection = json.loads((ROOT / "config/checkpoint_selection.json").read_text())
        expected = (ROOT / selection["selected_checkpoint"]).resolve()
        self.assertEqual(expected, (ROOT / "outputs/training/yolo11m_baseline_stage2/weights/best.pt").resolve())
        provenance = ROOT / "outputs/final_practice/interim_audit/checkpoint_provenance.json"
        if provenance.is_file():
            self.assertEqual(json.loads(provenance.read_text())["status"], "PASS")

    def test_matrix_has_expected_conditions(self) -> None:
        if not SNAPSHOT.is_file():
            self.skipTest("interim snapshot not generated")
        counts = pd.read_csv(SNAPSHOT, usecols=["image_path"])["image_path"].value_counts()
        self.assertEqual(set(counts.to_numpy()), {450})

    def test_no_duplicate_condition_rows(self) -> None:
        if not SNAPSHOT.is_file():
            self.skipTest("interim snapshot not generated")
        columns = ["image_path", "attack", "adaptive", "epsilon", "steps", "restart", "seed", "defense", "layer"]
        frame = pd.read_csv(SNAPSHOT, usecols=columns)
        self.assertFalse(frame.duplicated(columns).any())

    def test_fgsm_respects_linf_budget(self) -> None:
        clean = torch.full((1, 3, 8, 8), .5)
        batch = {"img": clean}
        with patch.object(attacks, "yolo_loss", simple_loss):
            result = attacks.fgsm(nn.Identity(), batch, 4.0, adaptive=False)
        delta = result.adversarial - clean
        self.assertLessEqual(float(delta.abs().max()), 4 / 255 + 1e-7)
        self.assertGreaterEqual(float(result.adversarial.min()), 0.0)
        self.assertLessEqual(float(result.adversarial.max()), 1.0)

    def test_pgd_respects_linf_budget(self) -> None:
        clean = torch.full((1, 3, 8, 8), .5)
        with patch.object(attacks, "yolo_loss", simple_loss):
            result = attacks.pgd(nn.Identity(), {"img": clean}, 2.0, 5, 42, False)
        self.assertLessEqual(float((result.adversarial - clean).abs().max()), 2 / 255 + 1e-7)

    def test_pgd_best_restart_selected(self) -> None:
        clean = torch.full((1, 3, 8, 8), .5)
        with patch.object(attacks, "yolo_loss", simple_loss):
            results = [attacks.pgd(nn.Identity(), {"img": clean}, 2.0, 3, seed, False)
                       for seed in (42, 123, 999)]
        selected = max(range(len(results)), key=lambda index: results[index].attack_loss)
        self.assertEqual(results[selected].attack_loss, max(item.attack_loss for item in results))

    def test_adaptive_gradient_reaches_input(self) -> None:
        image = torch.rand(1, 3, 8, 8)
        with patch.object(attacks, "yolo_loss", simple_loss):
            gradient, _ = attacks.loss_gradient(nn.Identity(), {"img": image}, image, True)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0.0)

    def test_adaptive_attack_uses_full_pipeline(self) -> None:
        source = inspect.getsource(attacks.defended_loss)
        self.assertIn("tnorm_filter", source)
        self.assertIn("checkpoint", source)
        self.assertNotIn(".detach()", source)

    def test_nms_timeout_not_silently_counted_as_empty(self) -> None:
        source = inspect.getsource(non_max_suppression)
        assignment = source.find("output[xi] = x[i]")
        timeout = source.find("(time.time() - t) > time_limit")
        self.assertGreaterEqual(assignment, 0)
        self.assertGreater(timeout, assignment)

    def test_detection_metrics_manual_example(self) -> None:
        ground_truth = torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]])
        classes = torch.tensor([0, 1])
        predictions = torch.tensor([
            [0., 0., 10., 10., .9, 0.],
            [0., 0., 10., 10., .8, 0.],
            [20., 20., 30., 30., .7, 2.],
        ])
        result = detection_metrics(predictions, ground_truth, classes, .35)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 2, 1))
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertAlmostEqual(result["recall"], 1 / 2)
        self.assertAlmostEqual(result["f1"], .4)

    def test_normalization_fit_clean_validation_only(self) -> None:
        protocol = load_protocol()
        self.assertEqual(protocol["normalization"]["fit_split"], "val")
        self.assertEqual(protocol["normalization"]["fit_inputs"], "clean_only")

    def test_normalization_separate_by_layer(self) -> None:
        samples = {"P3": torch.zeros(2, 10), "P4": torch.ones(2, 10), "P5": torch.ones(2, 10) * 2}
        statistics = fit_channel_statistics(samples)
        self.assertEqual(set(statistics), set(LAYERS))
        self.assertNotEqual(float(statistics["P3"]["mean"].mean()), float(statistics["P5"]["mean"].mean()))

    def test_tnorm_axioms(self) -> None:
        a = torch.rand(100)
        b = torch.rand(100)
        zero, one = torch.zeros_like(a), torch.ones_like(a)
        for operator in ("product", "godel", "lukasiewicz"):
            value = tnorm(a, b, operator)
            self.assertTrue(((value >= 0) & (value <= 1)).all())
            torch.testing.assert_close(value, tnorm(b, a, operator))
            torch.testing.assert_close(tnorm(a, one, operator), a)
            torch.testing.assert_close(tnorm(a, zero, operator), zero)
            self.assertTrue((tnorm(a, torch.maximum(a, b), operator) >= tnorm(a, torch.minimum(a, b), operator)).all())

    def test_tnorm_metrics_nonconstant(self) -> None:
        left = torch.linspace(0, 1, 100)
        right = torch.linspace(.2, .8, 100)
        for operator in ("product", "godel", "lukasiewicz"):
            self.assertGreater(int(tnorm(left, right, operator).unique().numel()), 1)

    def test_recovery_boundary_cases(self) -> None:
        self.assertAlmostEqual(similarity_recovery(.2, .2), 0.0)
        self.assertGreater(similarity_recovery(.2, .5), 0.0)
        self.assertLess(similarity_recovery(.5, .2), 0.0)
        self.assertAlmostEqual(similarity_recovery(.2, 1.0), 1.0, places=6)

    def test_g_raw_not_clipped(self) -> None:
        raw = recovery(.5, .2)
        self.assertLess(raw, 0.0)
        self.assertEqual(clipped_recovery(.5, .2), 0.0)

    def test_object_mask_alignment(self) -> None:
        batch = {
            "img": torch.zeros(1, 3, 10, 10),
            "bboxes": torch.tensor([[.5, .5, .4, .4]]),
            "batch_idx": torch.tensor([0]),
        }
        mask = object_masks(batch, 10, 10)[0]
        self.assertTrue(mask[5, 5])
        self.assertFalse(mask[0, 0])
        self.assertTrue(torch.equal(~mask, torch.logical_not(mask)))

    def test_independent_recomputation(self) -> None:
        a, r, p = .2, .5, .8
        g = (r - a) / (1 - a)
        self.assertAlmostEqual(g, .375)
        self.assertAlmostEqual(p * np.clip(g, 0, 1), .3)

    def test_groupkfold_uses_sequence_id(self) -> None:
        groups = ["a", "a", "b", "b", "c", "c"]
        folds = group_fold_assignments(groups, 3)
        for group in set(groups):
            indices = [index for index, value in enumerate(groups) if value == group]
            self.assertEqual(len(set(folds[indices])), 1)

    def test_cluster_bootstrap_uses_sequence_id(self) -> None:
        groups = np.array(["a", "a", "b", "b", "c"])
        sample = cluster_sample_plan(groups, 1, 42)[0]
        self.assertEqual(len(sample), 3)
        self.assertTrue(set(sample) <= {"a", "b", "c"})

    def test_no_test_tuning(self) -> None:
        for action in load_protocol()["split_policy"]["forbidden_test_actions"]:
            with self.assertRaises(RuntimeError):
                assert_split_action(action, "test")


if __name__ == "__main__":
    unittest.main()
