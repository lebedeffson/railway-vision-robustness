import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_final_practice import load_manifest, split_leakage
from revision_q1.normalization import (
    LAYERS,
    fit_channel_statistics,
    membership,
    normalized_quality_recovery,
)
from revision_q1.protocol import assert_split_action, load_protocol
from revision_q1.spatial import spatial_transform, transform_xywh_boxes
from revision_q1.statistics import (
    benjamini_hochberg,
    cluster_sample_plan,
    floor_effect_mode,
    group_fold_assignments,
    holm_bonferroni,
    materialize_cluster_sample,
    paired_cluster_delta_correlation,
)
from revision_q1.transfer import validate_transfer_pair


ROOT = Path(__file__).resolve().parents[1]


class RevisionQ1Test(unittest.TestCase):
    def test_no_sequence_leakage(self) -> None:
        leakage = split_leakage(load_manifest(ROOT / "data/yolo_osdar23/manifest.csv"))
        self.assertFalse(any(leakage.values()))

    def test_normalization_fit_on_validation_only(self) -> None:
        protocol = load_protocol()
        self.assertEqual(protocol["normalization"]["fit_split"], "val")
        self.assertEqual(protocol["normalization"]["fit_inputs"], "clean_only")

    def test_normalization_output_range(self) -> None:
        samples = {layer: torch.linspace(-3, 3, 64).repeat(2, 1) for layer in LAYERS}
        statistics = fit_channel_statistics(samples)
        feature = torch.randn(1, 2, 4, 4) * 3
        for mode in load_protocol()["normalization"]["variants"]:
            output = membership(feature, statistics["P3"], mode)
            self.assertGreaterEqual(float(output.min()), 0.0)
            self.assertLessEqual(float(output.max()), 1.0)

    def test_layer_statistics_are_separate(self) -> None:
        samples = {
            "P3": torch.zeros(2, 32),
            "P4": torch.ones(2, 32) * 5,
            "P5": torch.ones(2, 32) * 10,
        }
        statistics = fit_channel_statistics(samples)
        self.assertEqual(set(statistics), set(LAYERS))
        self.assertNotEqual(float(statistics["P3"]["mean"].mean()), float(statistics["P5"]["mean"].mean()))

    def test_bootstrap_clusters_by_sequence(self) -> None:
        groups = np.asarray(["a", "a", "b", "b", "c"])
        selected = cluster_sample_plan(groups, 1, 7)[0]
        indices = materialize_cluster_sample(groups, selected)
        for group in selected:
            expected = np.count_nonzero(groups == group)
            self.assertGreaterEqual(np.count_nonzero(groups[indices] == group), expected)

    def test_same_folds_for_all_metrics(self) -> None:
        groups = ["a", "a", "b", "b", "c", "c"]
        first = group_fold_assignments(groups, 3)
        second = group_fold_assignments(groups, 3)
        np.testing.assert_array_equal(first, second)

    def test_no_test_threshold_selection(self) -> None:
        for action in load_protocol()["split_policy"]["forbidden_test_actions"]:
            with self.assertRaises(RuntimeError):
                assert_split_action(action, "test")

    def test_paired_bootstrap_uses_same_samples(self) -> None:
        frame = pd.DataFrame({
            "sequence_id": np.repeat(["a", "b", "c"], 4),
            "target": np.arange(12, dtype=float),
            "tnorm": np.arange(12, dtype=float) + .1,
            "baseline": np.arange(12, dtype=float)[::-1],
        })
        result = paired_cluster_delta_correlation(
            frame, "target", "tnorm", "baseline", iterations=100, seed=8
        )
        self.assertEqual(result["bootstrap_iterations"], 100)
        self.assertEqual(result["sequences"], 3)

    def test_multiple_comparison_correction(self) -> None:
        p = [0.01, 0.02, 0.20]
        holm = holm_bonferroni(p)
        fdr = benjamini_hochberg(p)
        self.assertTrue(np.all(holm >= p))
        self.assertTrue(np.all(fdr >= p))
        self.assertTrue(np.all(holm <= 1))
        self.assertTrue(np.all(fdr <= 1))

    def test_stage1_stage2_sensitivity_protocol(self) -> None:
        protocol = load_protocol()
        self.assertIn("stage1", protocol["sensitivity"]["checkpoint"])
        self.assertIn("stage2", protocol["primary_checkpoint"])
        self.assertEqual(protocol["sensitivity"]["defenses"], ["none", "tnorm", "median", "bilateral"])

    def test_recovery_formula(self) -> None:
        value = normalized_quality_recovery(0.8, 0.2, 0.5)
        self.assertAlmostEqual(value, 0.5, places=6)
        negative = normalized_quality_recovery(0.8, 0.2, 0.1)
        self.assertLess(negative, 0.0)

    def test_floor_effect_switch(self) -> None:
        self.assertEqual(floor_effect_mode(0.01), "recall_f1_false_negatives")
        self.assertEqual(floor_effect_mode(0.02), "map_and_detection")

    def test_spatial_transform_reproducibility(self) -> None:
        image = torch.arange(3 * 8 * 8, dtype=torch.float32).view(1, 3, 8, 8)
        first = spatial_transform(image, angle_degrees=2, scale=1.0)
        second = spatial_transform(image, angle_degrees=2, scale=1.0)
        torch.testing.assert_close(first, second)
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
        torch.testing.assert_close(transform_xywh_boxes(boxes), boxes)

    def test_transfer_source_target_separation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            target = Path(directory) / "target.pt"
            source.write_bytes(b"source")
            target.write_bytes(b"target")
            resolved_source, resolved_target = validate_transfer_pair(source, target)
            self.assertNotEqual(resolved_source, resolved_target)
            with self.assertRaises(ValueError):
                validate_transfer_pair(source, source)


if __name__ == "__main__":
    unittest.main()
