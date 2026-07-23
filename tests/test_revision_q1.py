import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_final_practice import load_manifest, split_leakage
from revision_q1.normalization import (
    LAYERS,
    DIAGNOSTIC_QUANTILE_SAMPLE_MAX,
    distribution_diagnostics,
    fit_channel_statistics,
    membership,
    normalized_quality_recovery,
)
from revision_q1.protocol import assert_split_action, load_protocol
from revision_q1.analyze import paired_model_bootstrap, scene_macro_loso_table
from revision_q1.spatial import spatial_transform, transform_xywh_boxes
from revision_q1.statistics import (
    benjamini_hochberg,
    cluster_mean_interval,
    cluster_sample_plan,
    floor_effect_mode,
    group_fold_assignments,
    holm_bonferroni,
    materialize_cluster_sample,
    paired_cluster_delta_correlation,
    unique_cluster_samples,
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

    def test_large_normalization_diagnostics_use_deterministic_quantile_sample(self) -> None:
        values = torch.linspace(0, 1, DIAGNOSTIC_QUANTILE_SAMPLE_MAX + 17)
        diagnostics = distribution_diagnostics(values)
        self.assertLessEqual(
            diagnostics["quantile_sample_count"], DIAGNOSTIC_QUANTILE_SAMPLE_MAX
        )
        self.assertGreater(diagnostics["quantile_sample_count"], 0)
        self.assertTrue(diagnostics["quantiles_approximate"])
        self.assertAlmostEqual(diagnostics["mean_membership"], 0.5, places=5)

    def test_bootstrap_clusters_by_sequence(self) -> None:
        groups = np.asarray(["a", "a", "b", "b", "c"])
        selected = cluster_sample_plan(groups, 1, 7)[0]
        indices = materialize_cluster_sample(groups, selected)
        for group in selected:
            expected = np.count_nonzero(groups == group)
            self.assertGreaterEqual(np.count_nonzero(groups[indices] == group), expected)
        interval = cluster_mean_interval(
            pd.DataFrame({"sequence_id": groups, "value": np.arange(len(groups))}),
            "value", iterations=20, seed=7,
        )
        self.assertEqual(interval["sequences"], 3)
        unique = unique_cluster_samples(groups, 100, 7)
        self.assertEqual(sum(count for _, count in unique), 100)

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

    def test_delta_mae_and_reduction_have_opposite_improvement_signs(self) -> None:
        frame = pd.DataFrame({
            "sequence_id": np.repeat(["a", "b", "c"], 3),
            "target": np.tile([0.0, 1.0, 2.0], 3),
            "prediction_D2": np.tile([0.4, 1.4, 2.4], 3),
            "prediction_D3": np.tile([0.1, 1.1, 2.1], 3),
        })
        result = paired_model_bootstrap(
            frame, "target", (("D2", "D3"),), iterations=50, seed=7
        )
        delta = result[result["metric"] == "delta_mae"].iloc[0]
        self.assertLess(delta["estimate"], 0)
        self.assertGreater(delta["relative_mae_reduction"], 0)
        self.assertEqual(delta["relative_mae_reduction_unit"], "percent")

    def test_scene_macro_table_reports_three_loso_scenes(self) -> None:
        frame = pd.DataFrame({
            "sequence_id": np.repeat(["a", "b", "c"], 3),
            "target": np.tile([0.0, 1.0, 2.0], 3),
            "prediction_D2": np.tile([0.4, 1.4, 2.4], 3),
            "prediction_D3": np.tile([0.1, 1.1, 2.1], 3),
        })
        result = scene_macro_loso_table(frame, "target", (("D2", "D3"),))
        self.assertEqual((result["scope"] == "scene").sum(), 3)
        self.assertEqual((result["scope"] == "macro_average").sum(), 1)
        self.assertTrue(result["inference_warning"].str.contains("three_independent").all())

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
