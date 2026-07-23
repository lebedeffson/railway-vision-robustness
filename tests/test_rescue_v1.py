from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_dataset import bbox_checks  # noqa: E402
from audit_evaluator import (  # noqa: E402
    average_precision,
    class_aware_nms,
    golden_case,
    match_dataset,
    timeout_contract,
)
from rescue_common import load_protocol, sha256  # noqa: E402
from rescue_gate import assert_gate_allows  # noqa: E402


class RescueV1Test(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_protocol()

    def test_split_has_no_scene_leakage(self) -> None:
        import pandas as pd

        manifest = pd.read_csv(ROOT / self.protocol["split_manifest"])
        groups = {
            split: set(rows["grouped_scene_id"].astype(str))
            for split, rows in manifest.groupby("split")
        }
        self.assertEqual({name: len(value) for name, value in groups.items()}, {
            "train": 10, "val": 5, "test": 5,
        })
        self.assertFalse(groups["train"] & groups["val"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["val"] & groups["test"])

    def test_manifest_hash_matches(self) -> None:
        self.assertEqual(
            sha256(ROOT / self.protocol["split_manifest"]),
            self.protocol["split_manifest_sha256"],
        )

    def test_class_mapping_is_consistent_and_swap_is_detected(self) -> None:
        dataset = yaml.safe_load((ROOT / self.protocol["dataset"]).read_text())
        expected = {int(key): value for key, value in self.protocol["class_names"].items()}
        observed = {int(key): value for key, value in dataset["names"].items()}
        self.assertEqual(observed, expected)
        swapped = observed.copy()
        swapped[0], swapped[1] = swapped[1], swapped[0]
        self.assertNotEqual(swapped, expected)

    def test_bbox_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images/train/example.png"
            label = root / "labels/train/example.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            Image.new("RGB", (1920, 1080), "black").save(image)
            label.write_text("0 0.5 0.5 0.25 0.5\n", encoding="utf-8")
            errors, objects, _, _ = bbox_checks(
                image, label, 1, "train", "scene", self.protocol
            )
            self.assertFalse(errors)
            self.assertEqual(len(objects), 1)

    def test_letterbox_roundtrip(self) -> None:
        width, height, target = 1920, 1080, 1280
        scale = min(target / width, target / height)
        pad_x = (target - width * scale) / 2
        pad_y = (target - height * scale) / 2
        box = [100.5, 50.25, 700.75, 900.0]
        transformed = [
            box[0] * scale + pad_x, box[1] * scale + pad_y,
            box[2] * scale + pad_x, box[3] * scale + pad_y,
        ]
        inverse = [
            (transformed[0] - pad_x) / scale,
            (transformed[1] - pad_y) / scale,
            (transformed[2] - pad_x) / scale,
            (transformed[3] - pad_y) / scale,
        ]
        self.assertLess(max(abs(a - b) for a, b in zip(box, inverse)), 1e-9)

    def test_evaluator_reference_case(self) -> None:
        ground_truth, raw = golden_case()
        predictions = {
            image_id: class_aware_nms(rows)
            for image_id, rows in raw.items()
        }
        observed = match_dataset(ground_truth, predictions, 0.5)
        self.assertEqual((observed["tp"], observed["fp"], observed["fn"]), (2, 3, 2))
        self.assertAlmostEqual(observed["precision"], 0.4)
        self.assertAlmostEqual(observed["recall"], 0.5)
        self.assertAlmostEqual(observed["f1"], 4 / 9)

    def test_map_is_threshold_independent(self) -> None:
        ground_truth, raw = golden_case()
        predictions = {
            image_id: class_aware_nms(rows)
            for image_id, rows in raw.items()
        }
        map_before = sum(
            average_precision(ground_truth, predictions, class_id)
            for class_id in (0, 1)
        ) / 2
        operating_low = match_dataset(ground_truth, predictions, 0.5)
        operating_high = match_dataset(ground_truth, predictions, 0.85)
        map_after = sum(
            average_precision(ground_truth, predictions, class_id)
            for class_id in (0, 1)
        ) / 2
        self.assertAlmostEqual(map_before, 0.5)
        self.assertAlmostEqual(map_before, map_after)
        self.assertNotEqual(operating_low, operating_high)

    def test_timeout_is_not_empty_prediction(self) -> None:
        contract = timeout_contract()
        self.assertTrue(contract["timeout"])
        self.assertFalse(contract["valid_empty_prediction"])
        self.assertTrue(contract["must_be_excluded_or_rerun"])

    def test_checkpoint_hash_is_frozen(self) -> None:
        checkpoint = ROOT / self.protocol["failed_run"]["checkpoint"]
        self.assertEqual(
            sha256(checkpoint), self.protocol["failed_run"]["checkpoint_sha256"]
        )

    def test_test_stage_blocked_on_failed_gate(self) -> None:
        self._assert_role_blocked("test")

    def test_attack_stage_blocked_on_failed_gate(self) -> None:
        self._assert_role_blocked("attack")

    def test_finalization_blocked_on_failed_gate(self) -> None:
        self._assert_role_blocked("finalization")

    def _assert_role_blocked(self, role: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gate = Path(directory) / "quality_gate.json"
            gate.write_text(json.dumps({
                "protocol_id": self.protocol["protocol_id"],
                "split_manifest_sha256": self.protocol["split_manifest_sha256"],
                "quality_gate_passed": False,
                "test_opened": False,
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "quality gate did not pass"):
                assert_gate_allows(role, gate)

    def test_micro_overfit_contract(self) -> None:
        contract = self.protocol["micro_overfit"]
        self.assertEqual(contract["frames"], 32)
        self.assertGreaterEqual(contract["minimum_classes"], 3)
        self.assertGreaterEqual(contract["map50_min"], 0.90)
        self.assertGreaterEqual(contract["recall_min"], 0.90)
        for name in ("mosaic", "mixup", "translate", "scale", "perspective"):
            self.assertEqual(contract[name], 0.0)

    def test_micro_manifest_paths_map_train_and_val_aliases(self) -> None:
        from baseline_rescue_audit import manifest_scene_lookup
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            pd.DataFrame([{
                "split": "train",
                "frame_id": "1",
                "output_image": str(root / "source.png"),
                "sequence_id": "scene-a",
                "micro_train_image": str(root / "micro/train/frame.png"),
                "micro_val_image": str(root / "micro/val/frame.png"),
            }]).to_csv(manifest, index=False)
            train = manifest_scene_lookup(manifest, "train")
            val = manifest_scene_lookup(manifest, "val")
            self.assertEqual(
                train[str((root / "micro/train/frame.png").resolve())], "scene-a"
            )
            self.assertEqual(
                val[str((root / "micro/val/frame.png").resolve())], "scene-a"
            )

    def test_micro_failure_is_a_scientific_stop_not_a_crash(self) -> None:
        micro_source = (ROOT / "scripts/run_micro_overfit.py").read_text()
        runner_source = (ROOT / "scripts/run_rescue_pipeline.py").read_text()
        finalizer_source = (ROOT / "scripts/finalize_rescue.py").read_text()
        self.assertNotIn("raise RuntimeError(f\"Micro-overfit gate failed", micro_source)
        self.assertIn("stop_after_micro_failure(micro)", runner_source)
        self.assertIn('"--micro-failure"', runner_source)
        self.assertIn('"scripts/analyze_micro_failure.py"', runner_source)
        self.assertIn('"stopped_micro_overfit_failed"', runner_source)
        self.assertIn('"failure_stage": "micro_overfit"', finalizer_source)
        self.assertIn('"hypotheses_H1_H4": "not_evaluated"', finalizer_source)

    def test_gradient_logger_uses_dispatched_ultralytics_callback(self) -> None:
        micro_source = (ROOT / "scripts/run_micro_overfit.py").read_text()
        candidate_source = (ROOT / "scripts/train_rescue_candidate.py").read_text()
        self.assertIn(
            'add_callback("on_train_batch_end", gradient_logger.before_zero_grad)',
            micro_source,
        )
        self.assertIn(
            'add_callback("on_train_batch_end", logger.before_zero_grad)',
            candidate_source,
        )
        self.assertNotIn(
            'add_callback("on_before_zero_grad", gradient_logger.before_zero_grad)',
            micro_source,
        )

    def test_no_tbd_finalization_when_gate_failed(self) -> None:
        self._assert_role_blocked("finalization")

    def test_rescue_archive_contains_provenance_contract(self) -> None:
        source = (ROOT / "scripts/finalize_rescue.py").read_text(encoding="utf-8")
        for required in (
            '"checksums.sha256"', '"manifest.json"', '"weights_included": False',
            "configs/rescue", "data/yolo_osdar23_rescue_v1/manifest.csv",
        ):
            self.assertIn(required, source)
        self.assertIn('path.suffix.lower() not in allowed', source)

    def test_candidate_matrix_uses_three_frozen_seeds(self) -> None:
        self.assertEqual(
            self.protocol["selection"]["finalist_seeds"],
            [20260722, 20260723, 20260724],
        )
        source = (ROOT / "scripts/run_rescue_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('for seed in protocol["selection"]["finalist_seeds"]', source)
        self.assertIn("assert_test_sealed()", source)

    def test_checkpoint_interval_policy_is_frozen_before_training(self) -> None:
        policy = yaml.safe_load(
            (ROOT / "configs/rescue/candidate_execution_policy.yaml").read_text()
        )
        self.assertTrue(policy["frozen_before_candidate_training"])
        self.assertEqual(policy["checkpoint_interval_epochs"], 5)


if __name__ == "__main__":
    unittest.main()
