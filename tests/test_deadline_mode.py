from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import deadline_matrix_audit
import run_final_matrix


PROJECT_DIR = Path(__file__).resolve().parents[1]


class DeadlineModeTest(unittest.TestCase):
    def test_protocol_is_minimal_and_extended_work_is_deferred(self) -> None:
        protocol = yaml.safe_load(
            (PROJECT_DIR / "config/deadline_protocol.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(protocol["statistical_unit"], "sequence_id")
        self.assertEqual(protocol["final_bootstrap_iterations"], 5000)
        self.assertEqual(protocol["final_matrix"]["pgd_steps"], [20])
        self.assertIn("pgd_40", protocol["deferred_extended_analysis"])
        self.assertIn("full_spatial_stress", protocol["deferred_extended_analysis"])

    def test_minimal_stage2_grid_has_114_rows_per_image(self) -> None:
        args = argparse.Namespace(
            fgsm_eps=[1.0, 4.0], pgd_eps=[0.25, 1.0], pgd_steps=[20],
            adaptive_pgd=True, adaptive_pgd_eps=[1.0], adaptive_pgd_steps=[20],
            seeds=[42, 123, 999], defenses=["none", "tnorm", "bilateral", "median"],
        )
        self.assertEqual(run_final_matrix.expected_rows_per_image(args), 114)
        config = {
            "conditions": run_final_matrix.conditions(args), "seeds": args.seeds,
            "defenses": args.defenses,
        }
        self.assertEqual(deadline_matrix_audit.expected_rows(config), 114)

    def test_canonical_test_defaults_require_a_passing_validation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "canonical.csv"
            gate = root / "pilot/pilot_gate.json"
            stats = root / "normalization/layer_channel_statistics.pt"
            gate.parent.mkdir(parents=True)
            stats.parent.mkdir(parents=True)
            gate.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            stats.write_bytes(b"frozen-validation-statistics")
            args = argparse.Namespace(
                split="test", output=output, revision_stats=None,
                normalizations=[], fgsm_eps=[], pgd_eps=[], pgd_steps=[],
                adaptive_pgd_eps=None, adaptive_pgd_steps=[], seeds=[], defenses=[],
                checkpoint_name="", nms_max_time_img=.05,
            )
            with patch.object(run_final_matrix, "OUTPUT", output), patch.object(
                run_final_matrix, "DEADLINE_ROOT", root
            ):
                run_final_matrix.apply_deadline_defaults(args)
            self.assertEqual(args.normalizations, ["N1_quantile"])
            self.assertEqual(args.fgsm_eps, [1.0, 4.0])
            self.assertEqual(args.pgd_steps, [20])
            self.assertEqual(args.adaptive_pgd_eps, [1.0])
            self.assertEqual(args.defenses, ["none", "tnorm", "bilateral", "median"])
            self.assertEqual(args.nms_max_time_img, 10.0)

    def test_main_service_has_no_parallel_success_service(self) -> None:
        unit = (PROJECT_DIR / "systemd/tnorm-wait-train.service").read_text(encoding="utf-8")
        self.assertNotIn("OnSuccess=tnorm-deadline.service", unit)
        self.assertNotIn("OnSuccess=tnorm-revision-q1.service", unit)
        delivery = (PROJECT_DIR / "build_final_delivery.py").read_text(encoding="utf-8")
        self.assertIn('"deadline_finalize.py"', delivery)

    def test_stage1_sensitivity_is_reduced_and_uses_all_frozen_test_frames(self) -> None:
        protocol = yaml.safe_load(
            (PROJECT_DIR / "config/deadline_protocol.yaml").read_text(encoding="utf-8")
        )
        sensitivity = protocol["stage1_sensitivity"]
        self.assertEqual(sensitivity["split"], "test")
        self.assertEqual(sensitivity["frame_scope"], "all_frozen_test_frames")
        self.assertEqual(sensitivity["fgsm_epsilon_px"], [1])
        self.assertEqual(sensitivity["pgd_epsilon_px"], [1])
        self.assertEqual(sensitivity["defenses"], ["none", "tnorm"])


if __name__ == "__main__":
    unittest.main()
