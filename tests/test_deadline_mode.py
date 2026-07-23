from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import deadline_matrix_audit
import deadline_select_budgets
import recalculate_legacy_recovery
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
            budget = root / "config/canonical_budget_selection.json"
            gate.parent.mkdir(parents=True)
            stats.parent.mkdir(parents=True)
            budget.parent.mkdir(parents=True)
            gate.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            stats.write_bytes(b"frozen-validation-statistics")
            budget.write_text(json.dumps({
                "status": "PASS", "selected": {
                    "fgsm_epsilon_px": [0.1, 0.25],
                    "pgd_epsilon_px": [0.025, 0.05],
                    "adaptive_pgd_epsilon_px": [0.05],
                },
            }), encoding="utf-8")
            args = argparse.Namespace(
                split="test", output=output, revision_stats=None,
                normalizations=[], fgsm_eps=[], pgd_eps=[], pgd_steps=[],
                adaptive_pgd_eps=None, adaptive_pgd_steps=[], seeds=[], defenses=[],
                checkpoint_name="", nms_max_time_img=.05,
            )
            with patch.object(run_final_matrix, "OUTPUT", output), patch.object(
                run_final_matrix, "DEADLINE_ROOT", root
            ), patch.object(run_final_matrix, "PROJECT_DIR", root):
                run_final_matrix.apply_deadline_defaults(args)
            self.assertEqual(args.normalizations, ["N1_quantile"])
            self.assertEqual(args.fgsm_eps, [0.1, 0.25])
            self.assertEqual(args.pgd_steps, [20])
            self.assertEqual(args.adaptive_pgd_eps, [0.05])
            self.assertEqual(args.defenses, ["none", "tnorm", "bilateral", "median"])
            self.assertEqual(args.nms_max_time_img, 10.0)

    def test_main_service_has_no_parallel_success_service(self) -> None:
        unit = (PROJECT_DIR / "systemd/tnorm-wait-train.service").read_text(encoding="utf-8")
        self.assertNotIn("OnSuccess=tnorm-deadline.service", unit)
        self.assertNotIn("OnSuccess=tnorm-revision-q1.service", unit)
        delivery = (PROJECT_DIR / "build_final_delivery.py").read_text(encoding="utf-8")
        self.assertIn('"deadline_finalize.py"', delivery)

    def test_stage1_sensitivity_is_reduced_to_frozen_45_frames(self) -> None:
        protocol = yaml.safe_load(
            (PROJECT_DIR / "config/deadline_protocol.yaml").read_text(encoding="utf-8")
        )
        sensitivity = protocol["stage1_sensitivity"]
        self.assertEqual(sensitivity["split"], "test")
        self.assertEqual(
            sensitivity["frame_scope"],
            "deterministic_45_unique_frames_balanced_by_available_scene_size",
        )
        self.assertEqual(sensitivity["total_frames"], 45)
        self.assertEqual(sensitivity["fgsm_epsilon_px"], "frozen_at_runtime_from_validation")
        self.assertEqual(sensitivity["pgd_epsilon_px"], "frozen_at_runtime_from_validation")
        self.assertEqual(sensitivity["defenses"], ["none", "tnorm"])

    def test_budget_selection_rejects_floor_and_uses_validation_only(self) -> None:
        rows = []
        for attack, adaptive, epsilon, failures in (
            ("fgsm", False, .1, 1), ("pgd", False, .05, 1),
            ("pgd", True, .05, 1),
        ):
            for index in range(4):
                rows.append({
                    "sequence_id": f"s{index % 3}", "image_path": f"i{index}",
                    "attack": attack, "adaptive": adaptive, "epsilon_px": epsilon,
                    "steps": 1 if attack == "fgsm" else 20, "defense": "none",
                    "layer": "P3", "selected_best": True, "f1_clean": 1.0,
                    "recall_clean": 1.0, "f1_attack": 0.0 if index < failures else .2,
                    "recall_attack": 0.0 if index < failures else .2,
                })
        table = deadline_select_budgets.budget_table(__import__("pandas").DataFrame(rows))
        selected = deadline_select_budgets.select_budgets(table)
        self.assertEqual(selected["status"], "PASS")
        self.assertFalse(selected["test_used_for_selection"])

    def test_legacy_recovery_preserves_raw_and_clipped_g(self) -> None:
        import pandas as pd
        frame = pd.DataFrame({
            "a_attacked_similarity": [.8, .8],
            "r_restored_similarity": [.9, .7],
            "p_clean_preservation": [.95, .95],
            "g_recovery": [.5, 0.0], "c_def": [.475, 0.0],
            "a_godel": [.8, .8], "r_godel": [.9, .7], "p_godel": [.95, .95],
            "g_godel": [.5, 0.0], "c_def_godel": [.5, 0.0],
            "a_lukasiewicz": [.8, .8], "r_lukasiewicz": [.9, .7],
            "p_lukasiewicz": [.95, .95], "g_lukasiewicz": [.5, 0.0],
            "c_def_lukasiewicz": [.45, 0.0],
        })
        fixed = recalculate_legacy_recovery.recalculate(frame)
        self.assertGreater(fixed.loc[0, "g_recovery_raw"], 0)
        self.assertLess(fixed.loc[1, "g_recovery_raw"], 0)
        self.assertEqual(fixed.loc[1, "g_recovery_clipped"], 0)
        self.assertEqual(fixed.loc[1, "c_def"], 0)


if __name__ == "__main__":
    unittest.main()
