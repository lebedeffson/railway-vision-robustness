from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path

import yaml

from scripts.person_v4.build_train_only_proxy import centered_block
from scripts.person_v4.proxy import (
    classify_proxy_gate,
    synthetic_four_domain_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/person_v4_train_only_proxy.yaml"
PROTOCOL_ROOT = ROOT / "protocols/person_v4_train_only_proxy_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PersonV4ProxyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_protocol_is_train_only_and_test_sealed(self) -> None:
        scope = self.protocol["scope"]
        self.assertFalse(scope["may_open_external_fold0"])
        self.assertFalse(scope["may_open_official_validation"])
        self.assertFalse(scope["may_open_test"])
        self.assertFalse(scope["may_interrupt_active_A3"])
        self.assertFalse(scope["scientific_result"])
        self.assertFalse(scope["article_evidence"])

    def test_proxy_splits_never_use_outer_fold0_heldout(self) -> None:
        forbidden = set(
            self.protocol["forbidden"]["outer_fold0_heldout_scenes"]
        )
        self.assertEqual(len(self.protocol["proxy_splits"]), 3)
        for split in self.protocol["proxy_splits"]:
            sources = set(split["source_scenes"])
            heldout = split["heldout_scene"]
            self.assertEqual(len(sources), 4)
            self.assertNotIn(heldout, sources)
            self.assertFalse((sources | {heldout}) & forbidden)

    def test_frozen_manifests_are_isolated_and_complete(self) -> None:
        frame_path = PROTOCOL_ROOT / "proxy_frame_manifest.csv"
        tile_path = PROTOCOL_ROOT / "proxy_tile_manifest.csv"
        with frame_path.open(encoding="utf-8", newline="") as handle:
            frames = list(csv.DictReader(handle))
        with tile_path.open(encoding="utf-8", newline="") as handle:
            tiles = list(csv.DictReader(handle))
        self.assertEqual(len(frames), 2044)
        self.assertEqual(len(tiles), 8176)
        forbidden = set(
            self.protocol["forbidden"]["outer_fold0_heldout_scenes"]
        )
        self.assertFalse(
            {row["grouped_scene_id"] for row in frames} & forbidden
        )
        tile_counts: dict[tuple[str, str, str], int] = {}
        for row in tiles:
            key = (
                row["proxy_split"],
                row["role"],
                row["frame_stem"],
            )
            tile_counts[key] = tile_counts.get(key, 0) + 1
        self.assertEqual(len(tile_counts), len(frames))
        self.assertEqual(set(tile_counts.values()), {4})

    def test_protocol_lock_binds_config_and_implementation(self) -> None:
        lock = json.loads(
            (PROTOCOL_ROOT / "protocol_lock.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(lock["status"], "LOCKED")
        self.assertTrue(lock["test_sealed"])
        self.assertFalse(lock["active_A3_may_be_interrupted"])
        self.assertEqual(lock["protocol_sha256"], sha256(CONFIG))
        for relative, expected in lock["code_sha256"].items():
            self.assertEqual(sha256(ROOT / relative), expected)

    def test_temporal_block_selection_is_centered_and_deterministic(self) -> None:
        rows = [
            {
                "subsequence_id": "scene",
                "frame_id": f"{index:03d}",
                "source_image": f"{index:03d}.png",
            }
            for index in range(10)
        ]
        first = centered_block(rows, 4)
        second = centered_block(list(reversed(rows)), 4)
        self.assertEqual(first, second)
        self.assertEqual(
            [row["frame_id"] for row in first],
            ["003", "004", "005", "006"],
        )

    def test_proxy_gate_requires_paired_improvement(self) -> None:
        gate = self.protocol["proxy_gate"]["require_all"]
        rows = [
            {
                "delta_mAP50": value,
                "delta_recall": value,
                "delta_small_recall": value + 0.02,
                "NaN_Inf": 0,
                "lost_GT": 0,
                "evaluator_consistency": "PASS",
            }
            for value in (0.04, 0.05, 0.06)
        ]
        self.assertEqual(
            classify_proxy_gate(rows, gate)["status"], "PASS"
        )
        rows[0]["delta_recall"] = -0.04
        self.assertEqual(
            classify_proxy_gate(rows, gate)["status"], "FAIL"
        )

    def test_synthetic_four_domain_integration(self) -> None:
        report = synthetic_four_domain_benchmark()
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["article_evidence"])
        self.assertGreater(report["accuracy_improvement"], 0.20)


if __name__ == "__main__":
    unittest.main()
