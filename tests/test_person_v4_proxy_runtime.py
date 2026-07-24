from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from scripts.person_v4.run_train_only_proxy import (
    PROTOCOL_ROOT,
    TEST_MARKER,
    assert_runtime_isolation,
    prepare_datasets,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "configs/person_v4_train_only_proxy_runtime.yaml"
SERVICE = ROOT / "systemd/tnorm-person-v4-proxy.service"


class PersonV4ProxyRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = yaml.safe_load(RUNTIME.read_text(encoding="utf-8"))

    def test_runtime_is_frozen_before_proxy_metrics(self) -> None:
        self.assertTrue(self.runtime["clarified_before_proxy_metrics"])
        self.assertEqual(
            self.runtime["execution"]["variants"], ["A0", "A3"]
        )
        self.assertTrue(
            self.runtime["execution"]["require_all_six_runs"]
        )
        self.assertFalse(
            self.runtime["execution"]["early_reject_enabled"]
        )
        self.assertEqual(
            self.runtime["evaluation"]["threshold_rule"],
            "maximum_pooled_F1_across_three_proxy_splits_per_variant",
        )

    def test_runtime_contract_keeps_test_and_external_fold_closed(self) -> None:
        self.assertFalse(TEST_MARKER.exists())
        contract = assert_runtime_isolation()
        forbidden = set(
            self.runtime["isolation"]["forbidden_outer_fold0_scenes"]
        )
        self.assertFalse(
            set(contract["frames"]["grouped_scene_id"].astype(str))
            & forbidden
        )
        self.assertFalse(
            set(contract["tiles"]["grouped_scene_id"].astype(str))
            & forbidden
        )

    def test_proxy_dataset_lists_are_frozen_tile_subsets(self) -> None:
        contract = assert_runtime_isolation()
        datasets = prepare_datasets(contract)
        self.assertEqual(set(datasets), {"proxy_0", "proxy_1", "proxy_2"})
        for split, data_path in datasets.items():
            data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
            train = Path(data["train"]).read_text().splitlines()
            heldout = Path(data["val"]).read_text().splitlines()
            self.assertTrue(train)
            self.assertTrue(heldout)
            self.assertFalse(set(train) & set(heldout), split)

    def test_runtime_lock_binds_runtime_protocol(self) -> None:
        lock = json.loads(
            (PROTOCOL_ROOT / "protocol_lock.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            lock["runtime_protocol_path"],
            "configs/person_v4_train_only_proxy_runtime.yaml",
        )

    def test_service_is_train_only_and_conditioned_on_test_seal(self) -> None:
        text = SERVICE.read_text(encoding="utf-8")
        self.assertIn("run_train_only_proxy.py", text)
        self.assertIn("ConditionPathExists=!", text)
        self.assertNotIn("run_expedited.py", text)
        self.assertNotIn("test_matrix", text)


if __name__ == "__main__":
    unittest.main()
