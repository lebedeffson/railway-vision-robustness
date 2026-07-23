from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from audit_final_practice import audit, load_manifest, split_leakage


class AuditFinalPracticeTest(unittest.TestCase):
    def make_row(self, root: Path, split: str, group: str, sequence: str) -> dict[str, str]:
        image = root / "images" / split / f"{sequence}__0001.png"
        label = root / "labels" / split / f"{sequence}__0001.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), "black").save(image)
        label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        return {
            "split": split,
            "group": group,
            "sequence": sequence,
            "frame_id": "1",
            "output_image": str(image),
            "annotations": "1",
        }

    def write_manifest(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_group_is_independent_sequence_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.make_row(root, "train", "scene_3", "scene_3.1"),
                self.make_row(root, "train", "scene_3", "scene_3.2"),
                self.make_row(root, "val", "scene_4", "scene_4.1"),
                self.make_row(root, "test", "scene_5", "scene_5.1"),
            ]
            manifest = root / "manifest.csv"
            self.write_manifest(manifest, rows)
            loaded = load_manifest(manifest)
            self.assertEqual({row["sequence_id"] for row in loaded[:2]}, {"scene_3"})
            self.assertFalse(any(split_leakage(loaded).values()))
            summary = audit(manifest, root / "audit", sample_size=4, seed=1)
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(summary["sequence_ids"], 3)
            saved = json.loads((root / "audit/audit_summary.json").read_text())
            self.assertEqual(saved["statistical_unit"], "sequence_id")

    def test_leakage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                self.make_row(root, "train", "same_scene", "same_scene.1"),
                self.make_row(root, "test", "same_scene", "same_scene.2"),
            ]
            manifest = root / "manifest.csv"
            self.write_manifest(manifest, rows)
            summary = audit(manifest, root / "audit", sample_size=0, seed=1)
            self.assertEqual(summary["status"], "FAIL")
            self.assertEqual(summary["leakage"]["train_test"], ["same_scene"])


if __name__ == "__main__":
    unittest.main()
