import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_final_matrix import (
    expected_rows_per_condition, expected_rows_per_image,
    prepare_condition_partial, prepare_partial,
)


class FinalMatrixCheckpointTest(unittest.TestCase):
    @staticmethod
    def args() -> SimpleNamespace:
        return SimpleNamespace(
            fgsm_eps=[0.5], pgd_eps=[0.1], pgd_steps=[20], adaptive_pgd=True,
            adaptive_pgd_eps=None, adaptive_pgd_steps=[20, 40],
            seeds=[42, 123, 999],
            defenses=["none", "tnorm", "bilateral", "gaussian", "jpeg", "median"],
        )

    def test_partial_checkpoint_keeps_only_complete_images(self) -> None:
        args = self.args()
        expected = expected_rows_per_image(args)
        self.assertEqual(expected, 108)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv.tmp"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_path", "value"])
                writer.writeheader()
                writer.writerows(
                    [{"image_path": "complete.jpg", "value": index} for index in range(expected)]
                    + [{"image_path": "partial.jpg", "value": 1}]
                )
            completed, fields, rows = prepare_partial(path, expected)
            self.assertEqual(completed, {"complete.jpg"})
            self.assertEqual(fields, ["image_path", "value"])
            self.assertEqual(rows, expected)
            self.assertNotIn("partial.jpg", path.read_text(encoding="utf-8"))

    def test_condition_checkpoint_retains_complete_condition_only(self) -> None:
        args = self.args()
        expected = expected_rows_per_condition(args, "fgsm", False)
        fields = ["image_path", "attack", "adaptive", "epsilon_px", "steps", "value"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv.tmp"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows({
                    "image_path": "frame.jpg", "attack": "fgsm", "adaptive": False,
                    "epsilon_px": .5, "steps": 1, "value": index,
                } for index in range(expected))
                writer.writerow({
                    "image_path": "frame.jpg", "attack": "pgd", "adaptive": False,
                    "epsilon_px": .1, "steps": 20, "value": 1,
                })
            images, conditions, returned_fields, rows = prepare_condition_partial(path, args)
            self.assertFalse(images)
            self.assertEqual(len(conditions), 1)
            self.assertEqual(returned_fields, fields)
            self.assertEqual(rows, expected)
            self.assertNotIn(",pgd,", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
