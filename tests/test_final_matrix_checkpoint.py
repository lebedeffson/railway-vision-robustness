import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_final_matrix import expected_rows_per_image, prepare_partial


class FinalMatrixCheckpointTest(unittest.TestCase):
    def test_partial_checkpoint_keeps_only_complete_images(self) -> None:
        args = SimpleNamespace(
            fgsm_eps=[0.5], pgd_eps=[0.1], pgd_steps=[20], adaptive_pgd=True,
            adaptive_pgd_eps=None,
            adaptive_pgd_steps=[20, 40],
            seeds=[42, 123, 999],
            defenses=["none", "tnorm", "bilateral", "gaussian", "jpeg", "median"],
        )
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


if __name__ == "__main__":
    unittest.main()
