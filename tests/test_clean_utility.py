import unittest

import torch

from evaluate_clean_utility import correct_matrix, structural_similarity


class CleanUtilityTest(unittest.TestCase):
    def test_identical_images_have_unit_ssim(self) -> None:
        image = torch.rand(1, 3, 24, 24)
        self.assertAlmostEqual(structural_similarity(image, image), 1.0, places=5)

    def test_exact_box_is_correct_at_all_iou_levels(self) -> None:
        prediction = torch.tensor([[1.0, 1.0, 5.0, 5.0, 0.9, 2.0]])
        boxes = torch.tensor([[1.0, 1.0, 5.0, 5.0]])
        classes = torch.tensor([2])
        correct = correct_matrix(prediction, boxes, classes)
        self.assertEqual(correct.shape, (1, 10))
        self.assertTrue(correct.all())


if __name__ == "__main__":
    unittest.main()
