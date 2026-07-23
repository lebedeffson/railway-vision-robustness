from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from ultralytics import YOLO

from scripts.person_v4.losses import (
    normalized_wasserstein_similarity,
    quality_focal_loss,
    small_object_alpha,
    update_group_dro_weights,
)
from scripts.person_v4.swad import average_state_dicts, state_dict_sha256
from scripts.person_v4.trainer import (
    MixStyleSceneBank,
    PersonDGDetectionModel,
    SceneRoundRobinSampler,
)


ROOT = Path(__file__).resolve().parents[1]


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.active_scene_ids: list[str] = []


class PersonV4MathTest(unittest.TestCase):
    def test_nwd_is_symmetric_bounded_and_identity_is_one(self) -> None:
        a = torch.tensor([[1.0, 2.0, 5.0, 8.0]])
        b = torch.tensor([[2.0, 3.0, 6.0, 9.0]])
        identity = normalized_wasserstein_similarity(a, a, 10.0)
        ab = normalized_wasserstein_similarity(a, b, 10.0)
        ba = normalized_wasserstein_similarity(b, a, 10.0)
        self.assertTrue(torch.allclose(identity, torch.ones_like(identity)))
        self.assertTrue(torch.allclose(ab, ba))
        self.assertGreaterEqual(float(ab), 0.0)
        self.assertLessEqual(float(ab), 1.0)

    def test_nwd_is_smooth_for_non_overlapping_boxes(self) -> None:
        prediction = torch.tensor(
            [[10.0, 10.0, 12.0, 12.0]], requires_grad=True
        )
        target = torch.tensor([[20.0, 20.0, 22.0, 22.0]])
        loss = 1 - normalized_wasserstein_similarity(
            prediction, target, 8.0
        )
        loss.sum().backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)

    def test_nwd_decreases_smoothly_with_center_shift(self) -> None:
        target = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        shifts = [
            torch.tensor([[float(value), 0.0, float(value + 2), 2.0]])
            for value in (0, 1, 2)
        ]
        similarities = [
            float(normalized_wasserstein_similarity(target, box, 8.0))
            for box in shifts
        ]
        self.assertGreater(similarities[0], similarities[1])
        self.assertGreater(similarities[1], similarities[2])
        iou_loss_at_one_pixel = 1 - (2.0 / 6.0)
        nwd_loss_at_one_pixel = 1 - similarities[1]
        self.assertLess(nwd_loss_at_one_pixel, iou_loss_at_one_pixel)

    def test_nwd_is_finite_for_one_pixel_boxes(self) -> None:
        first = torch.tensor(
            [[0.0, 0.0, 1.0, 1.0]], requires_grad=True
        )
        second = torch.tensor([[1.0, 0.0, 2.0, 1.0]])
        similarity = normalized_wasserstein_similarity(first, second, 4.0)
        (1 - similarity).sum().backward()
        self.assertTrue(torch.isfinite(similarity).all())
        self.assertTrue(torch.isfinite(first.grad).all())

    def test_small_boxes_receive_more_nwd_weight(self) -> None:
        small = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
        large = torch.tensor([[0.0, 0.0, 64.0, 64.0]])
        self.assertGreater(
            float(small_object_alpha(small, 256.0, 1.0)),
            float(small_object_alpha(large, 256.0, 1.0)),
        )

    def test_quality_focal_loss_has_finite_gradient(self) -> None:
        logits = torch.tensor([[-2.0, 0.0, 2.0]], requires_grad=True)
        targets = torch.tensor([[0.0, 0.5, 1.0]])
        loss = quality_focal_loss(logits, targets, beta=2.0).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_quality_focal_gradient_moves_scores_toward_quality(self) -> None:
        positive = torch.tensor([-2.0], requires_grad=True)
        quality_focal_loss(
            positive, torch.tensor([0.8]), beta=2.0
        ).sum().backward()
        self.assertLess(float(positive.grad), 0.0)
        background = torch.tensor([2.0], requires_grad=True)
        quality_focal_loss(
            background, torch.tensor([0.0]), beta=2.0
        ).sum().backward()
        self.assertGreater(float(background.grad), 0.0)

    def test_quality_target_is_bounded_and_orders_localization(self) -> None:
        iou = torch.tensor([0.2, 0.8])
        nwd = torch.tensor([0.3, 0.9])
        target = 0.5 * iou + 0.5 * nwd
        self.assertTrue(torch.all((0 <= target) & (target <= 1)))
        self.assertGreater(float(target[1]), float(target[0]))

    def test_group_dro_upweights_the_worst_scene(self) -> None:
        weights = torch.full((3,), 1 / 3)
        for index, loss in enumerate((0.2, 0.5, 1.2)):
            weights = update_group_dro_weights(
                weights, index, loss, eta=0.5
            )
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertGreater(float(weights[2]), float(weights[1]))
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_scene_sampler_round_robins_groups(self) -> None:
        paths = ["/x/a_0.jpg", "/x/a_1.jpg", "/x/b_0.jpg", "/x/b_1.jpg"]
        scene = {"a_0": "a", "a_1": "a", "b_0": "b", "b_1": "b"}
        sampler = SceneRoundRobinSampler(paths, scene, seed=7)
        order = list(iter(sampler))
        groups = [scene[torch_path] for torch_path in [
            paths[index].rsplit("/", 1)[-1].removesuffix(".jpg")
            for index in order
        ]]
        self.assertEqual(groups, ["a", "b", "a", "b"])

    def test_mixstyle_uses_only_a_different_scene_bank(self) -> None:
        torch.manual_seed(3)
        model = _Model()
        hook = MixStyleSceneBank(
            model=model, probability=1.0, beta_alpha=0.1, epsilon=1.0e-6
        )
        model.train()
        model.active_scene_ids = ["scene_a"]
        first = torch.randn(1, 4, 3, 3)
        unchanged = hook(torch.nn.Identity(), (first,), first)
        self.assertTrue(torch.equal(first, unchanged))
        model.active_scene_ids = ["scene_b"]
        second = torch.randn(1, 4, 3, 3) + 10
        mixed = hook(torch.nn.Identity(), (second,), second)
        self.assertFalse(torch.equal(second, mixed))
        self.assertEqual(mixed.shape, second.shape)
        second_standardized = (
            second - second.mean(dim=(2, 3), keepdim=True)
        ) / second.std(dim=(2, 3), keepdim=True, unbiased=False)
        mixed_standardized = (
            mixed - mixed.mean(dim=(2, 3), keepdim=True)
        ) / mixed.std(dim=(2, 3), keepdim=True, unbiased=False)
        self.assertTrue(
            torch.allclose(
                second_standardized,
                mixed_standardized,
                atol=1.0e-4,
                rtol=1.0e-4,
            )
        )
        self.assertEqual(hook.mix_count, 1)
        model.eval()
        held = torch.randn(1, 4, 3, 3)
        self.assertTrue(
            torch.equal(
                held, hook(torch.nn.Identity(), (held,), held)
            )
        )

    def test_swad_average_and_hash_are_deterministic(self) -> None:
        states = [
            {
                "weight": torch.tensor([1.0, 3.0]),
                "bn.running_mean": torch.tensor([2.0]),
                "bn.num_batches_tracked": torch.tensor(4),
            },
            {
                "weight": torch.tensor([3.0, 5.0]),
                "bn.running_mean": torch.tensor([4.0]),
                "bn.num_batches_tracked": torch.tensor(7),
            },
        ]
        first = average_state_dicts(states)
        second = average_state_dicts(states)
        self.assertTrue(
            torch.equal(first["weight"], torch.tensor([2.0, 4.0]))
        )
        self.assertTrue(
            torch.equal(first["bn.running_mean"], torch.tensor([3.0]))
        )
        self.assertEqual(int(first["bn.num_batches_tracked"]), 7)
        self.assertEqual(
            state_dict_sha256(first), state_dict_sha256(second)
        )

    def test_custom_detection_loss_backward_on_real_yolo_graph(self) -> None:
        base = YOLO(str(ROOT / "yolo11m.pt")).model
        model = PersonDGDetectionModel(
            base.yaml, nc=1, ch=3, verbose=False
        )
        model.load(base)
        arguments = dict(base.args)
        arguments.update(box=7.5, cls=0.5, dfl=1.5)
        model.args = SimpleNamespace(**arguments)
        model.dg_loss_config = {
            "nwd_constant": 8.0,
            "nwd": {
                "reference_area": 256.0,
                "area_temperature": 1.0,
                "candidate_min_similarity": 0.5,
            },
            "qfl": {
                "beta": 2.0,
                "iou_weight": 0.5,
                "nwd_weight": 0.5,
            },
            "group_dro": {"enabled": False, "eta": 0.01},
        }
        model.dg_scene_names = ["scene_a", "scene_b"]
        model.dg_log_path = None
        model.train()
        batch = {
            "img": torch.rand(1, 3, 128, 128),
            "batch_idx": torch.tensor([0.0]),
            "cls": torch.tensor([[0.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
            "grouped_scene_id": ["scene_a"],
        }
        loss, _ = model(batch)
        loss.sum().backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreater(gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
