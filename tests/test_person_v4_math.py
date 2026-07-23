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
)
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
        self.assertEqual(hook.mix_count, 1)

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
