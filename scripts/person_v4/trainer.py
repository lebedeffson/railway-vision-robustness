from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Sampler
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK

from scripts.canonical_m4_trainer import DifferentialLRDetectionTrainer
from scripts.person_v4.losses import PersonDGDetectionLoss


def canonical_path(value: str | Path) -> str:
    return str(Path(value).absolute())


class SceneRoundRobinSampler(Sampler[int]):
    """Deterministically alternate scenes while shuffling within each scene."""

    def __init__(
        self,
        image_paths: list[str],
        scene_by_stem: dict[str, str],
        seed: int,
    ) -> None:
        self.seed = int(seed)
        self.iteration = 0
        groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, image in enumerate(image_paths):
            stem = Path(image).stem
            if stem not in scene_by_stem:
                raise RuntimeError(f"Scene is missing for training tile: {image}")
            groups[scene_by_stem[stem]].append(index)
        if len(groups) < 2:
            raise RuntimeError("Scene-aware sampler requires at least two scenes")
        self.groups = dict(sorted(groups.items()))
        self.length = len(image_paths)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.iteration)
        self.iteration += 1
        shuffled = {
            scene: [
                values[index]
                for index in torch.randperm(
                    len(values), generator=generator
                ).tolist()
            ]
            for scene, values in self.groups.items()
        }
        offsets = {scene: 0 for scene in shuffled}
        scenes = list(shuffled)
        emitted = 0
        while emitted < self.length:
            for scene in scenes:
                offset = offsets[scene]
                values = shuffled[scene]
                if offset >= len(values):
                    continue
                yield values[offset]
                offsets[scene] += 1
                emitted += 1


class MixStyleSceneBank:
    """Feature hook mixing channel statistics from distinct scenes."""

    def __init__(
        self,
        *,
        model: nn.Module,
        probability: float,
        beta_alpha: float,
        epsilon: float,
    ) -> None:
        self.model = model
        self.probability = float(probability)
        self.beta_alpha = float(beta_alpha)
        self.epsilon = float(epsilon)
        self.bank: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.mix_count = 0

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        del module, inputs
        if not self.model.training or output.ndim != 4:
            return output
        scenes = getattr(self.model, "active_scene_ids", None)
        if not scenes or len(scenes) != output.shape[0]:
            return output
        mean = output.mean(dim=(2, 3), keepdim=True)
        std = (
            output.var(dim=(2, 3), keepdim=True, unbiased=False)
            + self.epsilon
        ).sqrt()
        mixed = output
        for index, scene in enumerate(scenes):
            alternatives = sorted(name for name in self.bank if name != scene)
            apply_mix = bool(
                alternatives
                and torch.rand((), device=output.device) < self.probability
            )
            if apply_mix:
                choice = alternatives[
                    int(
                        torch.randint(
                            len(alternatives), (), device=output.device
                        )
                    )
                ]
                other_mean, other_std = self.bank[choice]
                other_mean = other_mean.to(
                    device=output.device, dtype=output.dtype
                )
                other_std = other_std.to(
                    device=output.device, dtype=output.dtype
                )
                beta = torch.distributions.Beta(
                    self.beta_alpha, self.beta_alpha
                )
                coefficient = beta.sample().to(
                    device=output.device, dtype=output.dtype
                )
                target_mean = (
                    coefficient * mean[index : index + 1]
                    + (1 - coefficient) * other_mean
                )
                target_std = (
                    coefficient * std[index : index + 1]
                    + (1 - coefficient) * other_std
                )
                normalized = (
                    output[index : index + 1] - mean[index : index + 1]
                ) / std[index : index + 1]
                replacement = normalized * target_std + target_mean
                if mixed is output:
                    mixed = output.clone()
                mixed[index : index + 1] = replacement
                self.mix_count += 1
            self.bank[scene] = (
                mean[index : index + 1].detach().cpu(),
                std[index : index + 1].detach().cpu(),
            )
        return mixed


class PersonDGDetectionModel(DetectionModel):
    """YOLO detection model with a prospectively configured custom loss."""

    dg_loss_config: dict[str, Any]
    dg_scene_names: list[str]
    dg_log_path: Path | None

    def init_criterion(self) -> PersonDGDetectionLoss:
        return PersonDGDetectionLoss(
            self,
            self.dg_loss_config,
            self.dg_scene_names,
            self.dg_log_path,
        )


class PersonDGTrainer(DifferentialLRDetectionTrainer):
    """Trainer for A1-A3 with scene metadata and optional DG mechanisms."""

    experiment_config: dict[str, Any] = {}
    scene_by_stem: dict[str, str] = {}
    scene_names: list[str] = []
    dro_log_path: Path | None = None
    augmentation_log: list[dict[str, Any]] = []

    def get_model(
        self,
        cfg: str | None = None,
        weights: str | None = None,
        verbose: bool = True,
    ) -> PersonDGDetectionModel:
        model = self.set_model_names_for_load(
            PersonDGDetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
            )
        )
        if weights:
            model.load(weights)
        model.dg_loss_config = dict(self.experiment_config)
        model.dg_scene_names = list(self.scene_names)
        model.dg_log_path = self.dro_log_path
        model.active_scene_ids = []
        return model

    def set_model_attributes(self) -> None:
        super().set_model_attributes()
        config = self.experiment_config
        if not config.get("mixstyle", {}).get("enabled"):
            return
        mix = config["mixstyle"]
        self.model.mixstyle_hooks = []
        self.model.mixstyle_banks = []
        for index in mix["layer_indices"]:
            bank = MixStyleSceneBank(
                model=self.model,
                probability=mix["probability"],
                beta_alpha=mix["beta_alpha"],
                epsilon=mix["epsilon"],
            )
            handle = self.model.model[int(index)].register_forward_hook(bank)
            self.model.mixstyle_hooks.append(handle)
            self.model.mixstyle_banks.append(bank)
        LOGGER.info(
            "Person v4 MixStyle enabled at backbone layers "
            f"{mix['layer_indices']} with a different-scene style bank"
        )

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        if mode != "train" or not self.experiment_config["scene_sampler"]:
            return DetectionTrainer.get_dataloader(
                self, dataset_path, batch_size, rank, mode
            )
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        sampler = SceneRoundRobinSampler(
            dataset.im_files,
            self.scene_by_stem,
            int(self.experiment_config["seed"]),
        )
        workers = min(
            self.args.workers,
            max(0, math.ceil(len(dataset) / batch_size)),
        )
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=workers,
            prefetch_factor=4 if workers > 0 else None,
            pin_memory=torch.cuda.device_count() > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(
                6148914691236517205 + RANK
            ),
            drop_last=False,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        paths = batch.get("im_file", [])
        scenes = [
            self.scene_by_stem.get(Path(path).stem, "heldout_or_unknown")
            for path in paths
        ]
        batch["grouped_scene_id"] = scenes
        self.model.active_scene_ids = scenes
        if self.experiment_config.get("scale_aware", {}).get("enabled"):
            self._scale_aware_zoom(batch)
        return batch

    def _scale_aware_zoom(self, batch: dict) -> None:
        config = self.experiment_config["scale_aware"]
        if batch["img"].shape[0] != 1 or not len(batch["bboxes"]):
            return
        if torch.rand((), device=batch["img"].device) >= float(
            config["probability"]
        ):
            return
        boxes = batch["bboxes"]
        areas = boxes[:, 2] * boxes[:, 3]
        target = boxes[int(areas.argmin())]
        zoom = float(
            torch.empty((), device=batch["img"].device)
            .uniform_(float(config["zoom_min"]), float(config["zoom_max"]))
            .item()
        )
        crop_width = 1.0 / zoom
        crop_height = 1.0 / zoom
        left = min(max(float(target[0]) - crop_width / 2, 0.0), 1 - crop_width)
        top = min(max(float(target[1]) - crop_height / 2, 0.0), 1 - crop_height)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        fully_visible = bool(
            (
                (x1 >= left)
                & (y1 >= top)
                & (x2 <= left + crop_width)
                & (y2 <= top + crop_height)
            )
            .all()
            .item()
        )
        if not fully_visible:
            self.augmentation_log.append(
                {"applied": False, "reason": "would_remove_GT"}
            )
            return
        height, width = batch["img"].shape[-2:]
        px_left = int(round(left * width))
        px_top = int(round(top * height))
        px_right = int(round((left + crop_width) * width))
        px_bottom = int(round((top + crop_height) * height))
        actual_left = px_left / width
        actual_top = px_top / height
        actual_width = (px_right - px_left) / width
        actual_height = (px_bottom - px_top) / height
        transformed = boxes.clone()
        transformed[:, 0] = (boxes[:, 0] - actual_left) / actual_width
        transformed[:, 1] = (boxes[:, 1] - actual_top) / actual_height
        transformed[:, 2] = boxes[:, 2] / actual_width
        transformed[:, 3] = boxes[:, 3] / actual_height
        minimum = float(config["minimum_box_px"])
        valid_size = bool(
            (
                (transformed[:, 2] * width >= minimum)
                & (transformed[:, 3] * height >= minimum)
            )
            .all()
            .item()
        )
        if not valid_size:
            self.augmentation_log.append(
                {"applied": False, "reason": "box_below_minimum"}
            )
            return
        crop = batch["img"][:, :, px_top:px_bottom, px_left:px_right]
        batch["img"] = F.interpolate(
            crop, size=(height, width), mode="bilinear", align_corners=False
        )
        batch["bboxes"] = transformed
        self.augmentation_log.append(
            {
                "applied": True,
                "zoom": zoom,
                "minimum_width_px_before": float(
                    (boxes[:, 2] * width).min()
                ),
                "minimum_height_px_before": float(
                    (boxes[:, 3] * height).min()
                ),
                "minimum_width_px_after": float(
                    (transformed[:, 2] * width).min()
                ),
                "minimum_height_px_after": float(
                    (transformed[:, 3] * height).min()
                ),
            }
        )
