from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import TaskAlignedAssigner, bbox2dist, make_anchors


def xyxy_gaussian_vector(boxes: torch.Tensor) -> torch.Tensor:
    """Map xyxy boxes to the axis-aligned Gaussian vector used by NWD."""
    centers = (boxes[..., :2] + boxes[..., 2:]) / 2
    half_sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0) / 2
    return torch.cat((centers, half_sizes), dim=-1)


def normalized_wasserstein_similarity(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    constant: float | torch.Tensor,
    epsilon: float = 1.0e-9,
) -> torch.Tensor:
    """Normalized Gaussian Wasserstein similarity for aligned xyxy boxes."""
    delta = xyxy_gaussian_vector(boxes_a) - xyxy_gaussian_vector(boxes_b)
    distance = delta.square().sum(dim=-1).clamp_min(epsilon).sqrt()
    scale = torch.as_tensor(
        constant, device=distance.device, dtype=distance.dtype
    ).clamp_min(epsilon)
    return torch.exp(-distance / scale)


def small_object_alpha(
    boxes: torch.Tensor,
    reference_area: float,
    temperature: float,
) -> torch.Tensor:
    """Smoothly increase the NWD contribution as box area decreases."""
    sizes = (boxes[..., 2:] - boxes[..., :2]).clamp_min(1.0e-6)
    log_area = sizes[..., 0].log() + sizes[..., 1].log()
    reference = math.log(float(reference_area))
    return torch.sigmoid((reference - log_area) / float(temperature))


def quality_focal_loss(
    logits: torch.Tensor,
    target_quality: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Quality Focal Loss with a continuous localization-quality target."""
    target = target_quality.to(dtype=logits.dtype)
    probability = logits.sigmoid()
    modulation = (target - probability).abs().pow(float(beta))
    return F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    ) * modulation


def update_group_dro_weights(
    weights: torch.Tensor,
    group_index: int,
    group_loss: float | torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Apply one deterministic exponentiated-gradient GroupDRO update."""
    if weights.ndim != 1 or weights.numel() == 0:
        raise ValueError("GroupDRO weights must be a non-empty vector")
    if not 0 <= int(group_index) < weights.numel():
        raise IndexError("GroupDRO group index is outside the weight vector")
    updated = weights.clone()
    loss = torch.as_tensor(
        group_loss, dtype=updated.dtype, device=updated.device
    ).clamp(max=50)
    updated[int(group_index)] *= torch.exp(float(eta) * loss)
    normalizer = updated.sum()
    if not torch.isfinite(normalizer) or float(normalizer) <= 0:
        raise FloatingPointError("GroupDRO weight update is not finite")
    return updated / normalizer


class NWDTaskAlignedAssigner(TaskAlignedAssigner):
    """Task-aligned assignment using a frozen CIoU/NWD hybrid metric."""

    def __init__(
        self,
        *,
        nwd_constant: float,
        reference_area: float,
        area_temperature: float,
        candidate_min_similarity: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.nwd_constant = float(nwd_constant)
        self.reference_area = float(reference_area)
        self.area_temperature = float(area_temperature)
        self.candidate_min_similarity = float(candidate_min_similarity)

    def pair_metrics(
        self,
        pd_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        na = pd_bboxes.shape[-2]
        predictions = pd_bboxes.unsqueeze(1).expand(
            -1, self.n_max_boxes, -1, -1
        )[valid]
        targets = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[valid]
        ciou = bbox_iou(
            targets, predictions, xywh=False, CIoU=True
        ).squeeze(-1).clamp_(0, 1)
        nwd = normalized_wasserstein_similarity(
            targets, predictions, self.nwd_constant
        )
        alpha = small_object_alpha(
            targets, self.reference_area, self.area_temperature
        )
        return ciou, nwd, alpha

    def get_box_metrics(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        na = pd_bboxes.shape[-2]
        valid = mask_gt.bool()
        hybrid = torch.zeros(
            [self.bs, self.n_max_boxes, na],
            dtype=pd_bboxes.dtype,
            device=pd_bboxes.device,
        )
        bbox_scores = torch.zeros(
            [self.bs, self.n_max_boxes, na],
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )
        indices = torch.zeros(
            [2, self.bs, self.n_max_boxes],
            dtype=torch.long,
            device=gt_labels.device,
        )
        indices[0] = torch.arange(
            self.bs, device=gt_labels.device
        ).view(-1, 1).expand(-1, self.n_max_boxes)
        indices[1] = gt_labels.squeeze(-1).long()
        bbox_scores[valid] = pd_scores[
            indices[0], :, indices[1]
        ][valid]
        if valid.any():
            ciou, nwd, alpha = self.pair_metrics(
                pd_bboxes, gt_bboxes, valid
            )
            hybrid[valid] = (1 - alpha) * ciou + alpha * nwd
        align_metric = bbox_scores.pow(self.alpha) * hybrid.pow(self.beta)
        return align_metric, hybrid

    def get_pos_mask(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inside = self.select_candidates_in_gts(
            anc_points, gt_bboxes, mask_gt
        )
        valid = mask_gt.expand(-1, -1, pd_bboxes.shape[-2]).bool()
        nwd_candidates = torch.zeros_like(inside, dtype=torch.bool)
        if valid.any():
            _, nwd, alpha = self.pair_metrics(
                pd_bboxes, gt_bboxes, valid
            )
            expanded_nwd = torch.zeros_like(inside, dtype=pd_bboxes.dtype)
            expanded_alpha = torch.zeros_like(
                inside, dtype=pd_bboxes.dtype
            )
            expanded_nwd[valid] = nwd
            expanded_alpha[valid] = alpha
            nwd_candidates = (
                expanded_nwd >= self.candidate_min_similarity
            ) & (expanded_alpha >= 0.5)
        candidates = (inside.bool() | nwd_candidates) & valid
        align_metric, overlaps = self.get_box_metrics(
            pd_scores,
            pd_bboxes,
            gt_labels,
            gt_bboxes,
            candidates,
        )
        mask_topk = self.select_topk_candidates(
            align_metric,
            topk_mask=mask_gt.expand(-1, -1, self.topk).bool(),
        )
        mask_pos = mask_topk * candidates * mask_gt
        return mask_pos, align_metric, overlaps


class HybridNWDBboxLoss(BboxLoss):
    """CIoU/NWD regression with more NWD weight for smaller boxes."""

    def __init__(
        self,
        reg_max: int,
        *,
        nwd_constant: float,
        reference_area: float,
        area_temperature: float,
    ) -> None:
        super().__init__(reg_max)
        self.nwd_constant = float(nwd_constant)
        self.reference_area = float(reference_area)
        self.area_temperature = float(area_temperature)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_grid = pred_bboxes[fg_mask]
        target_grid = target_bboxes[fg_mask]
        ciou = bbox_iou(
            pred_grid, target_grid, xywh=False, CIoU=True
        ).clamp(-1, 1)
        batch_stride = stride.view(1, -1, 1).expand(
            pred_bboxes.shape[0], -1, -1
        )[fg_mask]
        pred_px = pred_grid * batch_stride
        target_px = target_grid * batch_stride
        nwd = normalized_wasserstein_similarity(
            pred_px, target_px, self.nwd_constant
        ).unsqueeze(-1)
        alpha = small_object_alpha(
            target_px, self.reference_area, self.area_temperature
        ).unsqueeze(-1)
        hybrid = (1 - alpha) * (1 - ciou) + alpha * (1 - nwd)
        loss_box = (hybrid * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points,
                target_bboxes,
                self.dfl_loss.reg_max - 1,
            )
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = pred_dist.sum() * 0
        return loss_box, loss_dfl


class PersonDGDetectionLoss(v8DetectionLoss):
    """Person-only NWD/QFL loss with optional scene-wise GroupDRO."""

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        scene_names: list[str],
        log_path: Path | None,
    ) -> None:
        super().__init__(model)
        self.config = config
        self.nwd_constant = float(config["nwd_constant"])
        nwd = config["nwd"]
        self.assigner = NWDTaskAlignedAssigner(
            topk=10,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            nwd_constant=self.nwd_constant,
            reference_area=float(nwd["reference_area"]),
            area_temperature=float(nwd["area_temperature"]),
            candidate_min_similarity=float(
                nwd["candidate_min_similarity"]
            ),
        )
        self.bbox_loss = HybridNWDBboxLoss(
            self.reg_max,
            nwd_constant=self.nwd_constant,
            reference_area=float(nwd["reference_area"]),
            area_temperature=float(nwd["area_temperature"]),
        ).to(self.device)
        self.qfl_beta = float(config["qfl"]["beta"])
        self.iou_weight = float(config["qfl"]["iou_weight"])
        self.nwd_weight = float(config["qfl"]["nwd_weight"])
        self.use_group_dro = bool(config["group_dro"]["enabled"])
        self.group_eta = float(config["group_dro"]["eta"])
        self.scene_names = list(scene_names)
        self.scene_index = {
            name: index for index, name in enumerate(self.scene_names)
        }
        self.group_weights = torch.full(
            (max(len(self.scene_names), 1),),
            1.0 / max(len(self.scene_names), 1),
            device=self.device,
        )
        self.log_path = log_path
        self.step = 0
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "step",
                        "scene",
                        "unweighted_loss",
                        "scene_weight",
                        "objective_multiplier",
                    ]
                )

    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple:
        loss = torch.zeros(3, device=self.device)
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        anchor_points, stride_tensor = make_anchors(
            preds["feats"], self.stride, 0.5
        )
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(
                preds["feats"][0].shape[2:],
                device=self.device,
                dtype=dtype,
            )
            * self.stride[0]
        )
        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["cls"].view(-1, 1),
                batch["bboxes"],
            ),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        (
            _,
            target_bboxes,
            target_scores,
            fg_mask,
            target_gt_idx,
        ) = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        quality_target = torch.zeros_like(target_scores, dtype=dtype)
        if fg_mask.any():
            pred_grid = pred_bboxes[fg_mask]
            target_grid = (target_bboxes / stride_tensor)[fg_mask]
            ciou = bbox_iou(
                pred_grid.detach(),
                target_grid,
                xywh=False,
                CIoU=True,
            ).squeeze(-1).clamp_(0, 1)
            batch_stride = stride_tensor.view(1, -1, 1).expand(
                batch_size, -1, -1
            )[fg_mask]
            nwd = normalized_wasserstein_similarity(
                pred_grid.detach() * batch_stride,
                target_grid * batch_stride,
                self.nwd_constant,
            )
            quality = (
                self.iou_weight * ciou + self.nwd_weight * nwd
            ).clamp_(0, 1)
            quality_target[fg_mask] = quality.unsqueeze(-1)
        quality_sum = quality_target.sum().clamp_min(1.0)
        loss[1] = quality_focal_loss(
            pred_scores, quality_target, self.qfl_beta
        ).sum() / quality_sum
        if fg_mask.any():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                quality_target,
                quality_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (
                fg_mask,
                target_gt_idx,
                target_bboxes,
                anchor_points,
                stride_tensor,
            ),
            loss,
            loss.detach(),
        )

    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = preds["boxes"].shape[0]
        loss, detached = self.get_assigned_targets_and_loss(preds, batch)[1:]
        multiplier = 1.0
        scene = None
        scene_ids = batch.get("grouped_scene_id")
        if (
            self.use_group_dro
            and scene_ids
            and len(scene_ids) == 1
            and scene_ids[0] in self.scene_index
        ):
            scene = scene_ids[0]
            index = self.scene_index[scene]
            with torch.no_grad():
                unweighted = detached.sum().clamp(max=50)
                self.group_weights.copy_(
                    update_group_dro_weights(
                        self.group_weights,
                        index,
                        unweighted,
                        self.group_eta,
                    )
                )
                multiplier = float(
                    len(self.scene_names) * self.group_weights[index]
                )
        weighted = loss * float(multiplier)
        if scene is not None and self.log_path is not None:
            with self.log_path.open(
                "a", encoding="utf-8", newline=""
            ) as handle:
                csv.writer(handle).writerow(
                    [
                        self.step,
                        scene,
                        float(detached.sum()),
                        float(
                            self.group_weights[self.scene_index[scene]]
                        ),
                        float(multiplier),
                    ]
                )
        self.step += 1
        return weighted * batch_size, detached
