import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from collections import defaultdict

CORE_LOSS_KEY = "core_loss"


class MemoryFeatureAttackLoss(nn.Module):
    def __init__(
        self,
        weight_dict,
        weight_l2: float = 1.0,
        weight_repel_l2: float = 1.0,
        use_l2: bool = True,
        use_repel_l2: bool = True,
        sigmoid_scale: float = 20.0,
        sigmoid_bias: float = -10.0,
        use_spatial_uniform: bool = True,
        weight_spatial_uniform: float = 1.0,
    ):
        super().__init__()
        self.weight_dict = weight_dict
        self.weight_l2 = weight_l2
        self.weight_repel_l2 = weight_repel_l2
        self.use_l2 = use_l2
        self.use_repel_l2 = use_repel_l2
        self.sigmoid_scale = sigmoid_scale
        self.sigmoid_bias = sigmoid_bias
        self.use_spatial_uniform = use_spatial_uniform
        self.weight_spatial_uniform = weight_spatial_uniform

    def _generate_random_malicious_mask(self, batch_size, height, width, device):
        malicious_mask = torch.zeros(batch_size, 1, height, width, device=device)
        malicious_mask = malicious_mask * 20.0 - 10.0
        return malicious_mask

    def _compute_malicious_memory_feat(
        self,
        malicious_masks: torch.Tensor,
        pix_feat: torch.Tensor,
        maskmem_backbone: nn.Module,
        feat_sizes: List,
    ) -> torch.Tensor:
        mask_for_mem = torch.sigmoid(malicious_masks)
        if self.sigmoid_scale != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale
        if self.sigmoid_bias != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias
        maskmem_out = maskmem_backbone(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        return maskmem_out["vision_features"]

    def _compute_original_memory_feat(
        self,
        original_masks: torch.Tensor,
        original_pix_feat: torch.Tensor,
        maskmem_backbone: nn.Module,
        feat_sizes: List,
    ) -> torch.Tensor:
        mask_for_mem = torch.sigmoid(original_masks)
        if self.sigmoid_scale != 1.0:
            mask_for_mem = mask_for_mem * self.sigmoid_scale
        if self.sigmoid_bias != 0.0:
            mask_for_mem = mask_for_mem + self.sigmoid_bias
        maskmem_out = maskmem_backbone(
            original_pix_feat, mask_for_mem, skip_mask_sigmoid=True
        )
        return maskmem_out["vision_features"]

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        losses = defaultdict(float)
        total_frames = 0

        for frame_idx, outs in enumerate(outs_batch):
            actual_memory_feat = outs["maskmem_features"]
            B, C, H, W = actual_memory_feat.shape

            pix_feat = outs["pix_feat"]
            original_pix_feat = outs.get("original_pix_feat", pix_feat)
            maskmem_backbone = outs["maskmem_backbone"]
            feat_sizes = outs["feat_sizes"]

            mask_h, mask_w = outs["pred_masks_high_res"].shape[-2:]
            device = actual_memory_feat.device

            malicious_masks = self._generate_random_malicious_mask(
                B, mask_h, mask_w, device
            )
            malicious_memory_feat = self._compute_malicious_memory_feat(
                malicious_masks, original_pix_feat, maskmem_backbone, feat_sizes
            )

            original_masks = targets_batch[frame_idx].segments.unsqueeze(1).float()
            original_masks = original_masks * 20.0 - 10.0
            original_memory_feat = self._compute_original_memory_feat(
                original_masks, original_pix_feat, maskmem_backbone, feat_sizes
            )

            assert actual_memory_feat.shape == malicious_memory_feat.shape
            assert actual_memory_feat.shape == original_memory_feat.shape

            attention_map = outs["attention_map"]

            frame_losses = self._compute_memory_loss(
                actual_memory_feat,
                malicious_memory_feat,
                original_memory_feat,
                attention_map,
                frame_idx,
            )

            for k, v in frame_losses.items():
                losses[k] = (
                    losses.get(k, torch.tensor(0.0, device=v.device, dtype=v.dtype)) + v
                )

            total_frames += 1

        for k in losses:
            if k != "loss_memory" and k != CORE_LOSS_KEY:
                losses[k] = losses[k] / total_frames

        total_loss = sum(
            v for k, v in losses.items() if k not in ["loss_memory", CORE_LOSS_KEY]
        )
        losses["loss_memory"] = total_loss
        losses[CORE_LOSS_KEY] = total_loss

        return losses

    def _compute_memory_loss(
        self,
        actual_memory_feat: torch.Tensor,
        malicious_memory_feat: torch.Tensor,
        original_memory_feat: torch.Tensor,
        attention_map: torch.Tensor,
        frame_idx: int,
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        if frame_idx == 0:
            if self.use_l2:
                l2_loss = F.mse_loss(actual_memory_feat, malicious_memory_feat)
                losses["loss_memory_l2"] = l2_loss * self.weight_l2

            if self.use_repel_l2:

                mse_dist = F.mse_loss(actual_memory_feat, original_memory_feat)

                repel_l2_loss = torch.exp(-mse_dist * 10.0)
                losses["loss_memory_repel_l2"] = repel_l2_loss * self.weight_repel_l2

        if frame_idx > 0:
            if self.use_spatial_uniform:
                global_mean = attention_map.mean(dim=(2, 3), keepdim=True)
                spatial_var = ((attention_map - global_mean) ** 2).mean()
                losses["loss_spatial_uniform"] = (
                    spatial_var * self.weight_spatial_uniform
                )

        return losses
