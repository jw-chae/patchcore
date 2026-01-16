from __future__ import annotations

from typing import Optional, Tuple, Dict

import math
import torch
import torch.nn.functional as F

from patchcorex.models.backbones.backbone import build_backbone
from patchcorex.utils.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("vit_dual")
class ViTDualExtractor:
    is_dual = True

    def __init__(
        self,
        backbone_cfg: dict,
        seg_normalize: str | None = None,
        scr_normalize: str | None = None,
        scr_source: str = "cls",
    ) -> None:
        self.backbone = build_backbone(backbone_cfg)
        self.seg_normalize = seg_normalize
        self.scr_normalize = scr_normalize
        self.scr_source = scr_source
        self.last_grid_shape: Optional[Tuple[int, int]] = None
        self.enable_cam = False
        self.last_cam_tokens: Optional[torch.Tensor] = None

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not hasattr(self.backbone, "extract_tokens"):
            raise RuntimeError("ViT backbone does not implement extract_tokens")
        cls_token, patch_tokens = self.backbone.extract_tokens(x)

        b, patch_count, d = patch_tokens.shape
        grid = int(math.sqrt(patch_count))
        if grid * grid == patch_count:
            self.last_grid_shape = (grid, grid)
        else:
            self.last_grid_shape = None

        patches = patch_tokens
        if self.enable_cam:
            self.last_cam_tokens = patches
            self.last_cam_tokens.retain_grad()

        if self.seg_normalize == "l2":
            patches = F.normalize(patches, p=2, dim=-1)

        if self.scr_source == "cls":
            scr_vec = cls_token
        elif self.scr_source == "mean":
            scr_vec = patch_tokens.mean(dim=1)
        elif self.scr_source == "attn_pool":
            scale = patch_tokens.shape[-1] ** -0.5
            weights = (patch_tokens * cls_token.unsqueeze(1)).sum(dim=-1) * scale
            weights = torch.softmax(weights, dim=1)
            scr_vec = (weights.unsqueeze(-1) * patch_tokens).sum(dim=1)
        else:
            raise KeyError(f"Unsupported scr_source: {self.scr_source}")

        if self.scr_normalize == "l2":
            scr_vec = F.normalize(scr_vec, p=2, dim=-1)

        return {"seg": patches, "scr": scr_vec}
