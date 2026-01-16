from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F

from patchcorex.models.backbones.backbone import build_backbone
from patchcorex.utils.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("vit_patches")
class ViTPatchExtractor:
    def __init__(self, backbone_cfg: dict, include_cls: bool = False, normalize: str | None = None) -> None:
        self.backbone = build_backbone(backbone_cfg)
        self.include_cls = include_cls
        self.normalize = normalize
        self.last_grid_shape: Optional[Tuple[int, int]] = None
        self.enable_cam = False
        self.last_cam_tokens: Optional[torch.Tensor] = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.backbone, "extract_tokens"):
            raise RuntimeError("ViT backbone does not implement extract_tokens")
        cls_token, patch_tokens = self.backbone.extract_tokens(x)
        tokens = torch.cat([cls_token.unsqueeze(1), patch_tokens], dim=1)
        if not self.include_cls:
            tokens = patch_tokens

        patch_count = patch_tokens.shape[1]
        grid = int(math.sqrt(patch_count))
        if grid * grid == patch_count:
            self.last_grid_shape = (grid, grid)
        else:
            self.last_grid_shape = None

        patches = tokens
        if self.enable_cam:
            self.last_cam_tokens = patches
            self.last_cam_tokens.retain_grad()
        if self.normalize == "l2":
            patches = F.normalize(patches, p=2, dim=-1)
        return patches
