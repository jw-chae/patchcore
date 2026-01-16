from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from patchcorex.models.backbones.backbone import build_backbone
from patchcorex.utils.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("vit_scr")
class ViTScrExtractor:
    def __init__(
        self,
        backbone_cfg: dict,
        scr_normalize: str | None = None,
        scr_source: str = "cls",
    ) -> None:
        self.backbone = build_backbone(backbone_cfg)
        self.scr_normalize = scr_normalize
        self.scr_source = scr_source

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not hasattr(self.backbone, "extract_tokens"):
            raise RuntimeError("ViT backbone does not implement extract_tokens")
        cls_token, patch_tokens = self.backbone.extract_tokens(x)

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

        return {"scr": scr_vec}
