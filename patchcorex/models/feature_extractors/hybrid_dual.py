from __future__ import annotations

from typing import Dict, List

import torch

from patchcorex.models.feature_extractors.wrn_multilayer import WRNMultiLayerExtractor
from patchcorex.models.feature_extractors.vit_dual import ViTDualExtractor
from patchcorex.utils.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("hybrid_dual")
class HybridDualExtractor:
    is_dual = True

    def __init__(
        self,
        seg_layers: List[str],
        seg_align: str = "bilinear",
        seg_normalize: str | None = None,
        seg_pretrained: bool = True,
        scr_backbone_cfg: dict | None = None,
        scr_normalize: str | None = None,
        scr_source: str = "cls",
        **_: object,
    ) -> None:
        if scr_backbone_cfg is None:
            raise ValueError("hybrid_dual requires scr_backbone_cfg")
        self.seg_extractor = WRNMultiLayerExtractor(
            layers=seg_layers,
            align=seg_align,
            normalize=seg_normalize,
            pretrained=seg_pretrained,
        )
        self.scr_extractor = ViTDualExtractor(
            backbone_cfg=scr_backbone_cfg,
            seg_normalize=None,
            scr_normalize=scr_normalize,
            scr_source=scr_source,
        )
        self.last_grid_shape = None
        self.enable_cam = False
        self.last_cam_feature = None
        self.last_cam_tokens = None

    @property
    def backbones(self):
        return [self.seg_extractor.backbone, self.scr_extractor.backbone]

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.seg_extractor.enable_cam = False
        self.scr_extractor.enable_cam = bool(self.enable_cam)

        seg_patches = self.seg_extractor(x)
        scr_outputs = self.scr_extractor(x)

        self.last_grid_shape = self.seg_extractor.last_grid_shape
        self.last_cam_feature = getattr(self.scr_extractor, "last_cam_feature", None)
        self.last_cam_tokens = getattr(self.scr_extractor, "last_cam_tokens", None)

        return {"seg": seg_patches, "scr": scr_outputs["scr"]}
