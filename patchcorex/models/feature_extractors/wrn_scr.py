from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from patchcorex.models.backbones.wrn50_2 import build_wrn50_2
from patchcorex.utils.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("wrn_scr")
class WRNScrExtractor:
    def __init__(
        self,
        scr_layer: str,
        scr_pool: str = "avg",
        scr_normalize: str | None = None,
        pretrained: bool = True,
        **_: object,
    ) -> None:
        self.scr_layer = scr_layer
        self.scr_pool = scr_pool
        self.scr_normalize = scr_normalize
        self.enable_cam = False
        self.last_cam_feature = None
        self.backbone = build_wrn50_2(pretrained=pretrained)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self._features: Dict[str, torch.Tensor] = {}
        module = getattr(self.backbone, scr_layer)
        module.register_forward_hook(self._make_hook(scr_layer))

    def _make_hook(self, name: str):
        def _hook(_, __, output):
            self._features[name] = output
        return _hook

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.backbone(x)
        scr_feat = self._features[self.scr_layer]

        if self.enable_cam:
            self.last_cam_feature = scr_feat
            self.last_cam_feature.retain_grad()

        if self.scr_pool == "avg":
            scr_vec = F.adaptive_avg_pool2d(scr_feat, output_size=1).flatten(1)
        elif self.scr_pool == "max":
            scr_vec = F.adaptive_max_pool2d(scr_feat, output_size=1).flatten(1)
        else:
            raise KeyError(f"Unsupported scr_pool: {self.scr_pool}")
        if self.scr_normalize == "l2":
            scr_vec = F.normalize(scr_vec, p=2, dim=-1)
        return {"scr": scr_vec}
