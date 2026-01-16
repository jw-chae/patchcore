from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from patchcorex.models.backbones.backbone import build_backbone
from patchcorex.models.feature_extractors.torchvision_multilayer import _get_submodule
from patchcorex.utils.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("torchvision_scr")
class TorchvisionScrExtractor:
    def __init__(
        self,
        backbone_cfg: dict,
        scr_layer: str,
        scr_pool: str = "avg",
        scr_normalize: str | None = None,
        **_: object,
    ) -> None:
        self.scr_layer = scr_layer
        self.scr_pool = scr_pool
        self.scr_normalize = scr_normalize
        self.enable_cam = False
        self.last_cam_feature = None
        self.backbone = build_backbone(backbone_cfg)
        self.model = self.backbone.model
        self._features: Dict[str, torch.Tensor] = {}
        module = _get_submodule(self.model, scr_layer)
        module.register_forward_hook(self._make_hook(scr_layer))

    def _make_hook(self, name: str):
        def _hook(_, __, output):
            self._features[name] = output
        return _hook

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)
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
