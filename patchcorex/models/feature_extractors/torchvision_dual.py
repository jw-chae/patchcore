from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from patchcorex.models.backbones.backbone import build_backbone
from patchcorex.utils.registry import FEATURE_EXTRACTORS
from patchcorex.models.feature_extractors.torchvision_multilayer import _get_submodule


@FEATURE_EXTRACTORS.register("torchvision_dual")
class TorchvisionDualExtractor:
    is_dual = True

    def __init__(
        self,
        backbone_cfg: dict,
        seg_layers: List[str],
        scr_layer: str,
        align: str = "bilinear",
        seg_normalize: str | None = None,
        scr_normalize: str | None = None,
        scr_pool: str = "avg",
        **_: object,
    ) -> None:
        self.seg_layers = seg_layers
        self.scr_layer = scr_layer
        self.align = align
        self.seg_normalize = seg_normalize
        self.scr_normalize = scr_normalize
        self.scr_pool = scr_pool
        self.last_grid_shape = None
        self.enable_cam = False
        self.last_cam_feature = None
        self.backbone = build_backbone(backbone_cfg)
        self.model = self.backbone.model
        self._features: Dict[str, torch.Tensor] = {}
        for name in seg_layers + [scr_layer]:
            module = _get_submodule(self.model, name)
            module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def _hook(_, __, output):
            self._features[name] = output
        return _hook

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _ = self.model(x)

        seg_feats = [self._features[name] for name in self.seg_layers]
        scr_feat = self._features[self.scr_layer]

        max_h = max(f.shape[-2] for f in seg_feats)
        max_w = max(f.shape[-1] for f in seg_feats)
        aligned = []
        for f in seg_feats:
            if f.shape[-2:] != (max_h, max_w):
                f = F.interpolate(f, size=(max_h, max_w), mode=self.align, align_corners=False)
            aligned.append(f)
        concat = torch.cat(aligned, dim=1)
        b, c, h, w = concat.shape
        self.last_grid_shape = (h, w)
        patches = concat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if self.seg_normalize == "l2":
            patches = F.normalize(patches, p=2, dim=-1)

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

        return {"seg": patches, "scr": scr_vec}
