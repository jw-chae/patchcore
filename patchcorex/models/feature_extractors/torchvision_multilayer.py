from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F

from patchcorex.models.backbones.backbone import build_backbone
from patchcorex.utils.registry import FEATURE_EXTRACTORS


def _get_submodule(model: torch.nn.Module, name: str) -> torch.nn.Module:
    if hasattr(model, "get_submodule"):
        return model.get_submodule(name)
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Module not found: {name}")
    return modules[name]


@FEATURE_EXTRACTORS.register("torchvision_multilayer")
class TorchvisionMultiLayerExtractor:
    def __init__(
        self,
        backbone_cfg: dict,
        layers: List[str],
        align: str = "bilinear",
        normalize: str | None = None,
        **_: object,
    ) -> None:
        self.layers = layers
        self.align = align
        self.normalize = normalize
        self.last_grid_shape = None
        self.enable_cam = False
        self.last_cam_feature = None
        self.backbone = build_backbone(backbone_cfg)
        self.model = self.backbone.model
        self._features: Dict[str, torch.Tensor] = {}
        for name in layers:
            module = _get_submodule(self.model, name)
            module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def _hook(_, __, output):
            self._features[name] = output
        return _hook

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.model(x)
        feats = [self._features[name] for name in self.layers]
        max_h = max(f.shape[-2] for f in feats)
        max_w = max(f.shape[-1] for f in feats)
        aligned = []
        for f in feats:
            if f.shape[-2:] != (max_h, max_w):
                f = F.interpolate(f, size=(max_h, max_w), mode=self.align, align_corners=False)
            aligned.append(f)
        concat = torch.cat(aligned, dim=1)
        b, c, h, w = concat.shape
        self.last_grid_shape = (h, w)
        if self.enable_cam and aligned:
            self.last_cam_feature = aligned[-1]
            self.last_cam_feature.retain_grad()
        patches = concat.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if self.normalize == "l2":
            patches = F.normalize(patches, p=2, dim=-1)
        return patches
