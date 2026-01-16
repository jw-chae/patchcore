from __future__ import annotations

from typing import Dict

import torch

from patchcorex.engine import build_feature_extractor
from patchcorex.utils.registry import FEATURE_EXTRACTORS


@FEATURE_EXTRACTORS.register("dual_backbone")
class DualBackboneExtractor:
    is_dual = True

    def __init__(self, seg: dict, scr: dict) -> None:
        if "type" not in seg or "type" not in scr:
            raise ValueError("dual_backbone requires seg.type and scr.type")
        self.seg_extractor = build_feature_extractor(seg, backbone_cfg=seg.get("backbone_cfg"))
        self.scr_extractor = build_feature_extractor(scr, backbone_cfg=scr.get("backbone_cfg"))
        self.last_grid_shape = None
        self.enable_cam = False
        self.last_cam_feature = None
        self.last_cam_tokens = None

    @property
    def backbones(self):
        backbones = []
        for extractor in (self.seg_extractor, self.scr_extractor):
            if hasattr(extractor, "backbones"):
                backbones.extend(extractor.backbones)
            elif hasattr(extractor, "backbone"):
                backbones.append(extractor.backbone)
        return backbones

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if hasattr(self.seg_extractor, "enable_cam"):
            self.seg_extractor.enable_cam = False
        if hasattr(self.scr_extractor, "enable_cam"):
            self.scr_extractor.enable_cam = bool(self.enable_cam)

        seg_out = self.seg_extractor(x)
        scr_out = self.scr_extractor(x)

        if isinstance(seg_out, dict):
            seg_patches = seg_out.get("seg")
        else:
            seg_patches = seg_out

        if isinstance(scr_out, dict):
            scr_vec = scr_out.get("scr")
        else:
            raise RuntimeError("scr extractor must return dict with key 'scr'")

        self.last_grid_shape = getattr(self.seg_extractor, "last_grid_shape", None)
        self.last_cam_feature = getattr(self.scr_extractor, "last_cam_feature", None)
        self.last_cam_tokens = getattr(self.scr_extractor, "last_cam_tokens", None)

        return {"seg": seg_patches, "scr": scr_vec}
