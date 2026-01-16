from __future__ import annotations

from typing import Optional

from torchvision import models

from patchcorex.utils.registry import BACKBONES


def _resolve_swinv2_weights(weights: Optional[str]):
    if weights is None or str(weights).lower() in ("none", "false"):
        return None
    if str(weights).lower() == "default":
        return models.Swin_V2_B_Weights.DEFAULT
    if hasattr(models.Swin_V2_B_Weights, str(weights)):
        return getattr(models.Swin_V2_B_Weights, str(weights))
    raise ValueError(f"Unsupported SwinV2 weights: {weights}")


@BACKBONES.register("swinv2_base")
class SwinV2BaseBackbone:
    def __init__(self, weights: Optional[str] = "default") -> None:
        resolved = _resolve_swinv2_weights(weights)
        self.model = models.swin_v2_b(weights=resolved)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def to(self, device):
        self.model.to(device)
        return self
