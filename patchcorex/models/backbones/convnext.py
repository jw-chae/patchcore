from __future__ import annotations

from typing import Optional

from torchvision import models

from patchcorex.utils.registry import BACKBONES


def _resolve_convnext_weights(weights: Optional[str]):
    if weights is None or str(weights).lower() in ("none", "false"):
        return None
    if str(weights).lower() == "default":
        return models.ConvNeXt_Base_Weights.DEFAULT
    if hasattr(models.ConvNeXt_Base_Weights, str(weights)):
        return getattr(models.ConvNeXt_Base_Weights, str(weights))
    raise ValueError(f"Unsupported ConvNeXt weights: {weights}")


@BACKBONES.register("convnext_base")
class ConvNeXtBaseBackbone:
    def __init__(self, weights: Optional[str] = "default") -> None:
        resolved = _resolve_convnext_weights(weights)
        self.model = models.convnext_base(weights=resolved)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def to(self, device):
        self.model.to(device)
        return self
