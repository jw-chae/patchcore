from __future__ import annotations

from typing import Optional, Tuple

import torch

from patchcorex.utils.registry import BACKBONES


def _extract_tokens_from_features(features: object) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(features, dict):
        if "x_norm_clstoken" in features:
            cls_token = features["x_norm_clstoken"]
        elif "x_norm_cls_token" in features:
            cls_token = features["x_norm_cls_token"]
        elif "cls_token" in features:
            cls_token = features["cls_token"]
        else:
            cls_token = None
        if "x_norm_patchtokens" in features:
            patch_tokens = features["x_norm_patchtokens"]
        elif "x_norm_patch_tokens" in features:
            patch_tokens = features["x_norm_patch_tokens"]
        elif "patch_tokens" in features:
            patch_tokens = features["patch_tokens"]
        else:
            patch_tokens = None
        if isinstance(cls_token, torch.Tensor) and isinstance(patch_tokens, torch.Tensor):
            return cls_token, patch_tokens
    raise RuntimeError("Unsupported DINOv2 forward_features output")


@BACKBONES.register("dinov2_vit")
class DINOv2Backbone:
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        pretrained: bool = True,
        token_layer: Optional[int] = None,
    ) -> None:
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.token_layer_index = None
        self._tokens: Optional[torch.Tensor] = None
        if token_layer is not None:
            self.token_layer_index = max(int(token_layer) - 1, 0)
            blocks = getattr(self.model, "blocks", None)
            if blocks:
                idx = min(self.token_layer_index, len(blocks) - 1)
                blocks[idx].register_forward_hook(self._capture_tokens)

    def _capture_tokens(self, _module, _inputs, output) -> None:
        if isinstance(output, torch.Tensor):
            self._tokens = output

    def extract_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.token_layer_index is not None:
            self._tokens = None
            with torch.no_grad():
                _ = self.model(x)
            if self._tokens is None:
                raise RuntimeError("Failed to capture tokens from requested DINOv2 layer")
            tokens = self._tokens
            cls_token = tokens[:, 0, :]
            patch_tokens = tokens[:, 1:, :]
            return cls_token, patch_tokens

        features: Optional[object] = None
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(x)
        elif hasattr(self.model, "get_intermediate_layers"):
            layers = self.model.get_intermediate_layers(x, n=1, return_class_token=True)
            if layers:
                patch_tokens, cls_token = layers[0]
                return cls_token, patch_tokens
        if features is None:
            raise RuntimeError("DINOv2 backbone does not expose forward_features or get_intermediate_layers")
        return _extract_tokens_from_features(features)

    def to(self, device):
        self.model.to(device)
        return self
