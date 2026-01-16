from __future__ import annotations

from typing import Optional, Tuple

import torch

from patchcorex.utils.registry import BACKBONES


def _default_weights_url(model_name: str) -> str | None:
    mapping = {
        "dinov3_vitb16": "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    }
    return mapping.get(model_name)


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
    raise RuntimeError("Unsupported DINOv3 forward_features output")


@BACKBONES.register("dinov3_vit")
class DINOv3Backbone:
    def __init__(
        self,
        model_name: str,
        repo: str = "facebookresearch/dinov3",
        pretrained: bool = True,
        weights_url: str | None = None,
        weights_path: str | None = None,
    ) -> None:
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
            self.model = torch.hub.load(repo, model_name, pretrained=False)
            self.model.load_state_dict(state_dict)
        else:
            if weights_url is None:
                weights_url = _default_weights_url(model_name)
            try:
                self.model = torch.hub.load(repo, model_name, pretrained=pretrained)
            except Exception as exc:
                if not pretrained or weights_url is None:
                    raise
                self.model = torch.hub.load(repo, model_name, pretrained=False)
                state_dict = torch.hub.load_state_dict_from_url(weights_url, map_location="cpu")
                self.model.load_state_dict(state_dict)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def extract_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features: Optional[object] = None
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(x)
        elif hasattr(self.model, "get_intermediate_layers"):
            layers = self.model.get_intermediate_layers(x, n=1, return_class_token=True)
            if layers:
                patch_tokens, cls_token = layers[0]
                return cls_token, patch_tokens
        if features is None:
            raise RuntimeError("DINOv3 backbone does not expose forward_features or get_intermediate_layers")
        return _extract_tokens_from_features(features)

    def to(self, device):
        self.model.to(device)
        return self
