from __future__ import annotations

import types
from typing import List, Optional, Tuple

from patchcorex.utils.registry import BACKBONES


OPENCLIP_VIT_MODELS: List[str] = [
    "ViT-t-16",
    "ViT-S-32",
    "ViT-S-16",
    "ViT-B-32",
    "ViT-B-32-plus-256",
    "ViT-B-16",
    "ViT-B-16-plus-240",
    "ViT-B-16-SigLIP",
    "ViT-B-16-SigLIP-256",
    "ViT-B-16-SigLIP-384",
    "ViT-B-16-SigLIP-i18n-256",
    "ViT-M-32",
    "ViT-M-16",
    "ViT-L-16",
    "ViT-L-14",
    "ViT-L-14-336",
    "ViT-L-16-SigLIP-256",
    "ViT-L-16-SigLIP-384",
    "ViT-H-14",
    "ViT-H-14-378",
    "ViT-H-14-CLIPA",
    "ViT-g-14",
    "ViT-bigG-14",
    "ViT-SO400M-14-SigLIP",
    "ViT-SO400M-14-SigLIP-378",
    "ViT-SO400M-14-SigLIP-384",
    "ViT-p-16",
    "ViT-n-16",
]


def _resolve_pretrained(model_name: str, tag: str) -> str:
    if tag != "auto":
        return tag
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("open_clip_torch is required for openclip_vit backbone") from exc
    available = open_clip.list_pretrained_tags_by_model(model_name)
    if not available:
        raise RuntimeError(f"No pretrained weights found for model: {model_name}")
    return available[0]


@BACKBONES.register("openclip_vit")
class OpenCLIPViTBackbone:
    def __init__(self, model_name: str, pretrained: str = "auto", token_layer: Optional[int] = None) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("open_clip_torch is required for openclip_vit backbone") from exc
        self.token_layer = token_layer
        self.token_layer_index = None
        if token_layer is not None:
            # Convert to zero-based index while keeping None when unset
            self.token_layer_index = max(int(token_layer) - 1, 0)
        tag = _resolve_pretrained(model_name, pretrained)
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=tag)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._tokens: Optional["torch.Tensor"] = None
        self._has_cls_token = True

        visual = getattr(self.model, "visual", None)
        if visual is None:
            raise RuntimeError("OpenCLIP visual module not found; cannot extract patch tokens")

        if hasattr(visual, "transformer"):
            transformer = visual.transformer
            module = transformer
            blocks = getattr(transformer, "resblocks", None)
            if self.token_layer_index is not None and blocks:
                idx = min(self.token_layer_index, len(blocks) - 1)
                module = blocks[idx]
            module.register_forward_hook(self._capture_tokens)
        elif hasattr(visual, "trunk") and hasattr(visual.trunk, "forward_features"):
            trunk = visual.trunk
            blocks = getattr(trunk, "blocks", None)
            if self.token_layer_index is not None and blocks:
                idx = min(self.token_layer_index, len(blocks) - 1)
                blocks[idx].register_forward_hook(self._capture_tokens)
            else:
                self._wrap_timm_forward(trunk)
            cls_param = getattr(visual.trunk, "cls_token", None)
            self._has_cls_token = cls_param is not None
        else:
            raise RuntimeError("OpenCLIP vision backbone is not a ViT; cannot extract patch tokens")

    def _capture_tokens(self, _module, _inputs, output) -> None:
        self._set_tokens_from_output(output)

    def _wrap_timm_forward(self, trunk) -> None:
        original_forward = trunk.forward_features

        def forward_features_with_capture(module_self, *args, **kwargs):
            tokens = original_forward(*args, **kwargs)
            self._set_tokens_from_output(tokens)
            return tokens

        trunk.forward_features = types.MethodType(forward_features_with_capture, trunk)

    def _set_tokens_from_output(self, output) -> None:
        import torch

        if isinstance(output, torch.Tensor):
            self._tokens = output
        elif isinstance(output, (tuple, list)) and output:
            first = output[0]
            if isinstance(first, torch.Tensor):
                self._tokens = first

    def extract_tokens(self, x) -> Tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        self._tokens = None
        _ = self.model.visual(x)
        if self._tokens is None:
            raise RuntimeError("Failed to capture ViT tokens from OpenCLIP backbone")
        tokens = self._tokens
        if tokens.dim() != 3:
            raise RuntimeError("Unexpected token shape from OpenCLIP backbone")
        if self._has_cls_token:
            cls_token = tokens[:, 0, :]
            patch_tokens = tokens[:, 1:, :]
        else:
            cls_token = tokens.mean(dim=1)
            patch_tokens = tokens
        if not isinstance(cls_token, torch.Tensor) or not isinstance(patch_tokens, torch.Tensor):
            raise RuntimeError("Token extraction failed for OpenCLIP backbone")
        return cls_token, patch_tokens

    def to(self, device):
        self.model.to(device)
        return self
