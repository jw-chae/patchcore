from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from patchcorex.models.backbones.wrn50_2 import build_wrn50_2
from patchcorex.utils.registry import FEATURE_EXTRACTORS


class PatchMaker:
    def __init__(self, patchsize: int, stride: int | None = None) -> None:
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features: torch.Tensor, return_spatial_info: bool = False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded = unfolded.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded = unfolded.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded, number_of_total_patches
        return unfolded


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim: int) -> None:
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        self.preprocessing_modules = torch.nn.ModuleList(
            [MeanMapper(output_dim) for _ in input_dims]
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        mapped = []
        for module, feature in zip(self.preprocessing_modules, features):
            mapped.append(module(feature))
        return torch.stack(mapped, dim=1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim: int) -> None:
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


@FEATURE_EXTRACTORS.register("wrn_multilayer")
class WRNMultiLayerExtractor:
    def __init__(
        self,
        layers: List[str],
        align: str = "bilinear",
        normalize: str | None = None,
        pretrained: bool = True,
        use_avgpool: bool = True,
        pool_kernel: int = 3,
        pool_stride: int = 1,
        pool_padding: int = 1,
        patchsize: int | None = None,
        patchstride: int = 1,
        pretrain_embed_dimension: int | None = None,
        target_embed_dimension: int | None = None,
        **_: object,
    ) -> None:
        self.layers = layers
        self.align = align
        self.normalize = normalize
        self.last_grid_shape = None
        self.enable_cam = False
        self.last_cam_feature = None
        self.use_avgpool = use_avgpool
        self.pooler = torch.nn.AvgPool2d(pool_kernel, pool_stride, pool_padding) if use_avgpool else None
        if (pretrain_embed_dimension is None) != (target_embed_dimension is None):
            raise ValueError("Both pretrain_embed_dimension and target_embed_dimension must be set together.")
        self.patchcore_mode = pretrain_embed_dimension is not None and patchsize is not None
        self.patch_maker = PatchMaker(patchsize, stride=patchstride) if self.patchcore_mode else None
        self.preprocessing = (
            Preprocessing([0 for _ in layers], pretrain_embed_dimension)
            if self.patchcore_mode
            else None
        )
        self.aggregator = Aggregator(target_embed_dimension) if self.patchcore_mode else None
        self.backbone = build_wrn50_2(pretrained=pretrained)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self._features: Dict[str, torch.Tensor] = {}
        for name in layers:
            module = getattr(self.backbone, name)
            module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def _hook(_, __, output):
            self._features[name] = output
        return _hook

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.backbone(x)
        feats = [self._features[name] for name in self.layers]
        if self.pooler is not None:
            feats = [self.pooler(f) for f in feats]
        if self.patchcore_mode:
            patched = [self.patch_maker.patchify(f, return_spatial_info=True) for f in feats]
            patch_shapes = [p[1] for p in patched]
            feats = [p[0] for p in patched]
            ref_num_patches = patch_shapes[0]
            for i in range(1, len(feats)):
                _features = feats[i]
                patch_dims = patch_shapes[i]
                _features = _features.reshape(
                    _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
                )
                _features = _features.permute(0, -3, -2, -1, 1, 2)
                perm_base_shape = _features.shape
                _features = _features.reshape(-1, *_features.shape[-2:])
                _features = F.interpolate(
                    _features.unsqueeze(1),
                    size=(ref_num_patches[0], ref_num_patches[1]),
                    mode=self.align,
                    align_corners=False,
                )
                _features = _features.squeeze(1)
                _features = _features.reshape(
                    *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
                )
                _features = _features.permute(0, -2, -1, 1, 2, 3)
                _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
                feats[i] = _features
            feats = [f.reshape(-1, *f.shape[-3:]) for f in feats]
            features = self.preprocessing(feats)
            features = self.aggregator(features)
            b = x.shape[0]
            h, w = ref_num_patches
            self.last_grid_shape = (h, w)
            patches = features.reshape(b, h * w, -1)
            if self.normalize == "l2":
                patches = F.normalize(patches, p=2, dim=-1)
            return patches
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
