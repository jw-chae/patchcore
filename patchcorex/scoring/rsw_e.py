from __future__ import annotations

from typing import Tuple

import torch

from patchcorex.inference.knn_base import KNNBackendBase
from patchcorex.utils.registry import SCORERS


@SCORERS.register("rsw_e")
class RSWEScorer:
    def __init__(
        self,
        backend: KNNBackendBase,
        k: int = 1,
        image_agg: str = "min",
        image_topk_ratio: float = 0.01,
        **_: object,
    ) -> None:
        self.backend = backend
        self.k = k
        self.image_agg = image_agg
        self.image_topk_ratio = float(image_topk_ratio)

    def score(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if patches.dim() == 2:
            b, d = patches.shape
            queries = patches
            distances = self.backend.query(queries, k=self.k)
            if distances.ndim == 1:
                distances = distances.unsqueeze(1)
            image_scores = distances[:, 0]
            patch_scores = image_scores.unsqueeze(1)
            if self.image_agg not in ("none", "min", "max", "mean"):
                raise KeyError(f"Unsupported image_agg: {self.image_agg}")
            return patch_scores, image_scores

        b, p, d = patches.shape
        queries = patches.reshape(b * p, d)
        distances = self.backend.query(queries, k=self.k)
        if distances.ndim == 1:
            distances = distances.unsqueeze(1)
        patch_scores = distances[:, 0].reshape(b, p)
        if self.image_agg == "max":
            image_scores = patch_scores.max(dim=1).values
        elif self.image_agg == "min":
            image_scores = patch_scores.min(dim=1).values
        elif self.image_agg == "mean":
            image_scores = patch_scores.mean(dim=1)
        elif self.image_agg == "topk_mean":
            if not (0 < self.image_topk_ratio <= 1):
                raise ValueError("image_topk_ratio must be in (0, 1]")
            k = max(1, int(round(p * self.image_topk_ratio)))
            topk_vals = torch.topk(patch_scores, k, dim=1).values
            image_scores = topk_vals.mean(dim=1)
        elif self.image_agg == "none":
            raise ValueError("image_agg=none requires 2D image features, got patch tokens.")
        else:
            raise KeyError(f"Unsupported image_agg: {self.image_agg}")
        return patch_scores, image_scores
