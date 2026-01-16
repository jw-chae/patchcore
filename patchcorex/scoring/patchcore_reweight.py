from __future__ import annotations

from typing import Tuple

import torch
from torch.nn import functional as F  # noqa: N812

from patchcorex.utils.registry import SCORERS


@SCORERS.register("patchcore_reweight")
class PatchcoreReweightScorer:
    """PatchCore reweighting scorer aligned with anomalib's implementation."""

    def __init__(
        self,
        bank: torch.Tensor,
        num_neighbors: int = 9,
        **_: object,
    ) -> None:
        self.bank = bank
        self.num_neighbors = int(num_neighbors)

    @staticmethod
    def _euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        return res.clamp_min_(0).sqrt_()

    def _nearest_neighbors(self, embedding: torch.Tensor, bank: torch.Tensor, n_neighbors: int) -> Tuple[torch.Tensor, torch.Tensor]:
        distances = self._euclidean_dist(embedding, bank)
        if n_neighbors == 1:
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def score(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.bank.device != patches.device or self.bank.dtype != patches.dtype:
            self.bank = self.bank.to(device=patches.device, dtype=patches.dtype)
        bank = self.bank
        if patches.dim() == 2:
            # (B, D) image-level features; treat as single patch per image.
            patch_scores, locations = self._nearest_neighbors(patches, bank=bank, n_neighbors=1)
            patch_scores = patch_scores.unsqueeze(1)
            locations = locations.unsqueeze(1)
            image_scores = patch_scores.squeeze(1)
            return patch_scores, image_scores

        b, p, d = patches.shape
        queries = patches.reshape(b * p, d)
        patch_scores, locations = self._nearest_neighbors(queries, bank=bank, n_neighbors=1)
        patch_scores = patch_scores.reshape(b, p)
        locations = locations.reshape(b, p)

        if self.num_neighbors == 1:
            image_scores = patch_scores.amax(1)
            return patch_scores, image_scores

        # 1) max patch per image
        max_patches = torch.argmax(patch_scores, dim=1)
        max_patch_features = patches[torch.arange(b), max_patches]
        score = patch_scores[torch.arange(b), max_patches]
        nn_index = locations[torch.arange(b), max_patches]

        # 2) support samples of nn in memory bank
        nn_sample = bank[nn_index]
        bank_size = bank.shape[0]
        _, support_samples = self._nearest_neighbors(
            nn_sample,
            bank=bank,
            n_neighbors=min(self.num_neighbors, bank_size),
        )

        # 3) distance to support samples
        distances = self._euclidean_dist(max_patch_features.unsqueeze(1), bank[support_samples])
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        image_scores = weights * score
        return patch_scores, image_scores
