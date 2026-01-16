from __future__ import annotations

from typing import Tuple

import torch

from patchcorex.utils.registry import SCORERS


@SCORERS.register("manifold_1d")
class Manifold1DScorer:
    def __init__(self, bank: torch.Tensor, neighbor_rank: int = 1, image_agg: str = "max", normalize_l2: bool = False, **_: object) -> None:
        bank = bank.to(dtype=torch.float32, device="cpu")
        self.normalize_l2 = bool(normalize_l2)
        if self.normalize_l2:
            bank = torch.nn.functional.normalize(bank, p=2, dim=1)
        self.bank = bank
        self.image_agg = image_agg
        self.neighbor_rank = max(1, int(neighbor_rank))

        distances = torch.cdist(bank, bank, p=2)
        values, indices = torch.topk(distances, k=self.neighbor_rank + 1, dim=1, largest=False)
        self.neighbors = indices[:, self.neighbor_rank]

    def score(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, p, d = patches.shape
        queries = patches.reshape(b * p, d)

        bank = self.bank.to(device=queries.device, dtype=queries.dtype)
        if self.normalize_l2:
            queries = torch.nn.functional.normalize(queries, p=2, dim=1)
        distances = torch.cdist(queries, bank, p=2)
        nn_idx = torch.argmin(distances, dim=1)

        a = bank[nn_idx]
        b_idx = self.neighbors[nn_idx].to(device=queries.device)
        bpt = bank[b_idx]

        v = bpt - a
        denom = (v * v).sum(dim=1, keepdim=True).clamp_min(1e-12)
        t = ((queries - a) * v).sum(dim=1, keepdim=True) / denom
        t = t.clamp(0.0, 1.0)
        proj = a + t * v
        patch_scores = ((queries - proj) ** 2).sum(dim=1).reshape(b, p)

        if self.image_agg == "max":
            image_scores = patch_scores.max(dim=1).values
        elif self.image_agg == "mean":
            image_scores = patch_scores.mean(dim=1)
        else:
            raise KeyError(f"Unsupported image_agg: {self.image_agg}")

        return patch_scores, image_scores
