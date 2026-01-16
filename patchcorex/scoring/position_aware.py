from __future__ import annotations

from typing import Tuple

import torch

from patchcorex.utils.registry import SCORERS


@SCORERS.register("knn_pos")
class PositionAwareKNNScorer:
    def __init__(self, bank: torch.Tensor, positions: torch.Tensor | None, pos_lambda: float = 1.0, k: int = 1, image_agg: str = "max", normalize_l2: bool = False, **_: object) -> None:
        if positions is None:
            raise ValueError("Position-aware scoring requires memory bank positions")
        self.normalize_l2 = bool(normalize_l2)
        self.bank = bank
        self.bank_pos = positions
        self.pos_lambda = pos_lambda
        self.k = k
        self.image_agg = image_agg
        if self.normalize_l2:
            self.bank = torch.nn.functional.normalize(self.bank, p=2, dim=1)

    def score(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if positions is None:
            raise ValueError("Position-aware scoring requires query positions")
        b, p, d = patches.shape
        queries = patches.reshape(b * p, d)
        qpos = positions.reshape(b * p, 2)

        bank = self.bank.to(device=queries.device, dtype=queries.dtype)
        bank_pos = self.bank_pos.to(device=qpos.device, dtype=qpos.dtype)
        if self.normalize_l2:
            queries = torch.nn.functional.normalize(queries, p=2, dim=1)

        feat_dist = torch.cdist(queries, bank, p=2) ** 2
        pos_dist = torch.cdist(qpos, bank_pos, p=2) ** 2
        dist = feat_dist + self.pos_lambda * pos_dist

        values, _ = torch.topk(dist, k=self.k, dim=1, largest=False)
        patch_scores = values[:, 0].reshape(b, p)

        if self.image_agg == "max":
            image_scores = patch_scores.max(dim=1).values
        elif self.image_agg == "mean":
            image_scores = patch_scores.mean(dim=1)
        else:
            raise KeyError(f"Unsupported image_agg: {self.image_agg}")

        return patch_scores, image_scores
