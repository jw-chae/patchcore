from __future__ import annotations

from typing import Tuple

import torch


class DualScorer:
    supports_pixel_map = True

    def __init__(self, seg_scorer, scr_scorer, use_seg_image: bool = False) -> None:
        self.seg_scorer = seg_scorer
        self.scr_scorer = scr_scorer
        self.use_seg_image = use_seg_image

    def score(
        self,
        patches: torch.Tensor,
        positions: torch.Tensor | None = None,
        scr_features: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patch_scores, seg_image_scores = self.seg_scorer.score(patches, positions=positions)
        if self.use_seg_image:
            image_scores = seg_image_scores
        else:
            if scr_features is None:
                raise ValueError("DualScorer requires scr_features for screening score")
            _, image_scores = self.scr_scorer.score(scr_features, positions=None)
        return patch_scores, image_scores
