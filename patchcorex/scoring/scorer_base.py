from __future__ import annotations

from typing import Protocol, Tuple

import torch


class ScorerBase(Protocol):
    def score(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (patch_scores, image_scores)."""
        ...
