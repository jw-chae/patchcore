from __future__ import annotations

from typing import Protocol

import torch


class FeatureExtractorBase(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch features as (P, D) tensor."""
        ...