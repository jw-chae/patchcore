from __future__ import annotations

from typing import Protocol

import torch


class KNNBackendBase(Protocol):
    def query(self, queries: torch.Tensor, k: int) -> torch.Tensor:
        """Return kNN distances for each query as (N, k) tensor."""
        ...