from __future__ import annotations

from typing import List

import torch

from patchcorex.memory.bank import MemoryBank
from patchcorex.utils.dtype import resolve_dtype
from patchcorex.utils.registry import MEMORY_BUILDERS


@MEMORY_BUILDERS.register("kcenter")
class KCenterMemoryBuilder:
    def __init__(
        self,
        K: int | None = None,
        k: int | None = None,
        percentage: float | None = None,
        seed: int = 0,
        max_samples: int | None = None,
        dtype: str = "fp16",
    ) -> None:
        self.K = K if K is not None else k
        self.percentage = percentage
        self.seed = seed
        self.max_samples = max_samples
        self.dtype = dtype

    def __call__(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> MemoryBank:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use float32 for distance calculations to maintain precision/stability
        data = patches.to(dtype=torch.float32, device=device)
        pos = None if positions is None else positions.to(dtype=torch.float32, device=device)
        n = data.shape[0]

        if self.max_samples is not None and n > self.max_samples:
            gen = torch.Generator(device=device).manual_seed(self.seed)
            idx = torch.randperm(n, generator=gen)[: self.max_samples]
            data = data[idx]
            if pos is not None:
                pos = pos[idx]
            n = data.shape[0]

        # Determine K dynamically if percentage is given
        if self.percentage is not None:
            self.K = int(n * self.percentage)

        if self.K is None:
             raise ValueError("Neither 'K' (or 'k') nor 'percentage' was provided to KCenterMemoryBuilder.")

        if self.K >= n:
            embeddings = data.to(dtype=resolve_dtype(self.dtype)).cpu()
            pos_out = None if pos is None else pos.to(dtype=torch.float32).cpu()
            metadata = {
                "dtype": self.dtype,
                "count": str(embeddings.shape[0]),
                "method": "kcenter_full",
            }
            return MemoryBank(embeddings=embeddings, positions=pos_out, metadata=metadata)

        cpu_gen = torch.Generator(device="cpu").manual_seed(self.seed)
        centers: List[int] = []
        first = int(torch.randint(0, n, (1,), generator=cpu_gen).item())
        centers.append(first)

        # Pre-allocate min_dists on device
        min_dists = torch.cdist(data, data[first : first + 1]).squeeze(1)
        
        # Greedy K-Center Sampling
        for _ in range(1, self.K):
            next_idx = int(torch.argmax(min_dists).item())
            centers.append(next_idx)
            # Distance from all points to the newest center
            dist_new = torch.cdist(data, data[next_idx : next_idx + 1]).squeeze(1)
            min_dists = torch.minimum(min_dists, dist_new)

        embeddings = data[centers].to(dtype=resolve_dtype(self.dtype)).cpu()
        pos_out = None if pos is None else pos[centers].to(dtype=torch.float32).cpu()
        metadata = {
            "dtype": self.dtype,
            "count": str(embeddings.shape[0]),
            "method": "kcenter",
        }
        return MemoryBank(embeddings=embeddings, positions=pos_out, metadata=metadata)
