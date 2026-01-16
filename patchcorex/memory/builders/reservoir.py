from __future__ import annotations

import torch

from patchcorex.memory.bank import MemoryBank
from patchcorex.utils.dtype import resolve_dtype
from patchcorex.utils.registry import MEMORY_BUILDERS


@MEMORY_BUILDERS.register("reservoir")
class ReservoirMemoryBuilder:
    def __init__(self, K: int, seed: int = 0, dtype: str = "fp16") -> None:
        self.K = K
        self.seed = seed
        self.dtype = dtype

    def __call__(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> MemoryBank:
        data = patches.to(dtype=torch.float32).cpu()
        pos = None if positions is None else positions.to(dtype=torch.float32).cpu()
        n = data.shape[0]
        if self.K >= n:
            embeddings = data.to(dtype=resolve_dtype(self.dtype))
            metadata = {"dtype": self.dtype, "count": str(embeddings.shape[0]), "method": "reservoir_full"}
            pos_out = None if pos is None else pos
            return MemoryBank(embeddings=embeddings, positions=pos_out, metadata=metadata)

        gen = torch.Generator().manual_seed(self.seed)
        reservoir = data[: self.K].clone()
        pos_res = None if pos is None else pos[: self.K].clone()
        for i in range(self.K, n):
            j = int(torch.randint(0, i + 1, (1,), generator=gen).item())
            if j < self.K:
                reservoir[j] = data[i]
                if pos_res is not None:
                    pos_res[j] = pos[i]

        embeddings = reservoir.to(dtype=resolve_dtype(self.dtype))
        metadata = {"dtype": self.dtype, "count": str(embeddings.shape[0]), "method": "reservoir"}
        return MemoryBank(embeddings=embeddings, positions=pos_res, metadata=metadata)
