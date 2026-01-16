from __future__ import annotations

import torch

from patchcorex.memory.bank import MemoryBank
from patchcorex.utils.dtype import resolve_dtype
from patchcorex.utils.registry import MEMORY_BUILDERS


@MEMORY_BUILDERS.register("random")
class RandomMemoryBuilder:
    def __init__(self, K: int, seed: int = 0, dtype: str = "fp16") -> None:
        self.K = K
        self.seed = seed
        self.dtype = dtype

    def __call__(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> MemoryBank:
        device = patches.device
        data = patches.to(dtype=torch.float32)
        pos = None if positions is None else positions.to(dtype=torch.float32)
        n = data.shape[0]
        if self.K >= n:
            embeddings = data.to(dtype=resolve_dtype(self.dtype)).cpu()
            metadata = {"dtype": self.dtype, "count": str(embeddings.shape[0]), "method": "random_full"}
            pos_out = None if pos is None else pos.cpu()
            return MemoryBank(embeddings=embeddings, positions=pos_out, metadata=metadata)
        gen = torch.Generator(device=device).manual_seed(self.seed)
        idx = torch.randperm(n, generator=gen)[: self.K]
        embeddings = data[idx].to(dtype=resolve_dtype(self.dtype)).cpu()
        metadata = {"dtype": self.dtype, "count": str(embeddings.shape[0]), "method": "random"}
        pos_out = None if pos is None else pos[idx].cpu()
        return MemoryBank(embeddings=embeddings, positions=pos_out, metadata=metadata)
