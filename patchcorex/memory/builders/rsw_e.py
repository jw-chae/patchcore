from __future__ import annotations

import torch

from patchcorex.memory.bank import MemoryBank
from patchcorex.utils.dtype import resolve_dtype
from patchcorex.utils.registry import MEMORY_BUILDERS
from patchcorex.utils.rsw import rsw_embedding


@MEMORY_BUILDERS.register("rsw_e")
class RSWEMemoryBuilder:
    expects_per_image = True

    def __init__(self, num_dirs: int = 16, num_quantiles: int = 20, seed: int = 0, dtype: str = "fp16") -> None:
        self.num_dirs = num_dirs
        self.num_quantiles = num_quantiles
        self.seed = seed
        self.dtype = dtype

    def __call__(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> MemoryBank:
        # patches: (B, P, D)
        patches = patches.to(dtype=torch.float32, device=patches.device)
        embed = rsw_embedding(patches, self.num_dirs, self.num_quantiles, self.seed)
        embeddings = embed.to(dtype=resolve_dtype(self.dtype)).cpu()
        metadata = {
            "dtype": self.dtype,
            "count": str(embeddings.shape[0]),
            "num_dirs": str(self.num_dirs),
            "num_quantiles": str(self.num_quantiles),
            "seed": str(self.seed),
            "method": "rsw_e",
        }
        return MemoryBank(embeddings=embeddings, positions=None, metadata=metadata)