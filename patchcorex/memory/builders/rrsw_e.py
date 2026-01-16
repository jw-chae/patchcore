from __future__ import annotations

import torch

from patchcorex.memory.bank import MemoryBank
from patchcorex.utils.dtype import resolve_dtype
from patchcorex.utils.registry import MEMORY_BUILDERS
from patchcorex.utils.rsw import rsw_embedding


@MEMORY_BUILDERS.register("rrsw_e")
class RRSWEMemoryBuilder:
    expects_per_image = True

    def __init__(self, num_dirs: int = 16, num_quantiles: int = 20, seed: int = 0, shrinkage: float = 1e-5, dtype: str = "fp16") -> None:
        self.num_dirs = num_dirs
        self.num_quantiles = num_quantiles
        self.seed = seed
        self.shrinkage = shrinkage
        self.dtype = dtype

    def __call__(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> MemoryBank:
        # patches: (B, P, D)
        patches = patches.to(dtype=torch.float32, device=patches.device)
        embed = rsw_embedding(patches, self.num_dirs, self.num_quantiles, self.seed)
        embed_cpu = embed.cpu()

        mean = embed_cpu.mean(dim=0)
        centered = embed_cpu - mean
        cov = centered.t().matmul(centered) / max(embed_cpu.shape[0] - 1, 1)
        cov = cov + torch.eye(cov.shape[0]) * float(self.shrinkage)
        inv_cov = torch.linalg.pinv(cov, rcond=1e-4)
        if not torch.isfinite(inv_cov).all():
            diag = torch.diag(torch.diag(cov).clamp_min(1e-6))
            inv_cov = torch.linalg.pinv(diag, rcond=1e-4)

        embeddings = embed_cpu.to(dtype=resolve_dtype(self.dtype))
        stats = {"mean": mean, "inv_cov": inv_cov}
        metadata = {
            "dtype": self.dtype,
            "count": str(embeddings.shape[0]),
            "num_dirs": str(self.num_dirs),
            "num_quantiles": str(self.num_quantiles),
            "seed": str(self.seed),
            "method": "rrsw_e",
        }
        return MemoryBank(embeddings=embeddings, positions=None, stats=stats, metadata=metadata)
