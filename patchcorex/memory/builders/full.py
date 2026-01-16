from __future__ import annotations

from typing import Dict

import torch

from patchcorex.memory.bank import MemoryBank
from patchcorex.utils.dtype import resolve_dtype
from patchcorex.utils.registry import MEMORY_BUILDERS


@MEMORY_BUILDERS.register("full")
class FullMemoryBuilder:
    def __init__(self, dtype: str = "fp16") -> None:
        self.dtype = dtype

    def __call__(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> MemoryBank:
        embeddings = patches.to(dtype=resolve_dtype(self.dtype)).contiguous().cpu()
        pos = None if positions is None else positions.contiguous().cpu()
        metadata = {"dtype": self.dtype, "count": str(embeddings.shape[0])}
        return MemoryBank(embeddings=embeddings, positions=pos, metadata=metadata)
