from __future__ import annotations

import torch

from patchcorex.utils.registry import INFERENCE_BACKENDS


@INFERENCE_BACKENDS.register("faiss_cpu")
class FaissCPUBackend:
    def __init__(self, bank: torch.Tensor, normalize_l2: bool = False, **_: object) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required for faiss_cpu backend") from exc
        self.faiss = faiss
        self.normalize_l2 = bool(normalize_l2)
        bank = bank.to(dtype=torch.float32, device="cpu")
        if self.normalize_l2:
            bank = torch.nn.functional.normalize(bank, p=2, dim=1)
        self.index = faiss.IndexFlatL2(bank.shape[1])
        self.index.add(bank.numpy())

    def query(self, queries: torch.Tensor, k: int) -> torch.Tensor:
        if self.normalize_l2:
            queries = torch.nn.functional.normalize(queries, p=2, dim=1)
        distances, _ = self.index.search(queries.cpu().numpy(), k)
        return torch.from_numpy(distances)
