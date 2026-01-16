from __future__ import annotations

import torch

from patchcorex.utils.registry import INFERENCE_BACKENDS


@INFERENCE_BACKENDS.register("faiss_gpu")
class FaissGPUBackend:
    def __init__(self, bank: torch.Tensor, normalize_l2: bool = False, **_: object) -> None:
        try:
            import faiss
            import faiss.contrib.torch_utils  # Enables direct GPU tensor support
        except ImportError as exc:
            raise ImportError("faiss[gpu] and faiss.contrib.torch_utils are required for faiss_gpu backend") from exc
        
        self.faiss = faiss
        self.normalize_l2 = bool(normalize_l2)
        
        # Determine GPU device (default to 0)
        self.device_id = 0
        self.device = torch.device(f"cuda:{self.device_id}")
        
        # FAISS GPU initialization
        bank_cpu = bank.to(dtype=torch.float32, device="cpu")
        if self.normalize_l2:
            bank_cpu = torch.nn.functional.normalize(bank_cpu, p=2, dim=1)
            
        self.index = faiss.IndexFlatL2(bank_cpu.shape[1])
        if faiss.get_num_gpus() == 0:
            raise RuntimeError("faiss_gpu requested but no GPU available")
            
        res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(res, self.device_id, self.index)
        self.index.add(bank_cpu.numpy())

    def query(self, queries: torch.Tensor, k: int) -> torch.Tensor:
        if self.normalize_l2:
            queries = torch.nn.functional.normalize(queries, p=2, dim=1)
        
        # Ensure queries are on the correct GPU device
        if queries.device.type != "cuda":
            queries = queries.to(self.device)
            
        # faiss.contrib.torch_utils allows passing GPU tensors directly
        distances, _ = self.index.search(queries, k)
        
        # Return as tensor (stays on GPU if queries was on GPU)
        return distances
