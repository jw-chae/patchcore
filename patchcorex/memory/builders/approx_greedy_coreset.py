from __future__ import annotations

from typing import List

import numpy as np
import torch

from patchcorex.memory.bank import MemoryBank
from patchcorex.utils.dtype import resolve_dtype
from patchcorex.utils.registry import MEMORY_BUILDERS


def _batchwise_l2(matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
    a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
    b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
    a_times_b = matrix_a.mm(matrix_b.T)
    return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()


@MEMORY_BUILDERS.register("approx_greedy_coreset")
class ApproxGreedyCoresetBuilder:
    def __init__(
        self,
        percentage: float,
        seed: int = 0,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
        device: str | None = None,
        dtype: str = "fp32",
    ) -> None:
        if not 0 < percentage <= 1:
            raise ValueError("percentage must be in (0, 1].")
        self.percentage = percentage
        self.seed = seed
        self.number_of_starting_points = number_of_starting_points
        self.dimension_to_project_features_to = dimension_to_project_features_to
        self.device = device
        self.dtype = dtype

    def __call__(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> MemoryBank:
        preferred = self.device
        if preferred is None:
            preferred = "cuda" if torch.cuda.is_available() else "cpu"

        def _run_on(device: torch.device) -> MemoryBank:
            data = patches.to(dtype=torch.float32, device=device)
            pos = None if positions is None else positions.to(dtype=torch.float32, device=device)
            n = data.shape[0]

            if self.percentage >= 1:
                embeddings = data.to(dtype=resolve_dtype(self.dtype)).cpu()
                pos_out = None if pos is None else pos.to(dtype=torch.float32).cpu()
                metadata = {"dtype": self.dtype, "count": str(embeddings.shape[0]), "method": "approx_greedy_full"}
                return MemoryBank(embeddings=embeddings, positions=pos_out, metadata=metadata)

            reduced = data
            if data.shape[1] != self.dimension_to_project_features_to:
                mapper = torch.nn.Linear(data.shape[1], self.dimension_to_project_features_to, bias=False).to(device)
                reduced = mapper(data)

            num_start = int(np.clip(self.number_of_starting_points, 1, n))
            rng = np.random.default_rng(self.seed)
            start_points = rng.choice(n, num_start, replace=False).tolist()

            approx_dist = _batchwise_l2(reduced, reduced[start_points]).mean(dim=-1).reshape(-1, 1)
            num_samples = int(n * self.percentage)
            centers: List[int] = []

            for _ in range(num_samples):
                select_idx = int(torch.argmax(approx_dist).item())
                centers.append(select_idx)
                select_dist = _batchwise_l2(reduced, reduced[select_idx : select_idx + 1])
                approx_dist = torch.cat([approx_dist, select_dist], dim=-1)
                approx_dist = torch.min(approx_dist, dim=1).values.reshape(-1, 1)

            embeddings = data[centers].to(dtype=resolve_dtype(self.dtype)).cpu()
            pos_out = None if pos is None else pos[centers].to(dtype=torch.float32).cpu()
            metadata = {"dtype": self.dtype, "count": str(embeddings.shape[0]), "method": "approx_greedy_coreset"}
            return MemoryBank(embeddings=embeddings, positions=pos_out, metadata=metadata)

        device = torch.device(preferred)
        try:
            return _run_on(device)
        except torch.OutOfMemoryError:
            if device.type == "cuda":
                cpu_device = torch.device("cpu")
                return _run_on(cpu_device)
            raise
