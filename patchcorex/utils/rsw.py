from __future__ import annotations

from typing import Tuple

import torch


def make_orthogonal_directions(dim: int, num_dirs: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if num_dirs <= dim:
        a = torch.randn(dim, num_dirs, generator=gen, dtype=torch.float32)
        q, _ = torch.linalg.qr(a, mode="reduced")
        return q.t()
    u = torch.randn(num_dirs, dim, generator=gen, dtype=torch.float32)
    return torch.nn.functional.normalize(u, p=2, dim=1)


def rsw_embedding(patches: torch.Tensor, num_dirs: int, num_quantiles: int, seed: int) -> torch.Tensor:
    # patches: (B, P, D)
    b, p, d = patches.shape
    u = make_orthogonal_directions(d, num_dirs, seed).to(device=patches.device, dtype=patches.dtype)
    proj = torch.einsum("bpd,kd->bpk", patches, u)
    proj_sorted, _ = torch.sort(proj, dim=1)
    q = min(num_quantiles, p)
    idx = torch.linspace(0, p - 1, steps=q, device=patches.device).long()
    sampled = proj_sorted[:, idx, :]
    return sampled.reshape(b, -1)
