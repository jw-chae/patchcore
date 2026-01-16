from __future__ import annotations

from typing import Tuple

import torch

from patchcorex.utils.registry import SCORERS


@SCORERS.register("mahalanobis")
class MahalanobisScorer:
    def __init__(
        self,
        bank: torch.Tensor,
        shrinkage: float = 1e-5,
        cov_mode: str = "full",
        diag_eps: float = 1e-5,
        lw_eps: float = 1e-5,
        whiten: bool = False,
        image_agg: str = "max",
        **_: object,
    ) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bank = bank.to(dtype=torch.float32, device=device)
        self.mean = bank.mean(dim=0)
        centered = bank - self.mean
        n = max(centered.shape[0], 1)
        p = centered.shape[1]
        if cov_mode == "diag":
            var = centered.var(dim=0, unbiased=False)
            cov = torch.diag(var + float(diag_eps))
        else:
            if cov_mode == "lw":
                cov = centered.t().matmul(centered) / float(n)
                mu = torch.trace(cov) / float(p)
                frob = torch.sum(cov * cov)
                sum_sq = torch.sum(centered * centered, dim=1)
                beta = (torch.sum(sum_sq * sum_sq) / float(n) - frob) / float(n)
                delta = torch.sum((cov - torch.eye(p, device=cov.device, dtype=cov.dtype) * mu) ** 2)
                shrink = float(torch.clamp(beta / (delta + 1e-12), 0.0, 1.0))
                cov = (1.0 - shrink) * cov + shrink * mu * torch.eye(p, device=cov.device, dtype=cov.dtype)
                cov = cov + torch.eye(p, device=cov.device, dtype=cov.dtype) * float(lw_eps)
            elif cov_mode == "full":
                cov = centered.t().matmul(centered) / max(n - 1, 1)
                cov = cov + torch.eye(p, device=cov.device, dtype=cov.dtype) * float(shrinkage)
            else:
                raise KeyError(f"Unsupported cov_mode: {cov_mode}")

        self.cov = cov
        self.inv_cov = torch.linalg.inv(cov)
        self.whiten = bool(whiten)
        self.image_agg = image_agg

    def score(self, patches: torch.Tensor, positions: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = patches.device
        mean = self.mean.to(device=device, dtype=patches.dtype)
        inv_cov = self.inv_cov.to(device=device, dtype=patches.dtype)

        if patches.dim() == 2:
            delta = patches - mean
            if self.whiten:
                chol = torch.linalg.cholesky(self.cov.to(device=device, dtype=patches.dtype))
                z = torch.linalg.solve_triangular(chol, delta.t(), upper=False).t()
                image_scores = (z * z).sum(dim=-1)
            else:
                proj = torch.matmul(delta, inv_cov)
                image_scores = (proj * delta).sum(dim=-1)
            patch_scores = image_scores.unsqueeze(1)
            if self.image_agg not in ("none", "max", "mean"):
                raise KeyError(f"Unsupported image_agg: {self.image_agg}")
            return patch_scores, image_scores

        delta = patches - mean
        if self.whiten:
            chol = torch.linalg.cholesky(self.cov.to(device=device, dtype=patches.dtype))
            z = torch.linalg.solve_triangular(chol, delta.reshape(-1, delta.shape[-1]).t(), upper=False).t()
            patch_scores = (z * z).sum(dim=-1).reshape(delta.shape[0], delta.shape[1])
        else:
            proj = torch.matmul(delta, inv_cov)
            patch_scores = (proj * delta).sum(dim=-1)
        if self.image_agg == "max":
            image_scores = patch_scores.max(dim=1).values
        elif self.image_agg == "mean":
            image_scores = patch_scores.mean(dim=1)
        elif self.image_agg == "none":
            raise ValueError("image_agg=none requires 2D image features, got patch tokens.")
        else:
            raise KeyError(f"Unsupported image_agg: {self.image_agg}")
        return patch_scores, image_scores
