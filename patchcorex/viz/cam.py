from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def cam_from_cnn(feature: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    # feature/grad: (1, C, H, W)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feature).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    return cam


def cam_from_tokens(tokens: torch.Tensor, grad: torch.Tensor, grid_shape: Tuple[int, int]) -> torch.Tensor:
    # tokens/grad: (1, P, D)
    weights = grad.mean(dim=1, keepdim=True)
    cam = (weights * tokens).sum(dim=2).unsqueeze(1)
    cam = F.relu(cam)
    h, w = grid_shape
    cam = cam.reshape(1, 1, h, w)
    return cam


def normalize_cam(cam: torch.Tensor) -> torch.Tensor:
    cam = cam - cam.min()
    denom = cam.max().clamp_min(1e-8)
    return cam / denom