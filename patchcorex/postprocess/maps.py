from __future__ import annotations

from typing import Tuple

import math
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


def patches_to_map(patch_scores: torch.Tensor, grid_shape: Tuple[int, int], img_size: int) -> torch.Tensor:
    b, p = patch_scores.shape
    h, w = grid_shape
    if h * w != p:
        raise ValueError("grid shape does not match patch count")
    maps = patch_scores.reshape(b, 1, h, w)
    maps = F.interpolate(maps, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return maps


def blur_map(anomaly_map: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return anomaly_map
    k = int(max(3, 2 * math.ceil(2 * sigma) + 1))
    return gaussian_blur(anomaly_map, [k, k], sigma=[sigma, sigma])


def topk_pool_map(anomaly_map: torch.Tensor, kernel: int, topk: int) -> torch.Tensor:
    if kernel <= 1 or topk <= 0:
        return anomaly_map
    if kernel % 2 == 0:
        raise ValueError("topk_pool kernel must be odd")
    window = kernel * kernel
    if topk > window:
        raise ValueError("topk_pool topk cannot exceed kernel^2")
    b, c, h, w = anomaly_map.shape
    padded = F.pad(anomaly_map, (kernel // 2, kernel // 2, kernel // 2, kernel // 2), mode="reflect")
    cols = F.unfold(padded, kernel_size=kernel)
    cols = cols.reshape(b, c, window, h * w)
    topk_vals = cols.topk(topk, dim=2).values
    pooled = topk_vals.mean(dim=2)
    return pooled.reshape(b, c, h, w)
