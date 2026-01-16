from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _unnormalize(image: torch.Tensor) -> torch.Tensor:
    return image * _IMAGENET_STD + _IMAGENET_MEAN


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.clamp(0, 1)
    return (image.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def _colormap(values: np.ndarray) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 * v, 0, 1)
    g = np.clip(1.5 * (1 - np.abs(v - 0.5) * 2), 0, 1)
    b = np.clip(1.5 * (1 - v), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def save_overlay(image: torch.Tensor, anomaly_map: torch.Tensor, path: str | Path, alpha: float = 0.5) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image = _unnormalize(image)
    image_np = _to_uint8(image)
    amap = anomaly_map.squeeze(0).cpu().numpy()
    amap = (amap - amap.min()) / max(amap.max() - amap.min(), 1e-8)
    heat = _colormap(amap)
    overlay = (image_np * (1 - alpha) + heat * alpha).astype(np.uint8)
    Image.fromarray(overlay).save(path)