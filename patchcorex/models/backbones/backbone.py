from __future__ import annotations

from typing import Any, Dict

from patchcorex.utils.registry import BACKBONES


def build_backbone(cfg: Dict[str, Any]):
    if cfg is None:
        raise ValueError("backbone config is required")
    cls = BACKBONES.get(cfg["type"])
    args = dict(cfg)
    args.pop("type")
    return cls(**args)