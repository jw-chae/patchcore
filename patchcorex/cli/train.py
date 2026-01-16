from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from tqdm import tqdm

from patchcorex.data.loaders import build_loader
from patchcorex.engine import build_feature_extractor, build_memory_builder
from patchcorex.memory.bank import MemoryBank
from patchcorex.memory.dual_bank import DualMemoryBank
from patchcorex.utils.io import load_yaml, save_json, save_yaml
from patchcorex.utils.coords import make_patch_positions
from patchcorex.utils.seed import set_seed

# Ensure registries are populated
from patchcorex.data.datasets import mvtec  # noqa: F401
from patchcorex.models.feature_extractors import (  # noqa: F401
    wrn_multilayer,
    vit_patches,
    wrn_dual,
    vit_dual,
    vit_scr,
    wrn_scr,
    hybrid_dual,
    dual_backbone,
    torchvision_multilayer,
    torchvision_dual,
    torchvision_scr,
)
from patchcorex.models.backbones import openclip, dinov2, dinov3, convnext, swinv2  # noqa: F401
from patchcorex.memory.builders import approx_greedy_coreset, full, kcenter, random, reservoir, rsw_e, rrsw_e  # noqa: F401
from patchcorex.scoring import patchcore_reweight  # noqa: F401


def build_run_dir(cfg: Dict[str, Any]) -> Path:
    dataset = cfg["dataset"]["name"]
    category = cfg["dataset"]["category"]
    exp_name = cfg["experiment"]["name"]
    seed = cfg["experiment"].get("seed", 0)
    return Path("runs") / dataset / category / exp_name / f"seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("experiment", {}).get("seed"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = build_feature_extractor(cfg["features"], backbone_cfg=cfg.get("backbone"))
    if hasattr(extractor, "backbones"):
        for backbone in extractor.backbones:
            backbone.to(device)
    elif hasattr(extractor, "backbone"):
        extractor.backbone.to(device)
    loader = build_loader(cfg["dataset"], split="train", shuffle=False)

    dual_mode = getattr(extractor, "is_dual", False)
    all_patches = []
    per_image_patches = []
    all_positions = []
    scr_features = []
    for images, _, _, _ in tqdm(loader, desc="Extract train patches"):
        images = images.to(device)
        with torch.no_grad():
            outputs = extractor(images)
        if dual_mode:
            patches = outputs["seg"]
            scr_features.append(outputs["scr"].cpu())
        else:
            patches = outputs
        all_patches.append(patches.reshape(-1, patches.shape[-1]).cpu())
        per_image_patches.append(patches.cpu())
        if extractor.last_grid_shape is not None:
            grid = extractor.last_grid_shape
            coords = make_patch_positions(grid, device=patches.device, dtype=patches.dtype)
            coords = coords.repeat(patches.shape[0], 1)
            all_positions.append(coords.cpu())

    all_patches = torch.cat(all_patches, dim=0)
    positions = torch.cat(all_positions, dim=0) if all_positions else None
    per_image_patches = torch.cat(per_image_patches, dim=0)
    if dual_mode:
        scr_features = torch.cat(scr_features, dim=0)

    if dual_mode:
        memory_cfg = cfg["memory"]
        if "seg" not in memory_cfg or "scr" not in memory_cfg:
            raise KeyError("Dual extractor expects memory.seg and memory.scr configs")
        seg_builder = build_memory_builder(memory_cfg["seg"])
        scr_builder = build_memory_builder(memory_cfg["scr"])
        if getattr(seg_builder, "expects_per_image", False):
            seg_bank = seg_builder(per_image_patches, positions=None)
        else:
            seg_bank = seg_builder(all_patches, positions)
        scr_bank = scr_builder(scr_features, positions=None)
        bank = DualMemoryBank(seg=seg_bank, scr=scr_bank)
    else:
        builder = build_memory_builder(cfg["memory"])
        if getattr(builder, "expects_per_image", False):
            bank = builder(per_image_patches, positions=None)
        else:
            bank = builder(all_patches, positions)

    run_dir = build_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "config.yaml", cfg)
    if dual_mode:
        bank.save(run_dir / "dual_memory_bank.pt")
        seg_bytes = bank.seg.embeddings.element_size() * bank.seg.embeddings.nelement()
        scr_bytes = bank.scr.embeddings.element_size() * bank.scr.embeddings.nelement()
        bank_stats = {
            "seg": {
                "count": int(bank.seg.embeddings.shape[0]),
                "dim": int(bank.seg.embeddings.shape[1]),
                "dtype": str(bank.seg.embeddings.dtype),
                "size_gib": float(seg_bytes) / (1024**3),
            },
            "scr": {
                "count": int(bank.scr.embeddings.shape[0]),
                "dim": int(bank.scr.embeddings.shape[1]),
                "dtype": str(bank.scr.embeddings.dtype),
                "size_gib": float(scr_bytes) / (1024**3),
            },
        }
    else:
        bank.save(run_dir / "memory_bank.pt")
        bank_bytes = bank.embeddings.element_size() * bank.embeddings.nelement()
        bank_stats = {
            "count": int(bank.embeddings.shape[0]),
            "dim": int(bank.embeddings.shape[1]),
            "dtype": str(bank.embeddings.dtype),
            "size_gib": float(bank_bytes) / (1024**3),
        }
    save_json(run_dir / "bank_stats.json", bank_stats)


if __name__ == "__main__":
    main()
