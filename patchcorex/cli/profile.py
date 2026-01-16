from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch

from patchcorex.data.loaders import build_loader
from patchcorex.engine import build_feature_extractor, build_inference_backend, build_scorer
from patchcorex.memory.bank import MemoryBank
from patchcorex.memory.dual_bank import DualMemoryBank
from patchcorex.utils.gpu_mem import get_peak_memory_gib, reset_peak_memory
from patchcorex.utils.io import load_yaml, save_json
from patchcorex.utils.coords import make_patch_positions
from patchcorex.utils.timers import timer
from patchcorex.utils.seed import set_seed

# Ensure registries are populated
from patchcorex.data.datasets import mvtec  # noqa: F401
from patchcorex.models.feature_extractors import (  # noqa: F401
    wrn_multilayer,
    vit_patches,
    vit_patchcore,
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
from patchcorex.inference import torch_knn, faiss_gpu, faiss_cpu  # noqa: F401
from patchcorex.scoring import knn, mahalanobis, position_aware, manifold_1d, rsw_e, rrsw_e, patchcore_reweight  # noqa: F401
from patchcorex.scoring.dual import DualScorer


def build_run_dir(cfg: Dict[str, Any]) -> Path:
    dataset = cfg["dataset"]["name"]
    category = cfg["dataset"]["category"]
    exp_name = cfg["experiment"]["name"]
    seed = cfg["experiment"].get("seed", 0)
    return Path("runs") / dataset / category / exp_name / f"seed{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("experiment", {}).get("seed"))

    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(cfg)
    dual_bank_path = run_dir / "dual_memory_bank.pt"
    if dual_bank_path.exists():
        bank = DualMemoryBank.load(dual_bank_path)
        dual_mode = True
    else:
        bank = MemoryBank.load(run_dir / "memory_bank.pt")
        dual_mode = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = build_feature_extractor(cfg["features"], backbone_cfg=cfg.get("backbone"))
    if hasattr(extractor, "backbones"):
        for backbone in extractor.backbones:
            backbone.to(device)
    elif hasattr(extractor, "backbone"):
        extractor.backbone.to(device)
    loader = build_loader(cfg["dataset"], split="test", shuffle=False)

    scoring_cfg = cfg.get("scoring", {})

    if dual_mode:
        seg_backend = build_inference_backend(cfg["inference"]["seg"], bank.seg.embeddings)
        scr_backend = build_inference_backend(cfg["inference"]["scr"], bank.scr.embeddings)
        seg_scorer = build_scorer(cfg["scoring"]["seg"], backend=seg_backend, bank=bank.seg.embeddings, positions=bank.seg.positions, stats=bank.seg.stats)
        scr_scorer = build_scorer(cfg["scoring"]["scr"], backend=scr_backend, bank=bank.scr.embeddings, positions=bank.scr.positions, stats=bank.scr.stats)
        use_seg_image = bool(scoring_cfg.get("use_seg_image", False))
        scorer = DualScorer(seg_scorer=seg_scorer, scr_scorer=scr_scorer, use_seg_image=use_seg_image)
    else:
        backend = build_inference_backend(cfg["inference"], bank.embeddings)
        scorer = build_scorer(cfg["scoring"], backend=backend, bank=bank.embeddings, positions=bank.positions, stats=bank.stats)

    timings: Dict[str, float] = {}
    reset_peak_memory()

    images, _, _, _ = next(iter(loader))
    images = images.to(device)

    with timer(timings, "feature_extract_s"):
        with torch.no_grad():
            outputs = extractor(images)
            if dual_mode:
                patches = outputs["seg"]
                scr_features = outputs["scr"]
            else:
                patches = outputs
                scr_features = None

    with timer(timings, "knn_s"):
        with torch.no_grad():
            positions = None
            if extractor.last_grid_shape is not None:
                grid = extractor.last_grid_shape
                coords = make_patch_positions(grid, device=patches.device, dtype=patches.dtype)
                positions = coords.repeat(patches.shape[0], 1).reshape(patches.shape[0], -1, 2)
            if dual_mode:
                _ = scorer.score(patches, positions=positions, scr_features=scr_features)
            else:
                _ = scorer.score(patches, positions=positions)

    vram = get_peak_memory_gib()

    profile = {
        "timings": timings,
        **vram,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "profile.json", profile)


if __name__ == "__main__":
    main()
