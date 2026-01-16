from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from patchcorex.data.loaders import build_loader
from patchcorex.engine import build_feature_extractor, build_inference_backend, build_scorer
from patchcorex.eval.metrics import image_metrics, industrial_metrics, pixel_metrics
from patchcorex.memory.bank import MemoryBank
from patchcorex.memory.dual_bank import DualMemoryBank
from patchcorex.postprocess.maps import blur_map, patches_to_map, topk_pool_map
from patchcorex.postprocess.thresholds import choose_threshold
from patchcorex.utils.io import load_yaml, save_json, save_yaml
from patchcorex.utils.coords import make_patch_positions
from patchcorex.utils.seed import set_seed
from patchcorex.viz.heatmap import save_overlay
from patchcorex.viz.cam import cam_from_cnn, cam_from_tokens, normalize_cam

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


def patchcore_image_score(patch_scores: torch.Tensor) -> torch.Tensor:
    scores = patch_scores
    while scores.dim() > 1:
        scores = scores.max(dim=-1).values
    return scores


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
    parser.add_argument("--bank-dir", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("experiment", {}).get("seed"))

    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(cfg)
    bank_dir = Path(args.bank_dir) if args.bank_dir else run_dir
    dual_bank_path = bank_dir / "dual_memory_bank.pt"
    if dual_bank_path.exists():
        bank = DualMemoryBank.load(dual_bank_path)
        dual_mode = True
    else:
        bank = MemoryBank.load(bank_dir / "memory_bank.pt")
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
    image_score_mode = scoring_cfg.get("image_score", "scorer")

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

    all_scores: List[float] = []
    all_labels: List[int] = []
    pixel_maps: List[np.ndarray] = []
    pixel_masks: List[np.ndarray] = []
    top_fp: List[Tuple[float, str, torch.Tensor, torch.Tensor]] = []
    top_fn: List[Tuple[float, str, torch.Tensor, torch.Tensor]] = []
    viz_cfg = cfg.get("viz", {})
    viz_enabled = bool(viz_cfg.get("enabled", False))
    viz_topk = int(viz_cfg.get("topk", 8))
    viz_alpha = float(viz_cfg.get("alpha", 0.5))
    cam_cfg = cfg.get("cam", {})
    cam_enabled = bool(cam_cfg.get("enabled", False))
    cam_topk = int(cam_cfg.get("topk", viz_topk))
    cam_alpha = float(cam_cfg.get("alpha", viz_alpha))

    for images, labels, masks, paths in tqdm(loader, desc="Eval"):
        images = images.to(device)
        with torch.no_grad():
            outputs = extractor(images)
            if dual_mode:
                patches = outputs["seg"]
                scr_features = outputs["scr"]
            else:
                patches = outputs
                scr_features = None
            positions = None
            if extractor.last_grid_shape is not None:
                grid = extractor.last_grid_shape
                coords = make_patch_positions(grid, device=patches.device, dtype=patches.dtype)
                positions = coords.repeat(patches.shape[0], 1).reshape(patches.shape[0], -1, 2)
            if dual_mode:
                patch_scores, image_scores = scorer.score(patches, positions=positions, scr_features=scr_features)
            else:
                patch_scores, image_scores = scorer.score(patches, positions=positions)
        if image_score_mode == "patchcore":
            image_scores = patchcore_image_score(patch_scores)
        all_scores.extend(image_scores.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())

        if extractor.last_grid_shape is not None and getattr(scorer, "supports_pixel_map", True):
            # Keep on GPU for interpolation and blurring
            maps = patches_to_map(patch_scores, extractor.last_grid_shape, cfg["dataset"].get("img_size", 224))
            maps = blur_map(maps, float(cfg.get("postprocess", {}).get("blur_sigma", 0)))
            topk_cfg = cfg.get("postprocess", {}).get("topk_pool", None)
            if topk_cfg is not None:
                kernel = int(topk_cfg.get("kernel", 3))
                topk = int(topk_cfg.get("topk", 3))
                maps = topk_pool_map(maps, kernel=kernel, topk=topk)
            
            # Move to CPU for long-term storage and visualization
            maps_cpu = maps.cpu()
            for i in range(maps_cpu.shape[0]):
                if masks is not None and masks[i] is not None:
                    pixel_maps.append(maps_cpu[i].squeeze(0).numpy())
                    pixel_masks.append(masks[i].squeeze(0).numpy())
                if viz_enabled:
                    label = int(labels[i].item())
                    score = float(image_scores[i].item())
                    item = (score, paths[i], images[i].cpu(), maps[i].cpu())
                    if label == 0:
                        top_fp.append(item)
                        top_fp = sorted(top_fp, key=lambda x: x[0], reverse=True)[:viz_topk]
                    else:
                        top_fn.append(item)
                        top_fn = sorted(top_fn, key=lambda x: x[0])[:viz_topk]

    scores = np.asarray(all_scores, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int64)
    metrics = image_metrics(scores, labels)

    threshold_cfg = cfg.get("postprocess", {}).get("threshold", None)
    threshold_info = {}
    if threshold_cfg is not None:
        threshold_info = choose_threshold(scores, labels, threshold_cfg)
        metrics.update(threshold_info)
        metrics.update(industrial_metrics(scores, labels, threshold=threshold_info.get("threshold")))
    else:
        metrics.update(industrial_metrics(scores, labels))

    if pixel_maps and pixel_masks and getattr(scorer, "supports_pixel_map", True):
        maps_np = np.stack(pixel_maps, axis=0)
        masks_np = np.stack(pixel_masks, axis=0)
        fpr_limit = float(cfg.get("eval", {}).get("pixel_fpr_limit", 0.3))
        metrics.update(pixel_metrics(maps_np, masks_np, fpr_limit=fpr_limit))

    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(run_dir / "config.yaml", cfg)
    save_json(run_dir / "metrics.json", metrics)

    if viz_enabled:
        qual_dir = run_dir / "qualitative"
        for score, path, image, amap in top_fp:
            name = f"fp_score_{score:.4f}_" + Path(path).name
            save_overlay(image, amap, qual_dir / name, alpha=viz_alpha)
        for score, path, image, amap in top_fn:
            name = f"fn_score_{score:.4f}_" + Path(path).name
            save_overlay(image, amap, qual_dir / name, alpha=viz_alpha)

    if cam_enabled:
        if hasattr(extractor, "enable_cam"):
            extractor.enable_cam = True
        cam_dir = run_dir / "qualitative"
        def _cam_pass(items, prefix: str) -> None:
            for score, path, image, _ in items[:cam_topk]:
                image = image.unsqueeze(0).to(device)
                image.requires_grad_(True)
                if hasattr(extractor, "backbone"):
                    extractor.backbone.to(device)
                scorer_out = None
                with torch.enable_grad():
                    outputs = extractor(image)
                    if dual_mode:
                        patches = outputs["seg"]
                        scr_features = outputs["scr"]
                    else:
                        patches = outputs
                        scr_features = None
                    positions = None
                    if extractor.last_grid_shape is not None:
                        grid = extractor.last_grid_shape
                        coords = make_patch_positions(grid, device=patches.device, dtype=patches.dtype)
                        positions = coords.repeat(patches.shape[0], 1).reshape(patches.shape[0], -1, 2)
                    if dual_mode:
                        patch_scores, image_scores = scorer.score(patches, positions=positions, scr_features=scr_features)
                    else:
                        patch_scores, image_scores = scorer.score(patches, positions=positions)
                    scorer_out = image_scores[0]
                    scorer_out.backward()

                cam = None
                if getattr(extractor, "last_cam_feature", None) is not None and extractor.last_cam_feature.grad is not None:
                    cam = cam_from_cnn(extractor.last_cam_feature, extractor.last_cam_feature.grad)
                elif getattr(extractor, "last_cam_tokens", None) is not None and extractor.last_cam_tokens.grad is not None:
                    if extractor.last_grid_shape is None:
                        continue
                    cam = cam_from_tokens(extractor.last_cam_tokens, extractor.last_cam_tokens.grad, extractor.last_grid_shape)
                if cam is None:
                    continue
                if getattr(extractor, "last_cam_feature", None) is not None:
                    extractor.last_cam_feature.grad = None
                if getattr(extractor, "last_cam_tokens", None) is not None:
                    extractor.last_cam_tokens.grad = None
                cam = normalize_cam(cam.detach())
                cam = torch.nn.functional.interpolate(
                    cam, size=(cfg["dataset"].get("img_size", 224), cfg["dataset"].get("img_size", 224)),
                    mode="bilinear", align_corners=False
                )
                name = f"{prefix}_cam_score_{score:.4f}_" + Path(path).name
                save_overlay(image.squeeze(0).detach().cpu(), cam.squeeze(0).cpu(), cam_dir / name, alpha=cam_alpha)

        _cam_pass(sorted(top_fp, key=lambda x: x[0], reverse=True), "fp")
        _cam_pass(sorted(top_fn, key=lambda x: x[0]), "fn")
        if hasattr(extractor, "enable_cam"):
            extractor.enable_cam = False


if __name__ == "__main__":
    main()
