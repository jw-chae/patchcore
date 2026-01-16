from __future__ import annotations

import argparse
import json

from patchcorex.utils.registry import BACKBONES, DATASETS, FEATURE_EXTRACTORS, INFERENCE_BACKENDS, MEMORY_BUILDERS, SCORERS

# Ensure registries are populated
from patchcorex.data.datasets import mvtec  # noqa: F401
from patchcorex.models.feature_extractors import (  # noqa: F401
    wrn_multilayer,
    vit_patches,
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
from patchcorex.memory.builders import full, kcenter, random, reservoir, rsw_e, rrsw_e  # noqa: F401
from patchcorex.scoring import knn, mahalanobis, position_aware, manifold_1d, rsw_e, rrsw_e, patchcore_reweight  # noqa: F401
from patchcorex.inference import torch_knn, faiss_gpu, faiss_cpu  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["json"], default="json")
    args = parser.parse_args()

    options = {
        "datasets": DATASETS.keys(),
        "backbones": BACKBONES.keys(),
        "features": FEATURE_EXTRACTORS.keys(),
        "memory": MEMORY_BUILDERS.keys(),
        "scoring": SCORERS.keys(),
        "inference": INFERENCE_BACKENDS.keys(),
        "openclip_vit_models": openclip.OPENCLIP_VIT_MODELS,
        "schemas": {
            "backbone.openclip_vit": {
                "model_name": "ViT-L-14",
                "pretrained": "auto",
            },
            "backbone.dinov2_vit": {
                "model_name": "dinov2_vitb14",
                "pretrained": True,
            },
            "backbone.dinov3_vit": {
                "model_name": "dinov3_vitb16",
                "repo": "facebookresearch/dinov3",
                "pretrained": True,
                "weights_path": None,
            },
            "backbone.convnext_base": {
                "weights": "default",
            },
            "backbone.swinv2_base": {
                "weights": "default",
            },
            "features.wrn_multilayer": {
                "layers": ["layer2", "layer3"],
                "align": "bilinear",
                "normalize": "l2",
                "pretrained": True,
            },
            "features.vit_patches": {
                "include_cls": False,
                "normalize": "l2",
                "backbone_cfg": {"type": "openclip_vit", "model_name": "ViT-L-14", "pretrained": "auto"},
            },
            "features.vit_dual": {
                "seg_normalize": "l2",
                "scr_normalize": "l2",
                "scr_source": "cls",
                "backbone_cfg": {"type": "openclip_vit", "model_name": "ViT-L-14", "pretrained": "auto"},
            },
            "features.vit_scr": {
                "scr_normalize": "l2",
                "scr_source": "cls",
                "backbone_cfg": {"type": "openclip_vit", "model_name": "ViT-L-14", "pretrained": "auto"},
            },
            "features.wrn_scr": {
                "scr_layer": "layer4",
                "scr_pool": "max",
                "scr_normalize": "l2",
                "pretrained": True,
            },
            "features.dual_backbone": {
                "seg": {
                    "type": "vit_patches",
                    "backbone_cfg": {"type": "dinov2_vit", "model_name": "dinov2_vitb14", "pretrained": True},
                    "include_cls": False,
                    "normalize": "l2",
                },
                "scr": {
                    "type": "vit_scr",
                    "backbone_cfg": {"type": "openclip_vit", "model_name": "ViT-L-14", "pretrained": "auto"},
                    "scr_source": "cls",
                    "scr_normalize": "l2",
                },
            },
            "features.torchvision_multilayer": {
                "layers": ["features.2", "features.3"],
                "align": "bilinear",
                "normalize": "l2",
                "backbone_cfg": {"type": "convnext_base", "weights": "default"},
            },
            "features.torchvision_dual": {
                "seg_layers": ["features.2", "features.3"],
                "scr_layer": "features.3",
                "align": "bilinear",
                "seg_normalize": "l2",
                "scr_normalize": "l2",
                "scr_pool": "avg",
                "backbone_cfg": {"type": "convnext_base", "weights": "default"},
            },
            "features.torchvision_scr": {
                "scr_layer": "features.3",
                "scr_pool": "max",
                "scr_normalize": "l2",
                "backbone_cfg": {"type": "convnext_base", "weights": "default"},
            },
            "memory.full": {"dtype": "fp16"},
            "memory.kcenter": {"K": 10000, "seed": 0, "max_samples": 200000, "dtype": "fp16"},
            "memory.random": {"K": 10000, "seed": 0, "dtype": "fp16"},
            "memory.reservoir": {"K": 10000, "seed": 0, "dtype": "fp16"},
            "memory.rsw_e": {"num_dirs": 16, "num_quantiles": 20, "seed": 0, "dtype": "fp16"},
            "memory.rrsw_e": {"num_dirs": 16, "num_quantiles": 20, "seed": 0, "shrinkage": 1e-5, "dtype": "fp16"},
            "scoring.knn_l2": {"k": 1, "image_agg": "max"},
            "scoring.mahalanobis": {"shrinkage": 1e-5, "image_agg": "max"},
            "scoring.knn_pos": {"pos_lambda": 1.0, "k": 1, "image_agg": "max", "normalize_l2": False},
            "scoring.manifold_1d": {"neighbor_rank": 1, "image_agg": "max", "normalize_l2": False},
            "scoring.rsw_e": {"num_dirs": 16, "num_quantiles": 20, "seed": 0, "image_agg": "min"},
            "scoring.rrsw_e": {"num_dirs": 16, "num_quantiles": 20, "seed": 0, "image_agg": "min"},
            "scoring.patchcore_reweight": {"num_neighbors": 9},
            "inference.torch_knn": {"device": "cuda", "normalize_l2": False},
            "inference.faiss_gpu": {},
            "inference.faiss_cpu": {},
            "postprocess.threshold": {"type": "fpr_at_tpr", "tpr": 0.99},
            "eval": {"pixel_fpr_limit": 0.3},
            "viz": {"enabled": True, "topk": 8, "alpha": 0.5},
            "cam": {"enabled": False, "topk": 8, "alpha": 0.5},
        },
    }

    print(json.dumps(options, indent=2))


if __name__ == "__main__":
    main()
