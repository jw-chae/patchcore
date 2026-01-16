from typing import Dict, Any

from torch.utils.data import DataLoader

from patchcorex.utils.registry import DATASETS


def build_dataset(cfg: Dict[str, Any], split: str):
    dataset_cls = DATASETS.get(cfg["name"])
    return dataset_cls(
        root=cfg["root"],
        category=cfg["category"],
        split=split,
        img_size=cfg.get("img_size", 224),
        resize=cfg.get("resize"),
    )


def build_loader(cfg: Dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    dataset = build_dataset(cfg, split)
    return DataLoader(dataset, batch_size=cfg.get("batch_size", 8), shuffle=shuffle, num_workers=cfg.get("num_workers", 2))
