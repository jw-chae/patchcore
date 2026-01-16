from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from patchcorex.data.transforms import build_mask_transforms, build_transforms
from patchcorex.utils.registry import DATASETS


@dataclass
class MVTecSample:
    image_path: Path
    label: int
    mask_path: Optional[Path]


@DATASETS.register("mvtec")
class MVTecDataset(Dataset):
    def __init__(self, root: str, category: str, split: str, img_size: int) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.img_size = img_size
        self.transform = build_transforms(img_size)
        self.mask_transform = build_mask_transforms(img_size)
        self.samples = self._discover()

    def _discover(self) -> List[MVTecSample]:
        base = self.root / self.category / self.split
        if self.split == "train":
            good_dir = base / "good"
            return [MVTecSample(p, 0, None) for p in sorted(good_dir.glob("*.png"))]

        samples: List[MVTecSample] = []
        for defect_dir in sorted(base.iterdir()):
            if not defect_dir.is_dir():
                continue
            label = 0 if defect_dir.name == "good" else 1
            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path = None
                if label == 1:
                    mask_name = img_path.stem + "_mask.png"
                    mask_path = self.root / self.category / "ground_truth" / defect_dir.name / mask_name
                samples.append(MVTecSample(img_path, label, mask_path))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        image = self.transform(image)
        if sample.mask_path is not None and sample.mask_path.exists():
            mask = Image.open(sample.mask_path).convert("L")
            mask = self.mask_transform(mask)
        else:
            mask = torch.zeros(1, self.img_size, self.img_size)
        return image, sample.label, mask, str(sample.image_path)
