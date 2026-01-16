from typing import Tuple

import torchvision.transforms as T


def build_transforms(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_mask_transforms(img_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ]
    )