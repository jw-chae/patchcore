import torchvision.transforms as T


def build_transforms(img_size: int, resize: int | None = None) -> T.Compose:
    resize_size = resize if resize is not None else img_size
    return T.Compose(
        [
            T.Resize(resize_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_mask_transforms(img_size: int, resize: int | None = None) -> T.Compose:
    resize_size = resize if resize is not None else img_size
    return T.Compose(
        [
            T.Resize(resize_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
        ]
    )
