import torch


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise KeyError(f"Unsupported dtype: {name}")
    return DTYPE_MAP[name]