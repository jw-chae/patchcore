from typing import Dict

import torch


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_gib() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"vram_peak_gib": 0.0}
    peak_bytes = torch.cuda.max_memory_allocated()
    return {"vram_peak_gib": float(peak_bytes) / (1024**3)}