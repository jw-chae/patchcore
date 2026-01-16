from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from patchcorex.memory.bank import MemoryBank


@dataclass
class DualMemoryBank:
    seg: MemoryBank
    scr: MemoryBank

    @staticmethod
    def _dump_bank(bank: MemoryBank) -> Dict[str, object]:
        return {
            "embeddings": bank.embeddings,
            "positions": bank.positions,
            "stats": bank.stats,
            "metadata": bank.metadata,
        }

    @staticmethod
    def _load_bank(data: Dict[str, object]) -> MemoryBank:
        return MemoryBank(
            embeddings=data["embeddings"],
            positions=data.get("positions"),
            stats=data.get("stats"),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "seg": self._dump_bank(self.seg),
                "scr": self._dump_bank(self.scr),
            },
            path,
        )

    @staticmethod
    def load(path: str | Path) -> "DualMemoryBank":
        data = torch.load(path, map_location="cpu")
        return DualMemoryBank(
            seg=DualMemoryBank._load_bank(data["seg"]),
            scr=DualMemoryBank._load_bank(data["scr"]),
        )
