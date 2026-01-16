from __future__ import annotations

import argparse
import copy
import csv
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

from patchcorex.utils.io import load_yaml, save_yaml


def _set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def expand_sweep(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    sweep = cfg.get("sweep", {})
    params = sweep.get("params", {})
    if not params:
        return [cfg]
    keys = list(params.keys())
    values = [params[k] for k in keys]
    combos = []
    for vals in product(*values):
        new_cfg = copy.deepcopy(cfg)
        for k, v in zip(keys, vals):
            _set_by_path(new_cfg, k, v)
        combos.append(new_cfg)
    return combos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="runs/sweeps")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    configs = expand_sweep(cfg)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, cfg_i in enumerate(configs):
        path = out_dir / f"config_{i:04d}.yaml"
        save_yaml(path, cfg_i)
        rows.append({"index": i, "path": str(path)})

    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "path"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()