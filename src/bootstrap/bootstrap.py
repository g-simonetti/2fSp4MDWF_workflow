#!/usr/bin/env python3

import hashlib
from pathlib import Path

import numpy as np


def _normalise_path(path):
    p = Path(path).expanduser().resolve()
    if p.name in {"mesons", "log"}:
        p = p.parent
    elif p.is_file():
        p = p.parent
    return str(p)


def seed_from_path(path):
    """
    Deterministically convert an input path into a NumPy RNG seed.
    """
    key = _normalise_path(path)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def bootstrap_from_path(path, cfg_numbers, n_boot):
    """
    Build a deterministic bootstrap index matrix from a path-derived seed.

    Parameters
    ----------
    path:
        Directory or file path identifying the ensemble.
    cfg_numbers:
        Ordered list of selected configuration numbers. The bootstrap indices
        refer to positions in this list.
    n_boot:
        Number of bootstrap replicas.
    """
    cfg_numbers = [int(n) for n in cfg_numbers]
    n_boot = int(n_boot)
    if n_boot <= 0:
        raise ValueError("n_boot must be > 0")
    if len(cfg_numbers) == 0:
        raise ValueError("cfg_numbers must be non-empty")

    seed = seed_from_path(path)
    rng = np.random.default_rng(seed)
    n_cfg = len(cfg_numbers)
    boot_idx = rng.integers(0, n_cfg, size=(n_boot, n_cfg), dtype=np.int64)

    return {
        "path_key": _normalise_path(path),
        "seed": int(seed),
        "n_boot": n_boot,
        "n_cfg": n_cfg,
        "cfg_numbers": cfg_numbers,
        "boot_idx": boot_idx,
    }


def bootstrap_to_jsonable(bootstrap):
    return {
        "path_key": bootstrap["path_key"],
        "seed": int(bootstrap["seed"]),
        "n_boot": int(bootstrap["n_boot"]),
        "n_cfg": int(bootstrap["n_cfg"]),
        "cfg_numbers": [int(n) for n in bootstrap["cfg_numbers"]],
        "boot_idx": np.asarray(bootstrap["boot_idx"], dtype=int).tolist(),
    }


__all__ = [
    "bootstrap_from_path",
    "bootstrap_to_jsonable",
    "seed_from_path",
]
