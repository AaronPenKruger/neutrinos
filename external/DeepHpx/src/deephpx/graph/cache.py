"""Tiny caching helpers for graph objects.

Graph construction (especially estimating lmax) can be expensive for large
NSIDE. DeepHpx keeps caching optional and filesystem-based.

This module provides:
- default cache directory resolution
- save/load helpers for neighbour arrays and SciPy sparse matrices

We intentionally do *not* enforce a specific caching strategy here; the higher
level pipeline can decide what to cache.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import scipy.sparse as sp
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scipy is required for sparse caching. Install with `pip install deephpx[graph]`."
    ) from e


def default_cache_dir() -> Path:
    """Return the default cache directory.

    Uses the environment variable ``DEEHPX_CACHE_DIR`` if set, else
    ``~/.cache/deephpx``.
    """
    env = os.getenv("DEEHPX_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".cache" / "deephpx").resolve()


def ensure_cache_dir(cache_dir: Optional[os.PathLike] = None) -> Path:
    """Create and return a cache directory."""
    p = default_cache_dir() if cache_dir is None else Path(cache_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_neighbors(path: os.PathLike, neighbors: np.ndarray) -> None:
    """Save neighbours to a .npy file."""
    np.save(Path(path), np.asarray(neighbors, dtype=np.int64))


def load_neighbors(path: os.PathLike) -> np.ndarray:
    """Load neighbours from a .npy file."""
    arr = np.load(Path(path))
    return np.asarray(arr, dtype=np.int64)


def save_sparse_npz(path: os.PathLike, mat: "sp.spmatrix") -> None:
    """Save a SciPy sparse matrix as .npz."""
    sp.save_npz(Path(path), mat)


def load_sparse_npz(path: os.PathLike) -> "sp.spmatrix":
    """Load a SciPy sparse matrix saved via :func:`scipy.sparse.save_npz`."""
    return sp.load_npz(Path(path))
