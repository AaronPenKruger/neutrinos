"""Neighbour queries for HEALPix pixelizations.

We intentionally keep this file tiny: its job is only to produce a neighbour
list from a HEALPix tesselation, without building any heavy graph objects.

For Milestone 2, we use healpy's canonical routine:
    healpy.pixelfunc.get_all_neighbours

This returns the 8 nearest neighbours for each pixel index (SW, W, NW, N, NE,
E, SE, S). Missing neighbours are returned as -1.

The output format in DeepHpx is always:
    neighbours.shape == (npix, 8)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..healpix.geometry import nside2npix


def _require_healpy():
    try:
        import healpy as hp  # type: ignore
    except Exception as e:
        raise ImportError(
            "healpy is required for HEALPix neighbour queries. "
            "Install with `pip install deephpx[healpix]`."
        ) from e
    return hp


def neighbors_8(
    nside: int,
    *,
    nest: bool = True,
    pixels: Optional[np.ndarray] = None,
    dtype: np.dtype = np.int64,
) -> np.ndarray:
    """Return the 8-neighbour list for HEALPix pixels.

    Args:
        nside: HEALPix NSIDE.
        nest: if True, interpret pixel indices as NESTED ordering; else RING.
        pixels: optional 1D array of pixel indices. If None, compute neighbours
            for all pixels (0..npix-1).
        dtype: dtype for returned neighbour indices.

    Returns:
        neighbours: array of shape (N, 8) where N = len(pixels) or npix.
            Neighbour indices are in the same ordering as requested via `nest`.
            Missing neighbours are -1.
    """
    if nside <= 0:
        raise ValueError(f"nside must be positive, got {nside}")

    hp = _require_healpy()

    if pixels is None:
        npix = nside2npix(nside)
        pixels = np.arange(npix, dtype=np.int64)
    else:
        pixels = np.asarray(pixels, dtype=np.int64).ravel()

    # healpy.pixelfunc.get_all_neighbours(nside, theta, phi=None, nest=False, ...)
    # If phi is None, `theta` is interpreted as pixel number(s).
    neigh = hp.pixelfunc.get_all_neighbours(nside, pixels, nest=nest)  # type: ignore[attr-defined]

    neigh = np.asarray(neigh)

    # healpy convention: (8,) for scalar, (8, N) for vector input.
    if neigh.ndim == 1:
        neigh = neigh.reshape(1, 8)
    elif neigh.shape[0] == 8:
        neigh = neigh.T

    if neigh.ndim != 2 or neigh.shape[1] != 8:
        raise RuntimeError(f"Unexpected neighbour array shape from healpy: {neigh.shape}")

    return neigh.astype(dtype, copy=False)
