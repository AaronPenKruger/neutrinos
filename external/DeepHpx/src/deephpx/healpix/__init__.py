"""HEALPix utilities.

Milestone 1 + 2 entrypoints:
- :func:`deephpx.healpix.io.read_healpix_map`
- :func:`deephpx.healpix.ordering.to_nested` / :func:`deephpx.healpix.ordering.to_ring`
- :func:`deephpx.healpix.geometry.npix2nside` / :func:`deephpx.healpix.geometry.nside2npix`

The optional extra `deephpx[healpix]` installs `healpy`.
"""

from __future__ import annotations

from .io import read_healpix_map
from .ordering import reorder, to_nested, to_ring, normalize_ordering
from .geometry import npix2nside, nside2npix, HealpixMeta

__all__ = [
    "read_healpix_map",
    "reorder",
    "to_nested",
    "to_ring",
    "normalize_ordering",
    "npix2nside",
    "nside2npix",
    "HealpixMeta",
]
