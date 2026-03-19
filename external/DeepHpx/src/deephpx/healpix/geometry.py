"""Small HEALPix geometry helpers.

We keep these tiny utilities in DeepHpx so that basic checks (npix <-> nside)
work even when `healpy` is not installed. When `healpy` is available we prefer
its canonical implementations.

HEALPix definition used here:
    Npix = 12 * Nside^2

In standard HEALPix, Nside is a power of two.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Optional


def nside2npix(nside: int) -> int:
    """Return number of pixels for a given NSIDE."""
    if nside <= 0:
        raise ValueError(f"nside must be positive, got {nside}")
    return 12 * nside * nside


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def npix2nside(npix: int, *, strict: bool = True) -> int:
    """Infer NSIDE from Npix.

    Args:
        npix: number of HEALPix pixels.
        strict: if True, enforce the standard HEALPix constraints:
            - npix == 12 * nside^2
            - nside is a power of two

    Returns:
        nside

    Raises:
        ValueError: if `strict=True` and `npix` is not a valid HEALPix pixel count.
    """
    if npix <= 0:
        raise ValueError(f"npix must be positive, got {npix}")

    # Solve nside^2 = npix / 12
    if npix % 12 != 0:
        if strict:
            raise ValueError(f"Wrong pixel number (it is not 12*nside**2): npix={npix}")
        # best-effort
        return int((npix / 12) ** 0.5)

    nside_sq = npix // 12
    nside = isqrt(nside_sq)

    if nside * nside != nside_sq:
        if strict:
            raise ValueError(f"Wrong pixel number (it is not 12*nside**2): npix={npix}")
        return int((npix / 12) ** 0.5)

    if strict and not _is_power_of_two(nside):
        raise ValueError(f"Invalid nside (not power of 2): nside={nside} from npix={npix}")

    return nside


@dataclass(frozen=True)
class HealpixMeta:
    """Lightweight metadata container for loaded HEALPix maps."""

    path: str
    format: str
    npix: int
    nside: int
    ordering: Optional[str] = None  # 'RING' or 'NEST'
