"""HEALPix ordering normalization.

HEALPix supports two ordering schemes:
- RING
- NESTED (often abbreviated NEST)

`healpy` provides a canonical reorder function:
    healpy.pixelfunc.reorder(map_in, inp=..., out=...)

DeepHpx provides thin wrappers with:
- input validation
- support for 1D maps or 2D feature maps (npix, nch) / (nch, npix)
- a consistent internal naming convention: "RING" and "NEST"
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .geometry import npix2nside

Ordering = Literal["RING", "NEST"]


def normalize_ordering(ordering: Optional[str]) -> Optional[Ordering]:
    """Normalize user-provided ordering strings.

    Accepts: "RING", "NEST", "NESTED" (case-insensitive). Returns "RING" or "NEST".
    """
    if ordering is None:
        return None
    o = ordering.strip().upper()
    if o == "RING":
        return "RING"
    if o in {"NEST", "NESTED"}:
        return "NEST"
    raise ValueError(f"Unknown ordering {ordering!r}. Expected 'RING' or 'NEST'/'NESTED'.")


def _require_healpy():
    try:
        import healpy as hp  # type: ignore
    except Exception as e:
        raise ImportError(
            "healpy is required for ordering conversions. Install with `pip install deephpx[healpix]`."
        ) from e
    return hp


def _infer_pix_axis(arr: np.ndarray) -> int:
    """Infer which axis is the pixel axis for a 2D map.

    For 2D arrays we accept (npix, nch) or (nch, npix). We infer the pixel
    axis by checking which axis length corresponds to a valid HEALPix npix.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got ndim={arr.ndim}")

    candidates = []
    for ax, n in enumerate(arr.shape):
        try:
            _ = npix2nside(int(n), strict=True)
            candidates.append(ax)
        except Exception:
            continue

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise ValueError(
            f"Could not infer pixel axis from shape={arr.shape}. "
            "Neither axis length looks like a valid HEALPix npix (=12*nside^2)."
        )
    # Ambiguous: both axes look like valid npix (rare). Prefer axis 0.
    return 0


def reorder(map_in: Union[np.ndarray, np.ma.MaskedArray], *, inp: str, out: str):
    """Reorder a HEALPix map between RING and NEST.

    This is a robust wrapper around `healpy.pixelfunc.reorder`, supporting:
    - 1D maps of shape (npix,)
    - 2D maps of shape (npix, nch) or (nch, npix)

    Args:
        map_in: input map
        inp: input ordering ("RING" or "NEST"/"NESTED")
        out: output ordering ("RING" or "NEST"/"NESTED")

    Returns:
        Reordered map with the same shape as input.
    """
    hp = _require_healpy()

    inp_n = normalize_ordering(inp)
    out_n = normalize_ordering(out)

    if inp_n == out_n:
        return map_in

    arr = map_in

    if arr.ndim == 1:
        return hp.pixelfunc.reorder(arr, inp=inp_n, out=out_n)  # type: ignore[attr-defined]

    if arr.ndim != 2:
        raise ValueError(f"Only 1D or 2D maps are supported, got shape={arr.shape}")

    pix_axis = _infer_pix_axis(np.asarray(arr))

    # Reorder along the pixel axis, preserving original layout.
    if pix_axis == 0:
        nch = arr.shape[1]
        cols = [hp.pixelfunc.reorder(arr[:, i], inp=inp_n, out=out_n) for i in range(nch)]
        if isinstance(arr, np.ma.MaskedArray):
            return np.ma.stack(cols, axis=1)
        return np.stack(cols, axis=1)

    # pix_axis == 1
    nch = arr.shape[0]
    rows = [hp.pixelfunc.reorder(arr[i, :], inp=inp_n, out=out_n) for i in range(nch)]
    if isinstance(arr, np.ma.MaskedArray):
        return np.ma.stack(rows, axis=0)
    return np.stack(rows, axis=0)


def to_nested(map_in: Union[np.ndarray, np.ma.MaskedArray], *, inp: str = "RING"):
    """Convert a HEALPix map to NEST ordering."""
    return reorder(map_in, inp=inp, out="NEST")


def to_ring(map_in: Union[np.ndarray, np.ma.MaskedArray], *, inp: str = "NEST"):
    """Convert a HEALPix map to RING ordering."""
    return reorder(map_in, inp=inp, out="RING")
