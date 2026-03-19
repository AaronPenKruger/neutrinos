"""HEALPix map I/O.

This module provides a small, explicit API to load HEALPix maps into NumPy.

Supported formats:
- FITS: `.fits` / `.fits.gz` via `healpy.fitsfunc.read_map`
- NumPy: `.npy`, `.npz`

We keep this module separate from ordering conversions. If you want to ensure a
specific ordering, call :func:`deephpx.healpix.ordering.to_nested` or
:func:`deephpx.healpix.ordering.to_ring` after loading.

Notes on ordering:
- By default, `healpy.fitsfunc.read_map` converts maps to RING ordering unless
  you pass `nest=True`, or you keep ordering unchanged with `nest=None`.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .geometry import HealpixMeta, npix2nside
from .ordering import normalize_ordering


def _require_healpy():
    try:
        import healpy as hp  # type: ignore
    except Exception as e:
        raise ImportError(
            "healpy is required to read FITS HEALPix maps. Install with `pip install deephpx[healpix]`."
        ) from e
    return hp


def _is_fits_path(p: Path) -> bool:
    s = p.name.lower()
    return s.endswith(".fits") or s.endswith(".fit") or s.endswith(".fits.gz") or s.endswith(".fit.gz")


def _extract_header_value(header: Any, key: str) -> Optional[str]:
    """Best-effort extraction of a FITS header keyword from healpy's header return.

    `healpy.fitsfunc.read_map(..., h=True)` may return the header in different shapes
    depending on versions: often a list of (key, value, comment) tuples.

    We handle:
    - list/tuple of tuples
    - dict-like
    - list of strings like 'KEYWORD = VALUE / comment'
    """
    if header is None:
        return None

    # Dict-like (e.g., astropy.io.fits.Header)
    try:
        if hasattr(header, "get"):
            v = header.get(key)
            if v is not None:
                return str(v)
    except Exception:
        pass

    # Sequence forms
    if isinstance(header, (list, tuple)):
        for item in header:
            # (key, value, comment)
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                if str(item[0]).strip().upper() == key.upper():
                    return str(item[1])
            # 'KEY = VALUE'
            if isinstance(item, str):
                line = item.strip()
                if line.upper().startswith(key.upper()):
                    # naive parse: KEY = VALUE
                    if "=" in line:
                        rhs = line.split("=", 1)[1]
                        rhs = rhs.split("/", 1)[0]
                        return rhs.strip().strip("'")

    return None


def read_healpix_map(
    path: Union[str, Path],
    *,
    field: Union[int, Tuple[int, ...], None] = 0,
    dtype: Optional[np.dtype] = None,
    nest: Optional[bool] = None,
    partial: bool = False,
    memmap: bool = False,
    npz_key: Optional[str] = None,
    assume_ordering: Optional[str] = None,
    stack_fields: bool = True,
    ensure_2d: bool = False,
) -> Tuple[Union[np.ndarray, np.ma.MaskedArray], HealpixMeta, Dict[str, Any]]:
    """Load a HEALPix map from disk.

    Args:
        path: path to `.fits`/`.fits.gz`, `.npy`, or `.npz`.
        field: FITS column(s) to read (healpy convention). If None, read all fields.
        dtype: dtype override (passed to healpy for FITS; applied after load for NumPy).
        nest: FITS-only; passed to `healpy.fitsfunc.read_map`.
            - True: output is NEST ordering
            - False: output is RING ordering
            - None: no conversion (preserve file ordering)
        partial: FITS-only; partial-sky handling (passed to healpy).
        memmap: if True, allow memory mapping where supported.
        npz_key: `.npz` only; which array key to read. If None, uses 'map' if present
            else the first key.
        assume_ordering: for non-FITS formats (or when FITS ordering is unknown),
            set meta.ordering to this value ('RING' or 'NEST'/'NESTED').
        stack_fields: if FITS returns multiple fields (tuple of arrays), stack them
            into shape (npix, n_fields). If False, keep as a tuple (not recommended
            for ML).
        ensure_2d: if True, ensure output is at least 2D with shape (npix, n_channels)
            where n_channels==1 for single-field maps.

    Returns:
        (map, meta, extra)
        - map: ndarray or MaskedArray
        - meta: HealpixMeta with npix, nside, ordering (best-effort)
        - extra: dict with optional extra info (e.g., FITS header)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    extra: Dict[str, Any] = {}

    if _is_fits_path(p):
        hp = _require_healpy()
        out = hp.fitsfunc.read_map(
            p,
            field=field,
            dtype=dtype,
            nest=nest,
            partial=partial,
            h=True,
            memmap=memmap,
            verbose=False,
        )

        # Parse (maps..., header)
        header = None
        maps = None
        if isinstance(out, tuple) and len(out) >= 2:
            header = out[-1]
            maps = out[:-1]
            if len(maps) == 1:
                map_arr = maps[0]
            else:
                if not stack_fields:
                    map_arr = maps  # type: ignore[assignment]
                else:
                    # Stack to (npix, n_fields)
                    if any(isinstance(m, np.ma.MaskedArray) for m in maps):
                        map_arr = np.ma.stack(list(maps), axis=1)
                    else:
                        map_arr = np.stack(list(maps), axis=1)
        else:
            # Unexpected but handle
            map_arr = out

        extra["fits_header"] = header

        # Determine ordering of the *returned* map.
        # If nest is explicitly set, output ordering is known.
        ordering: Optional[str]
        if nest is True:
            ordering = "NEST"
        elif nest is False:
            ordering = "RING"
        else:
            # nest=None means no conversion; try to read ORDERING from header.
            ordering = _extract_header_value(header, "ORDERING")

        if ordering is not None:
            try:
                ordering = normalize_ordering(ordering)
            except Exception:
                # Keep best-effort value
                pass

        # infer npix
        npix = int(map_arr.shape[0]) if getattr(map_arr, "ndim", 1) >= 1 else int(len(map_arr))
        if getattr(map_arr, "ndim", 1) == 2 and map_arr.shape[0] < map_arr.shape[1]:
            # If healpy stacked to (n_fields, npix) (shouldn't happen in our code),
            # pick the axis that matches HEALPix npix.
            try:
                _ = npix2nside(int(map_arr.shape[1]), strict=True)
                npix = int(map_arr.shape[1])
            except Exception:
                pass

        nside = npix2nside(npix, strict=True)
        meta = HealpixMeta(path=str(p), format="fits", npix=npix, nside=nside, ordering=ordering)

        if ensure_2d and getattr(map_arr, "ndim", 1) == 1:
            map_arr = map_arr.reshape(-1, 1)

        return map_arr, meta, extra

    # NumPy formats
    suffix = p.suffix.lower()
    if suffix == ".npy":
        mmap_mode = "r" if memmap else None
        map_arr = np.load(p, allow_pickle=False, mmap_mode=mmap_mode)
        fmt = "npy"
    elif suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        key = npz_key
        if key is None:
            key = "map" if "map" in z.files else (z.files[0] if z.files else None)
        if key is None:
            raise ValueError(f"No arrays found in {p}")
        map_arr = z[key]
        fmt = "npz"
        extra["npz_key"] = key
    else:
        raise ValueError(
            f"Unsupported file extension for {p.name!r}. Supported: .fits/.fits.gz/.npy/.npz"
        )

    if dtype is not None:
        map_arr = map_arr.astype(dtype, copy=False)

    ordering = normalize_ordering(assume_ordering) if assume_ordering is not None else None

    # Determine npix from shape
    if map_arr.ndim == 1:
        npix = int(map_arr.shape[0])
    elif map_arr.ndim == 2:
        # Infer pixel axis by validity check.
        # Accept (npix, nch) or (nch, npix).
        try:
            _ = npix2nside(int(map_arr.shape[0]), strict=True)
            npix = int(map_arr.shape[0])
        except Exception:
            npix = int(map_arr.shape[1])
    else:
        raise ValueError(f"Only 1D or 2D maps are supported, got shape={map_arr.shape}")

    nside = npix2nside(npix, strict=True)
    meta = HealpixMeta(path=str(p), format=fmt, npix=npix, nside=nside, ordering=ordering)

    if ensure_2d and map_arr.ndim == 1:
        map_arr = map_arr.reshape(-1, 1)

    return map_arr, meta, extra
