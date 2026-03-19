"""Data-loading helpers for LtU-ILI.

LtU-ILI already ships with convenient in-memory and on-disk loaders such as
:class:`ili.dataloaders.NumpyLoader` and :class:`ili.dataloaders.StaticNumpyLoader`.

DeepHpx adds thin helpers focused on one thing:

- reading many HEALPix maps (FITS / NPY / NPZ) into a single numpy array
- normalizing ordering to NEST (recommended) for DeepHpx pooling/encoder
- creating an LtU-ILI ``NumpyLoader`` for training.

We keep ``ili`` as an optional dependency by importing it lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..healpix.io import read_healpix_map
from ..healpix.ordering import normalize_ordering, reorder


def _as_paths(paths: Sequence[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out.append(Path(p))
    return out


def load_healpix_maps_to_numpy(
    map_paths: Sequence[str | Path],
    *,
    field: int | tuple[int, ...] | None = 0,
    target_ordering: str = "NEST",
    fits_nest: Optional[bool] = False,
    assume_ordering_nonfits: Optional[str] = "RING",
    dtype: np.dtype = np.float32,
    ensure_2d: bool = True,
    fill_masked: bool = True,
    fill_value: float = 0.0,
    verbose: bool = False,
) -> np.ndarray:
    """Load a collection of HEALPix maps into a single numpy array.

    Parameters
    ----------
    map_paths:
        Sequence of paths to HEALPix maps (FITS/NPY/NPZ).
    field:
        FITS field(s) to read (healpy convention). If ``None``, read all fields.
    target_ordering:
        Output ordering. DeepHpx pooling expects **NEST**.
    fits_nest:
        Passed through to :func:`deephpx.healpix.io.read_healpix_map` for FITS files.
        - ``False`` (default): healpy converts FITS maps to RING ordering.
        - ``True``: healpy converts FITS maps to NEST ordering.
        - ``None``: keep file ordering (requires FITS header to indicate ORDERING).

        Practical recommendation: keep the default ``False`` and let this
        function reorder to ``target_ordering`` explicitly.
    assume_ordering_nonfits:
        Ordering to assume for non-FITS inputs if no explicit ordering metadata
        exists. If you save numpy arrays in NEST ordering, set this to ``"NEST"``.
    dtype:
        dtype for returned numpy array.
    ensure_2d:
        If True, force shape (npix, nch) even for single-channel maps.
    fill_masked:
        If True and the input is a masked array, fill masked values with
        ``fill_value``.

    Returns
    -------
    X:
        Array of shape (N, npix, nch).
    """

    tord = normalize_ordering(target_ordering)
    if tord is None:
        raise ValueError("target_ordering must be 'RING' or 'NEST'")

    map_paths_ = _as_paths(map_paths)
    if len(map_paths_) == 0:
        raise ValueError("map_paths must be a non-empty sequence")

    maps: list[np.ndarray] = []
    npix_ref: Optional[int] = None

    for i, p in enumerate(map_paths_):
        m, meta, _extra = read_healpix_map(
            p,
            field=field,
            dtype=dtype,
            nest=fits_nest,
            assume_ordering=assume_ordering_nonfits,
            ensure_2d=ensure_2d,
        )

        # Convert masked arrays (common for partial-sky maps) into dense arrays.
        if isinstance(m, np.ma.MaskedArray):
            if fill_masked:
                m = m.filled(fill_value)
            else:
                m = np.asarray(m)

        m = np.asarray(m, dtype=dtype)

        # Ensure (npix, nch)
        if ensure_2d and m.ndim == 1:
            m = m.reshape(-1, 1)

        if m.ndim != 2:
            raise ValueError(
                f"Expected each map to be 2D (npix, nch) after loading, got shape={m.shape} for {p}"
            )

        # Normalize ordering to target.
        inp = meta.ordering
        if inp is None:
            inp = normalize_ordering(assume_ordering_nonfits)
        if inp is None:
            raise ValueError(
                f"Could not determine ordering for {p}. Provide assume_ordering_nonfits='RING' or 'NEST'."
            )
        if inp != tord:
            m = reorder(m, inp=inp, out=tord)

        if npix_ref is None:
            npix_ref = int(m.shape[0])
        else:
            if int(m.shape[0]) != npix_ref:
                raise ValueError(
                    "All maps must have the same npix. "
                    f"Got npix_ref={npix_ref} but map {p} has npix={m.shape[0]}"
                )

        maps.append(m)

        if verbose and (i % 50 == 0 or i == len(map_paths_) - 1):
            print(f"Loaded {i+1}/{len(map_paths_)} maps")

    X = np.stack(maps, axis=0)
    return X


def make_ili_numpy_loader(
    X: np.ndarray,
    theta: np.ndarray,
    *,
    xobs: Optional[np.ndarray] = None,
    thetafid: Optional[np.ndarray] = None,
):
    """Create an LtU-ILI in-memory loader (``ili.dataloaders.NumpyLoader``).

    We import LtU-ILI lazily so DeepHpx can be used without it.
    """

    try:
        from ili.dataloaders import NumpyLoader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "LtU-ILI is required for make_ili_numpy_loader(). "
            "Install with `pip install 'deephpx[ili]'` (or install ltu-ili[pytorch] directly)."
        ) from e

    return NumpyLoader(X, theta, xobs=xobs, thetafid=thetafid)


def make_ili_numpy_loader_from_files(
    map_paths: Sequence[str | Path],
    theta_path: str | Path,
    *,
    theta_key: Optional[str] = None,
    field: int | tuple[int, ...] | None = 0,
    target_ordering: str = "NEST",
    fits_nest: Optional[bool] = False,
    assume_ordering_nonfits: Optional[str] = "RING",
    dtype: np.dtype = np.float32,
    ensure_2d: bool = True,
    fill_masked: bool = True,
    fill_value: float = 0.0,
    verbose: bool = False,
):
    """Convenience: load maps + theta arrays and return an LtU-ILI loader.

    ``theta_path`` must point to ``.npy`` or ``.npz``.
    """

    X = load_healpix_maps_to_numpy(
        map_paths,
        field=field,
        target_ordering=target_ordering,
        fits_nest=fits_nest,
        assume_ordering_nonfits=assume_ordering_nonfits,
        dtype=dtype,
        ensure_2d=ensure_2d,
        fill_masked=fill_masked,
        fill_value=fill_value,
        verbose=verbose,
    )

    theta_path = Path(theta_path)
    if not theta_path.exists():
        raise FileNotFoundError(str(theta_path))

    if theta_path.suffix.lower() == ".npy":
        theta = np.load(theta_path)
    elif theta_path.suffix.lower() == ".npz":
        z = np.load(theta_path)
        key = theta_key
        if key is None:
            key = "theta" if "theta" in z.files else (z.files[0] if z.files else None)
        if key is None:
            raise ValueError(f"No arrays found in {theta_path}")
        theta = z[key]
    else:
        raise ValueError("theta_path must be a .npy or .npz file")

    theta = np.asarray(theta, dtype=dtype)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)

    if len(theta) != len(X):
        raise ValueError(f"len(theta)={len(theta)} does not match len(X)={len(X)}")

    return make_ili_numpy_loader(X, theta)


__all__ = [
    "load_healpix_maps_to_numpy",
    "make_ili_numpy_loader",
    "make_ili_numpy_loader_from_files",
]
