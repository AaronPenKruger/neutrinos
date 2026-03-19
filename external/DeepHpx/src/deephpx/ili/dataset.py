"""PyTorch datasets for HEALPix SBI / ILI.

Milestone 7 introduces *streaming* (on-demand) ingestion of HEALPix maps.

The main entry point is :class:`HealpixFileDataset`, a map-style PyTorch
Dataset that:

- stores a list of file paths (FITS / npy / npz)
- stores a corresponding parameter array ``theta``
- reads a single HEALPix map on ``__getitem__``
- optionally enforces a target ordering (default: NEST)
- returns tensors suitable for ``torch.utils.data.DataLoader``

This is designed to pair naturally with LtU-ILI's ``TorchLoader`` wrapper,
which expects dataloaders that yield ``(x, theta)`` batches.

Notes
-----
* For FITS files, we can ask healpy to output the desired ordering directly
  via the ``nest=...`` argument in ``healpy.fitsfunc.read_map`` (DeepHpx wraps
  this through :func:`deephpx.healpix.io.read_healpix_map`).
* For NumPy files, ordering information is usually not embedded; in those
  cases, pass ``assume_ordering=...`` if you need a guaranteed reorder.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    """Atomically write a .npy file.

    This avoids partially-written cache files when using multi-worker
    DataLoaders.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    # Write to a deterministic name (no automatic .npy suffix)
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


class HealpixFileDataset:  # intentionally not importing torch at module import time
    """A streaming dataset reading HEALPix maps from disk.

    Parameters
    ----------
    paths:
        Sequence of FITS / npy / npz files. One file per sample.
    theta:
        Array-like of parameters with shape ``(N, D)``.
    field:
        FITS field(s) to read (healpy convention). Default: 0.
    dtype:
        Output dtype for maps (default: float32).
    output_ordering:
        Desired ordering for returned maps. "NEST" is recommended for DeepHpx
        pooling. If None, no reordering is performed.
    assume_ordering:
        If map files do not store ordering metadata (e.g., .npy), provide the
        assumed input ordering here ("RING" or "NEST").
    fits_nest:
        If not None, passes this to healpy for FITS reads.
        If left as None, we choose based on ``output_ordering``:
          - output_ordering == "NEST" -> fits_nest=True
          - output_ordering == "RING" -> fits_nest=False
          - output_ordering is None -> fits_nest=None
    memmap:
        For NumPy reads, use memory mapping where possible.
    cache_dir:
        Optional directory to cache preprocessed maps as .npy (after ordering
        conversion + transform). If provided, subsequent reads can avoid FITS
        parsing overhead.
    map_transform:
        Optional callable applied to the map array *after* ordering conversion.
        Signature: ``np.ndarray -> np.ndarray``.

    Returns
    -------
    Each sample is ``(x, theta)`` where:
      * ``x`` is ``torch.float32`` with shape ``(npix, n_chan)``
      * ``theta`` is ``torch.float32`` with shape ``(D,)``

    Notes
    -----
    This class is intentionally implemented without inheriting from
    ``torch.utils.data.Dataset`` at import time to keep ``deephpx`` lightweight
    when torch is not installed. At runtime we validate torch and then wrap the
    required methods.
    """

    def __init__(
        self,
        paths: Sequence[Union[str, Path]],
        theta: Union[np.ndarray, "torch.Tensor"],
        *,
        field: Union[int, Tuple[int, ...], None] = 0,
        dtype: np.dtype = np.float32,
        output_ordering: Optional[str] = "NEST",
        assume_ordering: Optional[str] = None,
        fits_nest: Optional[bool] = None,
        memmap: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        map_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        # Local import to avoid hard torch dependency at package import time.
        try:
            import torch  # noqa: WPS433
            from torch.utils.data import Dataset  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "HealpixFileDataset requires torch. Install with `pip install deephpx[torch]`."
            ) from e

        from deephpx.healpix.io import read_healpix_map
        from deephpx.healpix.ordering import normalize_ordering

        self._read_healpix_map = read_healpix_map
        self._normalize_ordering = normalize_ordering

        self.paths = [Path(p) for p in paths]
        if len(self.paths) == 0:
            raise ValueError("paths must be non-empty")

        self.field = field
        self.dtype = np.dtype(dtype)
        self.output_ordering = self._normalize_ordering(output_ordering) if output_ordering else None
        self.assume_ordering = self._normalize_ordering(assume_ordering) if assume_ordering else None

        if fits_nest is None:
            if self.output_ordering == "NEST":
                self.fits_nest = True
            elif self.output_ordering == "RING":
                self.fits_nest = False
            else:
                self.fits_nest = None
        else:
            self.fits_nest = bool(fits_nest)

        self.memmap = bool(memmap)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.map_transform = map_transform

        # store theta as torch Tensor on CPU
        if isinstance(theta, np.ndarray):
            theta_arr = theta
            if theta_arr.ndim == 1:
                theta_arr = theta_arr.reshape(-1, 1)
            self.theta = torch.as_tensor(theta_arr, dtype=torch.float32, device="cpu")
        else:
            self.theta = theta.detach().to(dtype=torch.float32, device="cpu")

        if len(self.theta) != len(self.paths):
            raise ValueError(
                f"theta length ({len(self.theta)}) must match number of paths ({len(self.paths)})."
            )

    # --- PyTorch Dataset protocol ---
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        import torch  # noqa: WPS433

        x_np = self._load_map(idx)
        # Ensure float32 and contiguous for torch.from_numpy
        x_np = np.ascontiguousarray(x_np, dtype=np.float32)
        x = torch.from_numpy(x_np)
        theta = self.theta[idx]
        return x, theta

    # --- Internal helpers ---
    def _cache_path_for(self, src: Path) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        # Include basic preprocessing knobs in the key so that caches don't get
        # silently reused across incompatible settings.
        key = _sha1(
            f"{src.resolve()}|field={self.field}|out={self.output_ordering}|dtype={self.dtype}"
        )
        return self.cache_dir / f"{key}.npy"

    def _load_map(self, idx: int) -> np.ndarray:
        from deephpx.healpix.ordering import reorder

        src = self.paths[idx]
        cache_path = self._cache_path_for(src)

        # 1) cache hit
        if cache_path is not None and cache_path.exists():
            arr = np.load(cache_path, allow_pickle=False, mmap_mode="r")
            arr = np.asarray(arr)
            return arr

        # 2) read raw map
        # For FITS, request desired ordering directly when possible.
        nest_arg = None
        if src.suffix.lower() in {".fits", ".fit"} or src.name.lower().endswith((".fits.gz", ".fit.gz")):
            nest_arg = self.fits_nest

        arr, meta, _extra = self._read_healpix_map(
            src,
            field=self.field,
            dtype=self.dtype,
            nest=nest_arg,
            memmap=self.memmap,
            assume_ordering=self.assume_ordering,
            ensure_2d=True,
        )

        # fill masked arrays with 0 by default (common for partial sky)
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled(0)

        arr = np.asarray(arr)

        # 3) enforce ordering if requested
        if self.output_ordering is not None:
            inp = meta.ordering or self.assume_ordering
            if inp is None:
                raise ValueError(
                    "Input ordering is unknown for map file "
                    f"{src}. Provide `assume_ordering=...` or store ordering metadata."
                )
            if inp != self.output_ordering:
                arr = reorder(arr, inp=inp, out=self.output_ordering)

        # 4) apply user transform
        if self.map_transform is not None:
            arr = self.map_transform(arr)
            arr = np.asarray(arr)

        # 5) write cache (best-effort)
        if cache_path is not None:
            try:
                _atomic_save_npy(cache_path, np.asarray(arr, dtype=np.float32))
            except FileExistsError:
                # benign race with another worker
                pass
            except Exception:
                # caching should never break training
                pass

        return np.asarray(arr, dtype=np.float32)


__all__ = ["HealpixFileDataset"]
