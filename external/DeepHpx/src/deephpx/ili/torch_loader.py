"""LtU-ILI TorchLoader helpers for streaming HEALPix datasets.

LtU-ILI provides a :class:`ili.dataloaders.TorchLoader` class that wraps
PyTorch ``DataLoader`` objects (train/val). This enables backends that can
consume streaming batches (notably the *lampe* backend) without requiring the
full dataset to be loaded into RAM.

Important caveat
---------------
LtU-ILI's *sbi* runner currently calls ``get_all_data``/``get_all_parameters``
on the loader, which (for large datasets) implies an in-memory representation.
The *lampe* runner, however, will use ``train_loader``/``val_loader`` directly
when present.

This module provides:

- :func:`make_dataloaders_from_files` to build PyTorch DataLoaders from a
  directory/glob of HEALPix maps + a theta array
- :func:`make_ili_torch_loader` to wrap those DataLoaders into an LtU-ILI
  ``TorchLoader`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np


def _require_torch():
    try:
        import torch  # type: ignore
        from torch.utils.data import DataLoader  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Torch is required for streaming loaders. Install with `pip install deephpx[torch]`."
        ) from e
    return torch


def _require_ili():
    try:
        import ili  # noqa: F401
        from ili.dataloaders import TorchLoader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "LtU-ILI is required for make_ili_torch_loader(). Install with `pip install deephpx[ili]`."
        ) from e
    return TorchLoader


@dataclass(frozen=True)
class SplitSpec:
    """Train/val split configuration."""

    val_fraction: float = 0.1
    seed: int = 0


def split_indices(n: int, *, val_fraction: float = 0.1, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Create deterministic train/val indices."""

    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_fraction * n))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def make_dataloaders_from_files(
    paths: Sequence[Union[str, Path]],
    theta: np.ndarray,
    *,
    batch_size: int = 32,
    split: SplitSpec = SplitSpec(),
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: Optional[int] = None,
    drop_last: bool = False,
    dataset_kwargs: Optional[dict] = None,
):
    """Create PyTorch DataLoaders for streaming HEALPix training.

    Parameters
    ----------
    paths:
        Sequence of map files (FITS/npy/npz), one per sample.
    theta:
        NumPy array of parameters with shape (N, D).
    batch_size:
        Batch size used by the returned DataLoaders.
    split:
        Train/val split spec.
    shuffle:
        Whether to shuffle training batches.
    num_workers:
        Number of DataLoader workers.
    pin_memory:
        Whether to enable pinned memory (useful when training on CUDA).
    persistent_workers:
        If True and num_workers>0, keep workers alive across epochs.
    prefetch_factor:
        Optional prefetch factor (PyTorch DataLoader). Only valid when num_workers>0.
    drop_last:
        Whether to drop the last incomplete batch in the training loader.
    dataset_kwargs:
        Extra keyword arguments forwarded to :class:`deephpx.ili.dataset.HealpixFileDataset`.

    Returns
    -------
    (train_loader, val_loader)
    """

    torch = _require_torch()
    from torch.utils.data import DataLoader, Subset

    from .dataset import HealpixFileDataset

    dataset_kwargs = {} if dataset_kwargs is None else dict(dataset_kwargs)

    ds = HealpixFileDataset(paths=paths, theta=theta, **dataset_kwargs)

    train_idx, val_idx = split_indices(len(ds), val_fraction=split.val_fraction, seed=split.seed)

    train_ds = Subset(ds, train_idx.tolist())
    val_ds = Subset(ds, val_idx.tolist()) if len(val_idx) > 0 else None

    dl_kwargs = dict(
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
    )

    # DataLoader kwargs with worker settings
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(train_ds, shuffle=bool(shuffle), **dl_kwargs)
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(val_ds, shuffle=False, **{k: v for k, v in dl_kwargs.items() if k != "drop_last"})

    return train_loader, val_loader


def make_ili_torch_loader(
    train_loader,
    val_loader=None,
    *,
    xobs: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    thetafid: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
):
    """Wrap PyTorch DataLoaders in an LtU-ILI TorchLoader."""

    TorchLoader = _require_ili()
    return TorchLoader(train_loader=train_loader, val_loader=val_loader, xobs=xobs, thetafid=thetafid)


def make_ili_torch_loader_from_files(
    *,
    maps: Union[str, Sequence[Union[str, Path]]],
    theta: Union[str, Path, np.ndarray],
    batch_size: int = 32,
    split: SplitSpec = SplitSpec(),
    num_workers: int = 0,
    pin_memory: bool = False,
    cache_dir: Optional[Union[str, Path]] = None,
    assume_ordering: Optional[str] = "NEST",
    output_ordering: Optional[str] = "NEST",
    field: Union[int, Tuple[int, ...], None] = 0,
    map_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    memmap: bool = False,
):
    """Convenience constructor: maps glob/paths + theta -> LtU-ILI TorchLoader.

    Parameters
    ----------
    maps:
        Either a glob pattern (str) or an explicit list of paths.
    theta:
        Either a path to a ``.npy``/``.npz`` array with shape (N, D) or an
        in-memory numpy array.

    Notes
    -----
    The returned object is an instance of LtU-ILI's ``TorchLoader``.
    """

    # Resolve map files
    if isinstance(maps, str):
        import glob

        m = str(Path(maps).expanduser())
        if any(ch in m for ch in "*?["):
            paths = [Path(p) for p in sorted(glob.glob(m, recursive=True))]
        else:
            p = Path(m)
            if p.is_dir():
                # Heuristic: load common map formats
                exts = ["*.fits", "*.fit", "*.fits.gz", "*.fit.gz", "*.npy", "*.npz"]
                paths = []
                for ext in exts:
                    paths.extend(sorted(p.glob(ext)))
            else:
                paths = [p]
    else:
        paths = [Path(p) for p in maps]

    if len(paths) == 0:
        raise ValueError("No map files found.")

    # Load theta
    if isinstance(theta, (str, Path)):
        tpath = Path(theta)
        if not tpath.exists():
            raise FileNotFoundError(str(tpath))
        if tpath.suffix.lower() == ".npy":
            theta_arr = np.load(tpath, allow_pickle=False)
        elif tpath.suffix.lower() == ".npz":
            z = np.load(tpath, allow_pickle=False)
            # Heuristic: use 'theta' if present else first key
            key = "theta" if "theta" in z.files else (z.files[0] if z.files else None)
            if key is None:
                raise ValueError(f"No arrays found in {tpath}")
            theta_arr = z[key]
        else:
            raise ValueError("theta must be a .npy/.npz file path or a numpy array")
    else:
        theta_arr = np.asarray(theta)

    if theta_arr.ndim == 1:
        theta_arr = theta_arr.reshape(-1, 1)

    dataset_kwargs = dict(
        field=field,
        output_ordering=output_ordering,
        assume_ordering=assume_ordering,
        memmap=memmap,
        cache_dir=cache_dir,
        map_transform=map_transform,
    )

    train_loader, val_loader = make_dataloaders_from_files(
        paths=paths,
        theta=theta_arr,
        batch_size=batch_size,
        split=split,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        dataset_kwargs=dataset_kwargs,
    )

    return make_ili_torch_loader(train_loader=train_loader, val_loader=val_loader)


__all__ = [
    "SplitSpec",
    "split_indices",
    "make_dataloaders_from_files",
    "make_ili_torch_loader",
    "make_ili_torch_loader_from_files",
]
