"""LtU-ILI integration helpers.

Milestone 6 adds a *thin* integration layer between DeepHpx (HEALPix graph /
encoder utilities) and the LtU-ILI pipeline (``ltu-ili``, imported as ``ili``).

Important: LtU-ILI and the HEALPix embedding network depend on optional
PyTorch/Healpy/Scipy stacks. To keep the base package lightweight, we
*intentionally* avoid importing torch/healpy at module import time.

See ``examples/05_ili_sbi_train_healpix.py`` for an end-to-end example.
"""

from __future__ import annotations

from typing import Any

# Lightweight (no torch/healpy required at import-time)
from .loader import (
    load_healpix_maps_to_numpy,
    make_ili_numpy_loader,
    make_ili_numpy_loader_from_files,
)
from .train import train_sbi_posterior


def __getattr__(name: str) -> Any:
    """Lazy-import torch/healpy-dependent components.

    This prevents ``import deephpx.ili`` from failing in environments that only
    need the file/array loading helpers.
    """

    if name in {"HealpixEmbeddingNet", "HealpixEmbeddingSpec", "build_laplacian_pyramid"}:
        from .embedding import (  # noqa: WPS433 (import inside function by design)
            HealpixEmbeddingNet,
            HealpixEmbeddingSpec,
            build_laplacian_pyramid,
        )

        return {
            "HealpixEmbeddingNet": HealpixEmbeddingNet,
            "HealpixEmbeddingSpec": HealpixEmbeddingSpec,
            "build_laplacian_pyramid": build_laplacian_pyramid,
        }[name]

    if name in {
        "HealpixFileDataset",
        "SplitSpec",
        "split_indices",
        "make_dataloaders_from_files",
        "make_ili_torch_loader",
        "make_ili_torch_loader_from_files",
        "train_lampe_posterior",
    }:
        # These require torch (and, for the returned LtU-ILI loader, ili).
        from .dataset import HealpixFileDataset  # noqa: WPS433
        from .torch_loader import (  # noqa: WPS433
            SplitSpec,
            split_indices,
            make_dataloaders_from_files,
            make_ili_torch_loader,
            make_ili_torch_loader_from_files,
        )
        from .train import train_lampe_posterior  # noqa: WPS433

        return {
            "HealpixFileDataset": HealpixFileDataset,
            "SplitSpec": SplitSpec,
            "split_indices": split_indices,
            "make_dataloaders_from_files": make_dataloaders_from_files,
            "make_ili_torch_loader": make_ili_torch_loader,
            "make_ili_torch_loader_from_files": make_ili_torch_loader_from_files,
            "train_lampe_posterior": train_lampe_posterior,
        }[name]

    raise AttributeError(f"module 'deephpx.ili' has no attribute {name!r}")


__all__ = [
    # loader
    "load_healpix_maps_to_numpy",
    "make_ili_numpy_loader",
    "make_ili_numpy_loader_from_files",
    # training
    "train_sbi_posterior",
    "train_lampe_posterior",
    # embedding (lazy)
    "HealpixEmbeddingNet",
    "HealpixEmbeddingSpec",
    "build_laplacian_pyramid",
    # streaming / torch dataloaders (lazy)
    "HealpixFileDataset",
    "SplitSpec",
    "split_indices",
    "make_dataloaders_from_files",
    "make_ili_torch_loader",
    "make_ili_torch_loader_from_files",
]
