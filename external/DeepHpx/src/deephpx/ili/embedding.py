"""Embedding networks for LtU-ILI / SBI that understand HEALPix.

This module provides a convenience wrapper around :class:`deephpx.nn.HealpixEncoder`
that constructs the required HEALPix Laplacian pyramid internally.

The motivation is to make it ergonomic to do

- HEALPix map (npix, C) / (B, npix, C)
- -> DeepHpx graph conv + pooling encoder
- -> fixed-size embedding
- -> conditional normalizing flow (via LtU-ILI / sbi)

without having to manually pre-compute Laplacians in every training script.

Notes
-----
- Pooling/unpooling in DeepHpx assumes **NESTED** ordering.
- Laplacians are constructed for NEST ordering (``nest=True``).

LtU-ILI reference:
- ``ili.utils.load_nde_sbi(..., embedding_net=...)`` supports arbitrary PyTorch
  modules as embedding nets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    raise ImportError(
        "torch is required for deephpx.ili.embedding. Install with `pip install deephpx[torch]`."
    ) from e

from ..graph.adjacency import adjacency_from_neighbors
from ..graph.cache import ensure_cache_dir, load_sparse_npz, save_sparse_npz
from ..graph.laplacian import (
    LaplacianKind,
    laplacian_from_adjacency,
    scale_laplacian_for_chebyshev,
    to_torch_sparse_coo,
)
from ..graph.neighbors import neighbors_8
from ..nn.encoder import HealpixEncoder


def _cache_key(*, nside: int, nest: bool, kind: str, scaled: bool = True) -> str:
    flag = "nest" if nest else "ring"
    s = "scaled" if scaled else "raw"
    return f"L_{kind}_{s}_{flag}_nside{int(nside)}.npz"


def build_laplacian_pyramid(
    nside: int,
    *,
    levels: int,
    nest: bool = True,
    kind: LaplacianKind = "normalized",
    lmax_method: str = "eigsh",
    cache_dir: Optional[str | Path] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device | str] = None,
) -> list[torch.Tensor]:
    """Build a list of *scaled* Laplacians for a HEALPix resolution pyramid.

    Each level halves NSIDE (so the number of pixels quarters), matching
    DeepHpx's pooling semantics.

    Parameters
    ----------
    nside:
        Highest-resolution NSIDE.
    levels:
        Number of pyramid levels / Laplacians to build.
    nest:
        Whether to build the graph in NEST ordering (recommended / default).
    kind:
        Laplacian type: ``'normalized'`` or ``'combinatorial'``.
    lmax_method:
        Method used to estimate lmax when scaling (``'eigsh'`` or ``'power'``).
        For ``kind='normalized'`` we use the known bound lmax=2.
    cache_dir:
        Optional directory to cache scaled Laplacians as SciPy ``.npz``.
    dtype, device:
        Torch dtype/device for the returned sparse tensors.

    Returns
    -------
    laplacians:
        List of torch sparse COO tensors, ordered from highest to lowest
        resolution.
    """

    if nside <= 0:
        raise ValueError(f"nside must be positive, got {nside}")
    if levels <= 0:
        raise ValueError(f"levels must be positive, got {levels}")

    # validate that the pyramid stays integer
    min_nside = nside // (2 ** (levels - 1))
    if min_nside < 1 or (nside % (2 ** (levels - 1)) != 0):
        raise ValueError(
            f"nside={nside} is not compatible with levels={levels}. "
            "Require nside divisible by 2**(levels-1)."
        )

    laplacians: list[torch.Tensor] = []

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = ensure_cache_dir(cache_dir)

    for i in range(levels):
        nside_i = nside // (2**i)

        # caching: store the *scaled* Laplacian in SciPy format.
        if cache_path is not None:
            fname = _cache_key(nside=nside_i, nest=nest, kind=kind, scaled=True)
            fpath = cache_path / fname
        else:
            fpath = None

        if fpath is not None and fpath.is_file():
            L_tilde_sp = load_sparse_npz(fpath)
        else:
            neigh = neighbors_8(nside_i, nest=nest)
            A = adjacency_from_neighbors(neigh, symmetric=True, remove_self_loops=True)
            L = laplacian_from_adjacency(A, kind=kind)
            L_tilde_sp = scale_laplacian_for_chebyshev(
                L,
                kind_hint=kind,
                lmax=None,
                lmax_method=lmax_method,  # only used if kind != 'normalized'
            )
            if fpath is not None:
                save_sparse_npz(fpath, L_tilde_sp)

        L_tilde = to_torch_sparse_coo(L_tilde_sp, dtype=dtype, device=device, coalesce=True)
        laplacians.append(L_tilde)

    return laplacians


@dataclass
class HealpixEmbeddingSpec:
    """A small config container for :class:`HealpixEmbeddingNet`."""

    nside: int
    levels: int
    in_channels: int
    conv_channels: Sequence[int]
    embedding_dim: int
    K: int = 3
    pool: str = "average"
    global_pool: str = "mean"
    norm: str = "layer"
    dropout: float = 0.0
    mlp_hidden: Sequence[int] = ()
    activation: str = "relu"
    laplacian_kind: LaplacianKind = "normalized"
    nest: bool = True
    lmax_method: str = "eigsh"
    cache_dir: Optional[str | Path] = None


class HealpixEmbeddingNet(nn.Module):
    """A HEALPix-aware embedding network (maps -> embeddings).

    This module is intended to be passed as an ``embedding_net`` to
    ``ili.utils.load_nde_sbi`` (LtU-ILI / sbi) so that a normalizing flow can be
    trained on HEALPix maps.

    The embedding network itself is just :class:`deephpx.nn.HealpixEncoder` with
    Laplacians built automatically from ``nside``/``levels``.

    Parameters
    ----------
    nside, levels:
        Define the HEALPix resolution pyramid (nside, nside/2, nside/4, ...).
    in_channels:
        Number of input channels per pixel.
    conv_channels:
        Output channels per level (must have length ``levels``).
    embedding_dim:
        Final embedding size.

    All remaining parameters match :class:`deephpx.nn.HealpixEncoder`.

    Notes
    -----
    - Pooling assumes NEST ordering. Convert maps from RING -> NEST before
      training.
    - The Laplacians are registered as buffers inside the encoder, so moving
      this module to CUDA will also move the sparse Laplacians.
    """

    def __init__(
        self,
        *,
        nside: int,
        levels: int,
        in_channels: int,
        conv_channels: Sequence[int],
        embedding_dim: int,
        K: int = 3,
        pool: str = "average",
        global_pool: str = "mean",
        norm: str = "layer",
        dropout: float = 0.0,
        mlp_hidden: Sequence[int] = (),
        activation: str = "relu",
        laplacian_kind: LaplacianKind = "normalized",
        nest: bool = True,
        lmax_method: str = "eigsh",
        cache_dir: Optional[str | Path] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()

        if len(conv_channels) != levels:
            raise ValueError(
                f"conv_channels must have length == levels. Got len(conv_channels)={len(conv_channels)} "
                f"and levels={levels}."
            )

        # Build laplacian pyramid (scaled for Chebyshev).
        laplacians = build_laplacian_pyramid(
            nside,
            levels=levels,
            nest=nest,
            kind=laplacian_kind,
            lmax_method=lmax_method,
            cache_dir=cache_dir,
            dtype=dtype,
            device=device,
        )

        self.encoder = HealpixEncoder(
            laplacians,
            in_channels=in_channels,
            conv_channels=conv_channels,
            embedding_dim=embedding_dim,
            K=K,
            pool=pool,  # type: ignore[arg-type]
            global_pool=global_pool,  # type: ignore[arg-type]
            norm=norm,  # type: ignore[arg-type]
            dropout=dropout,
            mlp_hidden=mlp_hidden,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.encoder(x)


__all__ = [
    "build_laplacian_pyramid",
    "HealpixEmbeddingSpec",
    "HealpixEmbeddingNet",
]
