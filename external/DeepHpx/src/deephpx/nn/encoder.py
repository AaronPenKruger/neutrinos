"""HEALPix encoders for SBI-style workflows.

This module provides a small *map -> embedding* network that stacks:

    (Chebyshev graph conv) -> (norm) -> (activation) -> (dropout) -> (pool)

repeated over a pyramid of HEALPix resolutions, followed by a global pooling
over pixels and an MLP head.

The encoder is designed to be used as an ``embedding_net`` in SBI/LtU-ILI
pipelines:

- Input:  HEALPix maps as tensors shaped ``(batch, pixels, channels)``.
- Output: a fixed-length embedding/context tensor shaped ``(batch, D)``.

Unlike DeepSphere, DeepHpx does *not* own any fragile PyGSP graph objects.
Instead, you build the (scaled) Laplacians via :mod:`deephpx.graph` and pass
them in as a list.

Important
---------
Pooling assumes **NESTED** ordering. If you read maps in RING ordering, convert
them first using :func:`deephpx.healpix.ordering.to_nested`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .chebyshev import SphericalChebConv
from .healpix_pooling import HealpixAvgPool, HealpixMaxPool


GlobalPool = Literal["mean", "max", "meanmax"]
PoolMode = Literal["average", "max"]
NormMode = Literal["layer", "none"]


def _canonicalize_map(x: Tensor) -> Tuple[Tensor, bool]:
    """Return (x3d, squeezed) where x3d is (B, P, C)."""
    if x.ndim == 3:
        return x, False
    if x.ndim == 2:
        # (P, C) -> (1, P, C)
        return x.unsqueeze(0), True
    if x.ndim == 1:
        # (P,) -> (1, P, 1)
        return x.view(1, -1, 1), True
    raise ValueError(
        f"Expected x shaped (B,P,C), (P,C), or (P,), got x.ndim={x.ndim} and shape={tuple(x.shape)}"
    )


def _make_activation(name: str) -> nn.Module:
    name = name.lower().strip()
    if name in {"relu", "relu_"}:
        return nn.ReLU()
    if name in {"gelu"}:
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unsupported activation {name!r}. Use 'relu', 'gelu', or 'silu'.")


def _global_pool(x: Tensor, mode: GlobalPool) -> Tensor:
    """Pool over pixels (dim=1) for x shaped (B, P, C)."""
    if x.ndim != 3:
        raise ValueError(f"Expected x to be 3D (B,P,C), got shape={tuple(x.shape)}")

    if mode == "mean":
        return x.mean(dim=1)
    if mode == "max":
        return x.max(dim=1).values
    if mode == "meanmax":
        return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
    raise ValueError(f"Unsupported global_pool {mode!r}")


def _make_mlp(
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int] = (),
    *,
    activation: nn.Module,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    dims = [int(in_dim), *[int(h) for h in hidden], int(out_dim)]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation.__class__())  # fresh module
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class HealpixEncoderSpec:
    """Configuration helper for :class:`HealpixEncoder`.

    This is optional; you can instantiate :class:`HealpixEncoder` directly.
    """

    conv_channels: Sequence[int]
    K: int = 3
    pool: PoolMode = "average"
    activation: str = "relu"
    norm: NormMode = "layer"
    dropout: float = 0.0
    global_pool: GlobalPool = "mean"
    mlp_hidden: Sequence[int] = ()


class HealpixEncoder(nn.Module):
    """Multi-resolution HEALPix encoder.

    Parameters
    ----------
    laplacians:
        List of *scaled* Laplacians (torch sparse), one per resolution level.
        The first Laplacian should match the highest resolution (largest npix).
        Each subsequent level is expected to have 1/4 the nodes of the previous
        level if you use HEALPix pooling.
    in_channels:
        Number of channels/features per pixel in the input map.
    conv_channels:
        Output channels of each ChebConv stage. Must have the same length as
        ``laplacians``.
    embedding_dim:
        Output embedding dimension.
    K:
        Chebyshev order (number of polynomial terms).
    pool:
        Pooling mode between stages: 'average' or 'max'. Pooling is applied
        after every stage except the last.
    global_pool:
        Pooling across pixels at the end: 'mean', 'max', or 'meanmax' (concat).
    norm:
        'layer' uses LayerNorm over the feature dimension; 'none' disables.
    dropout:
        Dropout probability applied after activation in each stage.
    mlp_hidden:
        Hidden layer sizes for the final MLP head.
    activation:
        Nonlinearity name: 'relu', 'gelu', or 'silu'.
    """

    def __init__(
        self,
        laplacians: Sequence[Tensor],
        *,
        in_channels: int,
        conv_channels: Sequence[int],
        embedding_dim: int,
        K: int = 3,
        pool: PoolMode = "average",
        global_pool: GlobalPool = "mean",
        norm: NormMode = "layer",
        dropout: float = 0.0,
        mlp_hidden: Sequence[int] = (),
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if len(laplacians) == 0:
            raise ValueError("laplacians must be a non-empty sequence")
        if len(conv_channels) != len(laplacians):
            raise ValueError(
                "conv_channels must have the same length as laplacians. "
                f"Got len(conv_channels)={len(conv_channels)} and len(laplacians)={len(laplacians)}."
            )
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if K <= 0:
            raise ValueError(f"K must be >= 1, got {K}")

        self.in_channels = int(in_channels)
        self.conv_channels = [int(c) for c in conv_channels]
        self.embedding_dim = int(embedding_dim)
        self.K = int(K)
        self.pool_mode: PoolMode = pool
        self.global_pool: GlobalPool = global_pool
        self.norm_mode: NormMode = norm
        self.dropout = float(dropout)

        act = _make_activation(activation)

        # Build conv stages.
        convs: List[nn.Module] = []
        norms: List[nn.Module] = []

        c_in = self.in_channels
        for L, c_out in zip(laplacians, self.conv_channels):
            convs.append(SphericalChebConv(L, in_channels=c_in, out_channels=c_out, K=self.K))
            if self.norm_mode == "layer":
                norms.append(nn.LayerNorm(c_out))
            elif self.norm_mode == "none":
                norms.append(nn.Identity())
            else:
                raise ValueError(f"Unsupported norm mode: {self.norm_mode!r}")
            c_in = c_out

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.act = act
        self.stage_dropout = nn.Dropout(p=self.dropout) if self.dropout > 0 else nn.Identity()

        if self.pool_mode == "average":
            self.pool = HealpixAvgPool()
        elif self.pool_mode == "max":
            self.pool = HealpixMaxPool(return_indices=False)
        else:
            raise ValueError(f"Unsupported pool mode: {self.pool_mode!r}")

        # Determine MLP input dim.
        last_c = self.conv_channels[-1]
        mlp_in = last_c * 2 if self.global_pool == "meanmax" else last_c
        self.mlp = _make_mlp(
            mlp_in,
            self.embedding_dim,
            hidden=mlp_hidden,
            activation=self.act,
            dropout=self.dropout,
        )

    @classmethod
    def from_spec(
        cls,
        laplacians: Sequence[Tensor],
        *,
        in_channels: int,
        embedding_dim: int,
        spec: HealpixEncoderSpec,
    ) -> "HealpixEncoder":
        return cls(
            laplacians,
            in_channels=in_channels,
            conv_channels=spec.conv_channels,
            embedding_dim=embedding_dim,
            K=spec.K,
            pool=spec.pool,
            global_pool=spec.global_pool,
            norm=spec.norm,
            dropout=spec.dropout,
            mlp_hidden=spec.mlp_hidden,
            activation=spec.activation,
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x3d, squeezed = _canonicalize_map(x)

        h = x3d
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h)
            h = norm(h)
            h = self.act(h)
            h = self.stage_dropout(h)
            if i < len(self.convs) - 1:
                h = self.pool(h)

        # h is (B, P_last, C_last)
        z = _global_pool(h, mode=self.global_pool)
        z = self.mlp(z)

        return z.squeeze(0) if squeezed else z
