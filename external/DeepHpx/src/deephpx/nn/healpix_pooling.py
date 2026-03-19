"""HEALPix pooling and unpooling layers.

This module implements the *sampling-side* pooling used in DeepSphere-style
HEALPix spherical CNNs.

Key idea
--------
Each pooling step divides **NSIDE by 2**. Since the number of pixels is

``npix = 12 * nside**2``,

this divides the number of pixels by **4** (and conversely unpooling multiplies
the number of pixels by 4).

These layers assume your map is in **NESTED** ordering.
In NESTED ordering, changing resolution corresponds to simple arithmetic on
pixel indices, which is exactly what makes fixed-kernel pooling feasible.

Shapes
------
DeepHpx follows DeepSphere's convention for spherical maps:

* ``(batch, pixels, features)`` for batched tensors
* ``(pixels, features)`` for unbatched tensors

Internally, PyTorch's 1D pooling layers operate on ``(batch, channels, length)``,
so we permute dimensions under the hood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import Tensor, nn


def _pixels_last_to_bcl(x: Tensor) -> Tuple[Tensor, Literal["3d", "2d", "1d"]]:
    """Convert (B,P,F) / (P,F) / (P,) into (B,C,L) for torch 1D pooling."""
    if x.ndim == 3:
        # (B, P, F) -> (B, F, P)
        return x.permute(0, 2, 1), "3d"
    if x.ndim == 2:
        # (P, F) -> (1, F, P)
        return x.t().unsqueeze(0), "2d"
    if x.ndim == 1:
        # (P,) -> (1, 1, P)
        return x.unsqueeze(0).unsqueeze(0), "1d"
    raise ValueError(
        "Expected x of shape (batch, pixels, features), (pixels, features), or (pixels,)"
    )


def _bcl_to_pixels_last(y: Tensor, mode: Literal["3d", "2d", "1d"]) -> Tensor:
    """Invert :func:`_pixels_last_to_bcl`."""
    if mode == "3d":
        # (B, F, P) -> (B, P, F)
        return y.permute(0, 2, 1)
    if mode == "2d":
        # (1, F, P) -> (P, F)
        return y.squeeze(0).t()
    if mode == "1d":
        # (1, 1, P) -> (P,)
        return y.squeeze(0).squeeze(0)
    raise ValueError(f"Invalid mode: {mode}")


def _check_divisible_by_4(num_pixels: int) -> None:
    if num_pixels % 4 != 0:
        raise ValueError(
            f"HEALPix pooling expects #pixels divisible by 4, got {num_pixels}. "
            "Did you ensure NESTED ordering and a power-of-two NSIDE?"
        )


class HealpixAvgPool(nn.AvgPool1d):
    """HEALPix average pooling.

    Semantics: group pixels in consecutive blocks of 4 and apply average pooling.

    This matches the DeepSphere documentation: HEALPix pooling divides NSIDE by 2
    (pixels / 4), corresponding to a fixed kernel size of 4 in 1D pooling.
    """

    def __init__(self) -> None:
        super().__init__(kernel_size=4, stride=4)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x_bcl, mode = _pixels_last_to_bcl(x)
        _check_divisible_by_4(x_bcl.shape[-1])
        y_bcl = super().forward(x_bcl)
        return _bcl_to_pixels_last(y_bcl, mode)


class HealpixAvgUnpool(nn.Module):
    """HEALPix average "unpooling" by simple repetition.

    DeepSphere's avg-unpool repeats values (like ``numpy.tile``) to go back to a
    higher resolution.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if x.ndim == 3:
            # (B, P, F) -> (B, 4P, F)
            return x.repeat_interleave(4, dim=1)
        if x.ndim == 2:
            # (P, F) -> (4P, F)
            return x.repeat_interleave(4, dim=0)
        if x.ndim == 1:
            return x.repeat_interleave(4)
        raise ValueError(
            "Expected x of shape (batch, pixels, features), (pixels, features), or (pixels,)"
        )


class HealpixMaxPool(nn.MaxPool1d):
    """HEALPix max pooling.

    If ``return_indices=True``, returns both pooled values and indices suitable
    for :class:`HealpixMaxUnpool`.
    """

    def __init__(self, *, return_indices: bool = False) -> None:
        super().__init__(kernel_size=4, stride=4, return_indices=return_indices)
        self._return_indices = bool(return_indices)

    def forward(self, x: Tensor):  # type: ignore[override]
        x_bcl, mode = _pixels_last_to_bcl(x)
        _check_divisible_by_4(x_bcl.shape[-1])

        if self._return_indices:
            y_bcl, idx_bcl = super().forward(x_bcl)
            y = _bcl_to_pixels_last(y_bcl, mode)
            idx = _bcl_to_pixels_last(idx_bcl, mode)
            return y, idx

        y_bcl = super().forward(x_bcl)
        return _bcl_to_pixels_last(y_bcl, mode)


class HealpixMaxUnpool(nn.MaxUnpool1d):
    """HEALPix max unpooling.

    Expects indices returned by :class:`HealpixMaxPool(return_indices=True)`.
    Note that max-unpooling fills non-max positions with zeros, mirroring
    PyTorch's :class:`torch.nn.MaxUnpool1d` semantics.
    """

    def __init__(self) -> None:
        super().__init__(kernel_size=4, stride=4)

    def forward(self, x: Tensor, indices: Tensor) -> Tensor:  # type: ignore[override]
        x_bcl, mode = _pixels_last_to_bcl(x)
        idx_bcl, mode_idx = _pixels_last_to_bcl(indices)
        if mode_idx != mode:
            raise ValueError(
                "x and indices must have compatible shapes. "
                f"Got x-mode={mode} and indices-mode={mode_idx}."
            )
        if idx_bcl.dtype != torch.int64:
            idx_bcl = idx_bcl.to(torch.int64)

        # With kernel=stride=4, output length is exactly 4 * input length.
        out_size = list(x_bcl.shape)
        out_size[-1] = out_size[-1] * 4
        y_bcl = super().forward(x_bcl, idx_bcl, output_size=tuple(out_size))
        return _bcl_to_pixels_last(y_bcl, mode)


@dataclass(frozen=True)
class Healpix:
    """Small convenience wrapper that groups pooling and unpooling.

    Mirrors the concept in DeepSphere's sampling modules.
    """

    mode: Literal["average", "max"] = "average"
    return_indices: bool = False

    @property
    def pooling(self):
        if self.mode == "average":
            return HealpixAvgPool()
        if self.mode == "max":
            return HealpixMaxPool(return_indices=self.return_indices)
        raise ValueError(f"Unsupported mode: {self.mode}")

    @property
    def unpooling(self):
        if self.mode == "average":
            return HealpixAvgUnpool()
        if self.mode == "max":
            return HealpixMaxUnpool()
        raise ValueError(f"Unsupported mode: {self.mode}")
