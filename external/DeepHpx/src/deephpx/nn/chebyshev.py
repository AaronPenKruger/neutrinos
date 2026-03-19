r"""Chebyshev (ChebNet-style) graph convolution.

This module implements the core Chebyshev polynomial approximation used in
spectral graph CNNs ("ChebNet"). The key idea is to operate on a *scaled*
Laplacian :math:`\tilde{L}` whose eigenvalues lie in [-1, 1].

Given an input signal X on the graph, the Chebyshev basis is defined by:

    T_0(X) = X
    T_1(X) = \tilde{L} X
    T_k(X) = 2 \tilde{L} T_{k-1}(X) - T_{k-2}(X)

The convolution output is:

    Y = \sum_{k=0}^{K-1} T_k(X) W_k + b

where W_k are learnable weight matrices.

Notes
-----
- This implementation is dependency-light: it uses only PyTorch, and expects
  the Laplacian as a ``torch.sparse`` matrix (COO or CSR).
- Use :func:`deephpx.graph.laplacian.scale_laplacian_for_chebyshev` to build
  the properly-scaled Laplacian from a SciPy sparse Laplacian.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math

import torch
from torch import nn


def _validate_laplacian(laplacian: torch.Tensor) -> Tuple[int, int]:
    if not laplacian.is_sparse:
        raise TypeError(
            "laplacian must be a torch sparse tensor (COO/CSR). "
            "Use deephpx.graph.laplacian.to_torch_sparse_coo(...) to convert from SciPy."
        )
    if laplacian.ndim != 2:
        raise ValueError(f"laplacian must be 2D, got ndim={laplacian.ndim}")
    n, m = laplacian.shape
    if n != m:
        raise ValueError(f"laplacian must be square, got shape={laplacian.shape}")
    # torch.sparse.mm expects sparse_dim=2 for COO; CSR is also supported.
    return n, m


def _canonicalize_x(x: torch.Tensor, n_nodes: int) -> Tuple[torch.Tensor, bool]:
    """Convert inputs to (B, N, Fin) and return (x3d, squeezed)."""
    if x.ndim == 3:
        if x.shape[1] != n_nodes:
            raise ValueError(
                f"Expected x.shape[1] == n_nodes ({n_nodes}), got x.shape={tuple(x.shape)}"
            )
        return x, False

    if x.ndim == 2:
        # Ambiguous case: (N, Fin) vs (B, N). Resolve using n_nodes.
        if x.shape[0] == n_nodes and x.shape[1] != n_nodes:
            return x.unsqueeze(0), True  # (1, N, Fin)
        if x.shape[1] == n_nodes and x.shape[0] != n_nodes:
            return x.unsqueeze(-1), False  # (B, N, 1)
        raise ValueError(
            "Ambiguous 2D input. Please pass x as (B, N, Fin) or (N, Fin) with Fin != N. "
            f"Got x.shape={tuple(x.shape)} and n_nodes={n_nodes}."
        )

    if x.ndim == 1:
        if x.shape[0] != n_nodes:
            raise ValueError(
                f"Expected x.shape[0] == n_nodes ({n_nodes}) for 1D input, got x.shape={tuple(x.shape)}"
            )
        return x.view(1, n_nodes, 1), True

    raise ValueError(f"x must be 1D/2D/3D, got ndim={x.ndim}")


def _spmm_batch(laplacian: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Sparse (N,N) @ dense (B,N,Fin) -> dense (B,N,Fin)."""
    n = laplacian.shape[0]
    if x.ndim != 3 or x.shape[1] != n:
        raise ValueError(
            f"x must have shape (B, N, Fin) with N={n}, got x.shape={tuple(x.shape)}"
        )

    # Ensure Laplacian is on the right device/dtype.
    if laplacian.device != x.device:
        laplacian = laplacian.to(device=x.device)
    if laplacian.dtype != x.dtype:
        laplacian = laplacian.to(dtype=x.dtype)

    # torch.sparse.mm expects 2D dense input.
    B, N, Fin = x.shape
    x2 = x.permute(1, 0, 2).reshape(N, B * Fin).contiguous()
    y2 = torch.sparse.mm(laplacian, x2)  # (N, B*Fin)
    y = y2.reshape(N, B, Fin).permute(1, 0, 2).contiguous()
    return y


def cheb_basis(laplacian: torch.Tensor, x: torch.Tensor, K: int) -> list[torch.Tensor]:
    """Compute Chebyshev basis {T_k(x)} for k=0..K-1.

    Args:
        laplacian: *scaled* Laplacian (torch sparse), shape (N, N).
        x: input features, shape (B, N, Fin).
        K: number of Chebyshev terms.

    Returns:
        basis: list of tensors, each shape (B, N, Fin).
    """
    if K <= 0:
        raise ValueError(f"K must be >= 1, got K={K}")

    _validate_laplacian(laplacian)
    if x.ndim != 3:
        raise ValueError(f"x must be 3D (B,N,Fin), got x.ndim={x.ndim}")
    if x.shape[1] != laplacian.shape[0]:
        raise ValueError(
            f"x.shape[1] must match laplacian size ({laplacian.shape[0]}), got x.shape={tuple(x.shape)}"
        )

    T0 = x
    if K == 1:
        return [T0]

    T1 = _spmm_batch(laplacian, x)
    basis = [T0, T1]

    for _ in range(2, K):
        T2 = 2.0 * _spmm_batch(laplacian, T1) - T0
        basis.append(T2)
        T0, T1 = T1, T2

    return basis


def cheb_conv(
    laplacian: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a Chebyshev graph convolution.

    Args:
        laplacian: *scaled* Laplacian (torch sparse), shape (N, N).
        x: input features. Supported shapes:
           - (B, N, Fin)
           - (N, Fin)  (interpreted as a single batch)
           - (B, N)    (interpreted as Fin=1)
           - (N,)      (interpreted as Fin=1 and a single batch)
        weight: tensor of shape (K, Fin, Fout).
        bias: optional bias of shape (Fout,).

    Returns:
        y: output features, shape matching input batching:
           - if x was (B, N, Fin) or (B, N): returns (B, N, Fout)
           - if x was (N, Fin) or (N,): returns (N, Fout)
    """
    n, _ = _validate_laplacian(laplacian)

    x3d, squeezed = _canonicalize_x(x, n_nodes=n)

    if weight.ndim != 3:
        raise ValueError(f"weight must have shape (K, Fin, Fout), got weight.ndim={weight.ndim}")
    K, Fin, Fout = weight.shape
    if x3d.shape[2] != Fin:
        raise ValueError(
            f"Input feature dim mismatch: x has Fin={x3d.shape[2]} but weight expects Fin={Fin}."
        )

    basis = cheb_basis(laplacian, x3d, K=K)
    y = torch.zeros((x3d.shape[0], n, Fout), dtype=x3d.dtype, device=x3d.device)

    for k in range(K):
        y = y + torch.matmul(basis[k], weight[k])

    if bias is not None:
        if bias.shape != (Fout,):
            raise ValueError(f"bias must have shape ({Fout},), got {tuple(bias.shape)}")
        y = y + bias.view(1, 1, Fout)

    return y.squeeze(0) if squeezed else y


class ChebConv(nn.Module):
    """Chebyshev graph convolution layer.

    This layer does not own a Laplacian; you pass it at forward time.
    If you want a fixed Laplacian (typical for HEALPix at a given NSIDE),
    use :class:`SphericalChebConv`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if K <= 0:
            raise ValueError(f"K must be >= 1, got K={K}")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.K = int(K)

        self.weight = nn.Parameter(torch.empty(self.K, self.in_channels, self.out_channels))
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier init per Chebyshev term.
        for k in range(self.K):
            nn.init.xavier_uniform_(self.weight[k])
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.out_channels)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, laplacian: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return cheb_conv(laplacian, x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, K={self.K}, bias={self.bias is not None}"


class SphericalChebConv(nn.Module):
    """Chebyshev convolution with a fixed Laplacian stored as a buffer."""

    def __init__(
        self,
        laplacian: torch.Tensor,
        in_channels: int,
        out_channels: int,
        K: int,
        *,
        bias: bool = True,
        coalesce_laplacian: bool = True,
    ) -> None:
        super().__init__()

        _validate_laplacian(laplacian)
        if coalesce_laplacian and laplacian.layout == torch.sparse_coo:
            laplacian = laplacian.coalesce()

        self.register_buffer("laplacian", laplacian)
        self.conv = ChebConv(in_channels, out_channels, K, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.laplacian, x)

    @property
    def weight(self) -> torch.Tensor:  # convenience
        return self.conv.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:  # convenience
        return self.conv.bias

    def extra_repr(self) -> str:
        n = int(self.laplacian.shape[0])
        return f"n_nodes={n}, {self.conv.extra_repr()}"
