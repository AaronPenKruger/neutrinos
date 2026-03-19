"""Milestone 3 smoke test: Chebyshev graph convolution forward pass.

This example builds a simple 8-neighbour HEALPix graph (no PyGSP), constructs a
scaled Laplacian, converts it to ``torch.sparse``, and runs a ChebConv layer.

Requires:
  pip install -e '.[healpix,graph,torch]'
"""

from __future__ import annotations

import argparse

import numpy as np

from deephpx.graph import (
    adjacency_from_neighbors,
    laplacian_from_adjacency,
    neighbors_8,
    scale_laplacian_for_chebyshev,
    to_torch_sparse_coo,
)
from deephpx.nn import SphericalChebConv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=16)
    ap.add_argument("--nest", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--kind", type=str, default="normalized", choices=["normalized", "combinatorial"])
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--in-ch", type=int, default=1)
    ap.add_argument("--out-ch", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    nest = args.nest.lower() == "true"

    neigh = neighbors_8(args.nside, nest=nest)
    A = adjacency_from_neighbors(neigh, symmetric=True)
    L = laplacian_from_adjacency(A, kind=args.kind)
    L_tilde = scale_laplacian_for_chebyshev(L, kind_hint=args.kind)

    import torch

    L_t = to_torch_sparse_coo(L_tilde, dtype=torch.float32).coalesce()
    N = L_t.shape[0]

    x = torch.randn(args.batch, N, args.in_ch)
    layer = SphericalChebConv(L_t, args.in_ch, args.out_ch, args.K)

    y = layer(x)
    loss = y.pow(2).mean()
    loss.backward()

    print(f"nside={args.nside} nest={nest} kind={args.kind} K={args.K}")
    print(f"x: {tuple(x.shape)} -> y: {tuple(y.shape)}")
    print(f"loss={float(loss.detach().cpu()):.6f}")


if __name__ == "__main__":
    main()
