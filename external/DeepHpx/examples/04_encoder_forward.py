"""Milestone 5 smoke test: HEALPixEncoder forward pass.

This script builds a *multi-resolution* HEALPix Laplacian pyramid and runs the
HealpixEncoder on random maps.

Dependencies:
  - deephpx[healpix,graph,torch]

Example:
  python examples/04_encoder_forward.py --nside 16 --levels 3 --channels 8,16,32 --embedding-dim 64
"""

from __future__ import annotations

import argparse

import torch

from deephpx.graph import (
    adjacency_from_neighbors,
    laplacian_from_adjacency,
    neighbors_8,
    scale_laplacian_for_chebyshev,
    to_torch_sparse_coo,
)
from deephpx.nn import HealpixEncoder
from deephpx.healpix.geometry import nside2npix


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_scaled_laplacian(nside: int, *, kind: str = "normalized") -> torch.Tensor:
    neigh = neighbors_8(nside, nest=True)
    A = adjacency_from_neighbors(neigh)
    L = laplacian_from_adjacency(A, kind=kind)  # SciPy CSR
    L_tilde = scale_laplacian_for_chebyshev(L, kind_hint=("normalized" if kind == "normalized" else None))
    return to_torch_sparse_coo(L_tilde, dtype=torch.float32).coalesce()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=16, help="Top-level NSIDE")
    p.add_argument("--levels", type=int, default=3, help="# of pyramid levels")
    p.add_argument("--channels", type=str, default="8,16,32", help="Conv channels per level")
    p.add_argument("--in-channels", type=int, default=1, help="Input channels per pixel")
    p.add_argument("--embedding-dim", type=int, default=64, help="Output embedding dim")
    p.add_argument("--K", type=int, default=3, help="Chebyshev order")
    p.add_argument("--pool", type=str, default="average", choices=["average", "max"])
    p.add_argument("--global-pool", type=str, default="mean", choices=["mean", "max", "meanmax"])
    p.add_argument("--kind", type=str, default="normalized", choices=["normalized", "combinatorial"])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")

    conv_channels = _parse_int_list(args.channels)
    if len(conv_channels) != args.levels:
        raise SystemExit("--channels must have the same number of entries as --levels")

    nsides = [args.nside // (2**i) for i in range(args.levels)]
    if any(n <= 0 for n in nsides):
        raise SystemExit(f"Invalid NSIDE pyramid computed: {nsides}")

    print(f"Building Laplacian pyramid for nsides={nsides} (kind={args.kind})")
    laplacians = [build_scaled_laplacian(n, kind=args.kind) for n in nsides]

    model = HealpixEncoder(
        laplacians,
        in_channels=args.in_channels,
        conv_channels=conv_channels,
        embedding_dim=args.embedding_dim,
        K=args.K,
        pool=args.pool,
        global_pool=args.global_pool,
        dropout=0.0,
        norm="layer",
        mlp_hidden=(),
        activation="relu",
    ).to(args.device)

    npix0 = nside2npix(nsides[0])
    x = torch.randn(args.batch, npix0, args.in_channels, device=args.device)

    z = model(x)
    print("x:", tuple(x.shape))
    print("z:", tuple(z.shape))

    # Backward sanity check
    loss = z.pow(2).mean()
    loss.backward()
    print("backward: ok")


if __name__ == "__main__":
    main()
