#!/usr/bin/env python
"""Build a HEALPix graph (Milestone 2 smoke test).

This example:
- builds the 8-neighbour list using healpy
- constructs a sparse adjacency
- constructs a Laplacian (normalized or combinatorial)
- optionally estimates lmax and produces a Chebyshev-scaled Laplacian

Run:
  python examples/01_build_graph.py --nside 32 --nest true --kind normalized
"""

from __future__ import annotations

import argparse

import numpy as np

from deephpx.healpix.geometry import nside2npix
from deephpx.graph import (
    neighbors_8,
    adjacency_from_neighbors,
    laplacian_from_adjacency,
    estimate_lmax,
    scale_laplacian_for_chebyshev,
)


def _str2bool(x: str) -> bool:
    x = x.strip().lower()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {x!r}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, required=True)
    p.add_argument("--nest", type=_str2bool, default=True)
    p.add_argument("--kind", choices=["normalized", "combinatorial"], default="normalized")
    p.add_argument("--estimate-lmax", action="store_true", help="Estimate lmax via scipy eigsh")
    p.add_argument("--lmax-method", choices=["eigsh", "power"], default="eigsh")
    args = p.parse_args()

    npix = nside2npix(args.nside)
    print(f"NSIDE={args.nside} -> NPIX={npix}")
    print(f"ordering={'NEST' if args.nest else 'RING'}")

    try:
        neigh = neighbors_8(args.nside, nest=args.nest)
    except ImportError as e:
        print("\nERROR: healpy not installed. Install with: pip install -e '.[healpix]'\n")
        raise

    print(f"neighbors shape: {neigh.shape}, dtype={neigh.dtype}")
    missing = int((neigh < 0).sum())
    print(f"missing neighbour entries: {missing}")

    A = adjacency_from_neighbors(neigh, symmetric=True)
    print(f"A: shape={A.shape}, nnz={A.nnz}")

    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    print(f"degree: min={deg.min():.0f}, mean={deg.mean():.2f}, max={deg.max():.0f}")

    L = laplacian_from_adjacency(A, kind=args.kind)
    print(f"L(kind={args.kind}): nnz={L.nnz}")

    lmax = None
    if args.estimate_lmax:
        lmax = estimate_lmax(L, method=args.lmax_method)
        print(f"estimated lmax ({args.lmax_method}): {lmax:.6g}")

    L_tilde = scale_laplacian_for_chebyshev(L, lmax=lmax, kind_hint=args.kind, lmax_method=args.lmax_method)
    print(f"L_tilde: nnz={L_tilde.nnz}")

    # Some quick sanity checks.
    sym_err = (L_tilde - L_tilde.T).power(2).sum()
    print(f"symmetry check (sum squared diff): {float(sym_err):.3e}")


if __name__ == "__main__":
    main()
