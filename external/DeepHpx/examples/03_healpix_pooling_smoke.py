"""Smoke test for HEALPix pooling / unpooling.

This script does *not* require healpy. It simply demonstrates the tensor shape
changes implied by HEALPix hierarchical pooling:

* Pooling: pixels -> pixels/4
* Unpooling: pixels -> pixels*4

In real usage, ensure your HEALPix maps are in **NESTED** ordering before using
these layers.
"""

from __future__ import annotations

import argparse

import torch

from deephpx.nn import (
    HealpixAvgPool,
    HealpixAvgUnpool,
    HealpixMaxPool,
    HealpixMaxUnpool,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npix", type=int, default=12 * 32 * 32)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--features", type=int, default=3)
    p.add_argument("--mode", choices=["average", "max"], default="average")
    args = p.parse_args()

    B, P, F = args.batch, args.npix, args.features
    x = torch.randn(B, P, F)
    print(f"Input shape:  {tuple(x.shape)}")

    if args.mode == "average":
        pool = HealpixAvgPool()
        unpool = HealpixAvgUnpool()
        y = pool(x)
        z = unpool(y)
        print(f"Pooled shape: {tuple(y.shape)}")
        print(f"Unpooled shape: {tuple(z.shape)}")
        # Note: average unpooling is a simple repetition (tile). It's not an
        # inverse of average pooling.

    else:
        pool = HealpixMaxPool(return_indices=True)
        unpool = HealpixMaxUnpool()
        y, idx = pool(x)
        z = unpool(y, idx)
        print(f"Pooled shape: {tuple(y.shape)}")
        print(f"Indices shape: {tuple(idx.shape)}")
        print(f"Unpooled shape: {tuple(z.shape)}")
        # Note: max-unpool leaves non-max positions as 0.0
        nonzero = (z != 0).float().mean().item()
        print(f"Fraction nonzero after max-unpool: {nonzero:.6f}")


if __name__ == "__main__":
    main()
