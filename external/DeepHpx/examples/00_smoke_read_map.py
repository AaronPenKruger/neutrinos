#!/usr/bin/env python
"""Milestone 1 smoke test: load a HEALPix map and (optionally) convert ordering.

Usage:
  python examples/00_smoke_read_map.py /path/to/map.fits --to NEST
  python examples/00_smoke_read_map.py map.npy --assume-input-order RING --to NEST

This script prints:
- inferred NSIDE / NPIX
- dtype / shape
- min/max (nan-aware)
- ordering metadata (best-effort)
"""

from __future__ import annotations

import argparse
import numpy as np

from deephpx.healpix import npix2nside, read_healpix_map, to_nested, to_ring


def _nan_aware_minmax(x: np.ndarray):
    if isinstance(x, np.ma.MaskedArray):
        if x.count() == 0:
            return None, None
        return float(x.min()), float(x.max())
    if x.ndim == 2:
        # flatten channels for summary
        x = x.reshape(-1)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return None, None
    return float(finite.min()), float(finite.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Path to a HEALPix map (.fits/.fits.gz/.npy/.npz)")
    ap.add_argument(
        "--field",
        type=str,
        default="0",
        help="FITS field to read. Use '0' or '0,1,2' or 'all' (maps to field=None).",
    )
    ap.add_argument(
        "--nest",
        type=str,
        default=None,
        choices=[None, "true", "false", "none"],
        help="FITS-only. Pass through to healpy.read_map nest: true/false/none (no conversion).",
    )
    ap.add_argument(
        "--assume-input-order",
        type=str,
        default=None,
        help="For non-FITS inputs (or unknown FITS ordering), declare ordering: RING or NEST/NESTED.",
    )
    ap.add_argument(
        "--to",
        type=str,
        default=None,
        help="Convert ordering after read: RING or NEST/NESTED.",
    )
    ap.add_argument(
        "--ensure-2d",
        action="store_true",
        help="Ensure output has shape (npix, nch).",
    )
    args = ap.parse_args()

    # Parse field argument
    field = 0
    if args.field.lower() in {"all", "none"}:
        field = None
    elif "," in args.field:
        field = tuple(int(x.strip()) for x in args.field.split(",") if x.strip())
    else:
        field = int(args.field)

    # Parse nest passthrough
    nest = None
    if args.nest is not None:
        if args.nest.lower() == "true":
            nest = True
        elif args.nest.lower() == "false":
            nest = False
        elif args.nest.lower() == "none":
            nest = None

    m, meta, extra = read_healpix_map(
        args.path,
        field=field,
        nest=nest,
        assume_ordering=args.assume_input_order,
        ensure_2d=args.ensure_2d,
    )

    print("=== DeepHpx Milestone 1: map load ===")
    print(f"path      : {meta.path}")
    print(f"format    : {meta.format}")
    print(f"shape     : {getattr(m, 'shape', None)}")
    print(f"dtype     : {getattr(m, 'dtype', None)}")
    print(f"npix      : {meta.npix}")
    print(f"nside     : {meta.nside}")
    print(f"ordering* : {meta.ordering}   (*best-effort)")

    # Validate NPIX/Nside relationship
    assert meta.npix == 12 * meta.nside * meta.nside, "npix != 12*nside^2"

    mn, mx = _nan_aware_minmax(m)
    print(f"min/max   : {mn} / {mx}")

    if args.to is not None:
        target = args.to
        inp = meta.ordering or args.assume_input_order
        if inp is None:
            raise SystemExit(
                "Cannot convert ordering because input ordering is unknown. "
                "Pass --assume-input-order RING|NEST (or read FITS with header ORDERING)."
            )
        if target.strip().upper() in {"NEST", "NESTED"}:
            m2 = to_nested(m, inp=inp)
            out = "NEST"
        elif target.strip().upper() == "RING":
            m2 = to_ring(m, inp=inp)
            out = "RING"
        else:
            raise SystemExit("--to must be RING or NEST/NESTED")

        print("=== Ordering conversion ===")
        print(f"inp -> out: {inp} -> {out}")
        print(f"shape     : {m2.shape}")
        mn2, mx2 = _nan_aware_minmax(m2)
        print(f"min/max   : {mn2} / {mx2}")


if __name__ == "__main__":
    main()
