#!/usr/bin/env python
"""Train an SBI normalizing flow on HEALPix maps via LtU-ILI.

This is a *minimal* end-to-end example showing how to combine:

- DeepHpx: HEALPix graph encoder (embedding_net)
- LtU-ILI: training harness (InferenceRunner / SBIRunner)
- sbi: conditional normalizing flow posterior

Two modes are supported:
1) ``--mode toy``: generate a small synthetic dataset of HEALPix maps.
2) ``--mode files``: load maps from disk (FITS/NPY/NPZ) + a theta.npy file.

Requirements
------------
- `pip install -e '.[healpix,graph,torch,ili]'`

Notes
-----
- DeepHpx pooling assumes **NESTED** ordering. We generate toy maps in NEST
  ordering and we default to converting loaded maps to NEST.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import glob

import numpy as np
import torch

from deephpx.ili import (
    HealpixEmbeddingNet,
    make_ili_numpy_loader,
    make_ili_numpy_loader_from_files,
    train_sbi_posterior,
)
from deephpx.healpix.geometry import npix2nside


def _require_healpy():
    try:
        import healpy as hp  # type: ignore
    except Exception as e:
        raise SystemExit(
            "This example requires healpy. Install with `pip install deephpx[healpix]`."
        ) from e
    return hp


def make_toy_dataset(*, nside: int, n_samples: int, noise: float, seed: int = 0):
    """Create (X, theta) where X is (N, npix, 1) in NEST ordering.

    We use a few low-order spherical patterns to make maps nontrivial.
    """
    hp = _require_healpy()

    rng = np.random.default_rng(seed)

    npix = hp.nside2npix(nside)
    pix = np.arange(npix, dtype=np.int64)
    # Angles for NEST ordering
    ang_theta, ang_phi = hp.pix2ang(nside, pix, nest=True)

    # Basis functions (simple, not a strict spherical-harmonic basis)
    b0 = np.cos(ang_theta)
    b1 = np.sin(ang_theta) * np.cos(ang_phi)
    b2 = np.sin(ang_theta) * np.sin(ang_phi)

    # Parameters in [-1, 1]^3
    theta = rng.uniform(-1.0, 1.0, size=(n_samples, 3)).astype(np.float32)

    X = np.empty((n_samples, npix, 1), dtype=np.float32)
    for i in range(n_samples):
        a0, a1, a2 = theta[i]
        m = a0 * b0 + a1 * b1 + a2 * b2
        m = m + rng.normal(0.0, noise, size=m.shape)
        X[i, :, 0] = m.astype(np.float32)

    return X, theta


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["toy", "files"], default="toy")
    parser.add_argument("--out-dir", type=str, default="./runs_m6")
    parser.add_argument("--device", type=str, default="cpu")

    # Data options
    parser.add_argument("--nside", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=512)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)

    # File mode options
    parser.add_argument("--maps-glob", type=str, default="")
    parser.add_argument("--theta-npy", type=str, default="")

    # Prior options (comma-separated). If omitted:
    # - toy mode defaults to [-1, 1]^D
    # - file mode uses min/max from theta.npy (with a small margin)
    parser.add_argument("--prior-low", type=str, default="")
    parser.add_argument("--prior-high", type=str, default="")

    # Encoder options
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--conv-channels", type=str, default="8,16,32")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--K", type=int, default=3)

    # LtU-ILI / SBI options
    parser.add_argument("--model", type=str, default="maf", choices=["maf", "nsf", "mdn"])
    parser.add_argument("--stop-after-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conv_channels = [int(s) for s in args.conv_channels.split(",") if s.strip()]

    # ---- Load / build data ----
    if args.mode == "toy":
        X, theta = make_toy_dataset(
            nside=args.nside,
            n_samples=args.n_samples,
            noise=args.noise,
            seed=args.seed,
        )
        loader = make_ili_numpy_loader(X, theta)
    else:
        if not args.maps_glob or not args.theta_npy:
            raise SystemExit("--mode files requires --maps-glob and --theta-npy")
        map_paths = sorted(glob.glob(args.maps_glob))
        if not map_paths:
            raise SystemExit(f"No files matched --maps-glob={args.maps_glob!r}")
        loader = make_ili_numpy_loader_from_files(map_paths, args.theta_npy)

    # ---- Inspect dataset shapes ----
    X_all = loader.get_all_data()
    theta_all = loader.get_all_parameters()

    if X_all.ndim != 3:
        raise SystemExit(f"Expected X to have shape (N, npix, nch), got {X_all.shape}")
    if theta_all.ndim == 1:
        theta_all = theta_all.reshape(-1, 1)
    if theta_all.ndim != 2:
        raise SystemExit(f"Expected theta to have shape (N, D), got {theta_all.shape}")

    nside_data = npix2nside(int(X_all.shape[1]), strict=True)
    in_channels = int(X_all.shape[2])
    param_dim = int(theta_all.shape[1])

    # ---- Build embedding network ----
    embedding_net = HealpixEmbeddingNet(
        nside=nside_data,
        levels=args.levels,
        in_channels=in_channels,
        conv_channels=conv_channels,
        embedding_dim=args.embedding_dim,
        K=args.K,
        pool="average",
        global_pool="mean",
        norm="layer",
        dropout=0.0,
        mlp_hidden=(128,),
        activation="relu",
        laplacian_kind="normalized",
        nest=True,
        cache_dir=out_dir / "laplacian_cache",
    )

    # ---- Prior over parameters ----
    if args.prior_low and args.prior_high:
        low_vals = [float(s) for s in args.prior_low.split(",") if s.strip()]
        high_vals = [float(s) for s in args.prior_high.split(",") if s.strip()]
        if len(low_vals) != param_dim or len(high_vals) != param_dim:
            raise SystemExit(
                f"--prior-low/--prior-high lengths must match param_dim={param_dim}. "
                f"Got len(low)={len(low_vals)}, len(high)={len(high_vals)}"
            )
        low = torch.tensor(low_vals)
        high = torch.tensor(high_vals)
    else:
        if args.mode == "toy":
            low = torch.full((param_dim,), -1.0)
            high = torch.full((param_dim,), 1.0)
        else:
            # Best-effort default: use min/max bounds from the provided theta.
            lo = np.min(theta_all, axis=0)
            hi = np.max(theta_all, axis=0)
            # Add a small margin to avoid zero-width bounds.
            eps = 1e-3
            low = torch.tensor(lo - eps, dtype=torch.float32)
            high = torch.tensor(hi + eps, dtype=torch.float32)
    prior = torch.distributions.Uniform(low=low, high=high)

    # ---- Train posterior via LtU-ILI ----
    train_args = {
        "stop_after_epochs": int(args.stop_after_epochs),
        "training_batch_size": int(args.batch_size),
        "validation_fraction": 0.1,
    }

    posterior, summaries, runner = train_sbi_posterior(
        loader,
        prior=prior,
        embedding_net=embedding_net,
        engine="NPE",
        model=args.model,
        train_args=train_args,
        out_dir=out_dir,
        device=args.device,
        name="deephpx_healpix",
        seed=args.seed,
    )

    print("Training complete. Posterior type:", type(posterior))

    # ---- Quick posterior sample sanity check ----
    # Use the first training example as a pseudo-observation.
    x0 = X_all[0]
    x0_t = torch.as_tensor(x0, dtype=torch.float32)

    with torch.no_grad():
        samples = posterior.sample(sample_shape=(1000,), x=x0_t)

    print("Posterior sample mean:", samples.mean(dim=0))
    print("Posterior sample std:", samples.std(dim=0))
    print("Done.")


if __name__ == "__main__":
    main()
