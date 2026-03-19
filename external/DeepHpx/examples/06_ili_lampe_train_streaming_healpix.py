"""Train a normalizing-flow posterior on streaming HEALPix maps (LtU-ILI + lampe).

This example demonstrates the *Milestone 7* feature set:

- stream HEALPix maps from disk via a PyTorch ``DataLoader``
- wrap the dataloaders in LtU-ILI's ``TorchLoader``
- train an NPE posterior using LtU-ILI's *lampe* runner
- use DeepHpx's HEALPix-aware embedding network

Why lampe?
----------
LtU-ILI's lampe backend consumes ``train_loader``/``val_loader`` directly when
present, enabling streaming training with large datasets (whereas the sbi runner
typically collects all training arrays into memory).

Run (toy dataset)
-----------------

```bash
pip install -e '.[healpix,graph,torch,ili]'
python examples/06_ili_lampe_train_streaming_healpix.py --mode toy --nside 16 --levels 3 \
  --channels 8,16,32 --embedding-dim 64 --num-samples 512 --batch-size 32 \
  --out-dir ./_out_streaming
```

Run (your own files)
--------------------

Assuming:
- you have maps on disk: ``/path/to/maps/*.fits`` (or ``.npy/.npz``)
- you have a parameter array: ``theta.npy`` with shape (N, D)

```bash
python examples/06_ili_lampe_train_streaming_healpix.py --mode files \
  --maps '/path/to/maps/*.fits' --theta /path/to/theta.npy \
  --levels 3 --channels 8,16,32 --embedding-dim 64 \
  --batch-size 32 --num-workers 4 --cache-dir /tmp/deephpx_cache
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _parse_int_list(csv: str) -> List[int]:
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


def _make_toy_files(out_dir: Path, *, nside: int, num_samples: int, theta_dim: int, seed: int = 0) -> Tuple[str, str]:
    """Create a toy dataset on disk: maps as .npy + theta.npy."""

    rng = np.random.default_rng(seed)

    npix = 12 * nside * nside

    data_dir = out_dir / "toy_data"
    maps_dir = data_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    # sample parameters
    theta = rng.uniform(low=-1.0, high=1.0, size=(num_samples, theta_dim)).astype(np.float32)

    # deterministic pixel coordinate to imprint parameter dependence
    pix = np.arange(npix, dtype=np.float32)
    base1 = np.sin(pix / 37.0)
    base2 = np.cos(pix / 53.0)

    for i in range(num_samples):
        t = theta[i]
        # 1-channel map
        m = (t[0] * base1 + (t[1] if theta_dim > 1 else 0.0) * base2)
        m = m + 0.10 * rng.normal(size=npix).astype(np.float32)
        np.save(maps_dir / f"map_{i:06d}.npy", m.astype(np.float32))

    theta_path = data_dir / "theta.npy"
    np.save(theta_path, theta)

    return str(maps_dir / "*.npy"), str(theta_path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["toy", "files"], default="toy")
    parser.add_argument("--out-dir", type=str, default="./_out_streaming")

    # data
    parser.add_argument("--maps", type=str, default="")
    parser.add_argument("--theta", type=str, default="")
    parser.add_argument("--cache-dir", type=str, default="")

    # healpix / embedding
    parser.add_argument("--nside", type=int, default=16)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--channels", type=str, default="8,16,32")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--K", type=int, default=3)

    # toy generation
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--theta-dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    # training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="maf", choices=["mdn", "maf", "nsf", "ncsf", "cnf", "nice", "gf", "sospf", "naf", "unaf"])
    parser.add_argument("--hidden-features", type=int, default=32)
    parser.add_argument("--num-transforms", type=int, default=3)

    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--stop-after-epochs", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=200)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Determine (maps_glob, theta_path)
    if args.mode == "toy":
        maps_glob, theta_path = _make_toy_files(
            out_dir,
            nside=args.nside,
            num_samples=args.num_samples,
            theta_dim=args.theta_dim,
            seed=args.seed,
        )
    else:
        if not args.maps or not args.theta:
            raise SystemExit("--maps and --theta are required in --mode files")
        maps_glob, theta_path = args.maps, args.theta

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # 2) Build an LtU-ILI TorchLoader from file paths
    from deephpx.ili import SplitSpec, make_ili_torch_loader_from_files

    loader = make_ili_torch_loader_from_files(
        maps=maps_glob,
        theta=theta_path,
        batch_size=args.batch_size,
        split=SplitSpec(val_fraction=0.1, seed=args.seed),
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
        cache_dir=cache_dir,
        # for .npy toy files we assume NEST ordering; for FITS this is inferred
        assume_ordering="NEST",
        output_ordering="NEST",
        field=0,
        memmap=False,
    )

    # 3) Build the DeepHpx HEALPix embedding network
    channels = _parse_int_list(args.channels)
    if len(channels) != args.levels:
        raise SystemExit("--channels must have exactly --levels entries")

    from deephpx.ili import HealpixEmbeddingNet

    embedding_net = HealpixEmbeddingNet(
        nside=args.nside,
        levels=args.levels,
        in_channels=1,
        conv_channels=channels,
        embedding_dim=args.embedding_dim,
        K=args.K,
        pool="average",
        global_pool="mean",
        laplacian_kind="normalized",
        nest=True,
        cache_dir=(out_dir / "laplacian_cache"),
        device="cpu",  # constructors in LtU-ILI call embedding on CPU first
    )

    # 4) Prior: uniform box
    # LtU-ILI exposes a convenience Uniform wrapper in ili.utils.
    import ili

    # Use a *vector* prior (shape == theta_dim). Passing scalar low/high to
    # an Independent distribution can be ambiguous.
    tpath = Path(theta_path)
    if tpath.suffix.lower() == ".npy":
        theta_arr = np.load(tpath, allow_pickle=False)
    elif tpath.suffix.lower() == ".npz":
        z = np.load(tpath, allow_pickle=False)
        key = "theta" if "theta" in z.files else (z.files[0] if z.files else None)
        if key is None:
            raise SystemExit(f"No arrays found in {tpath}")
        theta_arr = z[key]
    else:
        raise SystemExit("--theta must be a .npy or .npz file")

    if theta_arr.ndim == 1:
        theta_dim = 1
    else:
        theta_dim = int(theta_arr.shape[1])

    low = -np.ones(theta_dim, dtype=np.float32)
    high = np.ones(theta_dim, dtype=np.float32)
    prior = ili.utils.Uniform(low=low, high=high, device=args.device)

    # 5) Train with lampe backend
    from deephpx.ili import train_lampe_posterior

    posterior, summaries, runner = train_lampe_posterior(
        loader,
        prior=prior,
        embedding_net=embedding_net,
        engine="NPE",
        model=args.model,
        nde_kwargs={
            "hidden_features": args.hidden_features,
            "num_transforms": args.num_transforms,
            "x_normalize": True,
            "theta_normalize": True,
        },
        train_args={
            "learning_rate": args.learning_rate,
            "stop_after_epochs": args.stop_after_epochs,
            "max_epochs": args.max_epochs,
            # validation_fraction is unused when we provide train/val loaders,
            # but leaving it here for completeness.
            "validation_fraction": 0.1,
        },
        out_dir=out_dir,
        device=args.device,
        name="deephpx_streaming_",
        seed=args.seed,
    )

    print("\nTraining complete.")
    print("Posterior type:", type(posterior))
    print("Num summary entries:", len(summaries))

    # 6) Draw a few posterior samples for the first (val) sample
    # (Note: this is just a smoke test.)
    # We'll read one cached map via the dataset in the train loader.
    x0, _t0 = next(iter(loader.train_loader))
    x0 = x0[0].to(args.device)

    # posterior.sample signature in lampe wrapper is (shape, x)
    try:
        samples = posterior.sample((256,), x=x0)
    except TypeError:
        # Some wrappers follow the sbi signature: sample(sample_shape=(...), x=...)
        samples = posterior.sample(sample_shape=(256,), x=x0)
    print("Sample shape:", tuple(samples.shape))


if __name__ == "__main__":
    main()
