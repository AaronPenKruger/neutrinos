import importlib.util

import numpy as np
import pytest


torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    pytest.skip("torch not installed", allow_module_level=True)


def test_healpix_file_dataset_reads_npy_and_caches(tmp_path):
    import torch

    from deephpx.ili.dataset import HealpixFileDataset

    npix = 48
    n = 3

    # Create simple maps on disk
    paths = []
    for i in range(n):
        m = np.full(npix, float(i), dtype=np.float32)
        p = tmp_path / f"m{i}.npy"
        np.save(p, m)
        paths.append(p)

    theta = np.arange(n * 2, dtype=np.float32).reshape(n, 2)

    cache_dir = tmp_path / "cache"

    ds = HealpixFileDataset(
        paths=paths,
        theta=theta,
        cache_dir=cache_dir,
        # For .npy maps we must specify the assumed ordering if we want
        # the dataset to guarantee an output ordering.
        output_ordering="NEST",
        assume_ordering="NEST",
    )

    x0, t0 = ds[0]
    assert isinstance(x0, torch.Tensor)
    assert isinstance(t0, torch.Tensor)
    assert x0.shape == (npix, 1)
    assert t0.shape == (2,)

    assert torch.allclose(x0[:, 0], torch.zeros(npix, dtype=torch.float32))
    assert torch.allclose(t0, torch.tensor(theta[0], dtype=torch.float32))

    # Cache should be populated
    cached = list(cache_dir.glob("*.npy"))
    assert len(cached) >= 1

    # Remove original file; dataset should still work via cache hit
    paths[0].unlink()

    x0b, t0b = ds[0]
    assert torch.allclose(x0b, x0)
    assert torch.allclose(t0b, t0)
