import importlib.util

import numpy as np
import pytest


torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    pytest.skip("torch not installed", allow_module_level=True)


def test_make_dataloaders_from_files_smoke(tmp_path):
    from deephpx.ili.torch_loader import SplitSpec, make_dataloaders_from_files, make_ili_torch_loader

    npix = 48
    n = 10

    paths = []
    for i in range(n):
        m = np.random.default_rng(i).normal(size=npix).astype(np.float32)
        p = tmp_path / f"m{i}.npy"
        np.save(p, m)
        paths.append(p)

    theta = np.random.default_rng(0).normal(size=(n, 2)).astype(np.float32)

    train_loader, val_loader = make_dataloaders_from_files(
        paths=paths,
        theta=theta,
        batch_size=4,
        split=SplitSpec(val_fraction=0.2, seed=0),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        dataset_kwargs={
            "output_ordering": "NEST",
            "assume_ordering": "NEST",
        },
    )

    assert train_loader is not None
    assert val_loader is not None

    xb, thetab = next(iter(train_loader))
    assert xb.ndim == 3
    assert xb.shape[1:] == (npix, 1)
    assert thetab.shape[-1] == 2

    # Wrapping in an LtU-ILI TorchLoader requires ili. If it's not installed,
    # ensure we raise a helpful error.
    if importlib.util.find_spec("ili") is None:
        with pytest.raises(ImportError):
            _ = make_ili_torch_loader(train_loader=train_loader, val_loader=val_loader)
    else:
        loader = make_ili_torch_loader(train_loader=train_loader, val_loader=val_loader)
        assert hasattr(loader, "train_loader")
        assert loader.train_loader is train_loader
