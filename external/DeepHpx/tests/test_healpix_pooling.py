import pytest


torch = pytest.importorskip("torch")


from deephpx.nn import (
    HealpixAvgPool,
    HealpixAvgUnpool,
    HealpixMaxPool,
    HealpixMaxUnpool,
)


def test_healpix_avg_pool_shapes_3d():
    B, P, F = 2, 12 * 8 * 8, 4
    x = torch.randn(B, P, F)
    pool = HealpixAvgPool()
    y = pool(x)
    assert y.shape == (B, P // 4, F)


def test_healpix_avg_unpool_repeats_3d():
    B, P, F = 2, 12 * 8 * 8, 3
    x = torch.randn(B, P // 4, F)
    unpool = HealpixAvgUnpool()
    z = unpool(x)
    assert z.shape == (B, P, F)
    # Each pooled pixel should be repeated exactly 4 times.
    z4 = z.view(B, P // 4, 4, F)
    assert torch.allclose(z4[:, :, 0, :], x)
    assert torch.allclose(z4[:, :, 1, :], x)
    assert torch.allclose(z4[:, :, 2, :], x)
    assert torch.allclose(z4[:, :, 3, :], x)


def test_healpix_max_pool_unpool_roundtrip_shapes_and_sparsity():
    # Use nonnegative values so that the max-unpool sanity checks are meaningful.
    B, P, F = 2, 12 * 8 * 8, 5
    x = torch.arange(B * P * F, dtype=torch.float32).reshape(B, P, F)
    pool = HealpixMaxPool(return_indices=True)
    unpool = HealpixMaxUnpool()

    y, idx = pool(x)
    assert y.shape == (B, P // 4, F)
    assert idx.shape == (B, P // 4, F)

    z = unpool(y, idx)
    assert z.shape == (B, P, F)

    # Max-unpool sets non-max entries to 0. For strictly increasing x, in each
    # group of 4 pixels the last one is the max, so each group should have
    # exactly 1 nonzero per feature.
    z4 = z.view(B, P // 4, 4, F)
    nonzero_per_group = (z4 != 0).sum(dim=2)  # (B, P//4, F)
    assert torch.all(nonzero_per_group == 1)


def test_healpix_pool_requires_divisible_by_4():
    x = torch.randn(1, 10, 1)
    with pytest.raises(ValueError):
        HealpixAvgPool()(x)
