import numpy as np
import pytest

from deephpx.healpix.geometry import nside2npix
from deephpx.ili.loader import load_healpix_maps_to_numpy, make_ili_numpy_loader


def test_load_healpix_maps_to_numpy_ring_no_healpy(tmp_path):
    """Loading + stacking should work without healpy if no reorder is required."""
    nside = 4
    npix = nside2npix(nside)

    m0 = np.arange(npix, dtype=np.float32)
    m1 = np.arange(npix, dtype=np.float32)[::-1]

    p0 = tmp_path / "m0.npy"
    p1 = tmp_path / "m1.npy"
    np.save(p0, m0)
    np.save(p1, m1)

    X = load_healpix_maps_to_numpy(
        [p0, p1],
        target_ordering="RING",
        assume_ordering_nonfits="RING",
        ensure_2d=True,
    )

    assert X.shape == (2, npix, 1)
    assert np.allclose(X[0, :, 0], m0)
    assert np.allclose(X[1, :, 0], m1)


def test_load_healpix_maps_to_numpy_reorder_requires_healpy(tmp_path):
    """If a reorder is requested, healpy must be available."""
    hp = pytest.importorskip("healpy")

    nside = 2
    npix = nside2npix(nside)

    # save a simple RING-ordered map
    m_ring = np.arange(npix, dtype=np.float32)
    p = tmp_path / "m.npy"
    np.save(p, m_ring)

    X = load_healpix_maps_to_numpy(
        [p],
        target_ordering="NEST",
        assume_ordering_nonfits="RING",
        ensure_2d=True,
    )

    m_nest_ref = hp.pixelfunc.reorder(m_ring, inp="RING", out="NEST")
    assert X.shape == (1, npix, 1)
    assert np.allclose(X[0, :, 0], m_nest_ref)


def test_make_ili_numpy_loader_raises_if_missing_ili():
    X = np.zeros((2, 12, 1), dtype=np.float32)
    theta = np.zeros((2, 1), dtype=np.float32)

    # We *expect* ili not to be installed in the base test environment.
    with pytest.raises(ImportError):
        _ = make_ili_numpy_loader(X, theta)
