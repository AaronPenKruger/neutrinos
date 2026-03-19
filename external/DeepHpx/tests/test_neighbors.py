import numpy as np
import pytest


def test_neighbors_8_shape_and_range():
    hp = pytest.importorskip("healpy")  # noqa: F401

    from deephpx.graph.neighbors import neighbors_8
    from deephpx.healpix.geometry import nside2npix

    nside = 4
    npix = nside2npix(nside)

    neigh = neighbors_8(nside, nest=True)
    assert neigh.shape == (npix, 8)

    assert neigh.dtype.kind in {"i", "u"}

    # Values must be either -1 or a valid pixel index.
    assert np.all((neigh == -1) | ((neigh >= 0) & (neigh < npix)))

