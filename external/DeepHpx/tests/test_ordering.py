import numpy as np
import pytest

hp = pytest.importorskip("healpy")

from deephpx.healpix.geometry import nside2npix
from deephpx.healpix.ordering import reorder, to_nested, to_ring


def test_ordering_roundtrip_1d():
    nside = 4
    npix = nside2npix(nside)
    m = np.arange(npix, dtype=np.float64)

    m_nest = to_nested(m, inp="RING")
    m_ring = to_ring(m_nest, inp="NEST")

    assert m_ring.shape == m.shape
    assert np.all(m_ring == m)


def test_ordering_2d_pix_chan():
    nside = 2
    npix = nside2npix(nside)
    m = np.stack([np.arange(npix), np.arange(npix)[::-1]], axis=1)  # (npix,2)

    m_nest = reorder(m, inp="RING", out="NEST")
    m_ring = reorder(m_nest, inp="NEST", out="RING")

    assert m_ring.shape == m.shape
    assert np.all(m_ring == m)


def test_ordering_2d_chan_pix():
    nside = 2
    npix = nside2npix(nside)
    m = np.stack([np.arange(npix), np.arange(npix)[::-1]], axis=0)  # (2,npix)

    m_nest = reorder(m, inp="RING", out="NEST")
    m_ring = reorder(m_nest, inp="NEST", out="RING")

    assert m_ring.shape == m.shape
    assert np.all(m_ring == m)
