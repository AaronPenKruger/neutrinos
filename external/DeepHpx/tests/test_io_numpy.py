import numpy as np

from deephpx.healpix.geometry import nside2npix
from deephpx.healpix.io import read_healpix_map


def test_read_npy(tmp_path):
    nside = 4
    npix = nside2npix(nside)
    m = np.arange(npix, dtype=np.float32)
    p = tmp_path / "map.npy"
    np.save(p, m)

    out, meta, extra = read_healpix_map(p)
    assert out.shape == (npix,)
    assert meta.nside == nside
    assert meta.npix == npix
    assert meta.format == "npy"


def test_read_npz_default_key(tmp_path):
    nside = 2
    npix = nside2npix(nside)
    m = np.arange(npix, dtype=np.float64)
    p = tmp_path / "map.npz"
    np.savez(p, map=m)

    out, meta, extra = read_healpix_map(p)
    assert out.shape == (npix,)
    assert meta.nside == nside
    assert meta.format == "npz"
    assert extra.get("npz_key") == "map"
