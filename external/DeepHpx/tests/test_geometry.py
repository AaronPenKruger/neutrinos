import pytest

from deephpx.healpix.geometry import nside2npix, npix2nside


def test_nside_npix_roundtrip():
    for nside in [1, 2, 4, 8, 16, 32]:
        npix = nside2npix(nside)
        assert npix2nside(npix) == nside


def test_npix2nside_invalid_raises():
    with pytest.raises(ValueError):
        npix2nside(1000, strict=True)
