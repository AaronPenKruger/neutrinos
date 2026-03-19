# IceCube_expected_nu_counts.py
"""
Population -> IceCube expected-counts HEALPix maps (energy-binned), using icecube_tools public IRFs.

IMPORTANT UPDATE (2025-12):
---------------------------
Mean-map PSF deposition is DEPRECATED and REMOVED.

This module now supports only a consistent stochastic forward model:
  1) Build pre-PSF *mean* maps in TRUE direction (delta sources in pixels).
  2) Poisson-sample counts in true-direction pixels.
  3) Optionally apply event-level PSF jitter (IRF-weighted) and rebin into reconstructed pixels.

This is consistent with the atmospheric pipeline ("Poisson -> event-level PSF -> rebin") and avoids
mean-map smoothing shortcuts/pixelization artifacts.

Dependencies
------------
- nu_pop_core.py (CosmologyGrid, SpectrumParams, PopulationParams, k_gamma, sample_redshifts,
  expected_total_number_of_sources)
- icecube_tools >= 3.2 (EffectiveArea + R2021IRF)
- numpy, scipy, healpy

Notes on IRFs
-------------
- Effective area: EffectiveArea.from_dataset("20210126", period), values typically in m^2.
- Angular IRF: R2021IRF.from_period(period) provides sampling of angular error ("AngErr").
  We precompute a robust grid of median angular errors and build an interpolator σ(E, δ).

Output objects
--------------
- make_expected_counts_maps(...) returns dict with:
    'maps_mean_prepsf' : float [B, npix]    # true-direction expected counts, no PSF
    'psf_aux'          : dict               # caches for event-level PSF sampling (if enabled)
    plus metadata and the drawn catalog.

- poissonize_astro(...) returns dict with:
    'maps_counts'      : int [B, npix]      # reconstructed counts if PSF is used
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Dict, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    import healpy as hp
except ImportError as e:
    raise ImportError("Please `pip install healpy`.") from e

from nu_pop_core import (
    CosmologyGrid,
    SpectrumParams,
    PopulationParams,
    k_gamma,
    sample_redshifts,
    expected_total_number_of_sources,
)

from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.r2021 import R2021IRF


# -------------------- constants & helpers --------------------
CM_PER_M = 100.0
CM2_PER_M2 = CM_PER_M**2
GEV_PER_ERG = 624.150907446         # 1 erg = 624.1509 GeV
MPC_TO_CM = 3.085677581491367e24    # 1 Mpc in cm
FOURPI = 4.0 * np.pi
Z_FLOOR_DEFAULT = 1e-6               # avoid z=0 singularities


def safe_log10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log10(np.clip(x, 1e-300, None))


def _broadcast_to_common_size(*arrs: np.ndarray) -> Tuple[np.ndarray, ...]:
    sizes = [int(a.size) for a in arrs]
    n = max(sizes) if sizes else 0
    out = []
    for a in arrs:
        if a.size == n:
            out.append(a)
        elif a.size == 1 and n > 1:
            out.append(np.full(n, float(a.item()), dtype=float))
        else:
            raise ValueError(f"Incompatible sample_energy output sizes: {sizes}")
    return tuple(out)


def _parse_sample_energy_output(
    sample_out,
    coord_fallback: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize possible `R2021IRF.sample_energy` return signatures to:
      (ra_rec_rad, dec_rec_rad, ang_deg, Erec_GeV)
    """
    # Common tuple signatures:
    #   ((ra, dec), ang, Erec)
    #   ((ra, dec), ang, ..., Erec)
    #   (ra, dec, ang, Erec)
    #   (ra, dec, ang, ..., Erec)
    if isinstance(sample_out, tuple):
        if len(sample_out) >= 3 and isinstance(sample_out[0], (tuple, list)) and len(sample_out[0]) == 2:
            ra_rec = np.asarray(sample_out[0][0], dtype=float).reshape(-1)
            dec_rec = np.asarray(sample_out[0][1], dtype=float).reshape(-1)
            tail = sample_out[1:]
            if len(tail) == 1:
                ang_deg = np.zeros_like(ra_rec, dtype=float)
                Erec_GeV = np.asarray(tail[0], dtype=float).reshape(-1)
            else:
                ang_deg = np.asarray(tail[0], dtype=float).reshape(-1)
                Erec_GeV = np.asarray(tail[-1], dtype=float).reshape(-1)
            return _broadcast_to_common_size(ra_rec, dec_rec, ang_deg, Erec_GeV)
        if len(sample_out) >= 3:
            ra_rec = np.asarray(sample_out[0], dtype=float).reshape(-1)
            dec_rec = np.asarray(sample_out[1], dtype=float).reshape(-1)
            if len(sample_out) == 3:
                ang_deg = np.zeros_like(ra_rec, dtype=float)
                Erec_GeV = np.asarray(sample_out[2], dtype=float).reshape(-1)
            else:
                ang_deg = np.asarray(sample_out[2], dtype=float).reshape(-1)
                Erec_GeV = np.asarray(sample_out[-1], dtype=float).reshape(-1)
            return _broadcast_to_common_size(ra_rec, dec_rec, ang_deg, Erec_GeV)
        raise ValueError(f"Unexpected sample_energy tuple length: {len(sample_out)}")

    # Dict / mapping signature
    if isinstance(sample_out, dict):
        keys = {k.lower(): k for k in sample_out.keys()}
        def _pick(cands):
            for c in cands:
                if c in keys:
                    return sample_out[keys[c]]
            return None

        ra_v = _pick(["ra", "ra_reco", "ra_rec"])
        dec_v = _pick(["dec", "dec_reco", "dec_rec"])
        ang_v = _pick(["ang", "ang_deg", "angerr", "ang_err", "deflection", "deflection_deg"])
        erec_v = _pick(["erec", "e_rec", "e_reco", "ereco", "energy_reco", "reco_energy", "energy"])
        if ra_v is not None and dec_v is not None and erec_v is not None:
            ra_rec = np.asarray(ra_v, dtype=float).reshape(-1)
            dec_rec = np.asarray(dec_v, dtype=float).reshape(-1)
            ang_deg = np.asarray(ang_v if ang_v is not None else np.zeros_like(ra_rec), dtype=float).reshape(-1)
            Erec_GeV = np.asarray(erec_v, dtype=float).reshape(-1)
            return _broadcast_to_common_size(ra_rec, dec_rec, ang_deg, Erec_GeV)

    arr = np.asarray(sample_out)

    # Object-vector variant, e.g. np.array([ra, dec, ang, Erec], dtype=object)
    if arr.ndim == 1 and arr.dtype == object and arr.size > 0 and isinstance(arr[0], (tuple, list, np.ndarray)):
        rows = np.asarray(list(arr), dtype=float)
        if rows.ndim == 2 and rows.shape[1] >= 3:
            ra_rec = np.asarray(rows[:, 0], dtype=float).reshape(-1)
            dec_rec = np.asarray(rows[:, 1], dtype=float).reshape(-1)
            if rows.shape[1] == 3:
                Erec_GeV = np.asarray(rows[:, 2], dtype=float).reshape(-1)
                ang_deg = np.zeros_like(ra_rec, dtype=float)
            else:
                ang_deg = np.asarray(rows[:, 2], dtype=float).reshape(-1)
                Erec_GeV = np.asarray(rows[:, -1], dtype=float).reshape(-1)
            return _broadcast_to_common_size(ra_rec, dec_rec, ang_deg, Erec_GeV)

    if arr.ndim == 1 and arr.dtype == object and arr.size >= 4:
        ra_rec = np.asarray(arr[0], dtype=float).reshape(-1)
        dec_rec = np.asarray(arr[1], dtype=float).reshape(-1)
        ang_deg = np.asarray(arr[2], dtype=float).reshape(-1)
        Erec_GeV = np.asarray(arr[-1], dtype=float).reshape(-1)
        return _broadcast_to_common_size(ra_rec, dec_rec, ang_deg, Erec_GeV)
    if arr.ndim == 1 and arr.dtype == object and arr.size == 3:
        ra_rec = np.asarray(arr[0], dtype=float).reshape(-1)
        dec_rec = np.asarray(arr[1], dtype=float).reshape(-1)
        Erec_GeV = np.asarray(arr[2], dtype=float).reshape(-1)
        ang_deg = np.zeros_like(ra_rec, dtype=float)
        return _broadcast_to_common_size(ra_rec, dec_rec, ang_deg, Erec_GeV)

    # Structured array with named fields
    if arr.dtype.names:
        names = {n.lower(): n for n in arr.dtype.names}
        def _f(cands):
            for c in cands:
                if c in names:
                    return arr[names[c]]
            return None
        ra_v = _f(["ra", "ra_reco", "ra_rec"])
        dec_v = _f(["dec", "dec_reco", "dec_rec"])
        ang_v = _f(["ang", "ang_deg", "angerr", "ang_err", "deflection", "deflection_deg"])
        erec_v = _f(["erec", "e_rec", "e_reco", "ereco", "energy_reco", "reco_energy", "energy"])
        if ra_v is not None and dec_v is not None and erec_v is not None:
            ra_rec = np.asarray(ra_v, dtype=float).reshape(-1)
            dec_rec = np.asarray(dec_v, dtype=float).reshape(-1)
            ang_deg = np.asarray(ang_v if ang_v is not None else np.zeros_like(ra_rec), dtype=float).reshape(-1)
            Erec_GeV = np.asarray(erec_v, dtype=float).reshape(-1)
            return _broadcast_to_common_size(ra_rec, dec_rec, ang_deg, Erec_GeV)

    # Numeric 2D variants:
    #   shape (K, N): rows hold components
    #   shape (N, K): columns hold components
    if arr.ndim == 2:
        if arr.shape[0] >= 4:
            ra_rec = np.asarray(arr[0], dtype=float).reshape(-1)
            dec_rec = np.asarray(arr[1], dtype=float).reshape(-1)
            ang_deg = np.asarray(arr[2], dtype=float).reshape(-1)
            Erec_GeV = np.asarray(arr[-1], dtype=float).reshape(-1)
            return _broadcast_to_common_size(ra_rec, dec_rec, ang_deg, Erec_GeV)
        if arr.shape[1] >= 4:
            ra_rec = np.asarray(arr[:, 0], dtype=float).reshape(-1)
            dec_rec = np.asarray(arr[:, 1], dtype=float).reshape(-1)
            ang_deg = np.asarray(arr[:, 2], dtype=float).reshape(-1)
            Erec_GeV = np.asarray(arr[:, -1], dtype=float).reshape(-1)
            return _broadcast_to_common_size(ra_rec, dec_rec, ang_deg, Erec_GeV)
        if arr.shape[0] == 3:
            ra_rec = np.asarray(arr[0], dtype=float).reshape(-1)
            dec_rec = np.asarray(arr[1], dtype=float).reshape(-1)
            Erec_GeV = np.asarray(arr[2], dtype=float).reshape(-1)
            ang_deg = np.zeros_like(ra_rec, dtype=float)
            return _broadcast_to_common_size(ra_rec, dec_rec, ang_deg, Erec_GeV)
        if arr.shape[1] == 3:
            ra_rec = np.asarray(arr[:, 0], dtype=float).reshape(-1)
            dec_rec = np.asarray(arr[:, 1], dtype=float).reshape(-1)
            Erec_GeV = np.asarray(arr[:, 2], dtype=float).reshape(-1)
            ang_deg = np.zeros_like(ra_rec, dtype=float)
            return _broadcast_to_common_size(ra_rec, dec_rec, ang_deg, Erec_GeV)

    # Energy-only output variant: use true coords as reco coords fallback.
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number) and coord_fallback is not None:
        ra_fb = np.asarray(coord_fallback[0], dtype=float).reshape(-1)
        dec_fb = np.asarray(coord_fallback[1], dtype=float).reshape(-1)
        Erec_GeV = np.asarray(arr, dtype=float).reshape(-1)
        ang_deg = np.zeros_like(Erec_GeV, dtype=float)
        return _broadcast_to_common_size(ra_fb, dec_fb, ang_deg, Erec_GeV)

    raise TypeError(
        "Unsupported sample_energy return format. "
        f"type={type(sample_out)!r}, ndarray_shape={getattr(arr, 'shape', None)}"
    )


def coszen_from_dec(dec_rad) -> np.ndarray:
    """South Pole convention: cos(zenith) = -sin(dec)."""
    return -np.sin(np.asarray(dec_rad, dtype=float))


def ensure_spec_GeV(spec: SpectrumParams, L_unit: str = "erg/s") -> SpectrumParams:
    """Convert L to GeV/s so E[GeV] integrals are consistent."""
    if str(L_unit).lower().startswith("erg"):
        return replace(spec, L=float(spec.L) * GEV_PER_ERG)
    return spec


# -------------------- Eq. (6) in correct units (GeV^-1 cm^-2 s^-1) --------------------
def phi_earth_cm(E_GeV: np.ndarray, z: float, cosmo: CosmologyGrid, spec_GeV: SpectrumParams) -> np.ndarray:
    """
    Single-source flux at Earth:
      φ(E;z) = [ L * k_γ * (1+z)^(2-γ) / (4π D_L(z)^2) ] * E^{-γ}
    where L in GeV/s, E in GeV, D_L in cm.
    """
    E = np.asarray(E_GeV, dtype=float)
    Dl_cm = cosmo.luminosity_distance(np.atleast_1d(float(z)))[0] * MPC_TO_CM  # Mpc->cm
    kg = k_gamma(spec_GeV.gamma, spec_GeV.Emin, spec_GeV.Emax)
    pref = float(spec_GeV.L) * kg * (1.0 + float(z)) ** (2.0 - float(spec_GeV.gamma)) / (FOURPI * (Dl_cm**2))
    return pref * (E ** (-float(spec_GeV.gamma)))


# -------------------- Effective area interpolator --------------------
@dataclass
class AeffInterpolator:
    dataset: str = "20210126"
    period: str = "IC86_II"
    units: str = "m2"   # 'm2' (public tables) or 'cm2'

    def __post_init__(self):
        aeff = EffectiveArea.from_dataset(self.dataset, self.period)  # values usually in m^2
        self.E_edges = np.asarray(aeff.true_energy_bins, dtype=float)   # GeV (edges)
        self.cz_edges = np.asarray(aeff.cos_zenith_bins, dtype=float)   # [-1,1] (edges)
        self.values = np.asarray(aeff.values, dtype=float)              # (nE-1, nCZ-1)

        self.to_cm2 = (CM2_PER_M2 if str(self.units).lower() == "m2" else 1.0)

        self.logE_centers = safe_log10(0.5 * (self.E_edges[:-1] + self.E_edges[1:]))
        self.cz_centers = 0.5 * (self.cz_edges[:-1] + self.cz_edges[1:])

        self._interp = RegularGridInterpolator(
            (self.logE_centers, self.cz_centers),
            self.values,
            bounds_error=False,
            fill_value=0.0,
        )

    def __call__(self, E_GeV: np.ndarray, dec_rad) -> np.ndarray:
        E = np.asarray(E_GeV, dtype=float)
        cz = coszen_from_dec(dec_rad)
        cz_b = np.broadcast_to(cz, E.shape).ravel()
        pts = np.column_stack([safe_log10(E).ravel(), cz_b])
        vals = self._interp(pts).reshape(E.shape)
        return vals * self.to_cm2  # cm^2


# -------------------- PSF models (event-level only) --------------------
class PSFBase:
    def sigma_rad_for_bin(self, z: float, dec: float, Elo: float, Ehi: float) -> float:
        raise NotImplementedError


@dataclass
class FixedEventPSF(PSFBase):
    """Constant σ for event-level jitter (debug / quick checks)."""
    sigma_deg: float = 1.0

    def sigma_rad_for_bin(self, z: float, dec: float, Elo: float, Ehi: float) -> float:
        return float(np.deg2rad(self.sigma_deg))


@dataclass
class IRFWeightedPSF(PSFBase):
    """
    Flux-weighted σ(E, δ) computed from the IceCube R2021 public angular-resolution IRF.

    Implementation:
    - Precompute a grid of median angular errors (deg) over (log10E_true, δ) by sampling R2021IRF.sample.
    - Interpolate σ_deg(logE, δ).
    - For a bin [Elo,Ehi], compute:
        σ_bin^2(δ) = ∫ dE w(E,δ) σ^2(E,δ)  / ∫ dE w(E,δ)
      with weights w(E,δ) = A_eff(E,δ) * φ(E; z, spectrum).
    """
    period: str = "IC86_II"
    aeff: Optional[AeffInterpolator] = None
    cosmo: Optional[CosmologyGrid] = None
    spec_GeV: Optional[SpectrumParams] = None
    samples_per_node: int = 200
    energy_bins_per_decade: int = 6

    def __post_init__(self):
        if self.spec_GeV is None:
            raise ValueError("IRFWeightedPSF requires SpectrumParams in GeV/s (spec_GeV).")
        if self.cosmo is None:
            raise ValueError("IRFWeightedPSF requires a CosmologyGrid (cosmo).")

        self.irf = R2021IRF.from_period(self.period)

        if self.aeff is None:
            self.aeff = AeffInterpolator(dataset="20210126", period=self.period, units="m2")

        # declination rings implied by A_eff cos(zenith) edges
        self.cz_edges = self.aeff.cz_edges
        self.dec_edges = np.arcsin(-self.cz_edges)
        self.dec_centers = 0.5 * (self.dec_edges[:-1] + self.dec_edges[1:])

        # IRF energy axis (log10 GeV edges)
        self.logE_edges_irf = np.asarray(self.irf.true_energy_bins, dtype=float)
        self.logE_centers_irf = 0.5 * (self.logE_edges_irf[:-1] + self.logE_edges_irf[1:])

        # precompute σ_deg grid (median ang_err)
        rng = np.random.default_rng(42)
        self._sigma_deg_grid = np.zeros((self.logE_centers_irf.size, self.dec_centers.size), dtype=float)

        for iE, logE in enumerate(self.logE_centers_irf):
            for j, dec in enumerate(self.dec_centers):
                n = int(self.samples_per_node)
                coord = (np.full(n, 0.0), np.full(n, float(dec)))
                Etrue = np.full(n, float(logE))  # log10 GeV
                _, _, ang_err_deg, _ = self.irf.sample(
                    coord,
                    Etrue,
                    seed=int(rng.integers(1, 2**31 - 1)),
                    show_progress=False,
                )
                self._sigma_deg_grid[iE, j] = float(np.nanmedian(ang_err_deg))

        self._sigma_interp = RegularGridInterpolator(
            (self.logE_centers_irf, self.dec_centers),
            self._sigma_deg_grid,
            bounds_error=False,
            fill_value=None,
        )

        if not (np.isfinite(self.spec_GeV.Emin) and np.isfinite(self.spec_GeV.Emax)):
            raise ValueError("IRFWeightedPSF needs finite Emin/Emax.")

        logEmin, logEmax = np.log10(float(self.spec_GeV.Emin)), np.log10(float(self.spec_GeV.Emax))
        nE = max(8, int(self.energy_bins_per_decade * (logEmax - logEmin)))
        self._E_grid = np.logspace(logEmin, logEmax, nE)

    def _sigma_deg_vals(self, E_GeV: np.ndarray, dec: float) -> np.ndarray:
        E = np.asarray(E_GeV, dtype=float)
        pts = np.column_stack([safe_log10(E), np.full_like(E, float(dec))])
        return np.asarray(self._sigma_interp(pts), dtype=float)

    def sigma_rad_for_bin(self, z: float, dec: float, Elo: float, Ehi: float) -> float:
        Elo = float(Elo)
        Ehi = float(Ehi)
        dec = float(dec)
        z_eff = max(float(z), Z_FLOOR_DEFAULT)

        E = self._E_grid[(self._E_grid >= Elo) & (self._E_grid <= Ehi)]
        if E.size < 4:
            E = np.geomspace(Elo, Ehi, 8)

        # weights: A_eff * phi(E;z)
        phi = phi_earth_cm(E, z_eff, self.cosmo, self.spec_GeV)  # GeV^-1 cm^-2 s^-1
        A = self.aeff(E, dec)                                    # cm^2
        w = A * phi
        if not np.any(w > 0):
            return float(np.deg2rad(1.0))

        norm = float(np.trapz(w, E))
        if not np.isfinite(norm) or norm <= 0:
            return float(np.deg2rad(1.0))
        w = w / norm

        sig2 = float(np.trapz(w * (np.deg2rad(self._sigma_deg_vals(E, dec)) ** 2), E))
        return float(np.sqrt(max(sig2, 1e-16)))


# -------------------- Source sampling ∝ f(z) dV/dz --------------------
def sample_population_on_sky_and_z(
    Nsrc: int,
    pop: PopulationParams,
    cosmo: CosmologyGrid,
    rng: Optional[np.random.Generator] = None,
    z_floor: float = Z_FLOOR_DEFAULT,
    zmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample (ra, dec, z) with isotropic sky and redshifts drawn from p(z) ∝ f(z) dV/dz.
    """
    if rng is None:
        rng = np.random.default_rng(1234)

    z_samp = sample_redshifts(rng=rng, size=int(Nsrc), pop=pop, cosmo=cosmo, zmax=zmax)
    if z_samp.size == 0:
        z_samp = np.full(int(Nsrc), float(z_floor))
    z_samp = np.clip(z_samp, float(z_floor), float(cosmo.zmax if zmax is None else zmax))

    ra = 2.0 * np.pi * rng.random(int(Nsrc))
    sin_dec = rng.uniform(-1.0, 1.0, int(Nsrc))
    dec = np.arcsin(sin_dec)
    return ra, dec, z_samp


# -------------------- Expected counts per source / bin --------------------
def expected_counts_per_source_bin(
    z: float,
    dec: float,
    T_seconds: float,
    Elo: float,
    Ehi: float,
    spec_GeV: SpectrumParams,
    cosmo: CosmologyGrid,
    aeff: AeffInterpolator,
    nE: int = 120,
) -> float:
    """
    N̄_src(bin) = T ∫_{Elo}^{Ehi} dE A_eff(E,δ) φ(E;z)
    """
    z_eff = max(float(z), Z_FLOOR_DEFAULT)
    Elo = float(Elo)
    Ehi = float(Ehi)
    E = np.geomspace(Elo, Ehi, int(nE))
    phi = phi_earth_cm(E, z_eff, cosmo, spec_GeV)  # GeV^-1 cm^-2 s^-1
    A = aeff(E, float(dec))                        # cm^2
    rate = float(np.trapz(A * phi, E))             # s^-1
    return float(T_seconds * rate)


# -------------------- Map building + poissonization (event-level PSF) --------------------
def _unique_rings(nside: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return theta per pixel, unique theta values, and inverse indices pix->ring."""
    npix = hp.nside2npix(nside)
    theta, _phi = hp.pix2ang(nside, np.arange(npix))
    u_theta, inv = np.unique(theta, return_inverse=True)
    return theta, u_theta, inv


def _compute_sigma_ring_bin_for_event_psf(
    psf: PSFBase,
    u_theta: np.ndarray,
    bin_edges: np.ndarray,
    z_eff: float,
) -> np.ndarray:
    """
    Compute σ (radians) for each (bin, ring). Shape [B, R].
    For power-law spectra, z acts mainly as a normalization factor; z_eff is typically Z_FLOOR_DEFAULT.
    """
    B = bin_edges.size - 1
    R = u_theta.size
    sigma = np.zeros((B, R), dtype=float)
    for b in range(B):
        Elo, Ehi = float(bin_edges[b]), float(bin_edges[b + 1])
        for r, th in enumerate(u_theta):
            dec = (np.pi / 2.0) - float(th)
            sigma[b, r] = float(psf.sigma_rad_for_bin(z=float(z_eff), dec=float(dec), Elo=Elo, Ehi=Ehi))
    return sigma


@dataclass
class MapMakerConfig:
    dataset: str = "20210126"
    period: str = "IC86_II"
    aeff_units: str = "m2"          # 'm2' (default public IRFs) or 'cm2'
    time_years: float = 7.5
    nside: int = 64
    L_unit: str = "erg/s"           # convert L->GeV/s if 'erg/s'
    min_redshift: float = Z_FLOOR_DEFAULT
    # Event-level PSF only (mean-map PSF removed)
    psf_mode: str = "irf_weighted_event"   # 'none' | 'irf_weighted_event' | 'fixed_event'
    fixed_psf_sigma_deg: float = 1.0       # used only if psf_mode='fixed_event'
    irf_samples_per_node: int = 200
    energy_grid_per_decade: int = 8
    sources_to_draw: int = 5000
    energy_bin_edges_GeV: Optional[np.ndarray] = None
    scale_to_full_population: bool = True
    rng_seed_catalog: int = 12345

    def validate(self):
        if self.nside <= 0 or (self.nside & (self.nside - 1)) != 0:
            raise ValueError("nside must be a power of 2.")
        if self.sources_to_draw <= 0:
            raise ValueError("sources_to_draw must be > 0.")
        if self.psf_mode not in ("none", "irf_weighted_event", "fixed_event"):
            raise ValueError("psf_mode must be 'none', 'irf_weighted_event', or 'fixed_event'.")
        if self.energy_bin_edges_GeV is None:
            raise ValueError("Provide energy_bin_edges_GeV (strictly increasing, in GeV).")
        be = np.asarray(self.energy_bin_edges_GeV, dtype=float)
        if np.any(~np.isfinite(be)) or np.any(be <= 0) or np.any(np.diff(be) <= 0):
            raise ValueError("energy_bin_edges_GeV must be finite, positive, strictly increasing.")


def make_expected_counts_maps(
    spec: SpectrumParams,
    pop: PopulationParams,
    cosmo: CosmologyGrid,
    cfg: MapMakerConfig,
) -> Dict[str, object]:
    """
    Build pre-PSF expected-count maps for the astrophysical population.

    Returns dict with:
      - 'maps_mean_prepsf' : float [B, npix] true-direction expected counts (no PSF)
      - 'psf_aux' : caches for poissonize_astro (sigma_ring_bin, inv_ring) if psf_mode != 'none'
      - catalog arrays and metadata.
    """
    cfg.validate()

    # Units
    spec_GeV = ensure_spec_GeV(spec, cfg.L_unit)

    # Energy bins
    bin_edges = np.asarray(cfg.energy_bin_edges_GeV, dtype=float)
    B = bin_edges.size - 1

    # IRFs
    aeff = AeffInterpolator(dataset=cfg.dataset, period=cfg.period, units=cfg.aeff_units)

    # Exposure
    T = float(cfg.time_years) * 365.0 * 24.0 * 3600.0

    # Draw a catalog
    rng = np.random.default_rng(int(cfg.rng_seed_catalog))
    ra, dec, z = sample_population_on_sky_and_z(
        Nsrc=int(cfg.sources_to_draw),
        pop=pop,
        cosmo=cosmo,
        rng=rng,
        z_floor=float(cfg.min_redshift),
        zmax=None,
    )

    # Expected total number of sources (all-sky) up to the cosmology grid max (or zmax if you later add one)
    N_src_tot = float(expected_total_number_of_sources(pop=pop, cosmo=cosmo, zmax=None))

    # Per-source expected counts per bin
    mu_per_bin = np.zeros((z.size, B), dtype=float)
    for b in range(B):
        Elo, Ehi = float(bin_edges[b]), float(bin_edges[b + 1])
        nE = max(8, int(cfg.energy_grid_per_decade * (np.log10(Ehi) - np.log10(Elo))))
        for i in range(z.size):
            mu_per_bin[i, b] = expected_counts_per_source_bin(
                z[i], dec[i], T, Elo, Ehi, spec_GeV, cosmo, aeff, nE=nE
            )

    # Pre-PSF mean maps: delta sources in true-direction pixels
    npix = hp.nside2npix(int(cfg.nside))
    maps_mean_prepsf = np.zeros((B, npix), dtype=float)

    theta_src = (np.pi / 2.0) - dec
    pix_src = hp.ang2pix(int(cfg.nside), theta_src, ra)

    for b in range(B):
        # sum weights into pixels
        maps_mean_prepsf[b] = np.bincount(
            pix_src,
            weights=mu_per_bin[:, b],
            minlength=npix,
        ).astype(float)

    total_expected_per_bin = mu_per_bin.sum(axis=0)

    # Scale finite catalog to represent full population if requested
    scale = float(N_src_tot) / max(int(cfg.sources_to_draw), 1)
    if cfg.scale_to_full_population:
        maps_mean_prepsf *= scale
        total_expected_per_bin *= scale

    # PSF cache for poissonize_astro
    psf_aux: Dict[str, object] = {"mode": cfg.psf_mode, "nside": int(cfg.nside)}
    _theta_all, u_theta, inv_ring = _unique_rings(int(cfg.nside))

    if cfg.psf_mode == "fixed_event":
        psf = FixedEventPSF(sigma_deg=float(cfg.fixed_psf_sigma_deg))
        sigma_ring_bin = _compute_sigma_ring_bin_for_event_psf(psf, u_theta, bin_edges, z_eff=Z_FLOOR_DEFAULT)
        psf_aux.update({"sigma_ring_bin": sigma_ring_bin, "inv_ring": inv_ring, "u_theta": u_theta})

    elif cfg.psf_mode == "irf_weighted_event":
        psf = IRFWeightedPSF(
            period=cfg.period,
            aeff=aeff,
            cosmo=cosmo,
            spec_GeV=spec_GeV,
            samples_per_node=int(cfg.irf_samples_per_node),
            energy_bins_per_decade=max(4, int(cfg.energy_grid_per_decade)),
        )
        sigma_ring_bin = _compute_sigma_ring_bin_for_event_psf(psf, u_theta, bin_edges, z_eff=Z_FLOOR_DEFAULT)
        psf_aux.update({"sigma_ring_bin": sigma_ring_bin, "inv_ring": inv_ring, "u_theta": u_theta})

    # quick diagnostic magnitude check
    mid = z.size // 2
    mu_check_one_source = expected_counts_per_source_bin(
        z[mid], dec[mid], T, float(bin_edges[0]), float(bin_edges[-1]), spec_GeV, cosmo, aeff, nE=60
    )

    return {
        "maps_mean_prepsf": maps_mean_prepsf,
        "bin_edges_GeV": bin_edges,
        "total_expected_per_bin": total_expected_per_bin,
        "scale_factor": (scale if cfg.scale_to_full_population else 1.0),
        "ra": ra,
        "dec": dec,
        "z": z,
        "mu_per_bin": mu_per_bin,
        "N_src_tot": N_src_tot,
        "mu_check_one_source": float(mu_check_one_source),
        "spec_GeV": spec_GeV,
        "cosmo": cosmo,
        "aeff": aeff,
        "cfg": cfg,
        "psf_aux": psf_aux,
    }


def poissonize_astro(
    astro: Dict[str, object],
    use_psf: bool = True,
    rng_seed: Optional[int] = None,
) -> Dict[str, object]:
    """
    Draw a Poisson realization of the astrophysical component.

    - If psf_mode='none' OR use_psf=False:
        Poisson on pre-PSF mean map (true direction).
    - If psf_mode in {'fixed_event','irf_weighted_event'} AND use_psf=True:
        Poisson on pre-PSF means -> event-level jitter using σ(bin, ring) -> rebin.
    """
    cfg: MapMakerConfig = astro["cfg"]
    cfg.validate()

    be = np.asarray(astro["bin_edges_GeV"], dtype=float)
    nside = int(cfg.nside)
    lam_pre = np.asarray(astro["maps_mean_prepsf"], dtype=float)
    B, npix = lam_pre.shape

    seed_used = int(cfg.rng_seed_catalog + 1) if rng_seed is None else int(rng_seed)
    rng = np.random.default_rng(seed_used)

    if (not use_psf) or (cfg.psf_mode == "none"):
        counts = rng.poisson(lam_pre).astype(np.int64)
        return {
            "maps_counts": counts,
            "expected_total": lam_pre.sum(axis=1),
            "bin_edges_GeV": be,
            "nside": nside,
            "rng_seed_used": seed_used,
        }

    aux = astro.get("psf_aux") or {}
    if "sigma_ring_bin" not in aux or "inv_ring" not in aux:
        raise RuntimeError(
            "Missing PSF cache. Re-run make_expected_counts_maps(...) with psf_mode='fixed_event' "
            "or 'irf_weighted_event'."
        )

    sigma_rb = np.asarray(aux["sigma_ring_bin"], dtype=float)  # [B, R]
    inv_ring = np.asarray(aux["inv_ring"], dtype=np.int64)

    out = np.zeros((B, npix), dtype=np.int64)

    for b in range(B):
        mult = rng.poisson(lam_pre[b]).astype(np.int64)
        if mult.sum() == 0:
            continue

        pix_idx = np.repeat(np.arange(npix, dtype=np.int64), mult)
        theta0, phi0 = hp.pix2ang(nside, pix_idx)

        ring_idx = inv_ring[pix_idx]
        sigma = sigma_rb[b, ring_idx]

        # Rayleigh-distributed offset r, uniform azimuth alpha on tangent plane
        N = pix_idx.size
        u1 = np.clip(1.0 - rng.random(N), 1e-12, 1.0)
        r = sigma * np.sqrt(-2.0 * np.log(u1))
        alpha = rng.uniform(0.0, 2.0 * np.pi, size=N)

        # Tangent basis at (θ, φ)
        st, ct = np.sin(theta0), np.cos(theta0)
        sp, cp = np.sin(phi0), np.cos(phi0)
        e_th = np.column_stack([ct * cp, ct * sp, -st])
        e_ph = np.column_stack([-sp, cp, np.zeros_like(st)])
        uvec = np.column_stack([st * cp, st * sp, ct])

        cr, sr = np.cos(r), np.sin(r)
        ca, sa = np.cos(alpha), np.sin(alpha)
        v = (cr[:, None] * uvec) + (sr[:, None] * (ca[:, None] * e_th + sa[:, None] * e_ph))
        v /= np.linalg.norm(v, axis=1)[:, None]

        theta1 = np.arccos(np.clip(v[:, 2], -1.0, 1.0))
        phi1 = np.mod(np.arctan2(v[:, 1], v[:, 0]), 2.0 * np.pi)
        pix_out = hp.ang2pix(nside, theta1, phi1)

        out[b] = np.bincount(pix_out, minlength=npix).astype(np.int64)

    return {
        "maps_counts": out,
        "expected_total": lam_pre.sum(axis=1),
        "bin_edges_GeV": be,
        "nside": nside,
        "rng_seed_used": seed_used,
    }


def poissonize_astro_reco(
    astro_dict: Dict[str, object],
    reco_energy_edges_GeV: np.ndarray,
    rng_seed: int,
    batch_size: int = 200_000,
) -> Dict[str, object]:
    """
    Draw astrophysical events on true-energy bins, apply R2021 IRF energy+direction smearing,
    and bin into reconstructed-energy HEALPix maps.

    Notes
    -----
    - Uses `R2021IRF.sample_energy(...)` directly.
    - No additional PSF jitter is applied in this branch (avoid double-smearing).
    """
    cfg: MapMakerConfig = astro_dict["cfg"]
    cfg.validate()

    nside = int(cfg.nside)
    npix = hp.nside2npix(nside)

    true_edges = np.asarray(astro_dict["bin_edges_GeV"], dtype=float)
    reco_edges = np.asarray(reco_energy_edges_GeV, dtype=float)
    if reco_edges.ndim != 1 or reco_edges.size < 2 or np.any(np.diff(reco_edges) <= 0):
        raise ValueError("reco_energy_edges_GeV must be 1D, strictly increasing.")

    B_true = true_edges.size - 1
    B_reco = reco_edges.size - 1

    ra = np.asarray(astro_dict["ra"], dtype=float)
    dec = np.asarray(astro_dict["dec"], dtype=float)
    z = np.asarray(astro_dict["z"], dtype=float)
    mu_per_bin = np.asarray(astro_dict["mu_per_bin"], dtype=float)
    scale_factor = float(astro_dict.get("scale_factor", 1.0))

    if mu_per_bin.shape != (ra.size, B_true):
        raise RuntimeError(f"mu_per_bin has shape {mu_per_bin.shape}, expected {(ra.size, B_true)}.")

    spec_GeV = astro_dict.get("spec_GeV")
    cosmo = astro_dict.get("cosmo")
    aeff = astro_dict.get("aeff")
    if spec_GeV is None or cosmo is None or aeff is None:
        raise RuntimeError(
            "Missing spec_GeV/cosmo/aeff in astro_dict. "
            "Re-run make_expected_counts_maps(...) from this module."
        )

    irf = R2021IRF.from_period(cfg.period)
    if not hasattr(irf, "sample_energy"):
        raise RuntimeError(
            "R2021IRF.sample_energy is unavailable in this icecube_tools version. "
            "Please use the 2021 release-compatible API."
        )

    rng = np.random.default_rng(int(rng_seed))
    batch_size = max(1, int(batch_size))

    maps_counts_reco = np.zeros((B_reco, npix), dtype=np.int64)

    n_events_total = 0
    n_events_kept = 0
    n_events_migrated = 0

    ra_buf: list[np.ndarray] = []
    dec_buf: list[np.ndarray] = []
    loge_buf: list[np.ndarray] = []
    tbin_buf: list[np.ndarray] = []
    buf_size = 0

    def _flush_buffer() -> None:
        nonlocal buf_size, n_events_kept, n_events_migrated
        if buf_size == 0:
            return

        ra_evt = np.concatenate(ra_buf)
        dec_evt = np.concatenate(dec_buf)
        loge_evt = np.concatenate(loge_buf)
        tbin_evt = np.concatenate(tbin_buf)

        ra_buf.clear()
        dec_buf.clear()
        loge_buf.clear()
        tbin_buf.clear()
        buf_size = 0

        chunk_seed = int(rng.integers(1, 2**31 - 1))
        sample_out = irf.sample_energy(
            coord=(ra_evt, dec_evt),
            Etrue=loge_evt,
            seed=chunk_seed,
            show_progress=False,
        )
        ra_rec, dec_rec, _ang_deg, Erec_GeV = _parse_sample_energy_output(
            sample_out,
            coord_fallback=(ra_evt, dec_evt),
        )

        Erec = np.asarray(Erec_GeV, dtype=float)
        b_reco = np.searchsorted(reco_edges, Erec, side="right") - 1
        keep = (b_reco >= 0) & (b_reco < B_reco)
        if not np.any(keep):
            return

        b_kept = b_reco[keep].astype(np.int64)
        n_events_kept += int(keep.sum())
        n_events_migrated += int(np.sum(b_kept != tbin_evt[keep]))

        theta = (np.pi / 2.0) - np.asarray(dec_rec, dtype=float)[keep]
        phi = np.mod(np.asarray(ra_rec, dtype=float)[keep], 2.0 * np.pi)
        pix = hp.ang2pix(nside, theta, phi)

        flat = b_kept * npix + pix
        counts = np.bincount(flat, minlength=B_reco * npix).reshape(B_reco, npix)
        maps_counts_reco[:] += counts.astype(np.int64)

    # Source- and true-bin-aware event generation, then reco smearing in vectorized batches.
    for s in range(ra.size):
        dec_s = float(dec[s])
        ra_s = float(ra[s])
        z_s = max(float(z[s]), Z_FLOOR_DEFAULT)

        for k in range(B_true):
            lam = float(mu_per_bin[s, k] * scale_factor)
            if lam <= 0.0:
                continue

            n_evt = int(rng.poisson(lam))
            if n_evt <= 0:
                continue

            n_events_total += n_evt

            Elo = float(true_edges[k])
            Ehi = float(true_edges[k + 1])
            logE_grid = np.linspace(np.log10(Elo), np.log10(Ehi), 64)
            E_grid = np.power(10.0, logE_grid)

            w = np.asarray(aeff(E_grid, dec_s), dtype=float) * np.asarray(phi_earth_cm(E_grid, z_s, cosmo, spec_GeV), dtype=float)
            w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
            sw = float(w.sum())
            if sw <= 0.0:
                cdf = np.linspace(1.0 / E_grid.size, 1.0, E_grid.size)
            else:
                cdf = np.cumsum(w) / sw

            u = rng.random(n_evt)
            idx = np.searchsorted(cdf, u, side="right")
            idx = np.clip(idx, 0, E_grid.size - 1)
            E_true = E_grid[idx]

            ra_evt = np.full(n_evt, ra_s, dtype=float)
            dec_evt = np.full(n_evt, dec_s, dtype=float)
            loge_evt = np.log10(E_true)
            tbin_evt = np.full(n_evt, k, dtype=np.int64)

            ra_buf.append(ra_evt)
            dec_buf.append(dec_evt)
            loge_buf.append(loge_evt)
            tbin_buf.append(tbin_evt)
            buf_size += n_evt

            if buf_size >= batch_size:
                _flush_buffer()

    _flush_buffer()

    return {
        "maps_counts_reco": maps_counts_reco,
        "n_events_total": int(n_events_total),
        "n_events_kept": int(n_events_kept),
        "n_events_dropped_outside_reco_range": int(n_events_total - n_events_kept),
        "migration_fraction_kept": float(n_events_migrated / max(n_events_kept, 1)),
        "bin_edges_GeV": reco_edges,
        "nside": nside,
        "rng_seed_used": int(rng_seed),
    }


def combine_poisson_components(astro_counts: Dict[str, object], atmo_counts: Dict[str, object]) -> Dict[str, object]:
    """
    Sum two Poisson-realized components (e.g., astrophysical + atmospheric).
    """
    be_a = np.asarray(astro_counts["bin_edges_GeV"])
    be_b = np.asarray(atmo_counts["bin_edges_GeV"])
    if be_a.shape != be_b.shape or not np.allclose(be_a, be_b, rtol=0, atol=0):
        raise ValueError("Energy bin edges do not match.")
    if int(astro_counts["nside"]) != int(atmo_counts["nside"]):
        raise ValueError("NSIDE mismatch.")
    maps = (np.asarray(astro_counts["maps_counts"], dtype=np.int64)
            + np.asarray(atmo_counts["maps_counts"], dtype=np.int64))
    return {
        "maps_counts": maps,
        "bin_edges_GeV": be_a,
        "nside": int(astro_counts["nside"]),
        "expected_total": (np.asarray(astro_counts["expected_total"])
                           + np.asarray(atmo_counts["expected_total"])),
    }
