# Core functions to reproduce Capel, Mortlock, and Finley (2020) Fig. 1-style objects
# Formal, normalized implementation of Eqs. (1)-(10) with absolute normalization (n0, L, Emin, Emax).
#
# IMPORTANT FIXES (2025-12):
#  - Correct broadcasting in dNsrc_dEdtdA_earth(E, z, ...) for array z (previously used Dl[0]).
#  - Python 3.7 compatibility: remove PEP604 union syntax (X | None).
#  - Add reusable utilities: sample_redshifts(...) and expected_total_number_of_sources(...).

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Iterable, Dict, Any, Optional

import numpy as np
from scipy.integrate import quad

try:
    # SciPy >= 1.4
    from scipy.integrate import cumulative_trapezoid as _cumtrapz
except Exception:  # pragma: no cover
    # Older SciPy
    from scipy.integrate import cumtrapz as _cumtrapz  # type: ignore

# -------------------- Cosmology (flat ΛCDM) --------------------
_C_KM_S = 299_792.458  # km/s


@dataclass(frozen=True)
class CosmologyGrid:
    """
    Flat ΛCDM cosmology precomputed on a redshift grid.
    Provides:
      - luminosity distance D_L(z)
      - differential comoving volume element dV/dz (all-sky; includes 4π)
    """
    H0: float = 70.0
    Om: float = 0.3
    Ol: float = 0.7
    zmax: float = 6.0
    nz: int = 4000

    def __post_init__(self):
        z = np.linspace(0.0, float(self.zmax), int(self.nz))
        object.__setattr__(self, "z", z)

        Ez = np.sqrt(self.Om * (1.0 + z) ** 3 + self.Ol)
        object.__setattr__(self, "Ez", Ez)

        invEz = 1.0 / Ez
        Dc = (_C_KM_S / self.H0) * _cumtrapz(invEz, z, initial=0.0)  # comoving distance
        object.__setattr__(self, "Dc", Dc)

        Dl = (1.0 + z) * Dc  # luminosity distance
        object.__setattr__(self, "Dl", Dl)

        # All-sky differential comoving volume element (flat cosmology):
        # dV/dz = 4π * (c/H0) * D_M^2(z) / E(z)
        # with D_M = D_c for flat cosmology.
        # This is equivalent to dV/dz = 4π c D_L^2 / [H0 (1+z)^2 E(z)].
        dVdz = 4.0 * np.pi * (_C_KM_S / self.H0) * (Dc ** 2) / Ez
        object.__setattr__(self, "dVdz", dVdz)

    def luminosity_distance(self, z: np.ndarray) -> np.ndarray:
        """Vectorized D_L(z) interpolation over the internal grid."""
        z = np.clip(np.asarray(z, dtype=float), 0.0, float(self.zmax))
        return np.interp(z, self.z, self.Dl)

    def comoving_distance(self, z: np.ndarray) -> np.ndarray:
        """Vectorized D_c(z) interpolation over the internal grid."""
        z = np.clip(np.asarray(z, dtype=float), 0.0, float(self.zmax))
        return np.interp(z, self.z, self.Dc)

    def dVdz_of_z(self, z: np.ndarray) -> np.ndarray:
        """Vectorized all-sky dV/dz interpolation over the internal grid."""
        z = np.clip(np.asarray(z, dtype=float), 0.0, float(self.zmax))
        return np.interp(z, self.z, self.dVdz)


# -------------------- Population evolution f(z) --------------------
def fz_powerlaw(z: np.ndarray, p1: float, p2: float, zc: float, normalize: bool = True) -> np.ndarray:
    """
    Generic evolution:
      f(z) = (1+z)^p1 * (1 + z/zc)^p2
    Up to an overall factor absorbed by n0.
    """
    z = np.asarray(z, dtype=float)
    f = (1.0 + z) ** float(p1) * (1.0 + z / float(zc)) ** float(p2)
    return f / f[0] if normalize else f


def fz_flat(z: np.ndarray) -> np.ndarray:
    return np.ones_like(np.asarray(z, dtype=float))


def fz_negative(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return (1.0 + z) ** (-1.0)


def fz_sfr_md14(z: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Madau & Dickinson (2014) SFR proxy (often used as an evolution template):
      f(z) ∝ (1+z)^2.7 / [1 + ((1+z)/2.9)^5.6]
    Normalized to f(0)=1 by default.
    """
    z = np.asarray(z, dtype=float)
    num = (1.0 + z) ** 2.7
    den = 1.0 + ((1.0 + z) / 2.9) ** 5.6
    f = num / den
    return f / f[0] if normalize else f


# -------------------- Single-source spectrum & normalization --------------------
@dataclass(frozen=True)
class SpectrumParams:
    """
    Power-law spectrum normalized by energy luminosity L in [Emin, Emax].
    All energies must be in consistent units; L must match those implied energy units.
    """
    gamma: float
    Emin: float
    Emax: float
    L: float


def k_gamma(gamma: float, Emin: float, Emax: float) -> float:
    """
    Energy-normalization constant k_γ such that:
      L = ∫_{Emin}^{Emax} dE [ E * (L k_γ E^{-γ}) ] = L

    For γ ≠ 2:
      k_γ = (2-γ) / (Emax^{2-γ} - Emin^{2-γ})
    For γ = 2:
      k_2 = 1 / ln(Emax/Emin)
    """
    g = float(gamma)
    Emin = float(Emin)
    Emax = float(Emax)
    if Emin <= 0.0 or Emax <= Emin:
        raise ValueError(f"Invalid energy bounds: Emin={Emin}, Emax={Emax}")

    if np.isclose(g, 2.0, rtol=0.0, atol=1e-12):
        return 1.0 / np.log(Emax / Emin)

    return (2.0 - g) / (Emax ** (2.0 - g) - Emin ** (2.0 - g))


def dNsrc_dEdtdA_earth(E: np.ndarray, z: np.ndarray, cosmo: CosmologyGrid, spec: SpectrumParams) -> np.ndarray:
    """
    Single-source differential flux at Earth:
      dN̄^src / (dE dt dA) = [ L k_γ (1+z)^{2-γ} / (4π D_L^2(z)) ] * E^{-γ}

    Broadcasts over E and z.

    Returns:
      array with shape broadcast(E, z)
    """
    E = np.asarray(E, dtype=float)
    z = np.asarray(z, dtype=float)

    Dl = cosmo.luminosity_distance(z)  # same shape as z
    kg = k_gamma(spec.gamma, spec.Emin, spec.Emax)

    # Build broadcastable prefactor over z
    pref_z = spec.L * kg * (1.0 + z) ** (2.0 - spec.gamma) / (4.0 * np.pi * (Dl ** 2))

    # Broadcast E^{-γ} against prefactor
    return pref_z[..., None] * (E[None, ...] ** (-spec.gamma)) if pref_z.ndim == 1 else pref_z * (E ** (-spec.gamma))


# -------------------- Population flux & integrands --------------------
@dataclass(frozen=True)
class PopulationParams:
    """Population parameters: local density n0 and evolution f(z)."""
    n0: float  # Mpc^-3
    fz_fn: Callable[[np.ndarray], np.ndarray]


def population_differential_flux_integrand_per_solid_angle(
    E: np.ndarray,
    z: np.ndarray,
    cosmo: CosmologyGrid,
    spec: SpectrumParams,
    pop: PopulationParams,
) -> np.ndarray:
    """
    z-integrand for the *per-solid-angle* differential flux:
      dN̄^tot / (dE dt dA dΩ) = (1/4π) ∫ dz [ n0 f(z) (dV/dz) * dN̄^src/(dE dt dA) ].

    Returns:
      integrand over z with shape broadcast(z, E) (typically [Nz, NE]).
    """
    z = np.asarray(z, dtype=float)
    E = np.asarray(E, dtype=float)

    dVdz = cosmo.dVdz_of_z(z)
    fz = pop.fz_fn(z)

    # dNsrc returns broadcasted array. Ensure we get shape [Nz, NE].
    src = dNsrc_dEdtdA_earth(E=E, z=z, cosmo=cosmo, spec=spec)
    if src.ndim == 1:
        # scalar z, vector E -> [NE]
        src = src[None, :]

    weight_z = (pop.n0 * fz * dVdz) / (4.0 * np.pi)
    return weight_z[:, None] * src


def population_number_flux_integrand(z: np.ndarray, cosmo: CosmologyGrid, spec: SpectrumParams, pop: PopulationParams) -> np.ndarray:
    """
    A robust version of the z-integrand used in population number-flux-like quantities.

    Note: this is a project-specific convenience integrand; it is not a general cosmology identity.
    """
    z = np.asarray(z, dtype=float)
    fz = pop.fz_fn(z)
    Ez = np.sqrt(cosmo.Om * (1.0 + z) ** 3 + cosmo.Ol)
    return pop.n0 * fz * (_C_KM_S / cosmo.H0) * spec.L * (1.0 + z) ** (-spec.gamma) / Ez


def total_number_flux(spec: SpectrumParams, pop: PopulationParams, cosmo: CosmologyGrid) -> float:
    """Numerical integral over the internal grid of population_number_flux_integrand."""
    z = cosmo.z
    integrand = population_number_flux_integrand(z, cosmo, spec, pop)
    return float(np.trapz(integrand, z))


# -------------------- Expected counts --------------------
@dataclass(frozen=True)
class ObservationParams:
    """
    Observation configuration for expected counts.

    A_eff(E, delta) should return an effective area with the same units assumed by the flux.
    """
    T: float  # exposure time (s)
    A_eff: Callable[[np.ndarray, float], np.ndarray]


def expected_counts_from_one_source(
    z: float,
    delta: float,
    spec: SpectrumParams,
    obs: ObservationParams,
    cosmo: CosmologyGrid,
) -> float:
    """
    Expected detected counts from one source:
      N̄_src = T ∫_{Emin}^{Emax} dE A_eff(E, δ) [dN̄_src/(dE dt dA)]
    """
    Emin, Emax = float(spec.Emin), float(spec.Emax)

    def integrand(E_scalar: float) -> float:
        E_arr = np.asarray(E_scalar, dtype=float)
        aeff = obs.A_eff(E_arr, float(delta))
        # dNsrc expects arrays; feed scalar E and scalar z; return scalar
        dN = dNsrc_dEdtdA_earth(E=np.asarray([E_scalar], dtype=float), z=np.asarray([z], dtype=float), cosmo=cosmo, spec=spec)
        return float(np.asarray(aeff).reshape(-1)[0] * dN.reshape(-1)[0])

    val, _ = quad(integrand, Emin, Emax, epsabs=0.0, epsrel=1e-5, limit=200)
    return float(obs.T * val)


def expected_counts_all_sky_average(
    spec: SpectrumParams,
    pop: PopulationParams,
    obs: ObservationParams,
    cosmo: CosmologyGrid,
    delta_sampler: Optional[Callable[[int], np.ndarray]] = None,
    n_delta: int = 60,
) -> float:
    """
    Approximate total expected detected counts from the whole population:

      N̄_tot ≈ ∫ dz [ n0 f(z) dV/dz / (4π) ] * <N̄_src(z, δ)>_sky.

    WARNING: this can be expensive because it does many 1D quadratures.
    It's primarily for calibration / sanity checks, not bulk simulation.
    """
    zgrid = cosmo.z
    dVdz = cosmo.dVdz_of_z(zgrid)
    fz = pop.fz_fn(zgrid)

    if delta_sampler is None:
        u = np.linspace(-1.0, 1.0, int(n_delta))
        deltas = np.arcsin(u)  # uniform in sin(δ)
    else:
        deltas = np.asarray(delta_sampler(int(n_delta)), dtype=float)

    per_z = np.zeros_like(zgrid, dtype=float)
    for i, zi in enumerate(zgrid):
        counts = [expected_counts_from_one_source(float(zi), float(d), spec, obs, cosmo) for d in deltas]
        per_z[i] = float(np.mean(counts))

    integrand = (pop.n0 * fz * dVdz / (4.0 * np.pi)) * per_z
    return float(np.trapz(integrand, zgrid))


# -------------------- Fig. 1 objects by normalization --------------------
def fig1_Pz_and_cdf(
    z: np.ndarray,
    cosmo: CosmologyGrid,
    spec: SpectrumParams,
    pop: PopulationParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build P(z) and CDF P(<z) by normalizing population_number_flux_integrand(z).
    """
    z = np.asarray(z, dtype=float)
    I = population_number_flux_integrand(z, cosmo, spec, pop)
    area = float(np.trapz(I, z))
    P = I / (area + 1e-300)
    C = np.concatenate([[0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * (z[1:] - z[:-1]))])
    C[-1] = 1.0
    return P, C


def loop_models_family(
    z: np.ndarray,
    cosmo: CosmologyGrid,
    spec: SpectrumParams,
    families: Iterable[Dict[str, Any]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Convenience helper: compute P(z), C(z) for each model family entry.
    Each dict should include keys: "label", "n0", and "fz" (an array over z).
    """
    z = np.asarray(z, dtype=float)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for m in families:
        fz_arr = np.asarray(m["fz"], dtype=float)
        pop = PopulationParams(
            n0=float(m["n0"]),
            fz_fn=(lambda zz, _z=z, _arr=fz_arr: np.interp(np.asarray(zz, dtype=float), _z, _arr)),
        )
        P, C = fig1_Pz_and_cdf(z, cosmo, spec, pop)
        out[str(m.get("label", "model"))] = (P, C)
    return out


# -------------------- NEW: utilities for catalog simulation --------------------
def expected_total_number_of_sources(pop: PopulationParams, cosmo: CosmologyGrid, zmax: Optional[float] = None) -> float:
    """
    Expected total number of sources out to zmax:
      N_src(<zmax) = ∫_0^{zmax} dz [ n0 f(z) dV/dz ]   (all-sky dV/dz)

    This is often used to scale a finite drawn catalog to a full population.
    """
    if zmax is None:
        z = cosmo.z
        dVdz = cosmo.dVdz
    else:
        z = cosmo.z[cosmo.z <= float(zmax)]
        if z.size < 2:
            return 0.0
        dVdz = cosmo.dVdz_of_z(z)

    fz = pop.fz_fn(z)
    integrand = pop.n0 * fz * dVdz
    return float(np.trapz(integrand, z))


def sample_redshifts(
    rng: np.random.Generator,
    size: int,
    pop: PopulationParams,
    cosmo: CosmologyGrid,
    zmax: Optional[float] = None,
) -> np.ndarray:
    """
    Sample source redshifts from:
      p(z) ∝ f(z) dV/dz    on [0, zmax].

    Uses inverse-CDF sampling on the CosmologyGrid.
    """
    if size <= 0:
        return np.zeros((0,), dtype=float)

    if zmax is None:
        z = cosmo.z
        dVdz = cosmo.dVdz
    else:
        z = cosmo.z[cosmo.z <= float(zmax)]
        if z.size < 2:
            return np.zeros((size,), dtype=float)
        dVdz = cosmo.dVdz_of_z(z)

    w = np.clip(pop.fz_fn(z) * dVdz, 0.0, np.inf)
    # Build CDF with trapezoidal integration
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * (z[1:] - z[:-1]))])
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0.0:
        raise ValueError("Invalid redshift sampling weights: integral is non-positive or non-finite.")
    cdf /= cdf[-1]

    u = rng.random(int(size))
    return np.interp(u, cdf, z)


# -------------------- Calibration to a target N_obs --------------------
def calibrate_L_for_target_counts(
    target_N: float,
    n0: float,
    fz_fn: Callable[[np.ndarray], np.ndarray],
    spec: SpectrumParams,
    obs: ObservationParams,
    cosmo: CosmologyGrid,
) -> float:
    """
    Solve for luminosity L such that expected_counts_all_sky_average(...) == target_N (with given n0).
    Root-find in log10(L) for stability.
    """
    from math import isfinite

    target_N = float(target_N)
    pop_template = PopulationParams(n0=float(n0), fz_fn=fz_fn)

    def f_of_log10L(log10L: float) -> float:
        Lval = 10.0 ** float(log10L)
        sp = SpectrumParams(gamma=spec.gamma, Emin=spec.Emin, Emax=spec.Emax, L=Lval)
        return expected_counts_all_sky_average(sp, pop_template, obs, cosmo) - target_N

    # Bracket in log10-space (very broad; user can tighten if desired)
    lo, hi = 35.0, 48.0
    flo = f_of_log10L(lo)
    fhi = f_of_log10L(hi)

    # If not bracketed, still attempt bisection but results may be approximate.
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fmid = f_of_log10L(mid)
        if not isfinite(fmid):
            break

        # Standard bisection update if bracketed; otherwise steer toward decreasing |f|
        if np.sign(flo) != np.sign(fhi):
            if np.sign(fmid) == np.sign(flo):
                lo, flo = mid, fmid
            else:
                hi, fhi = mid, fmid
        else:
            # Not bracketed: pick the side that reduces magnitude
            if abs(fmid) < abs(flo):
                lo, flo = mid, fmid
            else:
                hi, fhi = mid, fmid

        if abs(fmid) / (target_N + 1e-12) < 1e-3:
            return 10.0 ** mid

    return 10.0 ** mid
