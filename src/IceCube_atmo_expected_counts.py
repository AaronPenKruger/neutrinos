# IceCube_atmo_expected_counts.py
"""
Atmospheric (conventional + prompt) νμ + ν̄μ background generator for IceCube through-going tracks
(IC86_* R2021 public IRFs), producing *pre-PSF mean maps* and *Poisson-realized reconstructed maps*
with a consistent event-level PSF model.

What this module does
---------------------
1) Compute *mean* expected-count HEALPix maps per energy bin in TRUE direction by folding
   φ_atm(E, cosZ) with IceCube A_eff(E, δ) and exposure T:

       μ_pix(prePSF) = T ∫ dE φ_atm(E, cosZ_pix) A_eff(E, δ_pix) ΔΩ_pix

   where φ_atm from `nuflux` is in GeV^-1 cm^-2 s^-1 sr^-1 and A_eff is in cm^2.

2) Sampling + PSF (DEPRECATED mean-map PSF REMOVED):
   - psf_mode='none':
       N_pix ~ Poisson( μ_pix(prePSF) ) (true direction; no PSF)
   - psf_mode='irf_weighted_event' (recommended):
       Draw Poisson counts in true-direction pixels, then jitter each event on the sphere using an
       IRF-weighted σ(E-bin, δ-ring), and rebin into reconstructed pixels.

   This implements the natural “Poisson → independent displacement → rebin” generative semantics.

Public API
----------
- make_atmo_expected_counts_maps(cfg) -> dict with:
    'maps_mean_prepsf' : float [B, npix]
    'bin_edges_GeV'    : float [B+1]
    'nside'            : int
    'omega_pix'        : float
    'expected_total_prepsf' : float [B]
    'cfg'              : AtmosphereConfig
    'psf_aux'          : dict containing caches for event-level PSF sampling

- poissonize_atmo(atmo_dict, use_psf=True, rng_seed=None) -> dict with:
    'maps_counts'      : int [B, npix]
    'expected_total'   : float [B]
    'bin_edges_GeV', 'nside', 'rng_seed_used'

- combine_poisson_components(astro_counts, atmo_counts) -> dict with summed maps

Dependencies
------------
- numpy, healpy
- nuflux (Honda/Bartol conventional; ERS/BERSS prompt) -> φ(E, cosZ)
- Your local modules:
    IceCube_expected_nu_counts: AeffInterpolator, coszen_from_dec, IRFWeightedPSF
    nu_pop_core: CosmologyGrid, SpectrumParams
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np

try:
    import healpy as hp
except ImportError as e:
    raise ImportError("Please `pip install healpy`.") from e

# ---- project IRF helpers ----
from IceCube_expected_nu_counts import (
    AeffInterpolator,
    coszen_from_dec,
    IRFWeightedPSF,
)
from nu_pop_core import CosmologyGrid, SpectrumParams
from icecube_tools.detector.r2021 import R2021IRF

# ---- atmospheric flux backend ----
try:
    import nuflux
except Exception as e:
    raise ImportError(
        "This module requires the IceCube `nuflux` package. Install with `pip install nuflux`."
    ) from e


# ----------------------------- Config -----------------------------

@dataclass
class AtmosphereConfig:
    # Detector / IRF
    dataset: str = "20210126"
    period: str = "IC86_II"
    aeff_units: str = "m2"  # public A_eff tables are typically in m^2 → converted to cm^2 internally

    # Sky / maps
    nside: int = 64

    # Time
    time_years: float = 7.5

    # Energy binning
    energy_bin_edges_GeV: Optional[np.ndarray] = None
    energy_grid_per_decade: int = 10

    # Flux models (nuflux keys)
    conventional_model: str = "honda2006"  # e.g., 'honda2015', 'bartol'
    include_prompt: bool = False
    prompt_model: str = "ers"              # used only if include_prompt=True

    # PSF handling (mean-map PSF deprecated and removed)
    psf_mode: str = "irf_weighted_event"   # 'none' | 'irf_weighted_event'
    atmo_gamma_proxy: float = 3.7          # spectral index to weight PSF(E, δ) within each bin
    irf_samples_per_node: int = 150        # for IRFWeightedPSF construction
    energy_bins_per_decade_psf: int = 6    # for PSF’s internal E-mix

    # RNG (used by poissonizer / event-level PSF sampling)
    rng_seed: int = 1234

    def validate(self):
        if self.energy_bin_edges_GeV is None:
            raise ValueError("Provide energy_bin_edges_GeV (strictly increasing, in GeV).")
        be = np.asarray(self.energy_bin_edges_GeV, dtype=float)
        if np.any(~np.isfinite(be)) or np.any(be <= 0) or np.any(np.diff(be) <= 0):
            raise ValueError("energy_bin_edges_GeV must be finite, positive, strictly increasing.")
        if self.nside <= 0 or (self.nside & (self.nside - 1)) != 0:
            raise ValueError("nside must be a power of 2.")

        if self.psf_mode not in ("none", "irf_weighted_event"):
            raise ValueError(
                "psf_mode must be 'none' or 'irf_weighted_event'. "
                "Mean-map PSF modes (e.g. 'isotropic_mean') are deprecated and removed."
            )


# ----------------------------- Flux builders -----------------------------

def _build_flux_functions(cfg: AtmosphereConfig):
    """Return callable(s): φ_conv(E, cosZ) and optionally φ_prompt(E, cosZ) in GeV^-1 cm^-2 s^-1 sr^-1."""
    f_conv = nuflux.makeFlux(cfg.conventional_model)
    f_prompt = nuflux.makeFlux(cfg.prompt_model) if cfg.include_prompt else None

    def phi_conv(E, cosz):
        return f_conv.getFlux(nuflux.NuMu, E, cosz) + f_conv.getFlux(nuflux.NuMuBar, E, cosz)

    if f_prompt is None:
        return phi_conv, None

    def phi_prompt(E, cosz):
        return f_prompt.getFlux(nuflux.NuMu, E, cosz) + f_prompt.getFlux(nuflux.NuMuBar, E, cosz)

    return phi_conv, phi_prompt


# ----------------------------- Geometry helpers -----------------------------

def _unique_rings(nside: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return theta for each pixel, the unique-theta values, and inverse indices pix→ring."""
    npix = hp.nside2npix(nside)
    theta, _phi = hp.pix2ang(nside, np.arange(npix))
    u_theta, inv = np.unique(theta, return_inverse=True)
    return theta, u_theta, inv


def _parse_sample_energy_output(
    sample_out,
    coord_fallback: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize possible `R2021IRF.sample_energy` return signatures to:
      (ra_rec_rad, dec_rec_rad, ang_deg, Erec_GeV)
    """
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


# ----------------------------- Core integral (means, pre-PSF) -----------------------------

def _counts_per_ring(
    u_theta: np.ndarray,
    bin_edges: np.ndarray,
    aeff: AeffInterpolator,
    phi_conv,   # callable(E, cosZ)
    phi_prompt, # callable(E, cosZ) or None
    T_seconds: float,
    energy_grid_per_decade: int,
) -> np.ndarray:
    """
    Compute expected counts per sr (not yet multiplied by Ω_pix) for each unique ring and energy bin.
    Output shape: [B, R]
    """
    n_rings = u_theta.size
    n_bins = bin_edges.size - 1
    counts_sr = np.zeros((n_bins, n_rings), dtype=float)

    dec = (np.pi / 2.0) - u_theta
    cosz = coszen_from_dec(dec)  # South-Pole convention: cosZ = -sin(δ)

    for b in range(n_bins):
        Elo, Ehi = float(bin_edges[b]), float(bin_edges[b + 1])
        # log-spacing with a minimum of ~8 nodes per bin
        nE = max(8, int(energy_grid_per_decade * (np.log10(Ehi) - np.log10(Elo))))
        E = np.logspace(np.log10(Elo), np.log10(Ehi), nE)

        for r in range(n_rings):
            cz = float(cosz[r])
            d = float(dec[r])

            # A_eff(E, δ) [cm^2]
            A = aeff(E, d)

            # φ_conv + optional φ_prompt
            phi_c = np.array([phi_conv(float(EE), cz) for EE in E], dtype=float)
            if phi_prompt is not None:
                phi_p = np.array([phi_prompt(float(EE), cz) for EE in E], dtype=float)
                phi_tot = phi_c + phi_p
            else:
                phi_tot = phi_c

            # Integrate over energy: [cm^2] × [GeV^-1 cm^-2 s^-1 sr^-1] dE → [s^-1 sr^-1]
            rate_per_sr = np.trapz(A * phi_tot, E)
            counts_sr[b, r] = T_seconds * rate_per_sr

    return counts_sr


# ----------------------------- PSF cache (event-level) -----------------------------

def _build_irf_psf(cfg: AtmosphereConfig, bin_edges: np.ndarray) -> IRFWeightedPSF:
    """
    Build IRFWeightedPSF with a steep spectrum proxy (γ≈3.7) appropriate for conventional atmo.
    Only affects energy weighting for σ(E, δ) within each bin.
    """
    spec_proxy = SpectrumParams(
        gamma=float(cfg.atmo_gamma_proxy),
        Emin=float(bin_edges[0]),
        Emax=float(bin_edges[-1]),
        L=1.0,
    )
    return IRFWeightedPSF(
        period=cfg.period,
        aeff=None,
        cosmo=CosmologyGrid(),
        spec_GeV=spec_proxy,
        samples_per_node=int(cfg.irf_samples_per_node),
        energy_bins_per_decade=max(4, int(cfg.energy_bins_per_decade_psf)),
    )


def _compute_sigma_ring_bin(psf: IRFWeightedPSF, u_theta: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Return σ (radians) for each (bin, ring) pair using IRFWeightedPSF. Shape [B, R]."""
    B = bin_edges.size - 1
    R = u_theta.size
    sigma = np.zeros((B, R), dtype=float)
    for b in range(B):
        Elo, Ehi = float(bin_edges[b]), float(bin_edges[b + 1])
        for r, th in enumerate(u_theta):
            dec = (np.pi / 2.0) - float(th)
            sigma[b, r] = psf.sigma_rad_for_bin(z=1e-6, dec=dec, Elo=Elo, Ehi=Ehi)
    return sigma


# ----------------------------- Public API -----------------------------

def make_atmo_expected_counts_maps(cfg: AtmosphereConfig) -> Dict[str, object]:
    """
    Build atmospheric expected-count maps (pre-PSF, true-direction), and (if requested)
    build the event-level PSF cache needed for poissonize_atmo(...).

    Returns dict with keys:
      - 'maps_mean_prepsf' : [B, npix] expected counts (true-direction), no PSF
      - 'bin_edges_GeV', 'nside', 'omega_pix', 'expected_total_prepsf'
      - 'cfg' : the config used (for provenance)
      - 'psf_aux' : auxiliary PSF info (sigma_ring_bin, inv_ring) for event-level PSF sampling
    """
    cfg.validate()
    be = np.asarray(cfg.energy_bin_edges_GeV, dtype=float)

    # IRFs and exposure
    aeff = AeffInterpolator(dataset=cfg.dataset, period=cfg.period, units=cfg.aeff_units)
    T_seconds = cfg.time_years * 365.0 * 24.0 * 3600.0

    # Fluxes
    phi_conv, phi_prompt = _build_flux_functions(cfg)

    # Geometry
    npix = hp.nside2npix(cfg.nside)
    _theta_all, u_theta, inv_ring = _unique_rings(cfg.nside)
    omega_pix = 4.0 * np.pi / npix  # sr

    # Mean (pre-PSF), ring-wise then broadcast to pixels
    counts_sr = _counts_per_ring(
        u_theta=u_theta,
        bin_edges=be,
        aeff=aeff,
        phi_conv=phi_conv,
        phi_prompt=phi_prompt,
        T_seconds=T_seconds,
        energy_grid_per_decade=cfg.energy_grid_per_decade,
    )  # [B, R]

    maps_mean_prepsf = (counts_sr[:, inv_ring] * omega_pix).astype(float)  # [B, npix]
    expected_total_prepsf = maps_mean_prepsf.sum(axis=1)

    # PSF cache for event-level sampling
    psf_aux: Dict[str, object] = {"mode": cfg.psf_mode, "nside": int(cfg.nside)}

    if cfg.psf_mode == "irf_weighted_event":
        psf = _build_irf_psf(cfg, be)
        sigma_rb = _compute_sigma_ring_bin(psf, u_theta, be)  # [B, R]
        psf_aux.update({
            "sigma_ring_bin": sigma_rb,
            "inv_ring": inv_ring,
            "u_theta": u_theta,
        })

    return {
        "maps_mean_prepsf": maps_mean_prepsf,
        "bin_edges_GeV": be,
        "nside": int(cfg.nside),
        "omega_pix": float(omega_pix),
        "expected_total_prepsf": expected_total_prepsf,
        "counts_sr_per_ring": counts_sr,
        "inv_ring": inv_ring,
        "u_theta": u_theta,
        "cfg": cfg,
        "psf_aux": psf_aux,
    }


# ----------------------------- Sampling -----------------------------

def poissonize_atmo(atmo: Dict[str, object], use_psf: bool = True, rng_seed: Optional[int] = None) -> Dict[str, object]:
    """
    Draw a Poisson realization of the atmospheric component.

    - If psf_mode='none' OR use_psf=False:
        Poisson on pre-PSF means (true-direction map).
    - If psf_mode='irf_weighted_event' AND use_psf=True:
        Poisson on pre-PSF means → event-level jitter using σ(bin, ring) → rebin.

    Returns dict with:
      - 'maps_counts' : int [B, npix]
      - 'expected_total' : float [B]  (sum of means used for the Poisson draw)
      - 'bin_edges_GeV', 'nside', 'rng_seed_used'
    """
    cfg: AtmosphereConfig = atmo["cfg"]
    cfg.validate()

    be = np.asarray(atmo["bin_edges_GeV"], dtype=float)
    nside = int(atmo["nside"])
    lam_pre = np.asarray(atmo["maps_mean_prepsf"], dtype=float)
    B, npix = lam_pre.shape

    seed_used = cfg.rng_seed if rng_seed is None else int(rng_seed)
    rng = np.random.default_rng(seed_used)

    # True-direction Poisson (no PSF)
    if (not use_psf) or (cfg.psf_mode == "none"):
        counts = rng.poisson(lam_pre).astype(np.int64)
        return {
            "maps_counts": counts,
            "expected_total": lam_pre.sum(axis=1),
            "bin_edges_GeV": be,
            "nside": nside,
            "rng_seed_used": seed_used,
        }

    # Event-level PSF (Poisson → jitter → rebin)
    aux = atmo.get("psf_aux") or {}
    if cfg.psf_mode != "irf_weighted_event":
        raise ValueError("psf_mode must be 'none' or 'irf_weighted_event' (mean-map PSF removed).")
    if "sigma_ring_bin" not in aux or "inv_ring" not in aux:
        raise RuntimeError(
            "Missing PSF cache. Re-run make_atmo_expected_counts_maps(cfg) with psf_mode='irf_weighted_event'."
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


def poissonize_atmo_reco(
    atmo_dict: Dict[str, object],
    reco_energy_edges_GeV: np.ndarray,
    rng_seed: int,
    batch_size: int = 200_000,
) -> Dict[str, object]:
    """
    Draw atmospheric events from true-energy mean maps, smear with R2021 IRF energy/direction,
    and bin into reconstructed-energy HEALPix maps.

    Notes
    -----
    - Uses `R2021IRF.sample_energy(...)` directly.
    - Does not apply extra PSF jitter after IRF smearing.
    """
    cfg: AtmosphereConfig = atmo_dict["cfg"]
    cfg.validate()

    true_edges = np.asarray(atmo_dict["bin_edges_GeV"], dtype=float)
    reco_edges = np.asarray(reco_energy_edges_GeV, dtype=float)
    if reco_edges.ndim != 1 or reco_edges.size < 2 or np.any(np.diff(reco_edges) <= 0):
        raise ValueError("reco_energy_edges_GeV must be 1D, strictly increasing.")

    nside = int(atmo_dict["nside"])
    npix = hp.nside2npix(nside)
    B_true = true_edges.size - 1
    B_reco = reco_edges.size - 1

    lam_pre = np.asarray(atmo_dict["maps_mean_prepsf"], dtype=float)
    if lam_pre.shape != (B_true, npix):
        raise RuntimeError(f"maps_mean_prepsf has shape {lam_pre.shape}, expected {(B_true, npix)}.")

    inv_ring = np.asarray(atmo_dict.get("inv_ring"), dtype=np.int64)
    u_theta = np.asarray(atmo_dict.get("u_theta"), dtype=float)
    if inv_ring.size != npix or u_theta.size == 0:
        raise RuntimeError("Missing ring geometry (inv_ring/u_theta). Re-run make_atmo_expected_counts_maps(...).")

    irf = R2021IRF.from_period(cfg.period)
    if not hasattr(irf, "sample_energy"):
        raise RuntimeError(
            "R2021IRF.sample_energy is unavailable in this icecube_tools version. "
            "Please use the 2021 release-compatible API."
        )

    rng = np.random.default_rng(int(rng_seed))
    batch_size = max(1, int(batch_size))

    aeff = AeffInterpolator(dataset=cfg.dataset, period=cfg.period, units=cfg.aeff_units)
    phi_conv, phi_prompt = _build_flux_functions(cfg)

    # Ring-wise E-true CDF cache per true bin (shared by all simulations for this atmo config).
    R = u_theta.size
    nE_grid = 64
    E_grid_bins = np.zeros((B_true, nE_grid), dtype=float)
    cdf_ring_bin = np.zeros((B_true, R, nE_grid), dtype=float)

    dec_rings = (np.pi / 2.0) - u_theta
    cosz_rings = coszen_from_dec(dec_rings)

    for b in range(B_true):
        Elo = float(true_edges[b])
        Ehi = float(true_edges[b + 1])
        E_grid = np.logspace(np.log10(Elo), np.log10(Ehi), nE_grid)
        E_grid_bins[b] = E_grid

        for r in range(R):
            dec_r = float(dec_rings[r])
            cosz_r = float(cosz_rings[r])
            A = np.asarray(aeff(E_grid, dec_r), dtype=float)
            phi_c = np.array([phi_conv(float(ee), cosz_r) for ee in E_grid], dtype=float)
            if phi_prompt is not None:
                phi_p = np.array([phi_prompt(float(ee), cosz_r) for ee in E_grid], dtype=float)
                w = A * (phi_c + phi_p)
            else:
                w = A * phi_c
            w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
            sw = float(w.sum())
            if sw <= 0.0:
                cdf_ring_bin[b, r] = np.linspace(1.0 / nE_grid, 1.0, nE_grid)
            else:
                cdf_ring_bin[b, r] = np.cumsum(w) / sw

    maps_counts_reco = np.zeros((B_reco, npix), dtype=np.int64)
    n_events_total = 0
    n_events_kept = 0
    n_events_migrated = 0

    for b in range(B_true):
        mult = rng.poisson(lam_pre[b]).astype(np.int64)
        pix_nonzero = np.flatnonzero(mult > 0)
        if pix_nonzero.size == 0:
            continue

        pix_evt = np.repeat(pix_nonzero.astype(np.int64), mult[pix_nonzero])
        n_events_total += int(pix_evt.size)
        ring_evt = inv_ring[pix_evt]

        for start in range(0, pix_evt.size, batch_size):
            stop = min(start + batch_size, pix_evt.size)
            pix_chunk = pix_evt[start:stop]
            ring_chunk = ring_evt[start:stop]

            E_chunk = np.empty(stop - start, dtype=float)
            u = rng.random(stop - start)
            for r in np.unique(ring_chunk):
                sel = (ring_chunk == r)
                cdf = cdf_ring_bin[b, int(r)]
                idx = np.searchsorted(cdf, u[sel], side="right")
                idx = np.clip(idx, 0, nE_grid - 1)
                E_chunk[sel] = E_grid_bins[b, idx]

            theta, phi = hp.pix2ang(nside, pix_chunk)
            ra_true = np.mod(phi, 2.0 * np.pi)
            dec_true = (np.pi / 2.0) - theta

            chunk_seed = int(rng.integers(1, 2**31 - 1))
            sample_out = irf.sample_energy(
                coord=(ra_true, dec_true),
                Etrue=np.log10(E_chunk),
                seed=chunk_seed,
                show_progress=False,
            )
            ra_rec, dec_rec, _ang_deg, Erec_GeV = _parse_sample_energy_output(
                sample_out,
                coord_fallback=(ra_true, dec_true),
            )

            Erec = np.asarray(Erec_GeV, dtype=float)
            b_reco = np.searchsorted(reco_edges, Erec, side="right") - 1
            keep = (b_reco >= 0) & (b_reco < B_reco)
            if not np.any(keep):
                continue

            b_kept = b_reco[keep].astype(np.int64)
            n_events_kept += int(keep.sum())
            n_events_migrated += int(np.sum(b_kept != b))

            theta_rec = (np.pi / 2.0) - np.asarray(dec_rec, dtype=float)[keep]
            phi_rec = np.mod(np.asarray(ra_rec, dtype=float)[keep], 2.0 * np.pi)
            pix_rec = hp.ang2pix(nside, theta_rec, phi_rec)

            flat = b_kept * npix + pix_rec
            counts = np.bincount(flat, minlength=B_reco * npix).reshape(B_reco, npix)
            maps_counts_reco[:] += counts.astype(np.int64)

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


# ----------------------------- Combining -----------------------------

def combine_poisson_components(astro_counts: Dict[str, object], atmo_counts: Dict[str, object]) -> Dict[str, object]:
    """
    Sum two Poisson-realized components (e.g., astrophysical + atmospheric).

    Both inputs must carry 'maps_counts' [B, npix] and identical 'bin_edges_GeV' and 'nside'.
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
