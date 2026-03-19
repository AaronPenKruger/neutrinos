"""generate_sims_capel_wide.py

Generate a bank of simulated IceCube-like HEALPix *count* maps for a smoke test of an SBI pipeline.

Physics model (Capel+2020)
-------------------------
- Extragalactic isotropic population of identical steady sources.
- Per-source spectrum: power-law with index gamma, normalized by an *energy luminosity* L integrated
  over [Emin, Emax] in the *Earth-frame* energy range.
- Comoving source density evolution:
    d\bar N_s^{tot}/dV = n0 * (1+z)^{p1} / (1+z/zc)^{p2}
  with theta=(p1,p2,zc). This is Eq. (4) in Capel+2020.

Forward model in this project
-----------------------------
- Astrophysical component: `IceCube_expected_nu_counts.make_expected_counts_maps` (pre-PSF means)
  + `IceCube_expected_nu_counts.poissonize_astro_reco` (Poisson in true-energy/source bins,
    then `R2021IRF.sample_energy` to reconstructed energy and direction).
- Atmospheric component: `IceCube_atmo_expected_counts.make_atmo_expected_counts_maps` (pre-PSF means)
  + `IceCube_atmo_expected_counts.poissonize_atmo_reco`.
- Combine by summing counts maps (per energy bin).

Design goal
-----------
A *fast* simulation bank for debugging the end-to-end SBI pipeline.
Therefore, we use a tightened, Capel-centered prior for (n0, L, gamma),
while using Capel's *wide* evolution prior for theta.

Outputs
-------
Writes counts maps as either:
  outdir/maps_counts.zarr  shape [N, B, npix]  dtype=uint32  (default)
or
  outdir/maps_counts.npy   shape [N, B, npix]  dtype=int32
and a CSV of parameters:
  outdir/theta.csv
and a JSON metadata file:
  outdir/meta.json
Optionally writes sparse pre-PSF hotspot truth:
  outdir/truth_hotspot_sparse.npz

This script is intended to be imported and driven from a notebook.

References
----------
- Capel, Mortlock, Finley (2020): arXiv:2005.02395, esp. Eq. (4) and Table II.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import csv
import json
import numpy as np

try:
    import zarr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    zarr = None

# Optional heavy dependencies are imported lazily so notebook drivers can import this module
# without immediate runtime dependency errors.
_SIM_IMPORT_ERROR: Optional[Exception] = None

try:
    from nu_pop_core import CosmologyGrid, SpectrumParams, PopulationParams, k_gamma
    from IceCube_expected_nu_counts import MapMakerConfig as AstroConfig
    from IceCube_expected_nu_counts import make_expected_counts_maps, poissonize_astro, poissonize_astro_reco
    from IceCube_atmo_expected_counts import (
        AtmosphereConfig,
        make_atmo_expected_counts_maps,
        poissonize_atmo,
        poissonize_atmo_reco,
    )
except (ImportError, ModuleNotFoundError) as _e:  # pragma: no cover - dependency availability is environment-specific
    _SIM_IMPORT_ERROR = _e
    CosmologyGrid = SpectrumParams = PopulationParams = k_gamma = None  # type: ignore[assignment]
    AstroConfig = AtmosphereConfig = None  # type: ignore[assignment]
    make_expected_counts_maps = poissonize_astro = poissonize_astro_reco = None  # type: ignore[assignment]
    make_atmo_expected_counts_maps = poissonize_atmo = poissonize_atmo_reco = None  # type: ignore[assignment]


def _require_sim_deps() -> None:
    if _SIM_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Simulation dependencies are not available. Install required packages "
            "(e.g. icecube_tools and nuflux) in the notebook environment."
        ) from _SIM_IMPORT_ERROR


# ----------------------------- Constants -----------------------------
SEC_PER_YEAR = 365.0 * 24.0 * 3600.0

# Energy binning: 5 logarithmic bins from 10 TeV to 10 PeV (in GeV)
DEFAULT_ENERGY_EDGES_GEV = np.logspace(4.0, 7.0, 6)

# Unit conversions
# 1 eV = 1.602176634e-12 erg -> 1 TeV = 1e12 eV = 1.602176634 erg
ERG_PER_TEV = 1.602176634


# ----------------------------- Capel evolution f(z) -----------------------------

def fz_capel(z: np.ndarray, p1: float, p2: float, zc: float, normalize: bool = True) -> np.ndarray:
    """Capel+2020 Eq. (4) evolution factor f(z, theta), normalized to f(0)=1 by default.

    f(z) = (1+z)^{p1} / (1 + z/zc)^{p2}.

    Parameters
    ----------
    z : array
        Redshift.
    p1, p2, zc : float
        Evolution parameters.
    normalize : bool
        If True, divide by f(0) so that f(0)=1.
    """
    z = np.asarray(z, dtype=float)
    f = (1.0 + z) ** float(p1) / (1.0 + z / float(zc)) ** float(p2)
    if normalize:
        f0 = f[0] if f.size else 1.0
        return f / (f0 if f0 != 0 else 1.0)
    return f


# ----------------------------- Tightened priors (smoke-test) -----------------------------

@dataclass(frozen=True)
class TightPriorConfig:
    """Capel-centered but tightened priors for a *smoke-test* simulation bank.

    Notes
    -----
    - Capel+ Table II uses L in TeV/s; we sample log10(L_TeV/s) then convert to erg/s.
    - Bounds here are intentionally much tighter than Capel's Table II to keep maps in a
      numerically well-behaved regime for early debugging.

    You should later widen these to reproduce Capel+ exactly.
    """

    # log10(n0 / Mpc^-3)
    log10_n0_min: float = -9.5
    log10_n0_max: float = -6.0

    # log10(L / TeV s^-1)
    # Centered to cover the Capel-favored region in their abstract (converted units),
    # while still allowing rare/bright tails.
    log10_L_TeV_s_min: float = 40.5
    log10_L_TeV_s_max: float = 45.5

    # gamma ~ truncated Normal
    gamma_mean: float = 2.2
    gamma_sigma: float = 0.30
    gamma_min: float = 1.5
    gamma_max: float = 3.2

    # Capel wide evolution prior (Table II):
    p1_min: float = 19.0
    p1_max: float = 21.0
    d_min: float = 0.0      # d = p2 - p1
    d_max: float = 6.0
    zc_min: float = 1.0
    zc_max: float = 1.8


def _sample_truncnorm(rng: np.random.Generator, mean: float, sigma: float, lo: float, hi: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    for _ in range(10_000):
        x = rng.normal(loc=mean, scale=sigma)
        if lo <= x <= hi:
            return float(x)
    raise RuntimeError("Failed to sample from truncated normal; check bounds.")


def sample_theta(rng: np.random.Generator, prior: TightPriorConfig) -> Dict[str, float]:
    """Sample one set of hyperparameters (n0, L, gamma, p1, p2, zc)."""
    log10_n0 = rng.uniform(prior.log10_n0_min, prior.log10_n0_max)
    n0 = 10.0 ** log10_n0

    log10_L_TeV_s = rng.uniform(prior.log10_L_TeV_s_min, prior.log10_L_TeV_s_max)
    L_TeV_s = 10.0 ** log10_L_TeV_s
    L_erg_s = L_TeV_s * ERG_PER_TEV

    gamma = _sample_truncnorm(rng, prior.gamma_mean, prior.gamma_sigma, prior.gamma_min, prior.gamma_max)

    p1 = rng.uniform(prior.p1_min, prior.p1_max)
    d = rng.uniform(prior.d_min, prior.d_max)
    p2 = p1 + d
    zc = rng.uniform(prior.zc_min, prior.zc_max)

    return {
        "n0_Mpc3": float(n0),
        "log10_n0": float(log10_n0),
        "L_TeV_s": float(L_TeV_s),
        "log10_L_TeV_s": float(log10_L_TeV_s),
        "L_erg_s": float(L_erg_s),
        "gamma": float(gamma),
        "p1": float(p1),
        "p2": float(p2),
        "zc": float(zc),
    }


# ----------------------------- Sanity checks (units + scaling) -----------------------------

@dataclass(frozen=True)
class SanityCheckReport:
    k_gamma_max_abs_err: float
    tev_to_erg_identity_rel_err: float
    linear_L_rel_err: float


def run_sanity_checks(
    energy_edges_GeV: np.ndarray = DEFAULT_ENERGY_EDGES_GEV,
    nside_quick: int = 16,
    sources_to_draw_quick: int = 400,
    time_years_quick: float = 1.0,
    rng_seed: int = 123,
) -> SanityCheckReport:
    """Fast unit/consistency checks.

    1) k_gamma normalization: \\int_{Emin}^{Emax} dE E * k_gamma * E^{-gamma} == 1.
    2) TeV<->erg conversion identity: (ERG_PER_TEV * 624.1509 GeV/erg) == 1000 GeV/TeV.
       (Your `IceCube_expected_nu_counts` uses 624.1509 GeV/erg internally.)
    3) Linearity in L: expected *mean* astrophysical counts scale linearly with L.

    Notes
    -----
    The linearity test uses a tiny nside and a small catalog, and checks totals of the
    *pre-PSF mean map* (no Poisson noise) to avoid stochastic variance.
    """

    _require_sim_deps()

    be = np.asarray(energy_edges_GeV, dtype=float)
    Emin, Emax = float(be[0]), float(be[-1])

    # (1) k_gamma normalization
    gammas = [1.7, 2.0, 2.6]
    Egrid = np.logspace(np.log10(Emin), np.log10(Emax), 5000)
    errs = []
    for g in gammas:
        kg = float(k_gamma(g, Emin, Emax))
        val = float(np.trapz(Egrid * kg * (Egrid ** (-g)), Egrid))
        errs.append(abs(val - 1.0))
    k_gamma_max_abs_err = float(np.max(errs))

    # (2) TeV<->erg identity for your internal conversion chain
    # If L is in TeV/s, then L[erg/s] = ERG_PER_TEV * L[TeV/s].
    # `IceCube_expected_nu_counts` converts erg/s -> GeV/s via (GeV/erg)=624.1509.
    # So: L[TeV/s] * ERG_PER_TEV * 624.1509 [GeV/erg] should equal L[TeV/s] * 1000 [GeV/TeV].
    GEV_PER_ERG_INTERNAL = 624.150907446  # must match IceCube_expected_nu_counts
    lhs = ERG_PER_TEV * GEV_PER_ERG_INTERNAL
    rhs = 1000.0
    tev_to_erg_identity_rel_err = float(abs(lhs - rhs) / rhs)

    # (3) Linearity in L (mean maps)
    rng = np.random.default_rng(int(rng_seed))
    cosmo = CosmologyGrid()

    # Use a representative theta
    theta = {
        "n0_Mpc3": 1e-7,
        "gamma": 2.2,
        "p1": 20.0,
        "p2": 23.0,
        "zc": 1.3,
    }
    fz_fn = lambda zz: fz_capel(zz, theta["p1"], theta["p2"], theta["zc"], normalize=True)
    pop = PopulationParams(n0=float(theta["n0_Mpc3"]), fz_fn=fz_fn)

    L1_TeV_s = 1e42
    L2_TeV_s = 2e42
    L1_erg_s = L1_TeV_s * ERG_PER_TEV
    L2_erg_s = L2_TeV_s * ERG_PER_TEV

    spec1 = SpectrumParams(gamma=float(theta["gamma"]), Emin=Emin, Emax=Emax, L=L1_erg_s)
    spec2 = SpectrumParams(gamma=float(theta["gamma"]), Emin=Emin, Emax=Emax, L=L2_erg_s)

    cfg = AstroConfig(
        time_years=float(time_years_quick),
        nside=int(nside_quick),
        psf_mode="none",  # mean map only
        sources_to_draw=int(sources_to_draw_quick),
        energy_bin_edges_GeV=be,
        scale_to_full_population=True,
        rng_seed_catalog=999,
    )

    astro1 = make_expected_counts_maps(spec=spec1, pop=pop, cosmo=cosmo, cfg=cfg)
    astro2 = make_expected_counts_maps(spec=spec2, pop=pop, cosmo=cosmo, cfg=cfg)

    tot1 = float(np.asarray(astro1["maps_mean_prepsf"]).sum())
    tot2 = float(np.asarray(astro2["maps_mean_prepsf"]).sum())

    ratio = tot2 / max(tot1, 1e-30)
    linear_L_rel_err = float(abs(ratio - 2.0) / 2.0)

    return SanityCheckReport(
        k_gamma_max_abs_err=k_gamma_max_abs_err,
        tev_to_erg_identity_rel_err=tev_to_erg_identity_rel_err,
        linear_L_rel_err=linear_L_rel_err,
    )


# ----------------------------- Simulation bank generation -----------------------------

@dataclass(frozen=True)
class SmokeSimConfig:
    n_sims: int = 1000
    seed: int = 0

    # Map settings
    nside: int = 64
    nest: bool = False
    output_format: Literal["zarr", "npy"] = "zarr"
    zarr_chunk_sims: int = 1
    time_years: float = 7.5
    energy_edges_GeV: Tuple[float, ...] = tuple(DEFAULT_ENERGY_EDGES_GEV.tolist())
    real_maps_npz: Optional[str] = None
    enforce_match_real_maps: bool = True
    allow_nside_override: bool = True
    allow_nest_override: bool = True

    # Optional sparse truth export for hotspot pretraining.
    truth_store: bool = False
    truth_topk: Optional[int] = None
    truth_min_expected_counts: float = 0.0

    # Astro map-maker tuning (speed vs fidelity)
    sources_to_draw: int = 2000
    energy_grid_per_decade: int = 6
    reco_batch_size: int = 200_000

    # PSF mode choice for smoke tests:
    # - 'none': fastest
    # - 'fixed_event': realistic smearing, fast
    # - 'irf_weighted_event': slow (builds IRF-weighted sigma caches)
    astro_psf_mode: str = "fixed_event"
    fixed_psf_sigma_deg: float = 1.0

    # Atmospheric settings
    atmo_psf_mode: str = "none"  # you can switch to 'irf_weighted_event' later
    conventional_model: str = "honda2006"
    include_prompt: bool = False


def generate_smoke_sim_bank(
    outdir: str | Path,
    sim_cfg: SmokeSimConfig = SmokeSimConfig(),
    prior: TightPriorConfig = TightPriorConfig(),
    overwrite: bool = False,
) -> Dict[str, Path]:
    """Generate N simulations and write to disk.

    Returns dict of output paths.
    """
    _require_sim_deps()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_format = str(sim_cfg.output_format).strip().lower()
    if output_format not in {"zarr", "npy"}:
        raise ValueError(f"Unsupported output_format={sim_cfg.output_format!r}. Use 'zarr' or 'npy'.")

    maps_path = outdir / ("maps_counts.zarr" if output_format == "zarr" else "maps_counts.npy")
    theta_csv_path = outdir / "theta.csv"
    meta_path = outdir / "meta.json"
    truth_sparse_path = outdir / "truth_hotspot_sparse.npz"

    if not overwrite:
        paths_to_check = [maps_path, theta_csv_path, meta_path]
        if bool(sim_cfg.truth_store):
            paths_to_check.append(truth_sparse_path)
        for p in paths_to_check:
            if p.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {p}")

    rng = np.random.default_rng(int(sim_cfg.seed))

    default_cfg = SmokeSimConfig()
    energy_edges = np.asarray(sim_cfg.energy_edges_GeV, dtype=float)
    if energy_edges.ndim != 1 or energy_edges.size < 2 or np.any(energy_edges <= 0) or np.any(np.diff(energy_edges) <= 0):
        raise ValueError("energy_edges_GeV must be 1D, positive, strictly increasing")

    nside_eff = int(sim_cfg.nside)
    nest_eff = bool(sim_cfg.nest)
    time_years_eff = float(sim_cfg.time_years)
    real_map_info: Optional[Dict[str, object]] = None
    mismatch_flags: Dict[str, bool] = {}

    if sim_cfg.real_maps_npz is not None:
        from data_to_maps import read_real_map_metadata

        real_map_info = read_real_map_metadata(Path(sim_cfg.real_maps_npz))
        real_edges = np.asarray(real_map_info["E_edges_GeV"], dtype=float)
        real_nside = int(real_map_info["nside"])
        real_nest = bool(real_map_info["nest"])
        time_years_eff = float(real_map_info["livetime_years_total"])

        if sim_cfg.enforce_match_real_maps:
            if nside_eff != real_nside:
                if int(sim_cfg.nside) == int(default_cfg.nside):
                    nside_eff = real_nside
                elif bool(sim_cfg.allow_nside_override):
                    # Keep explicit user override while still enforcing other real-map matches.
                    nside_eff = int(sim_cfg.nside)
                else:
                    raise ValueError(f"NSIDE mismatch: sim_cfg.nside={nside_eff} != real-map nside={real_nside}")

            if not np.allclose(energy_edges, real_edges, rtol=0.0, atol=1e-12):
                default_edges = np.asarray(default_cfg.energy_edges_GeV, dtype=float)
                if np.allclose(energy_edges, default_edges, rtol=0.0, atol=1e-12):
                    energy_edges = real_edges
                else:
                    raise ValueError("energy_edges_GeV mismatch between sim config and real maps metadata.")

            if nest_eff != real_nest:
                if bool(sim_cfg.nest) == bool(default_cfg.nest):
                    nest_eff = real_nest
                elif bool(sim_cfg.allow_nest_override):
                    nest_eff = bool(sim_cfg.nest)
                else:
                    raise ValueError(f"nest mismatch: sim_cfg.nest={nest_eff} != real-map nest={real_nest}")
        else:
            mismatch_flags = {
                "nside_mismatch": bool(nside_eff != real_nside),
                "energy_edges_mismatch": bool(not np.allclose(energy_edges, real_edges, rtol=0.0, atol=1e-12)),
                "nest_mismatch": bool(nest_eff != real_nest),
            }

    B = int(energy_edges.size - 1)
    if B != 5:
        raise ValueError(f"DeepHpx compatibility requires exactly 5 reconstructed-energy bins; got {B}.")
    npix = int(12 * nside_eff * nside_eff)

    # Precompute atmo mean maps once (theta-independent)
    atmo_cfg = AtmosphereConfig(
        nside=nside_eff,
        time_years=time_years_eff,
        energy_bin_edges_GeV=energy_edges,
        conventional_model=str(sim_cfg.conventional_model),
        include_prompt=bool(sim_cfg.include_prompt),
        psf_mode=str(sim_cfg.atmo_psf_mode),
    )
    atmo = make_atmo_expected_counts_maps(atmo_cfg)

    # Allocate output array.
    if output_format == "npy":
        maps_out = np.lib.format.open_memmap(
            maps_path,
            mode="w+",
            dtype=np.int32,
            shape=(int(sim_cfg.n_sims), B, npix),
        )
    else:
        if zarr is None:
            raise ImportError(
                "zarr is required for output_format='zarr'. Install with `pip install zarr numcodecs`."
            )
        try:
            from numcodecs import Blosc  # type: ignore

            compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        except Exception:
            compressor = None
        try:
            maps_out = zarr.open(
                str(maps_path),
                mode="w",
                shape=(int(sim_cfg.n_sims), B, npix),
                chunks=(max(1, int(sim_cfg.zarr_chunk_sims)), B, npix),
                dtype=np.uint32,
                compressor=compressor,
            )
        except TypeError:
            # zarr v3 may not accept `compressor=` in the same way as v2.
            maps_out = zarr.open(
                str(maps_path),
                mode="w",
                shape=(int(sim_cfg.n_sims), B, npix),
                chunks=(max(1, int(sim_cfg.zarr_chunk_sims)), B, npix),
                dtype=np.uint32,
            )

    # Optional sparse truth buffers (CSR-like).
    truth_indptr: List[int] = [0]
    truth_indices: List[int] = []
    truth_values: List[float] = []

    # Parameter CSV
    fieldnames = [
        "sim_id",
        "n0_Mpc3",
        "log10_n0",
        "L_TeV_s",
        "log10_L_TeV_s",
        "L_erg_s",
        "gamma",
        "p1",
        "p2",
        "zc",
        "seed_catalog",
        "seed_astro",
        "seed_atmo",
        "expected_astro_total",
        "expected_atmo_total",
        "astro_events_total",
        "astro_events_kept",
        "atmo_events_total",
        "atmo_events_kept",
    ]
    with theta_csv_path.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        cosmo = CosmologyGrid()

        for i in range(int(sim_cfg.n_sims)):
            th = sample_theta(rng, prior)

            Emin, Emax = float(energy_edges[0]), float(energy_edges[-1])
            spec = SpectrumParams(
                gamma=float(th["gamma"]),
                Emin=Emin,
                Emax=Emax,
                L=float(th["L_erg_s"]),  # erg/s (IceCube_expected_nu_counts converts to GeV/s)
            )

            fz_fn = (lambda zz, _p1=th["p1"], _p2=th["p2"], _zc=th["zc"]: fz_capel(zz, _p1, _p2, _zc, normalize=True))
            pop = PopulationParams(n0=float(th["n0_Mpc3"]), fz_fn=fz_fn)

            seed_catalog = int(10_000_000 + i)  # deterministic per-sim
            seed_astro = int(20_000_000 + i)
            seed_atmo = int(30_000_000 + i)

            astro_cfg = AstroConfig(
                time_years=time_years_eff,
                nside=nside_eff,
                psf_mode=str(sim_cfg.astro_psf_mode),
                fixed_psf_sigma_deg=float(sim_cfg.fixed_psf_sigma_deg),
                sources_to_draw=int(sim_cfg.sources_to_draw),
                energy_grid_per_decade=int(sim_cfg.energy_grid_per_decade),
                energy_bin_edges_GeV=energy_edges,
                scale_to_full_population=True,
                rng_seed_catalog=seed_catalog,
            )

            astro = make_expected_counts_maps(spec=spec, pop=pop, cosmo=cosmo, cfg=astro_cfg)
            astro_counts = poissonize_astro_reco(
                astro,
                reco_energy_edges_GeV=energy_edges,
                rng_seed=seed_astro,
                batch_size=int(sim_cfg.reco_batch_size),
            )

            atmo_counts = poissonize_atmo_reco(
                atmo,
                reco_energy_edges_GeV=energy_edges,
                rng_seed=seed_atmo,
                batch_size=int(sim_cfg.reco_batch_size),
            )

            total = (
                np.asarray(astro_counts["maps_counts_reco"], dtype=np.int32)
                + np.asarray(atmo_counts["maps_counts_reco"], dtype=np.int32)
            )
            if total.shape != (B, npix):
                raise RuntimeError(f"Unexpected map shape {total.shape}; expected {(B, npix)}")

            truth_hotspot = None
            if bool(sim_cfg.truth_store):
                truth_hotspot = np.asarray(astro["maps_mean_prepsf"], dtype=np.float64).sum(axis=0)

            if nest_eff:
                import healpy as hp

                total = np.stack([hp.reorder(total[b], r2n=True) for b in range(B)], axis=0).astype(np.int32, copy=False)
                if truth_hotspot is not None:
                    truth_hotspot = hp.reorder(truth_hotspot, r2n=True).astype(np.float64, copy=False)

            if output_format == "npy":
                maps_out[i] = total
            else:
                if np.any(total < 0):
                    raise RuntimeError("Counts map contains negative values, cannot store as uint32 in zarr.")
                maps_out[i] = np.asarray(total, dtype=np.uint32)

            if truth_hotspot is not None:
                thr = float(sim_cfg.truth_min_expected_counts)
                idx = np.flatnonzero(truth_hotspot >= thr)
                if idx.size > 0:
                    vals = truth_hotspot[idx]
                    topk = sim_cfg.truth_topk
                    if topk is not None and int(topk) > 0 and idx.size > int(topk):
                        sel = np.argpartition(vals, -int(topk))[-int(topk):]
                        idx = idx[sel]
                        vals = vals[sel]
                        order = np.argsort(vals)[::-1]
                        idx = idx[order]
                        vals = vals[order]
                    truth_indices.extend(idx.astype(np.int64).tolist())
                    truth_values.extend(vals.astype(np.float32).tolist())
                truth_indptr.append(len(truth_indices))

            writer.writerow({
                "sim_id": i,
                **{k: th[k] for k in ("n0_Mpc3", "log10_n0", "L_TeV_s", "log10_L_TeV_s", "L_erg_s", "gamma", "p1", "p2", "zc")},
                "seed_catalog": seed_catalog,
                "seed_astro": seed_astro,
                "seed_atmo": seed_atmo,
                "expected_astro_total": float(np.asarray(astro["maps_mean_prepsf"]).sum()),
                "expected_atmo_total": float(np.asarray(atmo["maps_mean_prepsf"]).sum()),
                "astro_events_total": int(astro_counts["n_events_total"]),
                "astro_events_kept": int(astro_counts["n_events_kept"]),
                "atmo_events_total": int(atmo_counts["n_events_total"]),
                "atmo_events_kept": int(atmo_counts["n_events_kept"]),
            })

    if output_format == "npy":
        maps_out.flush()

    truth_sparse_written = None
    if bool(sim_cfg.truth_store):
        np.savez_compressed(
            truth_sparse_path,
            indptr=np.asarray(truth_indptr, dtype=np.int64),
            indices=np.asarray(truth_indices, dtype=np.int64),
            values=np.asarray(truth_values, dtype=np.float32),
            nside=np.int64(nside_eff),
            npix=np.int64(npix),
            nest=np.bool_(nest_eff),
        )
        truth_sparse_written = truth_sparse_path

    meta = {
        "sim_cfg": asdict(sim_cfg),
        "prior": asdict(prior),
        "uses_reco_energy_smearing": True,
        "livetime_years_total": float(time_years_eff),
        "livetime_days_total": float(time_years_eff * 365.25),
        "paths": {
            "maps_counts": str(maps_path),
            "theta_csv": str(theta_csv_path),
            "truth_hotspot_sparse": (str(truth_sparse_written) if truth_sparse_written is not None else None),
        },
        "derived": {
            "B": B,
            "npix": npix,
            "nside": nside_eff,
            "energy_edges_GeV": energy_edges.tolist(),
            "nest": nest_eff,
        },
        "reco_pipeline": {
            "uses_reco_energy_smearing": True,
            "irf_method": "R2021IRF.sample_energy",
            "no_additional_psf_jitter": True,
        },
        "real_maps_match": {
            "real_maps_npz": (str(Path(sim_cfg.real_maps_npz).expanduser().resolve()) if sim_cfg.real_maps_npz else None),
            "livetime_days_total": (float(real_map_info["livetime_days_total"]) if real_map_info else None),
            "livetime_years_total": (float(real_map_info["livetime_years_total"]) if real_map_info else None),
            "seasons_found": ([str(s) for s in np.asarray(real_map_info["seasons_found"]).tolist()] if real_map_info else None),
            "season_livetime_days": (np.asarray(real_map_info["season_livetime_days"], dtype=float).tolist() if real_map_info else None),
            "energy_edges_match_exact": (bool(np.allclose(energy_edges, np.asarray(real_map_info["E_edges_GeV"], dtype=float), rtol=0.0, atol=1e-12)) if real_map_info else None),
            "nside_match": (bool(nside_eff == int(real_map_info["nside"])) if real_map_info else None),
            "nest_match": (bool(nest_eff == bool(real_map_info["nest"])) if real_map_info else None),
            "mismatches_when_not_enforced": mismatch_flags,
        },
        "notes": {
            "L_units": "We sample L in TeV/s (Capel Table II), convert to erg/s for the map generator.",
            "evolution": "Uses Capel Eq.(4): (1+z)^p1 / (1+z/zc)^p2, normalized to f(0)=1.",
            "output_format": "maps_counts.zarr stores uint32 when output_format='zarr'.",
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2))

    out = {"maps": maps_path, "maps_counts": maps_path, "theta_csv": theta_csv_path, "meta": meta_path}
    if truth_sparse_written is not None:
        out["truth_hotspot_sparse"] = truth_sparse_written
    return out


# ----------------------------- Notebook-friendly example -----------------------------

if __name__ == "__main__":
    # Minimal "it runs" example (kept small on purpose).
    rep = run_sanity_checks()
    print("Sanity checks:", rep)

    out = generate_smoke_sim_bank(
        outdir=Path("./smoke_sims_out"),
        sim_cfg=SmokeSimConfig(n_sims=10, nside=16, sources_to_draw=200, astro_psf_mode="none", atmo_psf_mode="none"),
        overwrite=True,
    )
    print("Wrote:", out)
