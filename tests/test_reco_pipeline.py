from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_to_maps import read_real_map_metadata
import generate_sims_capel_wide as simmod


def _write_fixture_real_maps(npz_path: Path) -> np.ndarray:
    edges = np.logspace(4.0, 7.0, 6)
    np.savez(
        npz_path,
        counts_map=np.zeros((5, 12 * 8 * 8), dtype=np.int64),
        E_edges_GeV=edges,
        nside=np.int64(8),
        nest=np.bool_(False),
        seasons_found=np.array(["IC86_I"], dtype="U"),
        season_livetime_days=np.array([365.25], dtype=float),
        global_summary_json=np.array("{}", dtype="U"),
    )
    return edges


def test_read_real_map_metadata_one_year(tmp_path: Path) -> None:
    fixture_npz = tmp_path / "real_maps_fixture.npz"
    edges = _write_fixture_real_maps(fixture_npz)

    meta = read_real_map_metadata(fixture_npz)

    assert meta["nside"] == 8
    assert meta["nest"] is False
    assert np.allclose(meta["E_edges_GeV"], edges, rtol=0.0, atol=0.0)
    assert np.allclose(meta["season_livetime_days"], [365.25], rtol=0.0, atol=0.0)
    assert np.isclose(meta["livetime_years_total"], 1.0, rtol=0.0, atol=1e-12)


def test_tiny_sim_run_with_real_maps_metadata(tmp_path: Path, monkeypatch) -> None:
    fixture_npz = tmp_path / "real_maps_fixture.npz"
    edges = _write_fixture_real_maps(fixture_npz)
    outdir = tmp_path / "sim_out"

    class DummyCosmologyGrid:
        pass

    class DummySpectrumParams:
        def __init__(self, gamma, Emin, Emax, L):
            self.gamma = gamma
            self.Emin = Emin
            self.Emax = Emax
            self.L = L

    class DummyPopulationParams:
        def __init__(self, n0, fz_fn):
            self.n0 = n0
            self.fz_fn = fz_fn

    class DummyAstroCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyAtmoCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def _dummy_make_expected_counts_maps(spec, pop, cosmo, cfg):
        npix = 12 * int(cfg.nside) * int(cfg.nside)
        B = len(cfg.energy_bin_edges_GeV) - 1
        return {
            "maps_mean_prepsf": np.ones((B, npix), dtype=float),
        }

    def _dummy_make_atmo_expected_counts_maps(cfg):
        npix = 12 * int(cfg.nside) * int(cfg.nside)
        B = len(cfg.energy_bin_edges_GeV) - 1
        return {
            "maps_mean_prepsf": np.ones((B, npix), dtype=float),
            "bin_edges_GeV": np.asarray(cfg.energy_bin_edges_GeV, dtype=float),
            "nside": int(cfg.nside),
            "cfg": cfg,
        }

    def _dummy_poissonize_astro_reco(astro_dict, reco_energy_edges_GeV, rng_seed, batch_size=200_000):
        rng = np.random.default_rng(int(rng_seed))
        B = len(reco_energy_edges_GeV) - 1
        npix = astro_dict["maps_mean_prepsf"].shape[1]
        maps = rng.poisson(0.1, size=(B, npix)).astype(np.int64)
        return {
            "maps_counts_reco": maps,
            "n_events_total": int(maps.sum()),
            "n_events_kept": int(maps.sum()),
        }

    def _dummy_poissonize_atmo_reco(atmo_dict, reco_energy_edges_GeV, rng_seed, batch_size=200_000):
        B = len(reco_energy_edges_GeV) - 1
        npix = int(atmo_dict["maps_mean_prepsf"].shape[1])
        maps = np.zeros((B, npix), dtype=np.int64)
        return {
            "maps_counts_reco": maps,
            "n_events_total": 0,
            "n_events_kept": 0,
        }

    monkeypatch.setattr(simmod, "_require_sim_deps", lambda: None)
    monkeypatch.setattr(simmod, "CosmologyGrid", DummyCosmologyGrid)
    monkeypatch.setattr(simmod, "SpectrumParams", DummySpectrumParams)
    monkeypatch.setattr(simmod, "PopulationParams", DummyPopulationParams)
    monkeypatch.setattr(simmod, "AstroConfig", DummyAstroCfg)
    monkeypatch.setattr(simmod, "AtmosphereConfig", DummyAtmoCfg)
    monkeypatch.setattr(simmod, "make_expected_counts_maps", _dummy_make_expected_counts_maps)
    monkeypatch.setattr(simmod, "make_atmo_expected_counts_maps", _dummy_make_atmo_expected_counts_maps)
    monkeypatch.setattr(simmod, "poissonize_astro_reco", _dummy_poissonize_astro_reco)
    monkeypatch.setattr(simmod, "poissonize_atmo_reco", _dummy_poissonize_atmo_reco)

    sim_cfg = simmod.SmokeSimConfig(
        n_sims=2,
        nside=8,
        nest=False,
        output_format="npy",
        energy_edges_GeV=tuple(edges.tolist()),
        real_maps_npz=str(fixture_npz),
        enforce_match_real_maps=True,
        reco_batch_size=8_000,
    )

    out = simmod.generate_smoke_sim_bank(outdir=outdir, sim_cfg=sim_cfg, overwrite=True)

    maps_path = Path(out["maps_counts"])
    meta_path = Path(out["meta"])
    assert maps_path.exists()
    assert meta_path.exists()

    maps = np.load(maps_path, mmap_mode="r")
    assert maps.shape == (2, 5, 12 * 8 * 8)
    assert np.issubdtype(maps.dtype, np.integer)

    meta = json.loads(meta_path.read_text())
    assert meta["uses_reco_energy_smearing"] is True
    assert np.isclose(meta["livetime_years_total"], 1.0, rtol=0.0, atol=1e-12)


def test_user_can_override_nside_with_real_maps_metadata(tmp_path: Path, monkeypatch) -> None:
    fixture_npz = tmp_path / "real_maps_fixture.npz"
    edges = _write_fixture_real_maps(fixture_npz)
    outdir = tmp_path / "sim_out_override_nside"

    class DummyCosmologyGrid:
        pass

    class DummySpectrumParams:
        def __init__(self, gamma, Emin, Emax, L):
            self.gamma = gamma
            self.Emin = Emin
            self.Emax = Emax
            self.L = L

    class DummyPopulationParams:
        def __init__(self, n0, fz_fn):
            self.n0 = n0
            self.fz_fn = fz_fn

    class DummyAstroCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyAtmoCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def _dummy_make_expected_counts_maps(spec, pop, cosmo, cfg):
        npix = 12 * int(cfg.nside) * int(cfg.nside)
        B = len(cfg.energy_bin_edges_GeV) - 1
        return {
            "maps_mean_prepsf": np.ones((B, npix), dtype=float),
        }

    def _dummy_make_atmo_expected_counts_maps(cfg):
        npix = 12 * int(cfg.nside) * int(cfg.nside)
        B = len(cfg.energy_bin_edges_GeV) - 1
        return {
            "maps_mean_prepsf": np.ones((B, npix), dtype=float),
            "bin_edges_GeV": np.asarray(cfg.energy_bin_edges_GeV, dtype=float),
            "nside": int(cfg.nside),
            "cfg": cfg,
        }

    def _dummy_poissonize_astro_reco(astro_dict, reco_energy_edges_GeV, rng_seed, batch_size=200_000):
        rng = np.random.default_rng(int(rng_seed))
        B = len(reco_energy_edges_GeV) - 1
        npix = astro_dict["maps_mean_prepsf"].shape[1]
        maps = rng.poisson(0.1, size=(B, npix)).astype(np.int64)
        return {
            "maps_counts_reco": maps,
            "n_events_total": int(maps.sum()),
            "n_events_kept": int(maps.sum()),
        }

    def _dummy_poissonize_atmo_reco(atmo_dict, reco_energy_edges_GeV, rng_seed, batch_size=200_000):
        B = len(reco_energy_edges_GeV) - 1
        npix = int(atmo_dict["maps_mean_prepsf"].shape[1])
        maps = np.zeros((B, npix), dtype=np.int64)
        return {
            "maps_counts_reco": maps,
            "n_events_total": 0,
            "n_events_kept": 0,
        }

    monkeypatch.setattr(simmod, "_require_sim_deps", lambda: None)
    monkeypatch.setattr(simmod, "CosmologyGrid", DummyCosmologyGrid)
    monkeypatch.setattr(simmod, "SpectrumParams", DummySpectrumParams)
    monkeypatch.setattr(simmod, "PopulationParams", DummyPopulationParams)
    monkeypatch.setattr(simmod, "AstroConfig", DummyAstroCfg)
    monkeypatch.setattr(simmod, "AtmosphereConfig", DummyAtmoCfg)
    monkeypatch.setattr(simmod, "make_expected_counts_maps", _dummy_make_expected_counts_maps)
    monkeypatch.setattr(simmod, "make_atmo_expected_counts_maps", _dummy_make_atmo_expected_counts_maps)
    monkeypatch.setattr(simmod, "poissonize_astro_reco", _dummy_poissonize_astro_reco)
    monkeypatch.setattr(simmod, "poissonize_atmo_reco", _dummy_poissonize_atmo_reco)

    sim_cfg = simmod.SmokeSimConfig(
        n_sims=2,
        nside=4,
        nest=False,
        output_format="npy",
        energy_edges_GeV=tuple(edges.tolist()),
        real_maps_npz=str(fixture_npz),
        enforce_match_real_maps=True,
        allow_nside_override=True,
        reco_batch_size=8_000,
    )

    out = simmod.generate_smoke_sim_bank(outdir=outdir, sim_cfg=sim_cfg, overwrite=True)

    maps = np.load(Path(out["maps_counts"]), mmap_mode="r")
    assert maps.shape == (2, 5, 12 * 4 * 4)

    meta = json.loads(Path(out["meta"]).read_text())
    assert int(meta["derived"]["nside"]) == 4
    assert meta["real_maps_match"]["nside_match"] is False


def test_zarr_output_and_sparse_truth_export(tmp_path: Path, monkeypatch) -> None:
    try:
        import zarr  # type: ignore  # noqa: F401
    except Exception:
        return

    fixture_npz = tmp_path / "real_maps_fixture.npz"
    edges = _write_fixture_real_maps(fixture_npz)
    outdir = tmp_path / "sim_out_zarr_truth"

    class DummyCosmologyGrid:
        pass

    class DummySpectrumParams:
        def __init__(self, gamma, Emin, Emax, L):
            self.gamma = gamma
            self.Emin = Emin
            self.Emax = Emax
            self.L = L

    class DummyPopulationParams:
        def __init__(self, n0, fz_fn):
            self.n0 = n0
            self.fz_fn = fz_fn

    class DummyAstroCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyAtmoCfg:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def _dummy_make_expected_counts_maps(spec, pop, cosmo, cfg):
        npix = 12 * int(cfg.nside) * int(cfg.nside)
        B = len(cfg.energy_bin_edges_GeV) - 1
        # Build a deterministic gradient map so top-k sparse truth is stable.
        base = np.linspace(0.0, 10.0, npix, dtype=float)
        maps = np.stack([base + float(j) for j in range(B)], axis=0)
        return {
            "maps_mean_prepsf": maps,
        }

    def _dummy_make_atmo_expected_counts_maps(cfg):
        npix = 12 * int(cfg.nside) * int(cfg.nside)
        B = len(cfg.energy_bin_edges_GeV) - 1
        return {
            "maps_mean_prepsf": np.ones((B, npix), dtype=float),
            "bin_edges_GeV": np.asarray(cfg.energy_bin_edges_GeV, dtype=float),
            "nside": int(cfg.nside),
            "cfg": cfg,
        }

    def _dummy_poissonize_astro_reco(astro_dict, reco_energy_edges_GeV, rng_seed, batch_size=200_000):
        rng = np.random.default_rng(int(rng_seed))
        B = len(reco_energy_edges_GeV) - 1
        npix = int(astro_dict["maps_mean_prepsf"].shape[1])
        maps = rng.poisson(0.2, size=(B, npix)).astype(np.int64)
        return {
            "maps_counts_reco": maps,
            "n_events_total": int(maps.sum()),
            "n_events_kept": int(maps.sum()),
        }

    def _dummy_poissonize_atmo_reco(atmo_dict, reco_energy_edges_GeV, rng_seed, batch_size=200_000):
        B = len(reco_energy_edges_GeV) - 1
        npix = int(atmo_dict["maps_mean_prepsf"].shape[1])
        maps = np.zeros((B, npix), dtype=np.int64)
        return {
            "maps_counts_reco": maps,
            "n_events_total": 0,
            "n_events_kept": 0,
        }

    monkeypatch.setattr(simmod, "_require_sim_deps", lambda: None)
    monkeypatch.setattr(simmod, "CosmologyGrid", DummyCosmologyGrid)
    monkeypatch.setattr(simmod, "SpectrumParams", DummySpectrumParams)
    monkeypatch.setattr(simmod, "PopulationParams", DummyPopulationParams)
    monkeypatch.setattr(simmod, "AstroConfig", DummyAstroCfg)
    monkeypatch.setattr(simmod, "AtmosphereConfig", DummyAtmoCfg)
    monkeypatch.setattr(simmod, "make_expected_counts_maps", _dummy_make_expected_counts_maps)
    monkeypatch.setattr(simmod, "make_atmo_expected_counts_maps", _dummy_make_atmo_expected_counts_maps)
    monkeypatch.setattr(simmod, "poissonize_astro_reco", _dummy_poissonize_astro_reco)
    monkeypatch.setattr(simmod, "poissonize_atmo_reco", _dummy_poissonize_atmo_reco)

    sim_cfg = simmod.SmokeSimConfig(
        n_sims=3,
        nside=8,
        nest=True,
        output_format="zarr",
        zarr_chunk_sims=1,
        energy_edges_GeV=tuple(edges.tolist()),
        real_maps_npz=str(fixture_npz),
        enforce_match_real_maps=True,
        truth_store=True,
        truth_topk=5,
        truth_min_expected_counts=0.0,
        reco_batch_size=8_000,
    )

    out = simmod.generate_smoke_sim_bank(outdir=outdir, sim_cfg=sim_cfg, overwrite=True)

    maps_path = Path(out["maps_counts"])
    assert maps_path.exists()

    import zarr  # type: ignore

    z = zarr.open(str(maps_path), mode="r")
    assert tuple(z.shape) == (3, 5, 12 * 8 * 8)
    assert np.issubdtype(np.dtype(z.dtype), np.unsignedinteger)

    truth_path = Path(out["truth_hotspot_sparse"])
    assert truth_path.exists()
    with np.load(truth_path) as truth:
        assert truth["indptr"].shape[0] == 4  # n_sims + 1
        assert truth["indices"].ndim == 1
        assert truth["values"].ndim == 1
        assert int(truth["nside"]) == 8
        assert bool(truth["nest"]) is True

    meta = json.loads(Path(out["meta"]).read_text())
    assert meta["paths"]["truth_hotspot_sparse"] is not None
