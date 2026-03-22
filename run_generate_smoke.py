from pathlib import Path
import os
import sys
import numpy as np

# Make local src/ imports work
sys.path.insert(0, os.path.abspath("src"))

from icecube_population_project import (
    SmokeSimConfig,
    TightPriorConfig,
    generate_smoke_sim_bank,
    read_real_map_metadata,
)

# Portable project/data paths
PROJECT_ROOT = Path.home() / "neutrinos"
DATA_ROOT = PROJECT_ROOT / "data"
REAL_MAPS_NPZ = DATA_ROOT / "Results" / "icecube10y_counts_maps.npz"
OUTDIR_NAME = os.environ.get("OUTDIR_NAME", "Capel_wide")
OUTDIR = DATA_ROOT / "Sims" / OUTDIR_NAME
OUTDIR.mkdir(parents=True, exist_ok=True)

# Production settings
N_SIMS = int(os.environ.get("N_SIMS", "200000"))
SEED = int(os.environ.get("SEED", "4698"))
OVERWRITE = os.environ.get("OVERWRITE", "true").lower() == "true"








print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_ROOT:", DATA_ROOT)
print("REAL_MAPS_NPZ:", REAL_MAPS_NPZ)
print("OUTDIR:", OUTDIR)
print("REAL_MAPS_NPZ exists:", REAL_MAPS_NPZ.exists())

if not REAL_MAPS_NPZ.exists():
    raise FileNotFoundError(f"Missing real maps NPZ: {REAL_MAPS_NPZ}")

real_meta = read_real_map_metadata(REAL_MAPS_NPZ)

sim_cfg = SmokeSimConfig(
    n_sims=N_SIMS,
    seed=SEED,
    nside=128,
    nest=True,
    output_format="zarr",
    zarr_chunk_sims=1,
    truth_store=True,
    truth_topk=64,
    truth_min_expected_counts=0.0,
    time_years=float(real_meta["livetime_years_total"]),
    energy_edges_GeV=tuple(np.asarray(real_meta["E_edges_GeV"], dtype=float).tolist()),
    real_maps_npz=str(REAL_MAPS_NPZ),
    enforce_match_real_maps=True,
    sources_to_draw=2000,
    energy_grid_per_decade=6,
    reco_batch_size=200_000,
    astro_psf_mode="fixed_event",
    fixed_psf_sigma_deg=1.0,
    atmo_psf_mode="none",
    conventional_model="honda2006",
    include_prompt=False,
)

prior_cfg = TightPriorConfig()

print("N simulations:", sim_cfg.n_sims)
print("NSIDE / NPIX:", sim_cfg.nside, 12 * sim_cfg.nside * sim_cfg.nside)
print("NEST ordering:", sim_cfg.nest)
print("Output format:", sim_cfg.output_format)
print("Store truth:", sim_cfg.truth_store)
print("Livetime [yr]:", sim_cfg.time_years)
print("Energy edges:", np.asarray(sim_cfg.energy_edges_GeV))

outputs = generate_smoke_sim_bank(
    outdir=OUTDIR,
    sim_cfg=sim_cfg,
    prior=prior_cfg,
    overwrite=OVERWRITE,
)

print("Done.")
print(outputs)
