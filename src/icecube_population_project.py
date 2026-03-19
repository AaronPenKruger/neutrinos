"""Flat public API for IceCube population project modules."""

from data_to_maps import build_maps, read_real_map_metadata, read_real_map_summary_json
from generate_sims_capel_wide import (
    SmokeSimConfig,
    TightPriorConfig,
    SanityCheckReport,
    fz_capel,
    generate_smoke_sim_bank,
    run_sanity_checks,
    sample_theta,
)
from pretrain_deephpx_embedding import (
    EmbeddingArchitecture,
    PretrainConfig,
    pretrain_deephpx_embedding,
    load_pretrained_embedding_net,
    load_history_csv,
    plot_training_diagnostics,
)

__all__ = [
    "build_maps",
    "read_real_map_metadata",
    "read_real_map_summary_json",
    "generate_smoke_sim_bank",
    "run_sanity_checks",
    "SmokeSimConfig",
    "TightPriorConfig",
    "SanityCheckReport",
    "sample_theta",
    "fz_capel",
    "EmbeddingArchitecture",
    "PretrainConfig",
    "pretrain_deephpx_embedding",
    "load_pretrained_embedding_net",
    "load_history_csv",
    "plot_training_diagnostics",
]
