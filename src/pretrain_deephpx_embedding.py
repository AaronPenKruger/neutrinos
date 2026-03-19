#!/usr/bin/env python3
"""pretrain_deephpx_embedding.py

Publication-grade pretraining for DeepHpx (DeepSphere-like) HEALPix embeddings.

This script is a **drop-in replacement** for the earlier regression-style
"compress map -> population parameters" pretraining.

Motivation (Eckner et al. 2025-style):
- Pretrain a **map-to-map U-Net** that performs **pixel-wise hotspot detection**
  on HEALPix count maps.
- Then reuse the **encoder** weights as initialization for the **map->embedding**
  network used by downstream SBI (here: NPE + MAF).

Key design goals:
- Use *your* DeepHpx building blocks:
  - Chebyshev graph convolutions on HEALPix graphs
  - HEALPix-aware max-pooling / unpooling in NESTED ordering
  - Cached Laplacians
- Keep a clean interface compatible with `icecube_population_project.py`:
  - EmbeddingArchitecture
  - PretrainConfig
  - pretrain_deephpx_embedding(...)
  - load_pretrained_embedding_net(...)
  - load_history_csv(...)
  - plot_training_diagnostics(...)

Expected simulation bank layout (compressed, recommended):
  <SIM_DIR>/maps_counts.zarr/         # Zarr group or array; dataset 'maps_counts' has shape (N, B, npix) dtype=uint32
  <SIM_DIR>/truth_hotspot_sparse.npz  # option B: sparse PSF-less truth (CSR: indptr, indices, values)
  <SIM_DIR>/theta.csv                # (N, ...)
  <SIM_DIR>/meta.json                # contains nside, nest, energy_edges_GeV

Backward-compatible layout is still supported:
  <SIM_DIR>/maps_counts.npy          # (N, B, npix) integer

Labeling options:
- Preferred (closest to Eckner+2025): provide a PSF-less point-source map
  (counts deposited only in the source pixel) and threshold it at C_th.
  This script will automatically use it if it finds one of:
    - maps_sources_prepsf.npy
    - maps_astro_prepsf.npy
    - maps_ps_prepsf.npy

- Fallback (no extra simulator outputs required): derive **pseudo-labels** from
  the reconstructed counts map via local-maxima detection on a weighted,
  optionally smoothed score map. You can calibrate the threshold such that the
  mean number of hotspots per map is ~K (e.g. 4).

Outputs:
- <OUTDIR>/checkpoint_unet.pt              # U-Net detector weights + config
- <OUTDIR>/checkpoint_embedding.pt         # encoder+head (map->embedding) weights
- <OUTDIR>/preprocessing.json              # normalization + label calibration
- <OUTDIR>/history.csv                     # training curves
- <OUTDIR>/metrics.json                    # final metrics
- <OUTDIR>/predictions_val_examples.npz     # small qualitative dump

Notes:
- DeepHpx pooling/unpooling assumes NESTED HEALPix ordering.
- Sparse Laplacians are *not* part of the model state_dict; they are rebuilt
  deterministically from (nside, laplacian_kind, ordering) during loading.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

# Optional deps (used if present)
try:
    import zarr  # type: ignore
except Exception:  # pragma: no cover
    zarr = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------------------
# DeepHpx imports (with robust path handling)
# --------------------------------------------------------------------------------------


def _maybe_add_deephpx_to_syspath(
    *,
    project_root: Optional[str | Path] = None,
    deephpx_src_override: Optional[str | Path] = None,
) -> None:
    """Best-effort: make `import deephpx` work.

    This mirrors typical layouts:
      - <project_root>/external/DeepHpx/src
      - <project_root>/DeepHpx/src
      - explicit override

    No error if it still fails; the caller will raise a clearer message.
    """
    import sys

    candidates: List[Path] = []

    if deephpx_src_override is not None:
        candidates.append(Path(deephpx_src_override).expanduser().resolve())

    if project_root is not None:
        pr = Path(project_root).expanduser().resolve()
        candidates.extend(
            [
                pr / "external" / "DeepHpx" / "src",
                pr / "DeepHpx" / "src",
                pr / "external" / "deephpx" / "src",
            ]
        )

    # Environment-variable override
    env = os.environ.get("DEEPHPX_SRC")
    if env:
        candidates.append(Path(env).expanduser().resolve())

    for p in candidates:
        if p.exists() and p.is_dir():
            sp = str(p)
            if sp not in sys.path:
                sys.path.insert(0, sp)


def _require_deephpx(
    *,
    project_root: Optional[str | Path] = None,
    deephpx_src_override: Optional[str | Path] = None,
) -> None:
    """Import DeepHpx and raise a helpful error if unavailable."""
    _maybe_add_deephpx_to_syspath(project_root=project_root, deephpx_src_override=deephpx_src_override)
    try:
        import deephpx  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "DeepHpx could not be imported. Ensure it is installed (pip) or add its `src/` "
            "directory via `project_root=...` or `deephpx_src_override=...` (or env DEEPHPX_SRC)."
        ) from e


# --------------------------------------------------------------------------------------
# Public API dataclasses (kept compatible with the earlier script)
# --------------------------------------------------------------------------------------


Ordering = Literal["RING", "NEST"]
LabelMode = Literal[
    "truth_sparse",  # preferred: sparse PSF-less truth (CSR indices/values) from generator (option B)
    "prepsf_threshold",  # use pre-PSF dense source map if available
    "localmax_threshold",  # pseudo-labels: local maxima above calibrated threshold
    "localmax_topk",  # pseudo-labels: top-K local maxima per map
]
LossMode = Literal["bce", "bce_dice", "focal", "focal_dice"]
NormMode = Literal["batch", "group", "none"]
PoolMode = Literal["max", "average"]


@dataclass(frozen=True)
class EmbeddingArchitecture:
    """Architecture knobs for the U-Net detector and the downstream embedding head.

    Defaults are chosen to mirror Eckner et al. (2025) Table III in spirit:
    - Chebyshev K=5
    - channel progression doubling up to 512
    - U-Net with skip connections

    Notes:
    - `first_hidden` controls the first conv output; stage-I uses two convs:
      `in -> first_hidden -> 2*first_hidden`.
    - `min_nside` is the coarsest resolution (default 2).

    The *embedding head* is a simple linear projection from the flattened
    bottleneck feature map, optionally concatenated with input mean/std (as in
    Eckner+2025).
    """

    # Graph convolution
    cheb_K: int = 5
    laplacian_kind: str = "normalized"  # 'normalized' or 'combinatorial'
    cheb_lmax: float = 2.0  # for normalized Laplacian, lmax<=2 exactly

    # Resolution hierarchy
    min_nside: int = 2

    # Channel schedule
    first_hidden: int = 16
    max_hidden: int = 512

    # Pooling / normalization / regularization
    pool: PoolMode = "max"  # max matches Eckner+2025
    norm: NormMode = "batch"  # batch matches Eckner+2025
    group_norm_groups: int = 16
    dropout: float = 0.0

    # Embedding head (for NPE/MAF)
    embedding_dim: int = 256
    append_input_stats: bool = True
    input_stats_mode: Literal["global", "per_channel"] = "per_channel"
    embedding_mlp_hidden: Tuple[int, ...] = ()  # optional extra FC layers before embedding_dim


@dataclass(frozen=True)
class PretrainConfig:
    """Training + data + label configuration.

    Some fields from the earlier regression pretrain remain for compatibility
    (`target_columns`, `ridge_alpha`) but are ignored by the new hotspot pretrain.
    """

    # --- compatibility fields (ignored) ---
    target_columns: Tuple[str, ...] = ()
    ridge_alpha: float = 1e-3

    # --- map preprocessing ---
    input_ordering: Ordering = "RING"  # how maps_counts.npy is stored
    target_ordering: Ordering = "NEST"  # DeepHpx pooling assumes NEST
    log1p_inputs: bool = False
    standardize_inputs: bool = True

    # --- splits ---
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    seed: int = 4698

    # --- training ---
    batch_size: int = 6
    num_workers: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 120
    early_stopping_patience: int = 20
    min_delta: float = 1e-4
    grad_clip_norm: float = 5.0
    lr_scheduler_patience: int = 6
    lr_scheduler_factor: float = 0.5
    amp: bool = True

    # --- labels (hotspots) ---
    label_mode: LabelMode = "truth_sparse"

    # If label_mode == prepsf_threshold, use this C_th (counts) unless None => auto-calibrate.
    counts_threshold: Optional[float] = None

    # If label_mode == localmax_threshold, auto-calibrate a threshold so mean peaks ~ target.
    target_mean_hotspots: int = 4

    # If label_mode == localmax_topk, take exactly this many peaks per map.
    top_k_hotspots: int = 4

    # If label_mode == truth_sparse, optionally downselect to the top-k stored truth pixels
    # per map (in descending truth value order). If None, use all stored pixels.
    truth_top_k: Optional[int] = None

    # Pseudo-label smoothing (very lightweight): apply this many neighbour-mean steps
    # to the aggregated score map before local-max detection.
    pseudo_smooth_steps: int = 1

    # Dilate the positive mask by this many neighbour hops (helps with PSF width)
    dilate_steps: int = 1

    # Aggregation weights across energy bins for pseudo labels:
    #  - 'inv_mean'  : weight bin b by 1/(mean_b + eps)
    #  - 'unity'     : all bins weight 1
    pseudo_score_weights: Literal["inv_mean", "unity"] = "inv_mean"

    # --- loss ---
    loss_mode: LossMode = "focal_dice"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    dice_weight: float = 0.2

    # If set, overrides auto pos_weight for BCE. (Used only in BCE modes.)
    bce_pos_weight: Optional[float] = None

    # --- caching ---
    cache_dir: Optional[str] = None  # Laplacians + labels cache; default: <outdir>/cache

    # How many maps to use for threshold calibration (None => min(4096, N_train)).
    calib_n_maps: Optional[int] = None


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class PretrainArtifacts:
    outdir: Path
    cache_dir: Path
    full_checkpoint: Path
    embedding_checkpoint: Path
    preprocessing_json: Path
    history_csv: Path
    metrics_json: Path
    predictions_npz: Path

    # --- Backwards-compatible attribute aliases (from the earlier regression pretrain) ---
    @property
    def config_json(self) -> Path:
        """Alias for older code that expected a config JSON path."""
        return self.preprocessing_json

    @property
    def prediction_npz(self) -> Path:
        """Alias for older code that expected a prediction .npz path."""
        return self.predictions_npz

    @property
    def ridge_weights_path(self) -> Optional[Path]:
        """Not produced by the hotspot pretrain (kept for API compatibility)."""
        return None


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def _require_pandas() -> None:
    if pd is None:
        raise ImportError("pandas is required for reading theta.csv. Install with `pip install pandas`.")


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


_SPARSE_HALF_CUDA_SUPPORTED: Optional[bool] = None


def _cuda_sparse_half_mm_supported() -> bool:
    """Whether torch.sparse.mm supports float16 on CUDA in this runtime."""
    global _SPARSE_HALF_CUDA_SUPPORTED
    if _SPARSE_HALF_CUDA_SUPPORTED is not None:
        return bool(_SPARSE_HALF_CUDA_SUPPORTED)

    if not torch.cuda.is_available():
        _SPARSE_HALF_CUDA_SUPPORTED = False
        return False

    try:
        device = torch.device("cuda")
        idx = torch.tensor([[0], [0]], dtype=torch.long, device=device)
        val = torch.tensor([1.0], dtype=torch.float16, device=device)
        lap = torch.sparse_coo_tensor(idx, val, (1, 1), device=device).coalesce()
        x = torch.ones((1, 1), dtype=torch.float16, device=device)
        _ = torch.sparse.mm(lap, x)
        _SPARSE_HALF_CUDA_SUPPORTED = True
    except Exception:
        _SPARSE_HALF_CUDA_SUPPORTED = False

    return bool(_SPARSE_HALF_CUDA_SUPPORTED)


def _infer_nside_from_npix(npix: int) -> int:
    # Prefer Deephpx if available (it has strict validation)
    try:
        from deephpx.healpix.geometry import npix2nside

        return int(npix2nside(int(npix), strict=True))
    except Exception:
        # Fallback: nside = sqrt(npix/12)
        nside_f = math.sqrt(float(npix) / 12.0)
        nside = int(round(nside_f))
        if 12 * nside * nside != int(npix):
            raise ValueError(f"npix={npix} is not a valid HEALPix pixel count")
        return nside


# --------------------------------------------------------------------------------------
# Zarr + compressed truth helpers
# --------------------------------------------------------------------------------------


def _resolve_sim_dir_and_maps_path(maps_counts_path: Path) -> Tuple[Path, Path]:
    """Resolve (sim_dir, maps_path) from a user-supplied path.

    Accepted inputs:
      - path to <SIM_DIR>/maps_counts.npy
      - path to <SIM_DIR>/maps_counts.zarr (directory)
      - path to <SIM_DIR> (directory), in which case we prefer maps_counts.zarr then maps_counts.npy
    """
    p = Path(maps_counts_path)
    if p.is_dir() and p.suffix != '.zarr':
        sim_dir = p
        cand_z = sim_dir / 'maps_counts.zarr'
        cand_n = sim_dir / 'maps_counts.npy'
        if cand_z.exists():
            return sim_dir, cand_z
        if cand_n.exists():
            return sim_dir, cand_n
        raise FileNotFoundError(
            f"Could not find maps_counts.zarr or maps_counts.npy in sim_dir={sim_dir}"
        )

    # p is a file OR a .zarr directory
    if p.is_dir() and p.suffix == '.zarr':
        return p.parent, p
    if p.is_file():
        return p.parent, p

    raise FileNotFoundError(f"maps_counts_path does not exist: {p}")


def _open_maps_counts_any(maps_path: Path):
    """Open maps_counts as either a NumPy memmap (npy) or a Zarr array.

    Returns an object supporting NumPy-style slicing. For Zarr stores, we accept both:
      - an array store directly at maps_path
      - a group store containing dataset 'maps_counts'

    Notes
    -----
    Zarr provides chunked, compressed N-D arrays and supports NumPy-like slicing.
    """
    maps_path = Path(maps_path)

    if maps_path.suffix == '.npy':
        return np.load(maps_path, mmap_mode='r')

    if maps_path.suffix == '.zarr' or maps_path.is_dir():
        if zarr is None:
            raise ImportError(
                "zarr is required to read maps_counts.zarr. Install with `pip install zarr numcodecs`."
            )
        obj = zarr.open(str(maps_path), mode='r')
        try:
            if hasattr(obj, 'keys') and 'maps_counts' in obj:
                obj = obj['maps_counts']
        except Exception:
            pass
        if not hasattr(obj, 'shape'):
            raise ValueError(f"Opened Zarr object has no shape attribute: {type(obj)}")
        return obj

    raise ValueError(f"Unrecognized maps path: {maps_path}")


def _find_truth_sparse_candidate(sim_dir: Path) -> Optional[Path]:
    """Find sparse PSF-less truth file produced by the compressed generator."""
    candidates = [
        sim_dir / 'truth_hotspot_sparse.npz',
        sim_dir / 'hotspot_truth_sparse.npz',
        sim_dir / 'truth_sparse_hotspots.npz',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_truth_sparse(truth_path: Path) -> Dict[str, np.ndarray]:
    """Load CSR-like sparse truth (indptr, indices, values) from an .npz."""
    truth_path = Path(truth_path)
    with np.load(truth_path, allow_pickle=False) as z:
        for k in ('indptr', 'indices', 'values'):
            if k not in z:
                raise KeyError(f"truth sparse file missing key '{k}': {truth_path}")
        out: Dict[str, np.ndarray] = {
            'indptr': np.asarray(z['indptr'], dtype=np.int64),
            'indices': np.asarray(z['indices'], dtype=np.int64),
            'values': np.asarray(z['values'], dtype=np.float32),
        }
        for k in ('nside', 'npix', 'nest'):
            if k in z:
                out[k] = np.asarray(z[k])
        return out


def _compute_truth_sparse_labels_packbits(
    *,
    truth: Dict[str, np.ndarray],
    indices: np.ndarray,
    npix: int,
    nside: int,
    reorder_index: Optional[np.ndarray],
    cfg: PretrainConfig,
    cache_path: Path,
    cache_meta_path: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Create packed-bit hotspot masks from sparse truth (option B).

    Each simulation i has stored pixels:
      pix = indices[indptr[i]:indptr[i+1]]
      val = values[indptr[i]:indptr[i+1]]

    The mask is built by selecting either:
      - all stored pixels (truth_top_k=None), or
      - the top-k pixels by truth value (truth_top_k=k),
    then optionally dilating by neighbour hops.

    Labels are constructed in NEST ordering.
    """

    _require_deephpx()
    from deephpx.graph import neighbors_8

    indptr = truth['indptr']
    pix_all = truth['indices']
    val_all = truth['values']

    if indptr.ndim != 1:
        raise ValueError('truth indptr must be 1D')

    n_sims = int(indptr.size - 1)
    if n_sims <= 0:
        raise ValueError('truth indptr implies zero simulations')

    # Consistency checks (if provided)
    if 'npix' in truth:
        try:
            npix_truth = int(np.asarray(truth['npix']).reshape(-1)[0])
            if npix_truth != int(npix):
                raise ValueError(f"truth npix={npix_truth} does not match maps npix={int(npix)}")
        except Exception as e:
            raise ValueError("Failed to interpret truth['npix']") from e
    if 'nside' in truth:
        try:
            nside_truth = int(np.asarray(truth['nside']).reshape(-1)[0])
            if nside_truth != int(nside):
                raise ValueError(f"truth nside={nside_truth} does not match maps nside={int(nside)}")
        except Exception as e:
            raise ValueError("Failed to interpret truth['nside']") from e

    # Optional: detect whether truth indices are in NEST ordering
    truth_nest = None
    if 'nest' in truth:
        try:
            truth_nest = bool(int(np.asarray(truth['nest']).reshape(-1)[0]))
        except Exception:
            truth_nest = None

    if truth_nest is False:
        # Convert ring->nest (requires healpy)
        import healpy as hp
        pix_all = hp.ring2nest(int(nside), pix_all.astype(np.int64, copy=False)).astype(np.int64, copy=False)

    # reorder_index is used for input RING->NEST reordering. Labels are always NEST.
    _ = reorder_index

    nbytes = int((int(npix) + 7) // 8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    labels_mm = np.memmap(cache_path, mode='w+', dtype=np.uint8, shape=(n_sims, nbytes))
    labels_mm[:] = 0

    neigh = neighbors_8(int(nside), nest=True)

    use_k = cfg.truth_top_k
    use_k = None if use_k is None else int(use_k)

    inds_write = np.asarray(indices, dtype=np.int64)
    for i in inds_write:
        i = int(i)
        a = int(indptr[i])
        b = int(indptr[i + 1])
        if b <= a:
            continue
        pix = pix_all[a:b]
        if use_k is not None and pix.size > use_k:
            vals = val_all[a:b]
            sel = _topk_indices(vals, use_k)
            pix = pix[sel]

        m = np.zeros((int(npix),), dtype=bool)
        pix = pix[(pix >= 0) & (pix < int(npix))]
        m[pix] = True
        m = _dilate_mask_neigh(m, neigh, steps=int(cfg.dilate_steps))
        labels_mm[i] = _packbits_mask(m)

    labels_mm.flush()

    meta = {
        'coverage': 'all',
        'label_mode': 'truth_sparse',
        'truth_top_k': (None if use_k is None else int(use_k)),
        'dilate_steps': int(cfg.dilate_steps),
        'truth_nest': truth_nest,
        'n_samples': int(n_sims),
        'npix': int(npix),
        'nbytes': int(nbytes),
    }
    cache_meta_path.write_text(json.dumps(meta, indent=2))

    labels_ro = np.memmap(cache_path, mode='r', dtype=np.uint8, shape=(n_sims, nbytes))
    return labels_ro, meta


def _nsides_from_nside(nside: int, min_nside: int) -> List[int]:
    nside = int(nside)
    min_nside = int(min_nside)
    if nside < min_nside:
        raise ValueError(f"nside={nside} < min_nside={min_nside}")
    if (nside & (nside - 1)) != 0 or (min_nside & (min_nside - 1)) != 0:
        raise ValueError("nside and min_nside must be powers of 2 for HEALPix pooling")
    if nside % min_nside != 0:
        raise ValueError("nside must be divisible by min_nside")

    out = [nside]
    while out[-1] > min_nside:
        out.append(out[-1] // 2)
    return out


def _default_channels_for_nsides(
    nsides: Sequence[int],
    *,
    first_hidden: int,
    max_hidden: int,
) -> List[int]:
    """Return channels per resolution (full->coarse) matching Eckner-like doubling."""
    if len(nsides) < 1:
        raise ValueError("nsides must be non-empty")

    # Stage-I ends at 2*first_hidden (e.g., 32)
    c0 = int(2 * first_hidden)
    ch: List[int] = [c0]
    for _ in range(1, len(nsides)):
        ch.append(int(min(max_hidden, ch[-1] * 2)))
    return ch


def _build_reorder_index_ring_to_nest(nside: int) -> np.ndarray:
    """Return index array idx such that map_nest = map_ring[idx]."""
    import healpy as hp

    npix = 12 * nside * nside
    ring_pix = np.arange(npix, dtype=np.int64)
    nest_pix = hp.ring2nest(nside, ring_pix)
    # nest_to_ring[nest_pix] = ring_pix
    nest_to_ring = np.empty_like(ring_pix)
    nest_to_ring[nest_pix] = ring_pix
    return nest_to_ring


# --------------------------------------------------------------------------------------
# Laplacian building / caching (DeepHpx backend)
# --------------------------------------------------------------------------------------

def _save_sparse_coo_tensor(t: torch.Tensor, path: Path) -> None:
    """Save a torch sparse COO tensor robustly.

    We avoid relying on DeepHpx cache helpers here because the DeepHpx
    cache module in this repository only guarantees SciPy sparse support.
    """
    t = t.coalesce().detach().cpu()
    payload = {
        'indices': t.indices(),
        'values': t.values(),
        'size': tuple(t.size()),
    }
    torch.save(payload, path)


def _load_sparse_coo_tensor(path: Path) -> torch.Tensor:
    """Load a torch sparse COO tensor saved by _save_sparse_coo_tensor."""
    payload = torch.load(path, map_location='cpu')
    t = torch.sparse_coo_tensor(payload['indices'], payload['values'], size=payload['size'])
    return t.coalesce()


def _build_or_load_scaled_laplacians(
    *,
    nsides: Sequence[int],
    arch: EmbeddingArchitecture,
    cache_dir: Path,
    device: torch.device,
    nest: bool = True,
) -> List[torch.Tensor]:
    """Return list of scaled Laplacians (torch sparse COO) for each nside in `nsides`.

    The tensors are moved to `device`.
    """

    _require_deephpx()
    from deephpx.graph import adjacency_from_neighbors, laplacian_from_adjacency, neighbors_8
    from deephpx.graph.cache import ensure_cache_dir
    from deephpx.graph.laplacian import scale_laplacian_for_chebyshev, to_torch_sparse_coo

    ensure_cache_dir(cache_dir)

    laplacians: List[torch.Tensor] = []

    for nside in nsides:
        # Chebyshev scaling needs an upper bound on the Laplacian spectrum.
        # For the *normalized* Laplacian, lambda_max <= 2 exactly.
        # For the combinatorial Laplacian on an 8-neighbour unweighted graph,
        # lambda_max <= 2*deg_max <= 16 (safe upper bound).
        if arch.laplacian_kind == 'normalized':
            lmax_used = 2.0
        else:
            lmax_used = float(arch.cheb_lmax)
            if lmax_used <= 2.0:
                lmax_used = 16.0

        tag = f"nside{int(nside)}_{arch.laplacian_kind}_nest{int(bool(nest))}_lmax{lmax_used:.3g}_K{arch.cheb_K}"
        fpath = cache_dir / f"laplacian_scaled_{tag}.pt"

        L_t: Optional[torch.Tensor] = None
        if fpath.exists():
            try:
                L_t = _load_sparse_coo_tensor(fpath)
            except Exception:
                L_t = None

        if L_t is None:
            neigh = neighbors_8(int(nside), nest=bool(nest))
            A = adjacency_from_neighbors(neigh, symmetric=True, remove_self_loops=True)
            L = laplacian_from_adjacency(A, kind=arch.laplacian_kind)

            # Avoid costly eigsh for the normalized Laplacian (spectrum in [0, 2]).
            lmax = float(lmax_used)
            L_scaled = scale_laplacian_for_chebyshev(L, lmax=lmax)
            L_t = to_torch_sparse_coo(L_scaled)

            _save_sparse_coo_tensor(L_t, fpath)

        if not L_t.is_sparse:
            raise RuntimeError("Expected sparse Laplacian tensor")

        laplacians.append(L_t.to(device))

    return laplacians


# --------------------------------------------------------------------------------------
# Hotspot label generation
# --------------------------------------------------------------------------------------


def _find_prepsf_maps_candidate(sim_dir: Path) -> Optional[Path]:
    """Search common filenames for a PSF-less source map."""
    candidates = [
        sim_dir / "maps_sources_prepsf.npy",
        sim_dir / "maps_astro_prepsf.npy",
        sim_dir / "maps_ps_prepsf.npy",
        sim_dir / "maps_mean_prepsf.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _dilate_mask_neigh(mask: np.ndarray, neigh: np.ndarray, steps: int) -> np.ndarray:
    """Dilate a boolean mask by neighbour hops using a neighbour list."""
    if steps <= 0:
        return mask
    m = mask.astype(bool, copy=True)
    for _ in range(int(steps)):
        idx = np.flatnonzero(m)
        if idx.size == 0:
            break
        nb = neigh[idx].reshape(-1)
        nb = nb[nb >= 0]
        m[nb] = True
    return m


def _smooth_score_neigh(score: np.ndarray, neigh: np.ndarray, steps: int) -> np.ndarray:
    """Very lightweight smoothing: neighbour mean iteration."""
    if steps <= 0:
        return score

    s = score.astype(np.float32, copy=True)

    # Precompute degrees (number of valid neighbours) for normalization.
    deg = (neigh >= 0).sum(axis=1).astype(np.float32)
    deg = np.maximum(deg, 1.0)

    for _ in range(int(steps)):
        # Gather neighbour values; handle -1 safely.
        neigh_safe = neigh.copy()
        neigh_safe[neigh_safe < 0] = 0
        nb_vals = s[neigh_safe]  # (npix, 8)
        nb_vals[neigh < 0] = 0.0
        s = (s + nb_vals.sum(axis=1)) / (1.0 + deg)

    return s


def _local_maxima_mask(score: np.ndarray, neigh: np.ndarray, *, strict: bool = True) -> np.ndarray:
    """Return boolean mask of pixels that are local maxima in `score`."""
    score = np.asarray(score)
    neigh_safe = neigh.copy()
    neigh_safe[neigh_safe < 0] = 0

    nb_vals = score[neigh_safe]
    # mask invalid neighbours
    if np.any(neigh < 0):
        nb_vals = nb_vals.copy()
        nb_vals[neigh < 0] = -np.inf

    nb_max = nb_vals.max(axis=1)
    if strict:
        return score > nb_max
    return score >= nb_max


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    k = int(k)
    if values.size <= k:
        return np.argsort(values)[::-1].astype(np.int64)
    # argpartition gives arbitrary order within partition; refine.
    idx = np.argpartition(values, -k)[-k:]
    idx = idx[np.argsort(values[idx])[::-1]]
    return idx.astype(np.int64)


def _packbits_mask(mask: np.ndarray) -> np.ndarray:
    """Pack a boolean mask (npix,) -> uint8 (ceil(npix/8),)."""
    bits = np.packbits(mask.astype(np.uint8), bitorder="little")
    return bits


def _unpackbits_mask(bits: np.ndarray, npix: int) -> np.ndarray:
    mask = np.unpackbits(bits, bitorder="little")
    return mask[:npix].astype(np.uint8)


def _compute_pseudo_labels_packbits(
    *,
    maps_counts: np.ndarray,
    indices: np.ndarray,
    calib_indices: Optional[np.ndarray] = None,
    npix: int,
    nside: int,
    in_channels: int,
    reorder_index: Optional[np.ndarray],
    cfg: PretrainConfig,
    cache_path: Path,
    cache_meta_path: Path,
    map_channel_mean_for_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute pseudo hotspot labels and save them as a packbits .npy.

    Returns:
      - labels_bits_memmap: memmap array shape (N, nbytes)
      - meta: dict with calibration details
    """

    _require_deephpx()
    from deephpx.graph import neighbors_8

    neigh = neighbors_8(nside, nest=True)  # labels always in NEST ordering

    # Weighting across energy bins
    inds_write = np.asarray(indices, dtype=np.int64)
    inds_calib = np.asarray(calib_indices if calib_indices is not None else indices, dtype=np.int64)

    if cfg.pseudo_score_weights == "unity":
        w = np.ones((in_channels,), dtype=np.float32)
    else:
        if map_channel_mean_for_weights is None:
            raise ValueError("map_channel_mean_for_weights is required for inv_mean weighting")
        w = 1.0 / (np.asarray(map_channel_mean_for_weights, dtype=np.float32) + 1e-6)

    # Calibration: pick threshold on local-max scores such that mean peaks ~ target.
    strict_max = True

    calib_n = cfg.calib_n_maps
    if calib_n is None:
        calib_n = int(min(4096, inds_calib.size))
    calib_n = int(max(256, min(calib_n, inds_calib.size)))

    rng = np.random.default_rng(int(cfg.seed) + 1337)
    calib_idx = rng.choice(inds_calib, size=calib_n, replace=False)

    kth_vals: List[float] = []

    def _score_for_map(i: int) -> np.ndarray:
        x = np.asarray(maps_counts[i], dtype=np.float32)  # (B,npix)
        if cfg.log1p_inputs:
            x = np.log1p(x)
        if reorder_index is not None:
            x = x[..., reorder_index]
        score = (w.reshape(-1, 1) * x).sum(axis=0)
        score = _smooth_score_neigh(score, neigh, steps=int(cfg.pseudo_smooth_steps))
        return score

    # Compute per-map kth local-max value
    k = int(max(1, cfg.target_mean_hotspots))
    for i in calib_idx:
        score = _score_for_map(int(i))
        lm = _local_maxima_mask(score, neigh, strict=strict_max)
        vals = score[lm]
        if vals.size >= k:
            kth = float(np.sort(vals)[-k])
            kth_vals.append(kth)
        elif vals.size > 0:
            kth_vals.append(float(vals.min()))
        else:
            kth_vals.append(float(np.max(score)))

    threshold = float(np.median(kth_vals))

    # If user explicitly provided counts_threshold, override
    if cfg.counts_threshold is not None:
        threshold = float(cfg.counts_threshold)

    # Allocate output packed bits
    nbytes = int((npix + 7) // 8)
    # We'll create a memmap file to avoid huge RAM usage
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    labels_mm = np.memmap(cache_path, mode="w+", dtype=np.uint8, shape=(int(maps_counts.shape[0]), nbytes))

    # Fill only requested indices; others remain 0
    labels_mm[:] = 0

    # Process in chunks
    chunk = 256
    inds = inds_write

    for start in range(0, inds.size, chunk):
        sl = inds[start : start + chunk]
        for i in sl:
            score = _score_for_map(int(i))
            lm = _local_maxima_mask(score, neigh, strict=strict_max)
            m = lm & (score >= threshold)

            if cfg.label_mode == "localmax_topk":
                cand = np.flatnonzero(lm)
                if cand.size > 0:
                    top_idx = _topk_indices(score[cand], int(cfg.top_k_hotspots))
                    keep = cand[top_idx]
                    m = np.zeros((npix,), dtype=bool)
                    m[keep] = True
                else:
                    m = np.zeros((npix,), dtype=bool)

            m = _dilate_mask_neigh(m, neigh, steps=int(cfg.dilate_steps))
            labels_mm[int(i)] = _packbits_mask(m)

    labels_mm.flush()

    meta = {
        "coverage": "all",
        "label_mode": cfg.label_mode,
        "pseudo_score_weights": cfg.pseudo_score_weights,
        "pseudo_smooth_steps": int(cfg.pseudo_smooth_steps),
        "dilate_steps": int(cfg.dilate_steps),
        "threshold": float(threshold),
        "strict_local_max": strict_max,
        "calib_n_maps": int(calib_n),
        "target_mean_hotspots": int(cfg.target_mean_hotspots),
        "top_k_hotspots": int(cfg.top_k_hotspots),
        "n_samples": int(maps_counts.shape[0]),
        "npix": int(npix),
        "nbytes": int(nbytes),
    }

    cache_meta_path.write_text(json.dumps(meta, indent=2))

    # Re-open in read-only mode for training
    labels_ro = np.memmap(cache_path, mode="r", dtype=np.uint8, shape=(int(maps_counts.shape[0]), nbytes))
    return labels_ro, meta


def _compute_prepsf_labels_packbits(
    *,
    prepsf_maps: np.ndarray,
    indices: np.ndarray,
    calib_indices: Optional[np.ndarray] = None,
    npix: int,
    reorder_index: Optional[np.ndarray],
    cfg: PretrainConfig,
    cache_path: Path,
    cache_meta_path: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute labels from a PSF-less source map by thresholding at C_th.

    Expected `prepsf_maps` shapes:
      - (N, B, npix) or (N, npix)

    We aggregate across B if present: counts_total = sum_b prepsf[b].

    If cfg.counts_threshold is None, calibrate an integer threshold so the mean
    number of hotspot pixels per map is ~ cfg.target_mean_hotspots.
    """

    inds_write = np.asarray(indices, dtype=np.int64)
    inds_calib = np.asarray(calib_indices if calib_indices is not None else indices, dtype=np.int64)

    if prepsf_maps.ndim == 3:
        # (N,B,npix) -> (N,npix)
        pre = np.asarray(prepsf_maps, dtype=np.float32).sum(axis=1)
    elif prepsf_maps.ndim == 2:
        pre = np.asarray(prepsf_maps, dtype=np.float32)
    else:
        raise ValueError(f"Unexpected prepsf_maps shape: {prepsf_maps.shape}")

    if pre.shape[1] != npix:
        raise ValueError(f"prepsf maps npix mismatch: {pre.shape[1]} vs {npix}")

    if reorder_index is not None:
        pre = pre[:, reorder_index]

    # Calibrate integer threshold
    if cfg.counts_threshold is None:
        # Use a subset of training indices for calibration
        calib_n = cfg.calib_n_maps
        if calib_n is None:
            calib_n = int(min(4096, inds_calib.size))
        calib_n = int(max(256, min(calib_n, inds_calib.size)))

        rng = np.random.default_rng(int(cfg.seed) + 2025)
        calib_idx = rng.choice(inds_calib, size=calib_n, replace=False)

        # Candidate thresholds from observed counts
        max_c = int(np.max(pre[calib_idx]))
        max_c = int(min(max_c, 10_000))
        # Scan thresholds 1..max_c
        target = float(max(1, cfg.target_mean_hotspots))
        best_t = 1
        best_err = float("inf")
        for t in range(1, max_c + 1):
            n_hot = (pre[calib_idx] >= float(t)).sum(axis=1).mean()
            err = abs(float(n_hot) - target)
            if err < best_err:
                best_err = err
                best_t = t
        threshold = float(best_t)
    else:
        threshold = float(cfg.counts_threshold)

    # Allocate packed output
    nbytes = int((npix + 7) // 8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    labels_mm = np.memmap(cache_path, mode="w+", dtype=np.uint8, shape=(int(pre.shape[0]), nbytes))
    labels_mm[:] = 0

    inds = inds_write
    for i in inds:
        m = pre[int(i)] >= threshold
        labels_mm[int(i)] = _packbits_mask(m)

    labels_mm.flush()

    meta = {
        "coverage": "all",
        "label_mode": "prepsf_threshold",
        "threshold": float(threshold),
        "target_mean_hotspots": int(cfg.target_mean_hotspots),
        "calib_n_maps": int(cfg.calib_n_maps) if cfg.calib_n_maps is not None else None,
        "n_samples": int(pre.shape[0]),
        "npix": int(npix),
        "nbytes": int(nbytes),
    }
    cache_meta_path.write_text(json.dumps(meta, indent=2))

    labels_ro = np.memmap(cache_path, mode="r", dtype=np.uint8, shape=(int(pre.shape[0]), nbytes))
    return labels_ro, meta


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class SimMapHotspotDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset for (map, hotspot_mask) training.

    Returns:
      x: float tensor (npix, B)
      y: float tensor (npix, 1) in {0,1}
      idx: int tensor scalar (global sim index)
    """

    def __init__(
        self,
        maps_counts: np.ndarray,
        labels_packbits: np.ndarray,
        indices: np.ndarray,
        *,
        npix: int,
        in_channels: int,
        map_channel_mean: np.ndarray,
        map_channel_std: np.ndarray,
        log1p_inputs: bool,
        reorder_index: Optional[np.ndarray],
    ) -> None:
        self.maps_counts = maps_counts
        self.labels_packbits = labels_packbits
        self.indices = np.asarray(indices, dtype=np.int64)
        self.npix = int(npix)
        self.in_channels = int(in_channels)
        self.map_channel_mean = np.asarray(map_channel_mean, dtype=np.float32)
        self.map_channel_std = np.asarray(map_channel_std, dtype=np.float32)
        self.log1p_inputs = bool(log1p_inputs)
        self.reorder_index = None if reorder_index is None else np.asarray(reorder_index, dtype=np.int64)

        if self.map_channel_mean.shape != (self.in_channels,) or self.map_channel_std.shape != (self.in_channels,):
            raise ValueError("map_channel_mean/std shape mismatch")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, local_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i = int(self.indices[int(local_idx)])

        x = np.asarray(self.maps_counts[i], dtype=np.float32)  # (B,npix)
        if self.reorder_index is not None:
            x = x[..., self.reorder_index]

        if self.log1p_inputs:
            x = np.log1p(x)

        # standardize per channel
        x = (x - self.map_channel_mean.reshape(-1, 1)) / (self.map_channel_std.reshape(-1, 1) + 1e-6)

        # (B,npix) -> (npix,B)
        x = np.swapaxes(x, 0, 1)

        bits = np.asarray(self.labels_packbits[i], dtype=np.uint8)
        y1 = _unpackbits_mask(bits, self.npix)  # (npix,)
        y = y1.reshape(-1, 1).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(y), torch.tensor(i, dtype=torch.int64)


# --------------------------------------------------------------------------------------
# Losses and metrics
# --------------------------------------------------------------------------------------


def _dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary segmentation.

    logits: (B,N,1) or (B,N)
    targets: same shape, float in {0,1}
    """
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits[..., 0]
    if targets.ndim == 3 and targets.shape[-1] == 1:
        targets = targets[..., 0]

    probs = torch.sigmoid(logits)
    targets = targets.float()

    num = 2.0 * (probs * targets).sum(dim=1)
    den = (probs + targets).sum(dim=1)
    dice = (num + eps) / (den + eps)
    return 1.0 - dice.mean()


class FocalLossWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.reduction = str(reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 3 and logits.shape[-1] == 1:
            logits = logits[..., 0]
        if targets.ndim == 3 and targets.shape[-1] == 1:
            targets = targets[..., 0]

        targets = targets.float()

        # BCE with logits per element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@torch.no_grad()
def _segmentation_metrics_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.0
) -> Dict[str, float]:
    """Compute precision/recall/F1/IoU at a logits threshold."""
    if logits.ndim == 3 and logits.shape[-1] == 1:
        logits = logits[..., 0]
    if targets.ndim == 3 and targets.shape[-1] == 1:
        targets = targets[..., 0]

    pred = logits >= float(threshold)
    true = targets >= 0.5

    tp = (pred & true).sum().item()
    fp = (pred & (~true)).sum().item()
    fn = ((~pred) & true).sum().item()

    tp_f = float(tp)
    fp_f = float(fp)
    fn_f = float(fn)

    prec = tp_f / (tp_f + fp_f + 1e-12)
    rec = tp_f / (tp_f + fn_f + 1e-12)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-12)
    iou = tp_f / (tp_f + fp_f + fn_f + 1e-12)

    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou}


# --------------------------------------------------------------------------------------
# DeepHpx U-Net (map->map)
# --------------------------------------------------------------------------------------


class _Norm(nn.Module):
    def __init__(self, mode: NormMode, n_channels: int, group_norm_groups: int = 16):
        super().__init__()
        self.mode = str(mode)
        c = int(n_channels)
        if self.mode == "batch":
            self.bn = nn.BatchNorm1d(c)
        elif self.mode == "group":
            g = int(group_norm_groups)
            g = max(1, min(g, c))
            # ensure divisibility; fallback to 1 group (LayerNorm-like)
            while c % g != 0 and g > 1:
                g -= 1
            self.bn = nn.GroupNorm(num_groups=g, num_channels=c)
        elif self.mode == "none":
            self.bn = None
        else:
            raise ValueError(f"Unknown norm mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,C)
        if self.bn is None:
            return x
        y = x.permute(0, 2, 1)  # (B,C,N)
        y = self.bn(y)
        return y.permute(0, 2, 1)


class GraphConvBlock(nn.Module):
    """GC -> Norm -> ReLU (+ optional dropout)."""

    def __init__(
        self,
        *,
        laplacian: torch.Tensor,
        K: int,
        in_channels: int,
        out_channels: int,
        norm: NormMode,
        group_norm_groups: int,
        dropout: float,
        use_bias: bool = True,
    ) -> None:
        super().__init__()

        from deephpx.nn.chebyshev import SphericalChebConv

        self.conv = SphericalChebConv(
            laplacian=laplacian,
            K=int(K),
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            bias=bool(use_bias),
        )
        self.norm = _Norm(norm, int(out_channels), group_norm_groups=group_norm_groups)
        self.dropout_p = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = F.relu(x)
        if self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x


class HealpixUNet(nn.Module):
    """Eckner-style DeepSphere U-Net on full-sky HEALPix graphs.

    Input:  (B, Npix, Cin)
    Output: (B, Npix, 1) logits

    Notes:
    - Pool/unpool assume NEST ordering.
    - We use DeepHpx HealpixMaxPool/Unpool or average equivalents.
    """

    def __init__(
        self,
        *,
        laplacians: Sequence[torch.Tensor],
        nsides: Sequence[int],
        in_channels: int,
        channels_per_res: Sequence[int],
        arch: EmbeddingArchitecture,
    ) -> None:
        super().__init__()

        if len(laplacians) != len(nsides):
            raise ValueError("laplacians and nsides must have same length")
        if len(channels_per_res) != len(nsides):
            raise ValueError("channels_per_res and nsides must have same length")

        self.nsides = tuple(int(s) for s in nsides)
        self.in_channels = int(in_channels)
        self.channels_per_res = tuple(int(c) for c in channels_per_res)
        self.arch = arch

        from deephpx.nn.healpix_pooling import (
            HealpixAvgPool,
            HealpixAvgUnpool,
            HealpixMaxPool,
            HealpixMaxUnpool,
        )

        if arch.pool == "max":
            self.pool = HealpixMaxPool(return_indices=True)
            self.unpool = HealpixMaxUnpool()
        else:
            self.pool = HealpixAvgPool()
            self.unpool = HealpixAvgUnpool()

        # Stage I: two convs at full resolution
        self.enc0a = GraphConvBlock(
            laplacian=laplacians[0],
            K=arch.cheb_K,
            in_channels=self.in_channels,
            out_channels=int(arch.first_hidden),
            norm=arch.norm,
            group_norm_groups=arch.group_norm_groups,
            dropout=arch.dropout,
        )
        self.enc0b = GraphConvBlock(
            laplacian=laplacians[0],
            K=arch.cheb_K,
            in_channels=int(arch.first_hidden),
            out_channels=self.channels_per_res[0],
            norm=arch.norm,
            group_norm_groups=arch.group_norm_groups,
            dropout=arch.dropout,
        )

        # Down path: conv at res i, then pool to res i+1
        self.down_convs = nn.ModuleList()
        for i in range(0, len(nsides) - 1):
            self.down_convs.append(
                GraphConvBlock(
                    laplacian=laplacians[i],
                    K=arch.cheb_K,
                    in_channels=self.channels_per_res[i],
                    out_channels=self.channels_per_res[i + 1],
                    norm=arch.norm,
                    group_norm_groups=arch.group_norm_groups,
                    dropout=arch.dropout,
                )
            )

        # Bottleneck conv at coarsest res
        self.bottleneck = GraphConvBlock(
            laplacian=laplacians[-1],
            K=arch.cheb_K,
            in_channels=self.channels_per_res[-1],
            out_channels=self.channels_per_res[-1],
            norm=arch.norm,
            group_norm_groups=arch.group_norm_groups,
            dropout=arch.dropout,
        )

        # Up path: conv at low res (i+1) to channels at high res (i), unpool, concat skip, conv refine
        self.up_convs_pre = nn.ModuleList()
        self.up_convs_post = nn.ModuleList()

        for i in reversed(range(0, len(nsides) - 1)):
            if arch.pool == "max":
                # Keep channel count before max-unpool so pooling indices remain compatible.
                pre_out = self.channels_per_res[i + 1]
                post_in = self.channels_per_res[i + 1] + self.channels_per_res[i]
            else:
                pre_out = self.channels_per_res[i]
                post_in = 2 * self.channels_per_res[i]

            # pre: at low resolution nsides[i+1]
            self.up_convs_pre.append(
                GraphConvBlock(
                    laplacian=laplacians[i + 1],
                    K=arch.cheb_K,
                    in_channels=self.channels_per_res[i + 1],
                    out_channels=pre_out,
                    norm=arch.norm,
                    group_norm_groups=arch.group_norm_groups,
                    dropout=arch.dropout,
                )
            )
            # post: after concat at high resolution nsides[i]
            self.up_convs_post.append(
                GraphConvBlock(
                    laplacian=laplacians[i],
                    K=arch.cheb_K,
                    in_channels=post_in,
                    out_channels=self.channels_per_res[i],
                    norm=arch.norm,
                    group_norm_groups=arch.group_norm_groups,
                    dropout=arch.dropout,
                )
            )

        # Output: BN? In Table III they have BN ◦ GC. We'll implement (GC only) for simplicity.
        from deephpx.nn.chebyshev import SphericalChebConv

        self.out_conv = SphericalChebConv(
            laplacian=laplacians[0],
            K=arch.cheb_K,
            in_channels=self.channels_per_res[0],
            out_channels=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,C)
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B,N,C), got {tuple(x.shape)}")

        # Encoder stage I
        x0 = self.enc0b(self.enc0a(x))  # (B,N0,C0)

        # Down path
        skips: List[torch.Tensor] = [x0]
        pool_idx: List[Optional[torch.Tensor]] = []

        h = x0
        for i, conv in enumerate(self.down_convs):
            h = conv(h)
            if self.arch.pool == "max":
                h, idx = self.pool(h)
                pool_idx.append(idx)
            else:
                h = self.pool(h)
                pool_idx.append(None)

            # Store skip for all but the coarsest (the last pooling leads to nsides[-1])
            if i < len(self.down_convs) - 1:
                skips.append(h)

        # Bottleneck
        h = self.bottleneck(h)

        # Up path (reverse)
        # Note: up_convs_* were appended in reverse resolution order.
        # We iterate them in that stored order.
        for j in range(len(self.up_convs_pre)):
            # Determine which pooling index / skip to use.
            # down_convs length = len(nsides)-1. pool_idx has same.
            # The j-th up block corresponds to i = (len(nsides)-2 - j).
            i = (len(self.nsides) - 2) - j

            h = self.up_convs_pre[j](h)

            if self.arch.pool == "max":
                idx = pool_idx[i]
                if idx is None:
                    raise RuntimeError("Missing pooling indices for max unpool")
                h = self.unpool(h, idx)
            else:
                h = self.unpool(h)

            # Concat skip at this resolution (if available)
            if i == 0:
                skip = skips[0]
            else:
                # skips[1] corresponds to nsides[1], ..., skips[-1] to nsides[-2]
                skip = skips[i]

            h = torch.cat([h, skip], dim=-1)
            h = self.up_convs_post[j](h)

        out = self.out_conv(h)  # (B,N0,1)
        return out


# --------------------------------------------------------------------------------------
# Embedding network wrapper (encoder reuse)
# --------------------------------------------------------------------------------------


class HealpixEncoderFromUNet(nn.Module):
    """Map->embedding network reusing the U-Net encoder weights.

    This follows Eckner+2025 Table III logic:
      - run encoder down to min_nside
      - flatten bottleneck feature map
      - append (mean, std) of input map
      - FC -> embedding_dim (optionally via hidden layers)

    Output: (B, embedding_dim)

    Note: This module does *not* include the U-Net decoder.
    """

    def __init__(
        self,
        *,
        unet: HealpixUNet,
        in_channels: int,
        embedding_dim: int,
        append_input_stats: bool,
        input_stats_mode: Literal["global", "per_channel"],
        mlp_hidden: Sequence[int] = (),
    ) -> None:
        super().__init__()

        self.in_channels = int(in_channels)
        self.embedding_dim = int(embedding_dim)
        self.append_input_stats = bool(append_input_stats)
        self.input_stats_mode = str(input_stats_mode)

        # Reuse encoder modules by reference.
        self.enc0a = unet.enc0a
        self.enc0b = unet.enc0b
        self.down_convs = unet.down_convs
        self.pool = unet.pool
        self.arch_pool = unet.arch.pool
        self.bottleneck = unet.bottleneck

        # Infer bottleneck feature size
        nside_min = int(unet.nsides[-1])
        npix_min = 12 * nside_min * nside_min
        c_min = int(unet.channels_per_res[-1])
        flat_dim = int(npix_min * c_min)

        stats_dim = 0
        if self.append_input_stats:
            if self.input_stats_mode == "global":
                stats_dim = 2
            elif self.input_stats_mode == "per_channel":
                stats_dim = 2 * self.in_channels
            else:
                raise ValueError(f"Unknown input_stats_mode: {self.input_stats_mode}")

        mlp_in = flat_dim + stats_dim

        layers: List[nn.Module] = []
        prev = mlp_in
        for h in mlp_hidden:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, int(self.embedding_dim)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,Npix,Cin)
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B,N,C), got {tuple(x.shape)}")

        # Input stats computed on the standardized input (consistent with Eckner).
        stats: Optional[torch.Tensor] = None
        if self.append_input_stats:
            if self.input_stats_mode == "global":
                mean = x.mean(dim=(1, 2), keepdim=False)
                std = x.std(dim=(1, 2), keepdim=False, unbiased=False)
                stats = torch.stack([mean, std], dim=1)
            else:
                mean = x.mean(dim=1)  # (B,C)
                std = x.std(dim=1, unbiased=False)  # (B,C)
                stats = torch.cat([mean, std], dim=1)  # (B,2C)

        h = self.enc0b(self.enc0a(x))

        for conv in self.down_convs:
            h = conv(h)
            if self.arch_pool == "max":
                h, _idx = self.pool(h)
            else:
                h = self.pool(h)

        h = self.bottleneck(h)
        h = h.reshape(h.shape[0], -1)

        if stats is not None:
            h = torch.cat([h, stats], dim=1)

        return self.mlp(h)


# --------------------------------------------------------------------------------------
# Training helpers
# --------------------------------------------------------------------------------------


def _make_split_indices(n: int, *, train_fraction: float, val_fraction: float, seed: int) -> SplitIndices:
    n = int(n)
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be in (0,1)")
    if not (0 <= val_fraction < 1):
        raise ValueError("val_fraction must be in [0,1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)

    n_train = int(round(train_fraction * n))
    n_val = int(round(val_fraction * n))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))

    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return SplitIndices(train=train, val=val, test=test)


def _compute_channel_stats(
    maps_counts: Any,
    indices: np.ndarray,
    *,
    log1p_inputs: bool,
    chunk_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std over (maps, pixels) on the given indices.

    Supports both NumPy memmaps and Zarr arrays.

    For Zarr backends, contiguous slice reads are typically faster than fancy indexing,
    so we aggregate over contiguous runs of sorted indices.
    """

    inds = np.asarray(indices, dtype=np.int64)
    n = int(inds.size)
    B = int(maps_counts.shape[1])
    npix = int(maps_counts.shape[2])

    if n == 0:
        return np.zeros((B,), dtype=np.float32), np.ones((B,), dtype=np.float32)

    s1 = np.zeros((B,), dtype=np.float64)
    s2 = np.zeros((B,), dtype=np.float64)
    total = 0

    inds_sorted = np.sort(inds)

    # contiguous runs
    runs: List[Tuple[int, int]] = []
    start = int(inds_sorted[0])
    prev = start
    for v in inds_sorted[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev + 1))
        start = v
        prev = v
    runs.append((start, prev + 1))

    max_block = int(max(1, chunk_size))

    for a, b in runs:
        cur = a
        while cur < b:
            end = min(b, cur + max_block)
            x = np.asarray(maps_counts[cur:end], dtype=np.float64)  # (bs,B,npix)
            if log1p_inputs:
                x = np.log1p(x)
            s1 += x.sum(axis=(0, 2))
            s2 += (x * x).sum(axis=(0, 2))
            total += int(end - cur) * npix
            cur = end

    mean = s1 / max(1, total)
    var = s2 / max(1, total) - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)


def _auto_pos_weight_from_labels(labels_packbits: np.ndarray, indices: np.ndarray, npix: int) -> float:
    """Compute pos_weight = Nneg/Npos over given indices (for BCEWithLogitsLoss).

    Implemented efficiently on *packed* labels via a uint8 popcount LUT.
    """
    inds = np.asarray(indices, dtype=np.int64)
    n = int(inds.size)
    if n == 0:
        return 1.0

    # Popcount LUT for uint8
    lut = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)

    nbytes = int((int(npix) + 7) // 8)
    extra_bits = int(nbytes * 8 - int(npix))
    last_mask: Optional[int] = None
    if extra_bits > 0:
        valid = 8 - extra_bits
        # With bitorder='little' the valid pixels occupy the *least significant* bits.
        last_mask = (1 << valid) - 1

    npos = 0
    chunk = 2048
    for start in range(0, n, chunk):
        sl = inds[start : start + chunk]
        b = np.asarray(labels_packbits[sl], dtype=np.uint8)  # (bs,nbytes)
        if b.ndim != 2 or b.shape[1] != nbytes:
            raise ValueError('labels_packbits has unexpected shape')
        if last_mask is None:
            npos += int(lut[b].sum())
        else:
            npos += int(lut[b[:, :-1]].sum())
            npos += int(lut[(b[:, -1] & last_mask)].sum())

    n_total = n * int(npix)
    nneg = max(0, n_total - npos)
    if npos <= 0:
        return 1.0
    return float(nneg / max(1, npos))

# --------------------------------------------------------------------------------------
# Main training entry point (public API)
# --------------------------------------------------------------------------------------


def pretrain_deephpx_embedding(
    *,
    maps_counts_path: str | Path,
    theta_csv_path: str | Path,
    outdir: str | Path,
    arch: EmbeddingArchitecture = EmbeddingArchitecture(),
    cfg: PretrainConfig = PretrainConfig(),
    device: str = "auto",
    project_root: Optional[str | Path] = None,
    deephpx_src_override: Optional[str | Path] = None,
    meta_json_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Train an Eckner-style hotspot U-Net and export a pretrained DeepHpx encoder.

    Args:
        maps_counts_path: path to maps_counts.npy or maps_counts.zarr (or sim_dir containing one of them)
        theta_csv_path: kept for compatibility (not required for hotspot pretrain)
        outdir: output directory
        arch: architecture config
        cfg: training config
        device: 'auto'|'cpu'|'cuda'|'cuda:0'...
        project_root: used to locate DeepHpx source if not installed
        deephpx_src_override: explicit DeepHpx src path
        meta_json_path: optional meta.json path (default: sibling of maps_counts)

    Returns:
        dict with keys:
          - artifacts: PretrainArtifacts
          - unet: trained HealpixUNet
          - embedding_net: HealpixEncoderFromUNet
          - preprocessing: dict
          - history: list[dict]
          - metrics: dict
    """

    _set_seed(int(cfg.seed))

    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    maps_counts_path = Path(maps_counts_path).expanduser().resolve()
    theta_csv_path = Path(theta_csv_path).expanduser().resolve()

    # DeepHpx import
    _require_deephpx(project_root=project_root, deephpx_src_override=deephpx_src_override)

    # Device
    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)
    amp_requested = bool(cfg.amp and device_t.type == "cuda")
    amp_enabled = bool(amp_requested)
    if amp_requested and not _cuda_sparse_half_mm_supported():
        amp_enabled = False
        print(
            "[pretrain] AMP requested, but CUDA sparse.mm(float16) is not supported "
            "in this PyTorch/CUDA build. Falling back to full precision."
        )

    # Resolve sim_dir and maps path (supports .npy, .zarr, or passing sim_dir)
    sim_dir, maps_path = _resolve_sim_dir_and_maps_path(maps_counts_path)
    maps_counts = _open_maps_counts_any(maps_path)

    if len(getattr(maps_counts, 'shape', ())) != 3:
        raise ValueError(f"Expected maps_counts with shape (N,B,npix), got {getattr(maps_counts, 'shape', None)}")

    n_samples = int(maps_counts.shape[0])
    in_channels = int(maps_counts.shape[1])
    npix = int(maps_counts.shape[2])
    nside = _infer_nside_from_npix(npix)

    # Meta.json
    if meta_json_path is None:
        meta_json_path = sim_dir / "meta.json"
    meta_json_path = Path(meta_json_path).expanduser().resolve()
    meta: Dict[str, Any] = {}
    if meta_json_path.exists():
        meta = json.loads(meta_json_path.read_text())

    # Infer map ordering from meta.json if present
    maps_are_nest = None
    try:
        maps_are_nest = bool(meta.get("derived", {}).get("nest"))
    except Exception:
        maps_are_nest = None

    input_ordering = cfg.input_ordering
    if maps_are_nest is not None:
        input_ordering = "NEST" if maps_are_nest else "RING"

    reorder_idx: Optional[np.ndarray] = None
    if input_ordering == "RING" and cfg.target_ordering == "NEST":
        reorder_idx = _build_reorder_index_ring_to_nest(nside)

    # Splits
    split = _make_split_indices(
        n_samples,
        train_fraction=float(cfg.train_fraction),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )

    # Channel stats for standardization (and pseudo-label weights)
    map_mean, map_std = _compute_channel_stats(
        maps_counts,
        split.train,
        log1p_inputs=bool(cfg.log1p_inputs),
        chunk_size=max(16, int(cfg.batch_size) * 4),
    )

    map_mean_for_weights = map_mean.copy()

    if not cfg.standardize_inputs:
        map_mean = np.zeros_like(map_mean)
        map_std = np.ones_like(map_std)

    # Cache directory
    cache_dir = Path(cfg.cache_dir).expanduser().resolve() if cfg.cache_dir else (outdir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Labels cache paths
    labels_path = cache_dir / f"hotspot_labels_{cfg.label_mode}_packbits_nside{nside}_npix{npix}.dat"
    labels_meta_path = cache_dir / f"hotspot_labels_{cfg.label_mode}_meta_nside{nside}_npix{npix}.json"

    labels_packbits: Optional[np.ndarray] = None
    label_meta: Dict[str, Any] = {}

    # Try preferred truth labels (compressed sparse truth, then dense pre-PSF, else pseudo-labels)
    truth_sparse_path = _find_truth_sparse_candidate(sim_dir)
    prepsf_path = _find_prepsf_maps_candidate(sim_dir)

    if cfg.label_mode == "truth_sparse":
        if truth_sparse_path is None:
            raise FileNotFoundError(
                "label_mode='truth_sparse' requested but no sparse truth file found. "
                "Expected truth_hotspot_sparse.npz next to maps_counts.zarr/.npy."
            )

    if cfg.label_mode == "prepsf_threshold":
        if prepsf_path is None:
            raise FileNotFoundError(
                "label_mode='prepsf_threshold' requested but no pre-PSF maps file found. "
                "Expected e.g. maps_astro_prepsf.npy next to maps_counts."
            )

    if labels_path.exists() and labels_meta_path.exists():
        # Load cached labels only if on-disk size and metadata match this run.
        nbytes = int((npix + 7) // 8)
        expected_bytes = int(n_samples) * int(nbytes)

        try:
            label_meta = json.loads(labels_meta_path.read_text())
        except Exception:
            label_meta = {}

        should_rebuild = False
        if label_meta.get("coverage") != "all":
            # Older/partial caches (e.g. train-only labels) are unsafe.
            should_rebuild = True

        if not should_rebuild:
            try:
                cached_bytes = int(labels_path.stat().st_size)
            except Exception:
                cached_bytes = -1
            if cached_bytes != expected_bytes:
                should_rebuild = True

        # If present, metadata sizes must match.
        if not should_rebuild and "n_samples" in label_meta:
            try:
                if int(label_meta["n_samples"]) != int(n_samples):
                    should_rebuild = True
            except Exception:
                should_rebuild = True
        if not should_rebuild and "npix" in label_meta:
            try:
                if int(label_meta["npix"]) != int(npix):
                    should_rebuild = True
            except Exception:
                should_rebuild = True
        if not should_rebuild and "nbytes" in label_meta:
            try:
                if int(label_meta["nbytes"]) != int(nbytes):
                    should_rebuild = True
            except Exception:
                should_rebuild = True

        if not should_rebuild:
            try:
                labels_packbits = np.memmap(labels_path, mode="r", dtype=np.uint8, shape=(n_samples, nbytes))
            except (OSError, ValueError):
                should_rebuild = True

        if should_rebuild:
            labels_packbits = None
            label_meta = {}
    if labels_packbits is None:
        # Build labels
        if cfg.label_mode == "truth_sparse":
            truth = _load_truth_sparse(truth_sparse_path)  # type: ignore[arg-type]
            labels_packbits, label_meta = _compute_truth_sparse_labels_packbits(
                truth=truth,
                indices=np.arange(n_samples, dtype=np.int64),
                npix=npix,
                nside=nside,
                reorder_index=reorder_idx,
                cfg=cfg,
                cache_path=labels_path,
                cache_meta_path=labels_meta_path,
            )

        elif cfg.label_mode == "prepsf_threshold":
            prepsf_maps = np.load(prepsf_path, mmap_mode="r")  # type: ignore[arg-type]
            labels_packbits, label_meta = _compute_prepsf_labels_packbits(
                prepsf_maps=prepsf_maps,
                indices=np.arange(n_samples, dtype=np.int64),
                calib_indices=split.train,
                npix=npix,
                reorder_index=reorder_idx,
                cfg=cfg,
                cache_path=labels_path,
                cache_meta_path=labels_meta_path,
            )
        else:
            labels_packbits, label_meta = _compute_pseudo_labels_packbits(
                maps_counts=maps_counts,
                indices=np.arange(n_samples, dtype=np.int64),
                calib_indices=split.train,
                npix=npix,
                nside=nside,
                in_channels=in_channels,
                reorder_index=reorder_idx,
                cfg=cfg,
                cache_path=labels_path,
                cache_meta_path=labels_meta_path,
                map_channel_mean_for_weights=map_mean_for_weights if cfg.pseudo_score_weights == "inv_mean" else None,
            )

    if labels_packbits is None:
        raise RuntimeError("Failed to create/load labels")

    # Datasets
    train_ds = SimMapHotspotDataset(
        maps_counts,
        labels_packbits,
        split.train,
        npix=npix,
        in_channels=in_channels,
        map_channel_mean=map_mean,
        map_channel_std=map_std,
        log1p_inputs=bool(cfg.log1p_inputs),
        reorder_index=reorder_idx,
    )
    val_ds = SimMapHotspotDataset(
        maps_counts,
        labels_packbits,
        split.val,
        npix=npix,
        in_channels=in_channels,
        map_channel_mean=map_mean,
        map_channel_std=map_std,
        log1p_inputs=bool(cfg.log1p_inputs),
        reorder_index=reorder_idx,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=(device_t.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=(device_t.type == "cuda"),
        drop_last=False,
    )

    # Build Laplacians (cached)
    nsides = _nsides_from_nside(nside, arch.min_nside)
    channels_per_res = _default_channels_for_nsides(nsides, first_hidden=arch.first_hidden, max_hidden=arch.max_hidden)

    laplacians = _build_or_load_scaled_laplacians(
        nsides=nsides,
        arch=arch,
        cache_dir=cache_dir,
        device=device_t,
        nest=True,
    )

    # Model
    unet = HealpixUNet(
        laplacians=laplacians,
        nsides=nsides,
        in_channels=in_channels,
        channels_per_res=channels_per_res,
        arch=arch,
    ).to(device_t)

    # Loss
    if cfg.loss_mode in ("bce", "bce_dice"):
        if cfg.bce_pos_weight is None:
            pos_w = _auto_pos_weight_from_labels(labels_packbits, split.train, npix)
        else:
            pos_w = float(cfg.bce_pos_weight)
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device_t))
        focal_loss = None
    else:
        bce_loss = None
        focal_loss = FocalLossWithLogits(gamma=float(cfg.focal_gamma), alpha=float(cfg.focal_alpha))
        pos_w = None

    # Optimizer
    opt = torch.optim.AdamW(unet.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=float(cfg.lr_scheduler_factor),
        patience=int(cfg.lr_scheduler_patience),
        verbose=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    # Artifacts
    artifacts = PretrainArtifacts(
        outdir=outdir,
        cache_dir=cache_dir,
        full_checkpoint=outdir / "checkpoint_unet.pt",
        embedding_checkpoint=outdir / "checkpoint_embedding.pt",
        preprocessing_json=outdir / "preprocessing.json",
        history_csv=outdir / "history.csv",
        metrics_json=outdir / "metrics.json",
        predictions_npz=outdir / "predictions_val_examples.npz",
    )

    # Training loop
    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    patience_left = int(cfg.early_stopping_patience)

    t_start = time.time()

    for epoch in range(1, int(cfg.max_epochs) + 1):
        epoch_t0 = time.time()
        unet.train()
        train_loss_acc = 0.0
        n_train_batches = 0

        for xb, yb, _ib in train_loader:
            xb = xb.to(device_t, non_blocking=True)
            yb = yb.to(device_t, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = unet(xb)
                if bce_loss is not None:
                    loss = bce_loss(logits, yb)
                else:
                    assert focal_loss is not None
                    loss = focal_loss(logits, yb)

                if cfg.loss_mode in ("bce_dice", "focal_dice") and float(cfg.dice_weight) > 0:
                    dloss = _dice_loss_from_logits(logits, yb)
                    loss = loss + float(cfg.dice_weight) * dloss

            scaler.scale(loss).backward()
            if float(cfg.grad_clip_norm) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=float(cfg.grad_clip_norm))
            scaler.step(opt)
            scaler.update()

            train_loss_acc += float(loss.detach().item())
            n_train_batches += 1

        train_loss = train_loss_acc / max(1, n_train_batches)

        # Validation
        unet.eval()
        val_loss_acc = 0.0
        n_val_batches = 0
        metric_acc = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

        with torch.no_grad():
            for xb, yb, _ib in val_loader:
                xb = xb.to(device_t, non_blocking=True)
                yb = yb.to(device_t, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = unet(xb)
                    if bce_loss is not None:
                        loss = bce_loss(logits, yb)
                    else:
                        assert focal_loss is not None
                        loss = focal_loss(logits, yb)

                    if cfg.loss_mode in ("bce_dice", "focal_dice") and float(cfg.dice_weight) > 0:
                        loss = loss + float(cfg.dice_weight) * _dice_loss_from_logits(logits, yb)

                val_loss_acc += float(loss.detach().item())
                n_val_batches += 1

                m = _segmentation_metrics_from_logits(logits, yb, threshold=0.0)
                for k in metric_acc:
                    metric_acc[k] += float(m[k])

        val_loss = val_loss_acc / max(1, n_val_batches)
        for k in metric_acc:
            metric_acc[k] /= max(1, n_val_batches)

        sched.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in metric_acc.items()},
            "lr": float(opt.param_groups[0]["lr"]),
            "epoch_seconds": float("nan"),
            "elapsed_seconds": float("nan"),
            "eta_early_seconds": float("nan"),
            "eta_max_seconds": float("nan"),
        }
        history.append(row)

        # Early stopping / checkpointing
        improved = val_loss < (best_val - float(cfg.min_delta))
        if improved:
            best_val = val_loss
            best_epoch = epoch
            patience_left = int(cfg.early_stopping_patience)

            torch.save(
                {
                    "state_dict": unet.state_dict(),
                    "arch": asdict(arch),
                    "cfg": asdict(cfg),
                    "nside": nside,
                    "npix": npix,
                    "in_channels": in_channels,
                    "nsides": nsides,
                    "channels_per_res": channels_per_res,
                    "preprocessing": {
                        "map_channel_mean": map_mean.tolist(),
                        "map_channel_std": map_std.tolist(),
                        "reorder_ring_to_nest": (reorder_idx.tolist() if reorder_idx is not None else None),
                        "label_meta": label_meta,
                        "meta_json": meta,
                    },
                    "history": history,
                    "best": {"epoch": best_epoch, "val_loss": best_val},
                },
                artifacts.full_checkpoint,
            )
        else:
            patience_left -= 1

        elapsed_seconds = float(time.time() - t_start)
        epoch_seconds = float(time.time() - epoch_t0)
        avg_epoch_seconds = elapsed_seconds / float(epoch)
        eta_max_seconds = avg_epoch_seconds * float(max(0, int(cfg.max_epochs) - epoch))
        eta_early_seconds = avg_epoch_seconds * float(max(0, patience_left))
        row["epoch_seconds"] = epoch_seconds
        row["elapsed_seconds"] = elapsed_seconds
        row["eta_early_seconds"] = eta_early_seconds
        row["eta_max_seconds"] = eta_max_seconds

        # Console log
        msg = (
            f"[epoch {epoch:04d}] train_loss={train_loss:.4g} val_loss={val_loss:.4g} "
            f"val_f1={metric_acc['f1']:.3f} lr={row['lr']:.2e} "
            f"elapsed={_format_duration(elapsed_seconds)} "
            f"eta_early={_format_duration(eta_early_seconds)} "
            f"eta_max={_format_duration(eta_max_seconds)}"
        )
        if improved:
            msg += "  (best)"
        print(msg)

        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best val_loss {best_val:.4g})")
            break

    t_end = time.time()

    # Load best checkpoint back (for export)
    ckpt = torch.load(artifacts.full_checkpoint, map_location=device_t)
    unet.load_state_dict(ckpt["state_dict"])

    # Build embedding net from U-Net encoder
    embedding_net = HealpixEncoderFromUNet(
        unet=unet,
        in_channels=in_channels,
        embedding_dim=int(arch.embedding_dim),
        append_input_stats=bool(arch.append_input_stats),
        input_stats_mode=arch.input_stats_mode,
        mlp_hidden=arch.embedding_mlp_hidden,
    ).to(device_t)

    # Save embedding checkpoint
    torch.save(
        {
            "state_dict": embedding_net.state_dict(),
            "arch": asdict(arch),
            "cfg": asdict(cfg),
            "nside": nside,
            "npix": npix,
            "in_channels": in_channels,
            "nsides": nsides,
            "channels_per_res": channels_per_res,
            "preprocessing": ckpt["preprocessing"],
            "best": ckpt.get("best", {}),
        },
        artifacts.embedding_checkpoint,
    )

    # Save preprocessing JSON
    preprocessing = ckpt["preprocessing"]
    artifacts.preprocessing_json.write_text(json.dumps(preprocessing, indent=2))

    # Save history CSV
    _write_history_csv(history, artifacts.history_csv)

    # Final metrics
    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "train_seconds": float(t_end - t_start),
        "device": str(device_t),
        "amp_requested": bool(amp_requested),
        "amp_enabled": bool(amp_enabled),
        "label_meta": label_meta,
        "final": history[-1] if history else {},
        "pos_weight": float(pos_w) if pos_w is not None else None,
    }
    artifacts.metrics_json.write_text(json.dumps(metrics, indent=2))

    # Small qualitative dump: first few val maps
    _dump_val_predictions(
        unet=unet,
        dataset=val_ds,
        device=device_t,
        outpath=artifacts.predictions_npz,
        n_examples=4,
    )

    return {
        "artifacts": artifacts,
        "unet": unet,
        "embedding_net": embedding_net,
        "preprocessing": preprocessing,
        "history": history,
        "metrics": metrics,
    }


def _write_history_csv(history: Sequence[Dict[str, Any]], path: Path) -> None:
    if not history:
        path.write_text("epoch,train_loss,val_loss\n")
        return

    keys = list(history[0].keys())
    lines = [",".join(keys)]
    for row in history:
        lines.append(
            ",".join(
                [
                    str(row.get(k, ""))
                    if not isinstance(row.get(k, ""), float)
                    else ("{:.10g}".format(float(row.get(k))))
                    for k in keys
                ]
            )
        )
    path.write_text("\n".join(lines))


@torch.no_grad()
def _dump_val_predictions(
    *,
    unet: HealpixUNet,
    dataset: SimMapHotspotDataset,
    device: torch.device,
    outpath: Path,
    n_examples: int = 4,
) -> None:
    n = min(int(n_examples), len(dataset))
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ps: List[np.ndarray] = []
    idxs: List[int] = []

    unet.eval()

    for j in range(n):
        x, y, idx = dataset[j]
        xb = x.unsqueeze(0).to(device)
        logits = unet(xb)
        prob = torch.sigmoid(logits)[0].detach().cpu().numpy()  # (npix,1)

        xs.append(x.numpy())
        ys.append(y.numpy())
        ps.append(prob)
        idxs.append(int(idx.item()))

    np.savez_compressed(
        outpath,
        x=np.stack(xs, axis=0),
        y=np.stack(ys, axis=0),
        p=np.stack(ps, axis=0),
        idx=np.asarray(idxs, dtype=np.int64),
    )


# --------------------------------------------------------------------------------------
# Loading helpers (public API)
# --------------------------------------------------------------------------------------


def load_pretrained_embedding_net(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    project_root: Optional[str | Path] = None,
    deephpx_src_override: Optional[str | Path] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load an embedding net checkpoint produced by this script.

    Returns (net, bundle) where bundle contains preprocessing + config.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    _require_deephpx(project_root=project_root, deephpx_src_override=deephpx_src_override)

    if device == "auto":
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)
    bundle = torch.load(checkpoint_path, map_location=device_t)

    arch = EmbeddingArchitecture(**bundle["arch"])  # type: ignore[arg-type]

    nside = int(bundle["nside"])
    npix = int(bundle["npix"])
    in_channels = int(bundle["in_channels"])

    nsides = list(bundle["nsides"])
    channels_per_res = list(bundle["channels_per_res"])

    cache_dir = checkpoint_path.parent / "cache"
    laplacians = _build_or_load_scaled_laplacians(
        nsides=nsides,
        arch=arch,
        cache_dir=cache_dir,
        device=device_t,
        nest=True,
    )

    # We need a U-Net instance to borrow encoder modules; build then wrap.
    unet = HealpixUNet(
        laplacians=laplacians,
        nsides=nsides,
        in_channels=in_channels,
        channels_per_res=channels_per_res,
        arch=arch,
    ).to(device_t)

    embedding_net = HealpixEncoderFromUNet(
        unet=unet,
        in_channels=in_channels,
        embedding_dim=int(arch.embedding_dim),
        append_input_stats=bool(arch.append_input_stats),
        input_stats_mode=arch.input_stats_mode,
        mlp_hidden=arch.embedding_mlp_hidden,
    ).to(device_t)

    embedding_net.load_state_dict(bundle["state_dict"])
    embedding_net.eval()

    return embedding_net, bundle


def load_history_csv(path: str | Path) -> List[Dict[str, float]]:
    """Load history CSV written by this script."""
    path = Path(path).expanduser().resolve()
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    out: List[Dict[str, float]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        row: Dict[str, float] = {}
        for k, v in zip(header, parts):
            try:
                row[k] = float(v)
            except Exception:
                # keep as NaN
                row[k] = float("nan")
        out.append(row)
    return out


def plot_training_diagnostics(
    history_csv: str | Path,
    *,
    outpath: Optional[str | Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """Plot training curves (loss + val F1) to a PNG.

    This function is intentionally dependency-light; it uses matplotlib.
    """
    import matplotlib.pyplot as plt

    hist = load_history_csv(history_csv)
    if not hist:
        raise ValueError("Empty history")

    epochs = [int(h.get("epoch", i + 1)) for i, h in enumerate(hist)]
    train_loss = [float(h.get("train_loss", float("nan"))) for h in hist]
    val_loss = [float(h.get("val_loss", float("nan"))) for h in hist]
    val_f1 = [float(h.get("val_f1", float("nan"))) for h in hist]

    fig = plt.figure(figsize=(8, 4.5))
    ax1 = fig.add_subplot(111)
    ax1.plot(epochs, train_loss, label="train_loss", color="tab:blue", linewidth=2.4)
    ax1.plot(epochs, val_loss, label="val_loss", color="tab:orange", linewidth=2.2)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1, label="val_f1", color="tab:green", linewidth=2.8)
    ax2.set_ylabel("val F1")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()

    if outpath is None:
        outpath = Path(history_csv).with_suffix(".png")
    outpath = Path(outpath).expanduser().resolve()
    fig.savefig(outpath, dpi=160)

    if show:
        plt.show()
    plt.close(fig)

    return outpath


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pretrain DeepHpx hotspot U-Net + export embedding net")
    p.add_argument("--sim-dir", type=str, required=True, help="Directory containing maps_counts.npy + theta.csv")
    p.add_argument("--outdir", type=str, required=True, help="Output directory")
    p.add_argument("--device", type=str, default="auto")

    # Label mode
    p.add_argument("--label-mode", type=str, default=PretrainConfig().label_mode)
    p.add_argument("--target-hotspots", type=int, default=PretrainConfig().target_mean_hotspots)
    p.add_argument("--topk-hotspots", type=int, default=PretrainConfig().top_k_hotspots)
    p.add_argument("--counts-threshold", type=float, default=float("nan"))

    # Training
    p.add_argument("--batch-size", type=int, default=PretrainConfig().batch_size)
    p.add_argument("--epochs", type=int, default=PretrainConfig().max_epochs)
    p.add_argument("--lr", type=float, default=PretrainConfig().learning_rate)
    p.add_argument("--seed", type=int, default=PretrainConfig().seed)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)

    sim_dir = Path(args.sim_dir).expanduser().resolve()
    maps_path = (sim_dir / "maps_counts.zarr") if (sim_dir / "maps_counts.zarr").exists() else (sim_dir / "maps_counts.npy")
    theta_path = sim_dir / "theta.csv"

    if not maps_path.exists():
        raise FileNotFoundError(str(maps_path))
    if not theta_path.exists():
        # keep compatibility but allow missing
        theta_path = maps_path

    cfg = PretrainConfig(
        label_mode=str(args.label_mode),  # type: ignore[arg-type]
        target_mean_hotspots=int(args.target_hotspots),
        top_k_hotspots=int(args.topk_hotspots),
        counts_threshold=(None if not np.isfinite(args.counts_threshold) else float(args.counts_threshold)),
        batch_size=int(args.batch_size),
        max_epochs=int(args.epochs),
        learning_rate=float(args.lr),
        seed=int(args.seed),
    )

    rep = pretrain_deephpx_embedding(
        maps_counts_path=maps_path,
        theta_csv_path=theta_path,
        outdir=args.outdir,
        cfg=cfg,
        device=args.device,
    )

    print("Done. Wrote:")
    art: PretrainArtifacts = rep["artifacts"]
    print(" -", art.full_checkpoint)
    print(" -", art.embedding_checkpoint)
    print(" -", art.history_csv)


if __name__ == "__main__":
    main()
