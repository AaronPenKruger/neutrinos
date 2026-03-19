from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pretrain_deephpx_embedding import (
    EmbeddingArchitecture,
    PretrainConfig,
    load_history_csv,
    load_pretrained_embedding_net,
    plot_training_diagnostics,
    pretrain_deephpx_embedding,
)


def _write_toy_sim_dir(root: Path, *, n: int = 24, nside: int = 4, B: int = 5) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)

    npix = 12 * nside * nside
    rng = np.random.default_rng(123)

    # maps_counts.zarr (N, B, npix)
    try:
        import zarr  # type: ignore
    except Exception as e:
        raise RuntimeError("zarr is required for this test") from e

    z = zarr.open(
        str(root / "maps_counts.zarr"),
        mode="w",
        shape=(n, B, npix),
        chunks=(1, B, npix),
        dtype=np.uint32,
    )
    for i in range(n):
        # Low-count toy maps with one bright pixel region varying per sample.
        m = rng.poisson(0.3, size=(B, npix)).astype(np.uint32)
        pix0 = (11 * i) % npix
        for b in range(B):
            m[b, pix0] = np.uint32(50 + 5 * b)
        z[i] = m

    # Sparse truth hotspots (CSR-like), already in NEST ordering.
    indptr = [0]
    indices = []
    values = []
    for i in range(n):
        pix0 = (11 * i) % npix
        pix1 = (pix0 + 7) % npix
        indices.extend([pix0, pix1])
        values.extend([100.0, 40.0])
        indptr.append(len(indices))

    np.savez_compressed(
        root / "truth_hotspot_sparse.npz",
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        values=np.asarray(values, dtype=np.float32),
        nside=np.int64(nside),
        npix=np.int64(npix),
        nest=np.bool_(True),
    )

    # meta + theta.csv (theta kept for compatibility)
    meta = {
        "derived": {
            "nside": int(nside),
            "npix": int(npix),
            "nest": False,
        }
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2))

    theta_csv = root / "theta.csv"
    with theta_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sim_id", "dummy"]) 
        writer.writeheader()
        for i in range(n):
            writer.writerow({"sim_id": i, "dummy": float(i)})

    return root, theta_csv


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_hotspot_pretrain_end_to_end(tmp_path: Path) -> None:
    deephpx_src = PROJECT_ROOT / "external" / "DeepHpx" / "src"
    if not deephpx_src.exists():
        pytest.skip(f"DeepHpx external source not found at {deephpx_src}")

    sim_dir, theta_csv = _write_toy_sim_dir(tmp_path / "toy_sim")
    outdir = tmp_path / "pretrain_out"

    arch = EmbeddingArchitecture(
        cheb_K=3,
        min_nside=2,
        first_hidden=4,
        max_hidden=16,
        embedding_dim=16,
        append_input_stats=True,
    )
    cfg = PretrainConfig(
        label_mode="truth_sparse",
        truth_top_k=1,
        dilate_steps=0,
        batch_size=4,
        max_epochs=2,
        early_stopping_patience=2,
        learning_rate=5e-4,
        num_workers=0,
        seed=7,
        train_fraction=0.7,
        val_fraction=0.2,
        input_ordering="RING",
        target_ordering="NEST",
        amp=False,
    )

    result = pretrain_deephpx_embedding(
        maps_counts_path=sim_dir,  # pass sim_dir directly
        theta_csv_path=theta_csv,
        outdir=outdir,
        arch=arch,
        cfg=cfg,
        device="auto",
        project_root=PROJECT_ROOT,
        deephpx_src_override=deephpx_src,
    )

    art = result["artifacts"]
    for p in (
        art.full_checkpoint,
        art.embedding_checkpoint,
        art.preprocessing_json,
        art.history_csv,
        art.metrics_json,
        art.predictions_npz,
    ):
        assert Path(p).exists(), f"missing artifact: {p}"

    metrics = json.loads(Path(art.metrics_json).read_text())
    assert "best_epoch" in metrics
    assert "best_val_loss" in metrics
    assert "label_meta" in metrics

    hist = load_history_csv(art.history_csv)
    assert len(hist) >= 1
    assert np.isfinite(float(hist[-1]["val_loss"]))

    diag_png = plot_training_diagnostics(art.history_csv, outpath=outdir / "diag.png", show=False)
    assert diag_png is not None and Path(diag_png).exists()

    emb_net, bundle = load_pretrained_embedding_net(
        art.embedding_checkpoint,
        device="auto",
        project_root=PROJECT_ROOT,
        deephpx_src_override=deephpx_src,
    )
    assert emb_net is not None
    assert int(bundle["nside"]) == 4

    # Forward smoke test for exported embedding.
    npix = int(bundle["npix"])
    in_ch = int(bundle["in_channels"])
    x = np.zeros((1, npix, in_ch), dtype=np.float32)
    import torch

    with torch.no_grad():
        z = emb_net(torch.from_numpy(x))
    assert int(z.shape[0]) == 1
    assert int(z.shape[1]) == int(arch.embedding_dim)
