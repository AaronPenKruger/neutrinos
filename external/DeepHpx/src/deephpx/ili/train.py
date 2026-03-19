"""Training helpers (LtU-ILI + DeepHpx).

This module is intentionally small: it provides a single convenience
function that wires together:

- an LtU-ILI data loader (e.g. ``ili.dataloaders.NumpyLoader``)
- a prior (torch distribution)
- a DeepHpx embedding network (e.g. :class:`deephpx.ili.HealpixEmbeddingNet`)
- an SBI normalizing flow model (e.g. MAF / NSF)

The heavy lifting (training loops, checkpointing, etc.) remains the
responsibility of LtU-ILI and the underlying backend.

References
----------
- LtU-ILI's suggested ``InferenceRunner.load(..., backend='sbi', engine='NPE', ...)``
  interface is documented in their docs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def train_sbi_posterior(
    loader: Any,
    *,
    prior: Any,
    embedding_net: Optional[Any] = None,
    engine: str = "NPE",
    model: str = "maf",
    nde_kwargs: Optional[Dict[str, Any]] = None,
    train_args: Optional[Dict[str, Any]] = None,
    out_dir: Optional[str | Path] = None,
    device: str = "cpu",
    proposal: Optional[Any] = None,
    name: str = "deephpx",
    signatures: Optional[list[str]] = None,
    seed: Optional[int] = None,
):
    """Train an SBI posterior via LtU-ILI (single-round).

    Parameters
    ----------
    loader:
        An LtU-ILI dataloader providing ``(x, theta)`` pairs (e.g.
        ``ili.dataloaders.NumpyLoader``).
    prior:
        A torch distribution (or LtU-ILI wrapper) over parameters.
    embedding_net:
        Optional embedding network applied to inputs ``x`` before the flow.
        For HEALPix maps, use :class:`deephpx.ili.HealpixEmbeddingNet`.
    engine:
        Inference engine string (default: ``'NPE'``).
    model:
        Density estimator architecture (default: ``'maf'``).
    nde_kwargs:
        Extra keyword arguments forwarded to ``ili.utils.load_nde_sbi``.
    train_args:
        Training hyperparameters forwarded to LtU-ILI's runner.
    out_dir:
        Optional output directory for LtU-ILI artifacts.
    device:
        ``'cpu'`` or ``'cuda'``.
    proposal:
        Optional proposal distribution (defaults to prior in LtU-ILI).
    name:
        Optional name used for saving.
    signatures:
        Optional list of net "signatures" for saving.
    seed:
        Optional torch seed forwarded to LtU-ILI runner.

    Returns
    -------
    posterior, summaries, runner:
        The trained sbi posterior, LtU-ILI summaries, and the runner instance.
    """

    try:
        import ili  # noqa: F401
        from ili.utils import load_nde_sbi  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "LtU-ILI is required for train_sbi_posterior(). "
            "Install with `pip install 'deephpx[ili]'` (or install ltu-ili[pytorch] directly)."
        ) from e

    nde_kwargs = {} if nde_kwargs is None else dict(nde_kwargs)
    train_args = {} if train_args is None else dict(train_args)

    net = load_nde_sbi(engine=engine, model=model, embedding_net=embedding_net, **nde_kwargs)

    # Prefer the universal InferenceRunner interface, but fall back to SBIRunner
    # directly if needed (e.g. older/newer LtU-ILI versions).
    runner = None
    try:
        from ili.inference import InferenceRunner  # type: ignore

        runner = InferenceRunner.load(
            backend="sbi",
            engine=engine,
            prior=prior,
            nets=[net],
            train_args=train_args,
            out_dir=out_dir,
            device=device,
            proposal=proposal,
            name=name,
            signatures=signatures,
        )
    except Exception:
        from ili.inference.runner_sbi import SBIRunner  # type: ignore

        runner = SBIRunner(
            prior=prior,
            engine=engine,
            nets=[net],
            train_args=train_args,
            out_dir=out_dir,
            device=device,
            proposal=proposal,
            name=name,
            signatures=signatures,
        )

    posterior, summaries = runner(loader, seed=seed)
    return posterior, summaries, runner


def train_lampe_posterior(
    loader: Any,
    *,
    prior: Any,
    embedding_net: Optional[Any] = None,
    engine: str = "NPE",
    model: str = "maf",
    nde_kwargs: Optional[Dict[str, Any]] = None,
    train_args: Optional[Dict[str, Any]] = None,
    out_dir: Optional[str | Path] = None,
    device: str = "cpu",
    proposal: Optional[Any] = None,
    name: str = "deephpx",
    signatures: Optional[list[str]] = None,
    seed: Optional[int] = None,
):
    """Train a *lampe* (streaming-friendly) NPE posterior via LtU-ILI.

    This mirrors :func:`train_sbi_posterior` but uses LtU-ILI's lampe backend.

    Notes
    -----
    LtU-ILI's lampe runner can consume a ``TorchLoader`` that wraps PyTorch
    DataLoaders (train/val), enabling streaming training. In contrast, the sbi
    runner typically loads all ``(x, theta)`` into memory.
    """

    try:
        import ili  # noqa: F401
        from ili.utils import load_nde_lampe  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "LtU-ILI is required for train_lampe_posterior(). "
            "Install with `pip install 'deephpx[ili]'` (or install ltu-ili[pytorch] directly)."
        ) from e

    nde_kwargs = {} if nde_kwargs is None else dict(nde_kwargs)
    train_args = {} if train_args is None else dict(train_args)

    # load_nde_lampe returns a list of constructors (callables)
    nets = load_nde_lampe(
        model=model,
        embedding_net=embedding_net,
        device=device,
        engine=engine,
        **nde_kwargs,
    )

    runner = None
    # Prefer the universal InferenceRunner interface if it supports lampe.
    try:
        from ili.inference import InferenceRunner  # type: ignore

        runner = InferenceRunner.load(
            backend="lampe",
            engine=engine,
            prior=prior,
            nets=nets,
            train_args=train_args,
            out_dir=out_dir,
            device=device,
            proposal=proposal,
            name=name,
            signatures=signatures,
        )
    except Exception:
        from ili.inference.runner_lampe import LampeRunner  # type: ignore

        runner = LampeRunner(
            prior=prior,
            nets=nets,
            engine=engine,
            train_args=train_args,
            out_dir=out_dir,
            device=device,
            proposal=proposal,
            name=name,
            signatures=signatures,
        )

    posterior, summaries = runner(loader, seed=seed)
    return posterior, summaries, runner


__all__ = ["train_sbi_posterior", "train_lampe_posterior"]
