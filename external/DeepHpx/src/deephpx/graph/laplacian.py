"""Graph Laplacians and Chebyshev scaling.

DeepSphere (and ChebNet-style graph CNNs) typically operate on a *scaled*
Laplacian so that its spectrum lies in [-1, 1], enabling stable Chebyshev
polynomial recursion.

This module provides:
- Combinatorial Laplacian: L = D - A
- Normalized Laplacian:    L = I - D^{-1/2} A D^{-1/2}
- Spectral radius (lambda_max) estimation
- Chebyshev scaling:       L_tilde = (2/lmax) * L - I
- Optional conversion to torch sparse COO tensor
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scipy is required for Laplacian construction. Install with `pip install deephpx[graph]`."
    ) from e


LaplacianKind = Literal["combinatorial", "normalized"]


def laplacian_from_adjacency(
    A: "sp.spmatrix",
    *,
    kind: LaplacianKind = "normalized",
    eps: float = 1e-12,
    force_symmetric: bool = True,
) -> "sp.csr_matrix":
    """Construct a graph Laplacian from an adjacency matrix.

    Args:
        A: adjacency matrix (sparse), expected shape (N, N).
        kind: 'combinatorial' or 'normalized'.
        eps: small value to treat degrees as nonzero.
        force_symmetric: if True, enforce symmetry by (M + M.T)/2.
            This can repair tiny numerical asymmetries.

    Returns:
        L: Laplacian in CSR format.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape={A.shape}")

    A = A.tocsr()

    # Remove diagonal explicitly (self-loops) if any slipped in.
    if A.diagonal().any():
        A = A - sp.diags(A.diagonal())
        A.eliminate_zeros()

    n = A.shape[0]

    # Degree vector
    d = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float64)

    if kind == "combinatorial":
        L = sp.diags(d, offsets=0, shape=(n, n), format="csr") - A
    elif kind == "normalized":
        inv_sqrt = np.zeros_like(d)
        mask = d > eps
        inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])
        D_inv_sqrt = sp.diags(inv_sqrt, offsets=0, shape=(n, n), format="csr")
        I = sp.identity(n, format="csr", dtype=A.dtype)
        L = I - (D_inv_sqrt @ A @ D_inv_sqrt)
    else:
        raise ValueError(f"Unknown Laplacian kind {kind!r}")

    L = L.tocsr()

    if force_symmetric:
        L = (L + L.T) * 0.5

    return L


def estimate_lmax(
    L: "sp.spmatrix",
    *,
    method: Literal["eigsh", "power"] = "eigsh",
    n_iter: int = 30,
    tol: float = 0.0,
    seed: int = 0,
) -> float:
    """Estimate the largest eigenvalue (spectral radius) of a symmetric Laplacian.

    Args:
        L: sparse symmetric matrix.
        method: 'eigsh' (ARPACK/Lanczos) or 'power' iteration.
        n_iter: power-iteration steps (only for method='power').
        tol: tolerance passed to eigsh (only for method='eigsh').
        seed: RNG seed for power iteration.

    Returns:
        lmax: estimated largest eigenvalue (float).

    Notes:
        - For the normalized Laplacian, the true maximum eigenvalue is <= 2.
        - 'eigsh' gives a good estimate but can be slower for large N.
    """
    if L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square, got shape={L.shape}")

    if method == "eigsh":
        try:
            # Largest magnitude eigenvalue for symmetric matrices.
            vals = spla.eigsh(
                L,
                k=1,
                which="LM",
                return_eigenvectors=False,
                tol=tol,
            )
            return float(np.real(vals[0]))
        except Exception:
            # Fall back to power iteration
            method = "power"

    if method == "power":
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(L.shape[0]).astype(np.float64)
        x /= np.linalg.norm(x) + 1e-30

        for _ in range(max(1, n_iter)):
            y = L @ x
            yn = np.linalg.norm(y)
            if yn <= 1e-30:
                return 0.0
            x = y / yn

        # Rayleigh quotient
        y = L @ x
        return float(np.dot(x, y))

    raise ValueError(f"Unknown method {method!r}")


def scale_laplacian_for_chebyshev(
    L: "sp.spmatrix",
    *,
    lmax: Optional[float] = None,
    kind_hint: Optional[LaplacianKind] = None,
    lmax_method: Literal["eigsh", "power"] = "eigsh",
) -> "sp.csr_matrix":
    """Scale Laplacian so that eigenvalues lie in [-1, 1].

    Uses the standard ChebNet scaling:
        L_tilde = (2 / lmax) * L - I

    Args:
        L: Laplacian (sparse).
        lmax: largest eigenvalue. If None, infer using `kind_hint` or estimate.
        kind_hint: if 'normalized' and lmax is None, we use lmax=2.
        lmax_method: estimation method if we need to estimate lmax.

    Returns:
        L_tilde: scaled Laplacian in CSR format.
    """
    if L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square, got shape={L.shape}")

    n = L.shape[0]

    if lmax is None:
        if kind_hint == "normalized":
            # The normalized Laplacian has spectrum in [0, 2].
            lmax = 2.0
        else:
            lmax = estimate_lmax(L.tocsr(), method=lmax_method)

    if lmax <= 0:
        raise ValueError(f"lmax must be positive, got {lmax}")

    I = sp.identity(n, format="csr", dtype=L.dtype)
    L_tilde = (2.0 / float(lmax)) * L - I

    return L_tilde.tocsr()


def to_torch_sparse_coo(
    mat: "sp.spmatrix",
    *,
    dtype: Optional["object"] = None,
    device: Optional["object"] = None,
    coalesce: bool = True,
):
    """Convert a SciPy sparse matrix to a torch sparse COO tensor.

    Args:
        mat: SciPy sparse matrix.
        dtype: optional torch dtype for values (e.g., torch.float32).
        device: optional torch device.
        coalesce: if True, coalesce the resulting tensor.

    Returns:
        torch_sparse: torch.sparse COO tensor.
    """
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "torch is required for to_torch_sparse_coo(). Install with `pip install deephpx[torch]`."
        ) from e

    coo = mat.tocoo()
    indices = np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
    values = coo.data

    t_indices = torch.from_numpy(indices).long()
    t_values = torch.from_numpy(values)

    if dtype is not None:
        t_values = t_values.to(dtype)
    if device is not None:
        t_indices = t_indices.to(device)
        t_values = t_values.to(device)

    t = torch.sparse_coo_tensor(t_indices, t_values, size=coo.shape)
    return t.coalesce() if coalesce else t
