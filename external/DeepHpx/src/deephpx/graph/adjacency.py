"""Adjacency matrix construction.

This module converts neighbour lists into sparse adjacency matrices.

We keep the data structure simple and dependency-light:

- input neighbours is a NumPy array of shape (N, K)
- output adjacency is a SciPy sparse CSR matrix of shape (N, N)

For Milestone 2, we implement binary (unweighted) adjacencies. This is enough
to reproduce the *core* DeepSphere pipeline for Chebyshev graph convolution:
neighbour list -> adjacency -> Laplacian.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

try:
    import scipy.sparse as sp
except Exception as e:  # pragma: no cover
    raise ImportError(
        "scipy is required for graph construction. Install with `pip install deephpx[graph]`."
    ) from e


WeightMode = Literal["binary"]


def adjacency_from_neighbors(
    neighbors: np.ndarray,
    *,
    symmetric: bool = True,
    remove_self_loops: bool = True,
    weight_mode: WeightMode = "binary",
    dtype: np.dtype = np.float32,
) -> "sp.csr_matrix":
    """Build a sparse adjacency matrix from a neighbour list.

    Args:
        neighbors: integer neighbour array of shape (N, K). Use -1 for missing.
        symmetric: if True, return an undirected adjacency by taking the union of
            edges with its transpose.
        remove_self_loops: if True, discard i->i entries.
        weight_mode: currently only 'binary' is supported.
        dtype: dtype for adjacency data.

    Returns:
        A: SciPy CSR sparse matrix of shape (N, N).
    """
    if weight_mode != "binary":
        raise NotImplementedError(
            f"Only weight_mode='binary' is supported in Milestone 2 (got {weight_mode!r})."
        )

    neigh = np.asarray(neighbors)
    if neigh.ndim != 2:
        raise ValueError(f"neighbors must be 2D of shape (N, K), got shape={neigh.shape}")

    n, k = neigh.shape

    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = neigh.astype(np.int64, copy=False).reshape(-1)

    mask = cols >= 0
    rows = rows[mask]
    cols = cols[mask]

    if remove_self_loops:
        mask2 = rows != cols
        rows = rows[mask2]
        cols = cols[mask2]

    data = np.ones(rows.shape[0], dtype=dtype)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))

    if symmetric:
        # union of directed edges, keeping values binary
        A = A.maximum(A.T)

    A = A.tocsr()
    A.eliminate_zeros()

    # enforce binary weights even if duplicates happened along construction
    if A.nnz > 0:
        A.data[:] = 1.0

    return A
