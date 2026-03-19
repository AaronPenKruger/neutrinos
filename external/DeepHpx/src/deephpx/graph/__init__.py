"""Graph utilities for HEALPix maps.

Milestone 2 provides a minimal, PyGSP-free graph backend:

- Build 8-neighbour connectivity on the HEALPix sphere via ``healpy``
- Convert neighbour lists to a SciPy sparse adjacency matrix
- Build combinatorial / normalized graph Laplacians
- Scale Laplacians for Chebyshev graph convolutions (ChebNet-style)
- Convert sparse matrices to ``torch.sparse`` COO tensors (optional)

The goal is to reproduce the *useful outputs* of DeepSphere's PyGSP-based
pipeline (adjacency/Laplacian) without any dependency on PyGSP.
"""

from __future__ import annotations

from .neighbors import neighbors_8
from .adjacency import adjacency_from_neighbors
from .laplacian import (
    laplacian_from_adjacency,
    estimate_lmax,
    scale_laplacian_for_chebyshev,
    to_torch_sparse_coo,
)

__all__ = [
    "neighbors_8",
    "adjacency_from_neighbors",
    "laplacian_from_adjacency",
    "estimate_lmax",
    "scale_laplacian_for_chebyshev",
    "to_torch_sparse_coo",
]
