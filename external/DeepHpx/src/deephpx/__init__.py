"""DeepHpx.

Milestone 1:
- HEALPix map ingestion
- Ordering normalization (RING <-> NEST)

Milestone 2:
- PyGSP-free HEALPix graph connectivity (neighbours, adjacency, Laplacian)

Milestone 3:
- Pure-PyTorch Chebyshev (ChebNet-style) graph convolution

Milestone 4:
- HEALPix pooling/unpooling layers (for hierarchical downsampling)

Milestone 5:
- Multi-resolution HEALPix encoder (maps -> fixed-length embeddings)

Milestone 6:
- Thin LtU-ILI integration helpers (HEALPix maps -> LtU-ILI loaders, plus an
  embedding_net that works with SBI flows)

The project is intentionally modular:
- Base installation depends only on NumPy.
- Extras:
  - `deephpx[healpix]` installs `healpy` (FITS I/O + ordering conversions + neighbour queries)
  - `deephpx[graph]` installs `scipy` (sparse adjacency/Laplacians)
  - `deephpx[torch]` installs `torch` (torch sparse conversion + Milestone 3 NN layers)
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.6.0"
