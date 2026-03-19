"""Neural network modules for DeepHpx.

Milestone 3 introduces the core building block needed to replace DeepSphere's
graph convolution stack without PyGSP / PyTorch Geometric dependencies:

- Chebyshev (ChebNet-style) graph convolution on HEALPix graphs.

Milestone 4 adds HEALPix sampling pooling / unpooling layers.

Milestone 5 adds a minimal multi-resolution HEALPix encoder that turns maps
into fixed-length embeddings suitable as the conditioner/context for
normalizing flows in SBI/LtU-ILI pipelines.

The intended workflow is:
1) Build a (scaled) Laplacian with :mod:`deephpx.graph.laplacian`.
2) Convert it to ``torch.sparse`` COO.
3) Use :class:`deephpx.nn.chebyshev.SphericalChebConv` inside your encoder.
"""

from __future__ import annotations

from .chebyshev import ChebConv, SphericalChebConv, cheb_conv
from .healpix_pooling import (
    Healpix,
    HealpixAvgPool,
    HealpixAvgUnpool,
    HealpixMaxPool,
    HealpixMaxUnpool,
)
from .encoder import HealpixEncoder, HealpixEncoderSpec

__all__ = [
    "cheb_conv",
    "ChebConv",
    "SphericalChebConv",
    "HealpixAvgPool",
    "HealpixAvgUnpool",
    "HealpixMaxPool",
    "HealpixMaxUnpool",
    "Healpix",
    "HealpixEncoder",
    "HealpixEncoderSpec",
]
