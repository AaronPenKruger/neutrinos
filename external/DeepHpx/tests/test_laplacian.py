import numpy as np
import pytest


def _cycle4_neighbors():
    # 4-node cycle: 0-1-2-3-0
    neigh = -np.ones((4, 8), dtype=np.int64)
    neigh[0, 0:2] = [1, 3]
    neigh[1, 0:2] = [0, 2]
    neigh[2, 0:2] = [1, 3]
    neigh[3, 0:2] = [2, 0]
    return neigh


def test_adjacency_and_laplacian_cycle4():
    pytest.importorskip("scipy")

    from deephpx.graph.adjacency import adjacency_from_neighbors
    from deephpx.graph.laplacian import laplacian_from_adjacency

    neigh = _cycle4_neighbors()
    A = adjacency_from_neighbors(neigh, symmetric=True)
    assert A.shape == (4, 4)
    assert A.nnz == 8  # undirected cycle has 4 edges -> 8 directed entries

    # Degree should be 2 everywhere.
    deg = np.asarray(A.sum(axis=1)).reshape(-1)
    assert np.allclose(deg, 2.0)

    # Combinatorial Laplacian
    L = laplacian_from_adjacency(A, kind="combinatorial")
    Ld = L.toarray()
    expected = np.array(
        [
            [2, -1, 0, -1],
            [-1, 2, -1, 0],
            [0, -1, 2, -1],
            [-1, 0, -1, 2],
        ],
        dtype=float,
    )
    assert np.allclose(Ld, expected)

    # Normalized Laplacian for a 2-regular graph: I - A/2
    Ln = laplacian_from_adjacency(A, kind="normalized")
    Lnd = Ln.toarray()
    expected_n = np.eye(4) - 0.5 * A.toarray()
    assert np.allclose(Lnd, expected_n)


def test_chebyshev_scaling_spectrum_cycle4():
    pytest.importorskip("scipy")

    from deephpx.graph.adjacency import adjacency_from_neighbors
    from deephpx.graph.laplacian import laplacian_from_adjacency, scale_laplacian_for_chebyshev

    neigh = _cycle4_neighbors()
    A = adjacency_from_neighbors(neigh, symmetric=True)
    Ln = laplacian_from_adjacency(A, kind="normalized")

    L_tilde = scale_laplacian_for_chebyshev(Ln, kind_hint="normalized")
    evals = np.linalg.eigvalsh(L_tilde.toarray())

    # Spectrum should live in [-1, 1] (up to numerical tolerance)
    assert evals.min() >= -1.0 - 1e-8
    assert evals.max() <= 1.0 + 1e-8


def test_torch_sparse_conversion():
    pytest.importorskip("scipy")
    pytest.importorskip("torch")

    from deephpx.graph.adjacency import adjacency_from_neighbors
    from deephpx.graph.laplacian import to_torch_sparse_coo

    neigh = _cycle4_neighbors()
    A = adjacency_from_neighbors(neigh, symmetric=True)

    tA = to_torch_sparse_coo(A)
    assert tuple(tA.shape) == (4, 4)
    assert tA._nnz() == A.nnz
