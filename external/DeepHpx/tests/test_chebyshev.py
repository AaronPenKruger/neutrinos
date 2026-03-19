from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")


def _sparse_identity(n: int, *, dtype=None):
    idx = torch.arange(n, dtype=torch.long)
    indices = torch.stack([idx, idx], dim=0)
    values = torch.ones(n, dtype=dtype or torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()


def test_cheb_conv_identity_matches_weight_sum():
    from deephpx.nn.chebyshev import cheb_conv

    torch.manual_seed(0)
    n = 17
    b = 3
    fin = 4
    fout = 5
    k = 6

    L = _sparse_identity(n)
    x = torch.randn(b, n, fin)
    w = torch.randn(k, fin, fout)
    y = cheb_conv(L, x, w)

    # For L = I, Chebyshev recursion yields T_k(x) = x for all k.
    y_ref = x @ w.sum(dim=0)
    assert y.shape == (b, n, fout)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)


def test_cheb_conv_bias_and_backward():
    from deephpx.nn.chebyshev import ChebConv

    torch.manual_seed(0)
    n = 9
    b = 2
    fin = 2
    fout = 3
    k = 3
    L = _sparse_identity(n)

    layer = ChebConv(fin, fout, k, bias=True)
    x = torch.randn(b, n, fin, requires_grad=True)
    y = layer(L, x)
    loss = y.pow(2).mean()
    loss.backward()

    assert y.shape == (b, n, fout)
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert layer.bias is not None and layer.bias.grad is not None


def test_cheb_conv_accepts_2d_input_n_fin():
    from deephpx.nn.chebyshev import cheb_conv

    torch.manual_seed(0)
    n = 11
    fin = 3
    fout = 2
    k = 2
    L = _sparse_identity(n)

    x = torch.randn(n, fin)
    w = torch.randn(k, fin, fout)
    y = cheb_conv(L, x, w)

    assert y.shape == (n, fout)
