import torch


def _sparse_eye(n: int) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.int64)
    indices = torch.stack([idx, idx], dim=0)
    values = torch.ones(n, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()


def test_healpix_encoder_shapes_and_backward():
    from deephpx.nn import HealpixEncoder

    # Synthetic multi-resolution pyramid: N -> N/4 -> N/16
    N0, N1, N2 = 64, 16, 4
    laplacians = [_sparse_eye(N0), _sparse_eye(N1), _sparse_eye(N2)]

    enc = HealpixEncoder(
        laplacians,
        in_channels=3,
        conv_channels=(8, 16, 32),
        embedding_dim=20,
        K=3,
        pool="average",
        global_pool="meanmax",
        norm="layer",
        dropout=0.1,
        mlp_hidden=(64,),
        activation="relu",
    )

    x = torch.randn(5, N0, 3, requires_grad=True)
    z = enc(x)
    assert z.shape == (5, 20)

    loss = z.sum()
    loss.backward()

    # Sanity: grads exist
    assert x.grad is not None
    assert enc.convs[0].weight.grad is not None


def test_healpix_encoder_unbatched_input():
    from deephpx.nn import HealpixEncoder

    N0, N1 = 32, 8
    laplacians = [_sparse_eye(N0), _sparse_eye(N1)]
    enc = HealpixEncoder(
        laplacians,
        in_channels=1,
        conv_channels=(4, 4),
        embedding_dim=7,
        K=2,
        pool="average",
        global_pool="mean",
        norm="none",
        activation="relu",
    )

    x = torch.randn(N0, 1)
    z = enc(x)
    assert z.shape == (7,)
