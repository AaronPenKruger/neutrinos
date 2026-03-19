# DeepHpx 

Heres is a **summary** of the core functionality of this project:

- **1)** HEALPix map ingestion + ordering normalization (RING <-> NEST)
- **2)** PyGSP-free HEALPix graph backend (neighbours -> adjacency -> Laplacian)
- **3)** Chebyshev (ChebNet-style) graph convolution in modern PyTorch
- **4)** HEALPix pooling/unpooling layers (NSIDE/2 <-> pixels/4)
- **5)** Multi-resolution HEALPix encoder (maps -> fixed-length embeddings)
- **6)** LtU-ILI integration (HEALPix maps -> LtU-ILI loaders + SBI flow training helpers)
- **7)** Streaming HEALPix ingestion (PyTorch DataLoader -> LtU-ILI TorchLoader) + lampe training helper

## Install (editable)

For development (pytest):

```bash
pip install -e '.[dev,healpix,graph,torch]'
pytest
```

## LtU-ILI integration 

DeepHpx keeps LtU-ILI as an *optional* dependency.

Install DeepHpx with LtU-ILI (PyTorch backend) enabled:

```bash
pip install -e '.[dev,healpix,graph,torch,ili]'
```

This installs ``ltu-ili[pytorch]`` and enables:

- ``deephpx.ili.HealpixEmbeddingNet`` (ready-to-use ``embedding_net`` for SBI flows)
- ``deephpx.ili.make_ili_numpy_loader(...)`` and friends
- ``deephpx.ili.train_sbi_posterior(...)``

## Version 0.7.0 features

- **Neighbours:** build 8-neighbour connectivity via `healpy.pixelfunc.get_all_neighbours`
- **Adjacency:** build a SciPy sparse CSR adjacency matrix from neighbour lists
- **Laplacian:** build combinatorial or normalized Laplacians and scale them for Chebyshev convolutions
- **Torch:** convert SciPy sparse matrices to `torch.sparse` COO tensors
- **ChebConv:** Chebyshev graph convolution (`deephpx.nn.ChebConv` / `deephpx.nn.SphericalChebConv`)
- **Pooling:** HEALPix AvgPool/MaxPool (+ corresponding unpool layers) for hierarchical downsampling
- **Encoder:** `deephpx.nn.HealpixEncoder` that stacks ChebConv + pooling and returns an embedding vector

We also added:

- **LtU-ILI integration:** ``deephpx.ili`` helpers for wrapping HEALPix maps
  into LtU-ILI loaders (``ili.dataloaders.NumpyLoader``)
- **Embedding net:** ``deephpx.ili.HealpixEmbeddingNet`` that internally builds
  the Laplacian pyramid and can be passed into ``ili.utils.load_nde_sbi``
- **Streaming dataset:** ``deephpx.ili.HealpixFileDataset`` for on-demand disk reads
- **TorchLoader wrappers:** ``deephpx.ili.make_ili_torch_loader_from_files(...)``
  to create LtU-ILI ``TorchLoader`` objects from map files + ``theta.npy``
- **Lampe training helper:** ``deephpx.ili.train_lampe_posterior(...)``

## Examples

Build a graph for a given NSIDE and print stats:

```bash
python examples/01_build_graph.py --nside 32 --nest true --kind normalized
```

Run a Chebyshev convolution forward pass on a HEALPix graph:

```bash
python examples/02_chebconv_forward.py --nside 16 --K 3
```

Smoke test the HEALPix pooling/unpooling layers:

```bash
python examples/03_healpix_pooling_smoke.py --npix 12288 --mode average
python examples/03_healpix_pooling_smoke.py --npix 12288 --mode max
```

Smoke test the multi-resolution encoder:

```bash
python examples/04_encoder_forward.py --nside 16 --levels 3 --channels 8,16,32 --embedding-dim 64
```

Train an SBI normalizing flow (NPE + MAF) via LtU-ILI on *toy* HEALPix maps:

```bash
python examples/05_ili_sbi_train_healpix.py --mode toy --nside 16 --levels 3
```

Train a flow with *streaming* data loading using LtU-ILI's **lampe** backend:

```bash
python examples/06_ili_lampe_train_streaming_healpix.py --mode toy --nside 16 --levels 3 \
  --channels 8,16,32 --embedding-dim 64 --num-samples 512 --batch-size 32 \
  --out-dir ./_out_streaming
```

