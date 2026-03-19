This repository contains scaffolded code and notebooks for expected astrophysical plus atmospheric neutrino count maps for IceCube.

Run notebooks after `pip install -e .`.

DeepHpx is included as an external git submodule in `external/DeepHpx`.
Initialize/update it with:

```bash
git submodule update --init --recursive
```

Optional editable install (recommended for local development):

```bash
pip install -e external/DeepHpx
```

Embedding pretraining for simulation maps is implemented in:
- `src/pretrain_deephpx_embedding.py`
- `notebooks/06_pretrain_deephpx_embedding_driver.ipynb`

Repository layout:
- `src/`: Python modules
- `notebooks/`: analysis notebooks
- `external/`: external libraries (DeepHpx submodule)
- `data/`: local input datasets (not committed)
- `outputs/`: generated artifacts (not committed)
