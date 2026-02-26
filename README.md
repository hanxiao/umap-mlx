# umap-mlx

UMAP in pure MLX for Apple Silicon. Entire pipeline runs on Metal GPU.

20-40x faster than umap-learn. Fashion-MNIST 60K in 1.9 seconds.

## Install

```bash
git clone https://github.com/hanxiao/umap-mlx.git && cd umap-mlx
uv venv .venv && source .venv/bin/activate
uv pip install -e .
```

## Usage

```python
from umap_mlx import UMAP
import numpy as np

X = np.random.randn(1000, 128).astype(np.float32)
Y = UMAP(n_components=2, n_neighbors=15).fit_transform(X)
```

Parameters:

- `n_components`: output dimensions (default 2)
- `n_neighbors`: k for k-NN graph (default 15)
- `min_dist`: minimum distance in low-dim space (default 0.1)
- `spread`: scale of embedded points (default 1.0)
- `n_epochs`: optimization epochs (default 200)
- `learning_rate`: SGD learning rate (default 1.0)
- `random_state`: seed for reproducibility
- `verbose`: print progress

## Performance (M3 Ultra, Fashion-MNIST)

```
N       umap-learn    MLX      speedup
1000    4.88s         0.15s    32x
2000    6.20s         0.16s    38x
5000    17.18s        0.22s    77x
10000   25.95s        0.32s    80x
20000   22.12s        0.56s    40x
60000   69.05s        2.26s    31x
```

Peak speedup at 5-10K (GPU parallelism sweet spot). At 60K the O(n^2) distance
matrix (14GB) becomes memory-bandwidth bound. umap-learn uses approximate KNN
(pynndescent) at large N, which has better asymptotic scaling.

## Comparison

Fashion-MNIST train (60,000 samples, 784 dims, 10 classes):

![comparison](comparison.png)

Fashion-MNIST created by Han Xiao et al.

## How it works

1. Chunked pairwise distances + `mx.argpartition` for k-NN on Metal GPU
2. Vectorized binary search for per-point sigma (all N points simultaneously)
3. Sparse graph symmetrization + edge pruning (matches umap-learn)
4. SGD on Metal GPU using `mx.array.at[].add()` for scatter accumulation
5. Negative sampling with repulsive forces

Dependencies: `mlx`, `numpy`, `scipy` (sparse graph ops).

## License

MIT
