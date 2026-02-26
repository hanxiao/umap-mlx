# umap-mlx

UMAP implementation in pure MLX for Apple Silicon. GPU-accelerated k-NN and optimization.

3-9x faster than umap-learn on M3 Ultra.

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

## Benchmark (M3 Ultra)

```
N       umap-learn    MLX      speedup
1000    3.31s         0.46s    7.2x
2000    2.87s         0.90s    3.2x
5000    9.11s         2.60s    3.5x
```

GPU acceleration helps most at smaller N. For N > 10K, umap-learn's parallelized graph construction catches up.

## Comparison

sklearn digits dataset (1797 samples, 64 dims, 10 classes):

![comparison](comparison.png)

Both produce well-separated digit clusters. MLX is 9.4x faster (0.75s vs 7.06s).

## How it works

1. **k-NN**: Exact pairwise distances on Metal GPU (`||x||^2 + ||y||^2 - 2x.y`)
2. **Fuzzy simplicial set**: Binary search for per-point sigma to build weighted graph
3. **Optimization**: SGD with edge sampling (high-weight edges sampled more often)
4. Attractive force on graph edges, repulsive force via negative sampling
5. Uses `scipy.sparse` for graph and `numpy` for SGD (MLX doesn't have scatter_add yet)

## License

MIT
