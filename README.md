# umap-mlx

UMAP in pure MLX for Apple Silicon. Entire pipeline runs on Metal GPU.

30-46x faster than umap-learn. Fashion-MNIST 70K in 2.6 seconds.



https://github.com/user-attachments/assets/e7677d25-6a14-46c5-afff-55311989dccc



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
- `n_epochs`: optimization epochs (default: 500 for N<=10K, 200 for larger)
- `learning_rate`: SGD learning rate (default 1.0)
- `random_state`: seed for reproducibility
- `verbose`: print progress

## Performance (M3 Ultra, Fashion-MNIST)

```
N       umap-learn    MLX      speedup
1000    4.87s         0.40s    12x
2000    6.18s         0.36s    17x
5000    17.22s        0.44s    40x
10000   25.85s        0.56s    46x
20000   22.01s        0.54s    41x
60000   68.99s        2.04s    34x
70000   81.40s        2.65s    31x
```

Bottleneck at large N is exact k-NN (O(n^2) pairwise distances). umap-learn
uses approximate KNN (pynndescent) with better asymptotic scaling.

## Comparison

Fashion-MNIST full (70,000 samples, 784 dims, 10 classes):

Spectral initialization:

![init](comparison_init.png?raw=true)

After optimization:

![comparison](comparison.png?raw=true)

## How it works

1. Chunked pairwise distances + `mx.argpartition` for exact k-NN on Metal GPU
2. Vectorized binary search for per-point sigma (all N points simultaneously on GPU)
3. Sparse graph symmetrization via GPU `_searchsorted` + edge pruning
4. Spectral initialization via power iteration on normalized graph Laplacian
5. Compiled SGD on Metal GPU with `mx.array.at[].add()` for scatter accumulation
6. Negative sampling with repulsive forces

Dependencies: `mlx` and `numpy` only. No scipy, no PyTorch.

## License

MIT
