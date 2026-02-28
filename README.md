# umap-mlx

UMAP in pure MLX for Apple Silicon. Entire pipeline runs on Metal GPU.

30-46x faster than umap-learn. Fashion-MNIST 70K in 3.4 seconds for 500 epochs.



https://github.com/user-attachments/assets/1e0bf0ca-0d4b-43ba-9e97-14e1e0fc0a4c




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

## Possible future directions

**Approximate KNN via NNDescent.** Current exact KNN computes O(n^2) pairwise
distances, which dominates runtime at large N (e.g. 2.0s of 3.3s at 70K).
NNDescent builds an approximate k-NN graph in O(n log n) by iteratively
improving neighbors through neighbor-of-neighbor exploration. The challenge
is that NNDescent is inherently iterative with random access patterns
(heap operations, conditional updates per candidate), which maps poorly to
GPU SIMD execution. umap-learn's implementation is 1500 lines of Numba JIT.
A practical MLX version would need to replace heaps with fixed-size sorted
buffers and batch the neighbor-of-neighbor lookups as matrix gathers.
Likely only worthwhile above 100K+ samples where the O(n^2) distance matrix
exceeds memory.

**Custom Metal kernel for SGD.** The optimization loop currently dispatches
~10 separate MLX operations per epoch (gather, subtract, multiply, power,
clip, scatter-add x3), each as an independent Metal kernel. A fused Metal
shader via `mx.fast.metal_kernel()` could handle one edge per thread: read
Y[head] and Y[tail], compute the gradient in registers, and atomic-add the
update back to Y. This eliminates intermediate buffer allocations and reduces
kernel dispatch overhead from ~5000 launches to ~500 (one per epoch). Metal 3.0
supports `atomic_fetch_add_explicit` for float on all Apple Silicon. Estimated
improvement: ~10% on total runtime (SGD portion from 1.3s to ~1.0s at 70K).
The 500-epoch Python loop cannot be collapsed into a single kernel because
Metal lacks global barriers across threadgroups -- each epoch's scatter-add
must complete before the next epoch reads Y.

## See also

- [pacmap-mlx](https://github.com/hanxiao/pacmap-mlx) -- PaCMAP in pure MLX. Same idea, different algorithm: uses 3 pair types (near/mid-near/far) with phase-scheduled weights instead of fuzzy graphs. Often better at preserving global structure.

## License

MIT
