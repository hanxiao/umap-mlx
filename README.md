# UMAP-MLX

Pure MLX implementation of UMAP (Uniform Manifold Approximation and Projection) for Apple Silicon. Runs dimensionality reduction on the Metal GPU without PyTorch or CUDA dependencies.

## Installation

```bash
pip install umap-mlx
```

Or from source:

```bash
git clone https://github.com/hanxiao/umap-mlx.git
cd umap-mlx
pip install -e .
```

## Usage

```python
from umap_mlx import UMAP
import numpy as np

X = np.random.randn(5000, 128)

embedding = UMAP(n_components=2, n_neighbors=15, min_dist=0.1).fit_transform(X)
print(embedding.shape)  # (5000, 2)
```

## API

```python
UMAP(
    n_components=2,           # Target dimensionality
    n_neighbors=15,           # Neighbors for manifold approximation
    min_dist=0.1,             # Minimum distance in embedding space
    spread=1.0,               # Effective scale of embedded points
    n_epochs=500,             # Optimization epochs
    learning_rate=1.0,        # SGD learning rate
    negative_sample_rate=5,   # Negative samples per positive edge
    random_state=None         # Random seed
)
```

## How it works

1. **KNN graph** - Pairwise Euclidean distances computed via MLX matrix ops on Metal GPU, then k-nearest neighbors extracted
2. **Fuzzy simplicial set** - Smooth KNN distances via binary search for per-point bandwidth (sigma), then symmetrized membership graph
3. **Embedding optimization** - Force-directed layout via SGD with attractive forces on graph edges and repulsive forces via negative sampling. Spectral initialization from graph Laplacian eigenvectors.

## Benchmark

Apple M3 Ultra, 512GB unified memory. Random data with 64 dimensions, 200 epochs.

| Samples | umap-learn | UMAP-MLX | Speedup |
|---------|------------|----------|---------|
| 1,000   | 0.64s      | 0.43s    | 1.5x    |
| 10,000  | 7.95s      | 4.77s    | 1.7x    |
| 50,000  | 19.0s      | 23.7s    | 0.8x    |

MLX acceleration is most effective at medium scale (5K-10K points) where GPU-accelerated KNN dominates. At 50K+ the O(n^2) pairwise distance computation becomes the bottleneck; approximate nearest neighbor methods would help here.

## Comparison

MNIST digits (1797 samples, 500 epochs). Both produce topologically similar cluster structures:

![Comparison](comparison.png)

## Dependencies

- mlx >= 0.20.0
- numpy >= 1.20.0
- scipy >= 1.7.0

## License

MIT
