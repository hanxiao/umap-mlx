"""UMAP implementation in pure MLX for Apple Silicon."""

import mlx.core as mx
import numpy as np
from scipy.sparse import coo_matrix


class UMAP:
    """UMAP dimensionality reduction using MLX on Metal GPU.

    Parameters:
        n_components: Dimension of the embedded space (default 2).
        n_neighbors: Number of nearest neighbors (default 15).
        min_dist: Minimum distance in low-dimensional space (default 0.1).
        spread: Effective scale of embedded points (default 1.0).
        n_epochs: Number of optimization epochs (default 200).
        learning_rate: SGD learning rate (default 1.0).
        negative_sample_rate: Negative samples per positive edge (default 5).
        random_state: Random seed for reproducibility.
        verbose: Print progress.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: int = 200,
        learning_rate: float = 1.0,
        negative_sample_rate: int = 5,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.verbose = verbose
        self.embedding_ = None

    def fit_transform(self, X) -> np.ndarray:
        """Fit UMAP and return the embedding.

        Args:
            X: Input data, shape (n_samples, n_features). np.ndarray or mx.array.

        Returns:
            Embedding as np.ndarray, shape (n_samples, n_components).
        """
        if isinstance(X, mx.array):
            X_np = np.array(X)
        else:
            X_np = np.asarray(X, dtype=np.float32)

        n = X_np.shape[0]

        # Step 1: Find k-nearest neighbors (numpy, exact)
        if self.verbose:
            print("Computing nearest neighbors...")
        knn_indices, knn_dists = self._compute_knn(X_np)

        # Step 2: Compute fuzzy simplicial set (graph weights)
        if self.verbose:
            print("Building fuzzy simplicial set...")
        graph = self._fuzzy_simplicial_set(knn_indices, knn_dists, n)

        # Step 3: Find a/b parameters for the embedding distance function
        a, b = self._find_ab_params(self.spread, self.min_dist)

        # Step 4: Initialize embedding
        if self.verbose:
            print("Optimizing embedding...")
        if self.random_state is not None:
            mx.random.seed(self.random_state)
        # Spectral-like init: use first 2 eigenvectors of graph Laplacian
        # Fallback to small random init
        Y = mx.random.normal((n, self.n_components)) * 0.01
        mx.eval(Y)

        # Step 5: Optimize
        Y = self._optimize(graph, Y, a, b)
        mx.eval(Y)

        self.embedding_ = np.array(Y)
        return self.embedding_

    def _compute_knn(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact k-nearest neighbors using pairwise distances on GPU."""
        X_mx = mx.array(X)
        sum_sq = mx.sum(X_mx * X_mx, axis=1)
        D = sum_sq[:, None] + sum_sq[None, :] - 2.0 * (X_mx @ X_mx.T)
        D = mx.maximum(D, 0.0)
        mx.eval(D)

        # Get k+1 nearest (including self), then exclude self
        D_np = np.array(D)
        k = self.n_neighbors
        n = X.shape[0]

        # Set diagonal to inf so self is never a neighbor
        np.fill_diagonal(D_np, np.inf)

        # argpartition for top-k, then sort within
        indices = np.argpartition(D_np, k, axis=1)[:, :k]
        # Gather distances and sort
        row_idx = np.arange(n)[:, None]
        knn_dists_sq = D_np[row_idx, indices]
        sort_order = np.argsort(knn_dists_sq, axis=1)
        knn_indices = np.take_along_axis(indices, sort_order, axis=1).astype(np.int32)
        knn_dists = np.sqrt(np.maximum(np.take_along_axis(knn_dists_sq, sort_order, axis=1), 0)).astype(np.float32)

        return knn_indices, knn_dists

    def _fuzzy_simplicial_set(
        self, knn_indices: np.ndarray, knn_dists: np.ndarray, n: int
    ) -> coo_matrix:
        """Build the fuzzy simplicial set (weighted graph) from KNN."""
        k = self.n_neighbors

        # Binary search for sigma per point
        sigmas = np.zeros(n, dtype=np.float32)
        rhos = np.zeros(n, dtype=np.float32)
        target = np.log2(k)

        for i in range(n):
            dists = knn_dists[i]
            rhos[i] = max(dists[dists > 0].min(), 1e-8) if np.any(dists > 0) else 0

            lo, hi = 1e-20, 1e3
            sigma = 1.0

            for _ in range(64):
                vals = np.exp(-(np.maximum(dists - rhos[i], 0)) / sigma)
                vals_sum = vals.sum()

                if abs(vals_sum - target) < 1e-5:
                    break

                if vals_sum > target:
                    hi = sigma
                    sigma = (lo + hi) / 2
                else:
                    lo = sigma
                    if hi >= 1e3:
                        sigma *= 2
                    else:
                        sigma = (lo + hi) / 2

            sigmas[i] = sigma

        # Build sparse graph
        rows, cols, vals = [], [], []
        for i in range(n):
            for j_idx in range(k):
                j = int(knn_indices[i, j_idx])
                d = max(knn_dists[i, j_idx] - rhos[i], 0)
                w = np.exp(-d / max(sigmas[i], 1e-10))
                rows.append(i)
                cols.append(j)
                vals.append(w)

        graph = coo_matrix((vals, (rows, cols)), shape=(n, n))

        # Symmetrize: P = A + A^T - A * A^T
        graph_t = graph.T
        prod = graph.multiply(graph_t)
        graph = graph + graph_t - prod

        graph = graph.tocoo()

        return graph

    @staticmethod
    def _find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
        """Find a, b parameters for the smooth approximation to the membership function."""
        from scipy.optimize import curve_fit

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        xv = np.linspace(0, spread * 3, 300)
        yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))

        (a, b), _ = curve_fit(curve, xv, yv)
        return float(a), float(b)

    def _optimize(self, graph, Y: mx.array, a: float, b: float) -> mx.array:
        """Optimize embedding with SGD using edge sampling."""
        n = Y.shape[0]
        d = self.n_components

        # Work in numpy for the SGD loop (scatter_add not available in MLX)
        Y_np = np.array(Y)

        # Extract edges and weights
        graph = graph.tocoo()
        edge_from = graph.row.astype(np.int32)
        edge_to = graph.col.astype(np.int32)
        weights = graph.data.astype(np.float32)

        # Epochs per edge: high-weight edges sampled more often
        max_weight = weights.max()
        n_samples = self.n_epochs * (weights / max_weight)
        epochs_per_sample = np.where(n_samples > 0, self.n_epochs / n_samples, -1.0)
        epochs_per_next = epochs_per_sample.copy()

        alpha = self.learning_rate
        rng = np.random.RandomState(self.random_state if self.random_state else 42)

        for epoch in range(self.n_epochs):
            alpha_epoch = alpha * (1.0 - epoch / self.n_epochs)

            # Active edges this epoch
            active = np.where(epochs_per_next <= epoch)[0]
            if len(active) == 0:
                continue

            ef = edge_from[active]
            et = edge_to[active]
            n_active = len(active)

            # Positive forces (attract)
            diff = Y_np[ef] - Y_np[et]  # (E, d)
            dist_sq = np.sum(diff * diff, axis=1, keepdims=True)  # (E, 1)
            dist_sq = np.maximum(dist_sq, 1e-6)

            pow_val = np.power(dist_sq, b)
            grad_coeff = -2.0 * a * b * np.power(dist_sq, b - 1.0) / (1.0 + a * pow_val)
            pos_grad = np.clip(grad_coeff * diff, -4.0, 4.0)

            # Apply positive gradients
            np.add.at(Y_np, ef, alpha_epoch * pos_grad)
            np.add.at(Y_np, et, -alpha_epoch * pos_grad)

            # Negative sampling
            n_neg = self.negative_sample_rate * n_active
            neg_from = ef[np.arange(n_neg) % n_active]
            neg_to = rng.randint(0, n, n_neg)

            neg_diff = Y_np[neg_from] - Y_np[neg_to]
            neg_dist_sq = np.sum(neg_diff * neg_diff, axis=1, keepdims=True)
            neg_dist_sq = np.maximum(neg_dist_sq, 1e-6)

            neg_pow = np.power(neg_dist_sq, b)
            neg_grad_coeff = 2.0 * b / ((0.001 + neg_dist_sq) * (1.0 + a * neg_pow))
            neg_grad = np.clip(neg_grad_coeff * neg_diff, -4.0, 4.0)

            np.add.at(Y_np, neg_from, alpha_epoch * neg_grad)

            # Schedule next epoch for processed edges
            epochs_per_next[active] += epochs_per_sample[active]

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}")

        return mx.array(Y_np)
