"""UMAP implementation in pure MLX for Apple Silicon."""

import mlx.core as mx
import numpy as np


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
            X = np.array(X)
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]

        # Step 1: KNN on GPU
        if self.verbose:
            print("Computing nearest neighbors...")
        knn_indices, knn_dists = self._compute_knn(X)

        # Step 2: Fuzzy simplicial set
        if self.verbose:
            print("Building fuzzy simplicial set...")
        graph_rows, graph_cols, graph_vals = self._fuzzy_simplicial_set(knn_indices, knn_dists, n)

        # Step 3: a/b parameters
        a, b = self._find_ab_params(self.spread, self.min_dist)

        # Step 4: Initialize
        if self.verbose:
            print("Optimizing embedding...")
        if self.random_state is not None:
            mx.random.seed(self.random_state)
        Y = mx.random.normal((n, self.n_components)) * 0.01
        mx.eval(Y)

        # Step 5: Optimize (pure MLX)
        Y = self._optimize(graph_rows, graph_cols, graph_vals, Y, a, b, n)
        mx.eval(Y)

        self.embedding_ = np.array(Y)
        return self.embedding_

    def _compute_knn(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact k-nearest neighbors using pairwise distances on GPU."""
        X_mx = mx.array(X)
        sum_sq = mx.sum(X_mx * X_mx, axis=1)
        D = mx.maximum(sum_sq[:, None] + sum_sq[None, :] - 2.0 * (X_mx @ X_mx.T), 0.0)
        mx.eval(D)

        D_np = np.array(D)
        n = X.shape[0]
        k = self.n_neighbors

        np.fill_diagonal(D_np, np.inf)
        indices = np.argpartition(D_np, k, axis=1)[:, :k]
        row_idx = np.arange(n)[:, None]
        knn_dists_sq = D_np[row_idx, indices]
        sort_order = np.argsort(knn_dists_sq, axis=1)
        knn_indices = np.take_along_axis(indices, sort_order, axis=1).astype(np.int32)
        knn_dists = np.sqrt(np.maximum(
            np.take_along_axis(knn_dists_sq, sort_order, axis=1), 0
        )).astype(np.float32)

        return knn_indices, knn_dists

    def _fuzzy_simplicial_set(
        self, knn_indices: np.ndarray, knn_dists: np.ndarray, n: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Build fuzzy simplicial set. Returns (rows, cols, vals) as MLX arrays."""
        k = self.n_neighbors
        target = np.log2(k)

        sigmas = np.zeros(n, dtype=np.float32)
        rhos = np.zeros(n, dtype=np.float32)

        for i in range(n):
            dists = knn_dists[i]
            rhos[i] = max(dists[dists > 0].min(), 1e-8) if np.any(dists > 0) else 0

            lo, hi = 1e-20, 1e3
            sigma = 1.0
            for _ in range(64):
                vals = np.exp(-np.maximum(dists - rhos[i], 0) / sigma)
                vals_sum = vals.sum()
                if abs(vals_sum - target) < 1e-5:
                    break
                if vals_sum > target:
                    hi = sigma
                    sigma = (lo + hi) / 2
                else:
                    lo = sigma
                    sigma = sigma * 2 if hi >= 1e3 else (lo + hi) / 2
            sigmas[i] = sigma

        # Build graph edges
        rows = np.zeros(n * k, dtype=np.int32)
        cols = np.zeros(n * k, dtype=np.int32)
        vals = np.zeros(n * k, dtype=np.float32)

        idx = 0
        for i in range(n):
            for j_idx in range(k):
                j = int(knn_indices[i, j_idx])
                d = max(knn_dists[i, j_idx] - rhos[i], 0)
                w = np.exp(-d / max(sigmas[i], 1e-10))
                rows[idx] = i
                cols[idx] = j
                vals[idx] = w
                idx += 1

        # Symmetrize: build dense, symmetrize, extract edges
        # P = A + A^T - A * A^T
        W = np.zeros((n, n), dtype=np.float32)
        for e in range(idx):
            W[rows[e], cols[e]] = vals[e]

        W_sym = W + W.T - W * W.T

        # Extract non-zero edges
        nz = np.nonzero(W_sym)
        return (
            mx.array(nz[0].astype(np.int32)),
            mx.array(nz[1].astype(np.int32)),
            mx.array(W_sym[nz].astype(np.float32)),
        )

    @staticmethod
    def _find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
        """Find a, b parameters via curve fitting (numpy only, no scipy)."""
        # Levenberg-Marquardt is overkill; simple grid search + refinement
        xv = np.linspace(0, spread * 3, 300)
        yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))

        best_a, best_b, best_err = 1.0, 1.0, float('inf')

        # Coarse grid
        for a in np.linspace(0.1, 5.0, 50):
            for b in np.linspace(0.1, 2.0, 20):
                pred = 1.0 / (1.0 + a * np.power(xv, 2 * b))
                err = np.sum((pred - yv) ** 2)
                if err < best_err:
                    best_a, best_b, best_err = a, b, err

        # Fine refinement
        for a in np.linspace(max(best_a - 0.5, 0.01), best_a + 0.5, 100):
            for b in np.linspace(max(best_b - 0.2, 0.01), best_b + 0.2, 40):
                pred = 1.0 / (1.0 + a * np.power(xv, 2 * b))
                err = np.sum((pred - yv) ** 2)
                if err < best_err:
                    best_a, best_b, best_err = a, b, err

        return float(best_a), float(best_b)

    def _optimize(
        self, edge_from: mx.array, edge_to: mx.array, edge_weights: mx.array,
        Y: mx.array, a: float, b: float, n: int
    ) -> mx.array:
        """SGD optimization using pure MLX."""
        n_edges = edge_from.shape[0]

        # Compute epochs_per_sample
        max_weight = float(mx.max(edge_weights).item())
        weights_np = np.array(edge_weights)
        n_samples = self.n_epochs * (weights_np / max_weight)
        epochs_per_sample = np.where(n_samples > 0, self.n_epochs / n_samples, -1.0)
        epochs_per_next = epochs_per_sample.copy()

        alpha = self.learning_rate

        for epoch in range(self.n_epochs):
            alpha_epoch = alpha * (1.0 - epoch / self.n_epochs)

            # Active edges
            active = np.where(epochs_per_next <= epoch)[0]
            if len(active) == 0:
                continue

            active_mx = mx.array(active.astype(np.int32))
            ef = edge_from[active_mx]
            et = edge_to[active_mx]
            n_active = len(active)

            # Positive forces (attract)
            y_from = Y[ef]
            y_to = Y[et]
            diff = y_from - y_to
            dist_sq = mx.maximum(mx.sum(diff * diff, axis=1, keepdims=True), 1e-6)

            pow_val = mx.power(dist_sq, b)
            grad_coeff = -2.0 * a * b * mx.power(dist_sq, b - 1.0) / (1.0 + a * pow_val)
            pos_grad = mx.clip(grad_coeff * diff, -4.0, 4.0)

            # Negative sampling
            n_neg = self.negative_sample_rate * n_active
            neg_from_idx = ef[mx.arange(n_neg) % n_active]
            neg_to_idx = mx.random.randint(0, n, (n_neg,))

            y_neg_from = Y[neg_from_idx]
            y_neg_to = Y[neg_to_idx]
            neg_diff = y_neg_from - y_neg_to
            neg_dist_sq = mx.maximum(mx.sum(neg_diff * neg_diff, axis=1, keepdims=True), 1e-6)

            neg_pow = mx.power(neg_dist_sq, b)
            neg_grad_coeff = 2.0 * b / ((0.001 + neg_dist_sq) * (1.0 + a * neg_pow))
            neg_grad = mx.clip(neg_grad_coeff * neg_diff, -4.0, 4.0)

            # Apply updates: since MLX doesn't have scatter_add,
            # convert to numpy for the accumulation step only
            mx.eval(pos_grad, neg_grad)
            Y_np = np.array(Y)
            ef_np = np.array(ef)
            et_np = np.array(et)
            neg_from_np = np.array(neg_from_idx)

            pos_np = np.array(pos_grad)
            neg_np = np.array(neg_grad)

            np.add.at(Y_np, ef_np, alpha_epoch * pos_np)
            np.add.at(Y_np, et_np, -alpha_epoch * pos_np)
            np.add.at(Y_np, neg_from_np, alpha_epoch * neg_np)

            Y = mx.array(Y_np)
            mx.eval(Y)

            epochs_per_next[active] += epochs_per_sample[active]

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}")

        return Y
