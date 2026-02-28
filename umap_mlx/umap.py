"""UMAP implementation in pure MLX for Apple Silicon."""

import mlx.core as mx
import numpy as np
from functools import partial


def _searchsorted(sorted_array: mx.array, values: mx.array) -> mx.array:
    """Vectorized binary search on GPU (equivalent to np.searchsorted)."""
    n = sorted_array.shape[0]
    m = values.shape[0]
    lo = mx.zeros((m,), dtype=mx.int32)
    hi = mx.full((m,), n, dtype=mx.int32)

    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        mid = (lo + hi) // 2
        mid_clamped = mx.minimum(mid, n - 1)
        go_right = sorted_array[mid_clamped] < values
        lo = mx.where(go_right, mid + 1, lo)
        hi = mx.where(go_right, hi, mid)

    return lo


class UMAP:
    """UMAP dimensionality reduction using MLX on Metal GPU.

    Parameters:
        n_components: Dimension of the embedded space (default 2).
        n_neighbors: Number of nearest neighbors (default 15).
        min_dist: Minimum distance in low-dimensional space (default 0.1).
        spread: Effective scale of embedded points (default 1.0).
        n_epochs: Number of optimization epochs (default: 500 for N<=10K, 200 for larger).
        learning_rate: SGD learning rate (default 1.0).
        negative_sample_rate: Negative samples per positive edge (default 5).
        random_state: Random seed for reproducibility.
        verbose: Print progress.
        pca_dim: PCA preprocessing dimension (default None = no PCA).
            Set to e.g. 100 to reduce high-dimensional inputs before KNN.
            Reduces curse of dimensionality and speeds up distance computation.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: int | None = None,
        learning_rate: float = 1.0,
        negative_sample_rate: int = 5,
        random_state: int | None = None,
        verbose: bool = False,
        pca_dim: int | None = None,
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
        self.pca_dim = pca_dim
        self.embedding_ = None

    def fit_transform(self, X, epoch_callback=None) -> np.ndarray:
        """Fit UMAP and return the embedding.

        Args:
            X: Input data, shape (n_samples, n_features).
            epoch_callback: Optional callable(epoch, Y_numpy) for snapshots.
        """
        if isinstance(X, mx.array):
            X = np.array(X)
        X = np.asarray(X, dtype=np.float32)
        n, dim = X.shape

        # Optional PCA preprocessing for high-dimensional data
        if self.pca_dim is not None and dim > self.pca_dim:
            if self.verbose:
                print(f"Applying PCA: {dim} -> {self.pca_dim} dims...")
            X_mx = mx.array(X)
            mean = mx.mean(X_mx, axis=0)
            X_centered = X_mx - mean
            cov = (X_centered.T @ X_centered) / (n - 1)
            mx.eval(cov)
            eigvals, eigvecs = mx.linalg.eigh(cov, stream=mx.cpu)
            mx.eval(eigvals, eigvecs)
            # Take top pca_dim components (eigh returns ascending order)
            proj = eigvecs[:, -self.pca_dim:][:, ::-1]
            X_pca = X_centered @ proj
            mx.eval(X_pca)
            X_for_knn = np.array(X_pca)
            if self.verbose:
                # Report variance retained
                total_var = mx.sum(eigvals).item()
                retained_var = mx.sum(eigvals[-self.pca_dim:]).item()
                print(f"Variance retained: {retained_var/total_var*100:.1f}%")
        else:
            X_for_knn = X

        if self.verbose:
            print("Computing nearest neighbors...")
        knn_indices, knn_dists = self._compute_knn(X_for_knn)

        if self.n_epochs is None:
            self.n_epochs = 500 if n <= 10000 else 200

        if self.verbose:
            print("Building fuzzy simplicial set...")
        graph_rows, graph_cols, graph_vals = self._fuzzy_simplicial_set(knn_indices, knn_dists, n)

        a, b = self._find_ab_params(self.spread, self.min_dist)

        if self.verbose:
            print("Initializing embedding...")
        if self.random_state is not None:
            mx.random.seed(self.random_state)
        Y = self._spectral_init(graph_rows, graph_cols, graph_vals, n)
        mx.eval(Y)

        Y = self._optimize(graph_rows, graph_cols, graph_vals, Y, a, b, n,
                           epoch_callback=epoch_callback)
        mx.eval(Y)

        self.embedding_ = np.array(Y)
        return self.embedding_

    def _compute_knn(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact k-nearest neighbors on Metal GPU.

        Uses chunked pairwise distance computation with async pipeline.
        """
        n = X.shape[0]
        k = self.n_neighbors
        X_mx = mx.array(X)
        sum_sq = mx.sum(X_mx * X_mx, axis=1)
        mx.eval(sum_sq)

        chunk_size = min(n, max(1000, 500_000_000 // (n * 4)))
        knn_indices = np.zeros((n, k), dtype=np.int32)
        knn_dists = np.zeros((n, k), dtype=np.float32)

        prev_idx = prev_dists = prev_start = prev_end = None

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            X_chunk = X_mx[start:end]
            sum_sq_chunk = sum_sq[start:end]

            D_chunk = mx.maximum(
                sum_sq_chunk[:, None] + sum_sq[None, :] - 2.0 * (X_chunk @ X_mx.T),
                0.0
            )

            # Set self-distance to inf
            if end - start == n:
                D_chunk = D_chunk + mx.eye(n) * 1e30
            else:
                arange_chunk = mx.arange(start, end)[:, None]
                arange_all = mx.arange(n)[None, :]
                D_chunk = D_chunk + (arange_chunk == arange_all).astype(mx.float32) * 1e30

            sorted_all = mx.argsort(D_chunk, axis=1)[:, :k]
            sorted_dists = mx.take_along_axis(D_chunk, sorted_all, axis=1)

            # Pipeline: collect previous while current computes
            if prev_idx is not None:
                mx.eval(prev_idx, prev_dists)
                knn_indices[prev_start:prev_end] = np.array(prev_idx).astype(np.int32)
                knn_dists[prev_start:prev_end] = np.sqrt(np.maximum(np.array(prev_dists), 0)).astype(np.float32)

            prev_idx, prev_dists, prev_start, prev_end = sorted_all, sorted_dists, start, end

        if prev_idx is not None:
            mx.eval(prev_idx, prev_dists)
            knn_indices[prev_start:prev_end] = np.array(prev_idx).astype(np.int32)
            knn_dists[prev_start:prev_end] = np.sqrt(np.maximum(np.array(prev_dists), 0)).astype(np.float32)

        return knn_indices, knn_dists

    def _fuzzy_simplicial_set(
        self, knn_indices: np.ndarray, knn_dists: np.ndarray, n: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Build fuzzy simplicial set on GPU."""
        k = self.n_neighbors
        target = np.log2(k)

        knn_dists_mx = mx.array(knn_dists)

        # Rho: distance to nearest non-zero neighbor
        mask = knn_dists_mx > 0
        rhos = mx.where(mask, knn_dists_mx, mx.array(float('inf')))
        rhos = mx.maximum(mx.min(rhos, axis=1), mx.array(1e-8))
        mx.eval(rhos)

        # Binary search for sigma (vectorized over all N points on GPU)
        lo = mx.full((n,), 1e-20)
        hi = mx.full((n,), 1e3)
        sigma = mx.ones((n,))
        dists_shifted = mx.maximum(knn_dists_mx - rhos[:, None], 0.0)
        dists_shifted_tail = dists_shifted[:, 1:]  # skip j=0 (rho contributor)

        for _ in range(64):
            vals = mx.exp(-dists_shifted_tail / sigma[:, None])
            vals_sum = mx.sum(vals, axis=1)

            converged = mx.abs(vals_sum - target) < 1e-5
            too_high = (vals_sum > target) & ~converged
            too_low = (vals_sum < target) & ~converged

            hi = mx.where(too_high, sigma, hi)
            lo = mx.where(too_low, sigma, lo)
            sigma = mx.where(too_high, (lo + sigma) / 2.0, sigma)
            sigma = mx.where(too_low, mx.where(hi >= 1e3, sigma * 2.0, (sigma + hi) / 2.0), sigma)

            mx.eval(sigma)
            if bool(mx.all(converged)):
                break

        # Edge weights
        weights = mx.exp(-dists_shifted / mx.maximum(sigma[:, None], 1e-10))
        mx.eval(weights)

        # Build sparse edges (rows/cols from numpy, vals from MLX)
        rows_np = np.repeat(np.arange(n, dtype=np.int32), k)
        cols_np = knn_indices.ravel().astype(np.int32)
        rows_mx = mx.array(rows_np)
        cols_mx = mx.array(cols_np)
        vals_mx = weights.reshape(-1)

        # Symmetrize on GPU: P = A + A^T - A * A^T
        fwd_keys = rows_mx.astype(mx.int64) * n + cols_mx.astype(mx.int64)
        rev_keys = cols_mx.astype(mx.int64) * n + rows_mx.astype(mx.int64)

        sort_idx = mx.argsort(fwd_keys)
        sorted_keys = fwd_keys[sort_idx]
        sorted_vals = vals_mx[sort_idx]

        pos = _searchsorted(sorted_keys, rev_keys)
        pos = mx.minimum(pos, sorted_keys.shape[0] - 1)
        matched = sorted_keys[pos] == rev_keys
        w_rev = mx.where(matched, sorted_vals[pos], 0.0)

        w_sym = vals_mx + w_rev - vals_mx * w_rev

        # Prune weak edges
        threshold = mx.max(w_sym) / self.n_epochs
        mx.eval(w_sym, threshold)
        mask_np = np.array(w_sym >= threshold)
        active = np.nonzero(mask_np)[0].astype(np.int32)
        active_mx = mx.array(active)
        return rows_mx[active_mx], cols_mx[active_mx], w_sym[active_mx]

    @staticmethod
    def _find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
        """Find a, b parameters via Gauss-Newton optimization (no scipy)."""
        xv = np.linspace(0, spread * 3, 300)
        yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))

        a, b = 1.0, 1.0
        for _ in range(100):
            x2b = np.power(xv, 2 * b)
            denom = 1.0 + a * x2b
            pred = 1.0 / denom
            residual = pred - yv

            da = -x2b / (denom * denom)
            db = -a * 2.0 * np.log(np.maximum(xv, 1e-20)) * x2b / (denom * denom)

            J = np.column_stack([da, db])
            step = np.linalg.lstsq(J, -residual, rcond=None)[0]
            a += step[0]
            b += step[1]

            if np.sum(step ** 2) < 1e-12:
                break

        return float(a), float(b)

    def _spectral_init(self, rows, cols, vals, n):
        """Spectral initialization via power iteration on normalized Laplacian (pure MLX)."""
        try:
            k = self.n_components + 1

            mx.eval(rows, cols, vals)

            # Compute degrees and normalize on GPU
            # Scatter-add for degrees
            degrees = mx.zeros((n,))
            degrees = degrees.at[rows].add(vals)
            mx.eval(degrees)
            degrees = mx.maximum(degrees, 1e-10)
            d_inv_sqrt = 1.0 / mx.sqrt(degrees)

            # Normalized edge weights
            w_norm = vals * d_inv_sqrt[rows] * d_inv_sqrt[cols]
            mx.eval(w_norm)

            # Sparse matvec on GPU
            def sparse_matvec(x):
                gathered = w_norm[:, None] * x[cols]
                result = mx.zeros_like(x)
                result = result.at[rows].add(gathered)
                return result

            # Power iteration for top-k eigenvectors
            V = mx.random.normal((n, k))
            mx.eval(V)

            for _ in range(100):
                V = sparse_matvec(V)
                # Modified Gram-Schmidt (batched eval)
                for j in range(k):
                    for i in range(j):
                        proj = mx.sum(V[:, j] * V[:, i])
                        V = V.at[:, j].add(-proj * V[:, i])
                    norm = mx.sqrt(mx.sum(V[:, j] * V[:, j]) + 1e-10)
                    V = V.at[:, j].multiply(1.0 / norm)
                mx.eval(V)

            embedding = V[:, 1:k]
            mx.eval(embedding)

            # Scale to [0, 10] + noise
            embed_np = np.array(embedding)
            rng = np.random.RandomState(self.random_state or 42)
            expansion = 10.0 / np.max(np.abs(embed_np))
            embed_np = (embed_np * expansion).astype(np.float32)
            embed_np += rng.normal(scale=0.0001, size=embed_np.shape).astype(np.float32)
            embed_np = 10.0 * (embed_np - embed_np.min(axis=0)) / (embed_np.max(axis=0) - embed_np.min(axis=0) + 1e-10)

            if self.verbose:
                print("Using spectral initialization")
            return mx.array(embed_np)

        except Exception as e:
            if self.verbose:
                print(f"Spectral init failed ({e}), using random initialization")
            return mx.random.normal((n, self.n_components)) * 0.01

    @staticmethod
    def _sgd_step(Y, ef, et, neg_from_idx, neg_to_idx, alpha_epoch, a, b):
        """Single SGD step -- pure MLX, ready for mx.compile."""
        # Positive forces (attract)
        y_from = Y[ef]
        y_to = Y[et]
        diff = y_from - y_to
        dist_sq = mx.maximum(mx.sum(diff * diff, axis=1, keepdims=True), 1e-6)

        pow_val = mx.power(dist_sq, b)
        grad_coeff = -2.0 * a * b * mx.power(dist_sq, b - 1.0) / (1.0 + a * pow_val)
        pos_grad = mx.clip(grad_coeff * diff, -4.0, 4.0) * alpha_epoch

        # Negative forces (repel)
        y_neg_from = Y[neg_from_idx]
        y_neg_to = Y[neg_to_idx]
        neg_diff = y_neg_from - y_neg_to
        neg_dist_sq = mx.maximum(mx.sum(neg_diff * neg_diff, axis=1, keepdims=True), 1e-6)

        neg_pow = mx.power(neg_dist_sq, b)
        neg_grad_coeff = 2.0 * b / ((0.001 + neg_dist_sq) * (1.0 + a * neg_pow))
        neg_grad = mx.clip(neg_grad_coeff * neg_diff, -4.0, 4.0) * alpha_epoch

        # Scatter updates (move both head and tail)
        Y = Y.at[ef].add(pos_grad)
        Y = Y.at[et].add(-pos_grad)
        Y = Y.at[neg_from_idx].add(neg_grad)
        return Y

    def _optimize(
        self, edge_from: mx.array, edge_to: mx.array, edge_weights: mx.array,
        Y: mx.array, a: float, b: float, n: int,
        epoch_callback=None,
    ) -> mx.array:
        """Pure MLX SGD optimization with pre-computed scheduling.

        Args:
            epoch_callback: Optional callable(epoch, Y_numpy) called after each epoch.
        """
        mx.eval(edge_weights)
        max_weight = float(mx.max(edge_weights).item())
        weights_np = np.array(edge_weights)
        n_samples = self.n_epochs * (weights_np / max_weight)
        epochs_per_sample = np.where(n_samples > 0, self.n_epochs / n_samples, -1.0)
        epochs_per_next = epochs_per_sample.copy()

        # Pre-compute active edge sets (all numpy work upfront)
        active_sets = []
        for epoch in range(self.n_epochs):
            active = np.where(epochs_per_next <= epoch)[0]
            if len(active) > 0:
                epochs_per_next[active] += epochs_per_sample[active]
                active_sets.append(mx.array(active.astype(np.int32)))
            else:
                active_sets.append(None)

        a_mx = mx.array(a)
        b_mx = mx.array(b)
        alpha = self.learning_rate

        if epoch_callback is not None:
            epoch_callback(0, np.array(Y))

        for epoch in range(self.n_epochs):
            if active_sets[epoch] is None:
                continue

            active_mx = active_sets[epoch]
            ef = edge_from[active_mx]
            et = edge_to[active_mx]
            n_active = active_mx.shape[0]
            alpha_epoch = mx.array(alpha * (1.0 - epoch / self.n_epochs))

            # Negative sampling: fixed rate per active edge (pure MLX)
            n_neg = self.negative_sample_rate * n_active
            neg_from_idx = ef[mx.arange(n_neg) % n_active]
            neg_to_idx = mx.random.randint(0, n, (n_neg,))

            Y = self._sgd_step(Y, ef, et, neg_from_idx, neg_to_idx, alpha_epoch, a_mx, b_mx)

            if epoch_callback is not None:
                mx.eval(Y)
                epoch_callback(epoch + 1, np.array(Y))
            elif (epoch + 1) % 10 == 0 or epoch == self.n_epochs - 1:
                mx.eval(Y)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}")

        return Y
