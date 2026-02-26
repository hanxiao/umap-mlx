"""UMAP implementation in pure MLX for Apple Silicon."""

import mlx.core as mx
import numpy as np


def _searchsorted(sorted_array: mx.array, values: mx.array) -> mx.array:
    """Vectorized binary search on GPU (equivalent to np.searchsorted).

    For each value in `values`, find the insertion index in `sorted_array`.
    """
    n = sorted_array.shape[0]
    m = values.shape[0]
    lo = mx.zeros((m,), dtype=mx.int32)
    hi = mx.full((m,), n, dtype=mx.int32)

    # ceil(log2(n)) iterations suffice
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        mid = (lo + hi) // 2
        # Clamp mid to valid range for indexing
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

        # Step 2: auto epochs (same as umap-learn: 500 for N<=10K, 200 otherwise)
        if self.n_epochs is None:
            self.n_epochs = 500 if n <= 10000 else 200

        # Step 3: Fuzzy simplicial set
        if self.verbose:
            print("Building fuzzy simplicial set...")
        graph_rows, graph_cols, graph_vals = self._fuzzy_simplicial_set(knn_indices, knn_dists, n)

        # Step 4: a/b parameters
        a, b = self._find_ab_params(self.spread, self.min_dist)

        # Step 4: Initialize (spectral via normalized Laplacian, fallback to random)
        if self.verbose:
            print("Initializing embedding...")
        if self.random_state is not None:
            mx.random.seed(self.random_state)
        Y = self._spectral_init(graph_rows, graph_cols, graph_vals, n)
        mx.eval(Y)

        # Step 5: Optimize (pure MLX)
        Y = self._optimize(graph_rows, graph_cols, graph_vals, Y, a, b, n)
        mx.eval(Y)

        self.embedding_ = np.array(Y)
        return self.embedding_

    def _compute_knn(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact k-nearest neighbors, entirely on Metal GPU.

        Uses chunked distance computation to handle large datasets,
        with mx.argpartition + mx.argsort for top-k selection.
        """
        n = X.shape[0]
        k = self.n_neighbors
        X_mx = mx.array(X)
        sum_sq = mx.sum(X_mx * X_mx, axis=1)
        mx.eval(sum_sq)

        # Process in chunks to control GPU memory
        # Each chunk: (chunk_size, n) distance matrix, ~2GB per chunk
        chunk_size = min(n, max(1000, 500_000_000 // (n * 4)))
        knn_indices = np.zeros((n, k), dtype=np.int32)
        knn_dists = np.zeros((n, k), dtype=np.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            X_chunk = X_mx[start:end]
            sum_sq_chunk = sum_sq[start:end]

            # Pairwise distances: chunk vs all
            D_chunk = mx.maximum(
                sum_sq_chunk[:, None] + sum_sq[None, :] - 2.0 * (X_chunk @ X_mx.T),
                0.0
            )

            # Set self-distance to inf
            # Create mask for diagonal block
            if end - start == n:
                D_chunk = D_chunk + mx.eye(n) * 1e30
            else:
                arange_chunk = mx.arange(start, end)[:, None]
                arange_all = mx.arange(n)[None, :]
                self_mask = (arange_chunk == arange_all).astype(mx.float32)
                D_chunk = D_chunk + self_mask * 1e30

            # Top-k on GPU: argpartition then sort within k
            top_k_idx = mx.argpartition(D_chunk, kth=k, axis=1)[:, :k]
            top_k_dists = mx.take_along_axis(D_chunk, top_k_idx, axis=1)
            sort_order = mx.argsort(top_k_dists, axis=1)
            sorted_idx = mx.take_along_axis(top_k_idx, sort_order, axis=1)
            sorted_dists = mx.take_along_axis(top_k_dists, sort_order, axis=1)

            mx.eval(sorted_idx, sorted_dists)

            knn_indices[start:end] = np.array(sorted_idx).astype(np.int32)
            knn_dists[start:end] = np.sqrt(np.maximum(np.array(sorted_dists), 0)).astype(np.float32)

        return knn_indices, knn_dists

    def _fuzzy_simplicial_set(
        self, knn_indices: np.ndarray, knn_dists: np.ndarray, n: int
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Build fuzzy simplicial set. Vectorized binary search for sigma."""
        k = self.n_neighbors
        target = np.log2(k)

        # Vectorized rho + binary search for sigma on MLX GPU
        knn_dists_mx = mx.array(knn_dists)

        # Rho: nearest non-zero distance per point
        mask = knn_dists_mx > 0
        rhos = mx.where(mask, knn_dists_mx, mx.array(float('inf')))
        rhos = mx.maximum(mx.min(rhos, axis=1), mx.array(1e-8))
        mx.eval(rhos)

        # Binary search for sigma (all N points simultaneously on GPU)
        lo = mx.full((n,), 1e-20)
        hi = mx.full((n,), 1e3)
        sigma = mx.ones((n,))
        dists_shifted = mx.maximum(knn_dists_mx - rhos[:, None], 0.0)

        for _ in range(64):
            vals = mx.exp(-dists_shifted / sigma[:, None])
            vals_sum = mx.sum(vals, axis=1)

            converged = mx.abs(vals_sum - target) < 1e-5
            too_high = (vals_sum > target) & ~converged
            too_low = (vals_sum < target) & ~converged

            hi = mx.where(too_high, sigma, hi)
            lo = mx.where(too_low, sigma, lo)

            sigma = mx.where(too_high, (lo + sigma) / 2, sigma)
            sigma = mx.where(too_low, mx.where(hi >= 1e3, sigma * 2, (sigma + hi) / 2), sigma)

            mx.eval(sigma)
            if bool(mx.all(converged)):
                break

        # Edge weights on GPU
        weights = mx.exp(-dists_shifted / mx.maximum(sigma[:, None], 1e-10))
        mx.eval(weights)

        # Build sparse edges
        rows = np.repeat(np.arange(n, dtype=np.int32), k)
        cols = knn_indices.ravel().astype(np.int32)
        vals = np.array(weights.reshape(-1)).astype(np.float32)

        # Symmetrize: P = A + A^T - A * A^T

        # For each edge (r,c), find reverse edge (c,r) -- all on MLX GPU
        rows_mx = mx.array(rows)
        cols_mx = mx.array(cols)
        vals_mx = mx.array(vals)

        # Encode edges as single int: r * n + c
        fwd_keys = rows_mx.astype(mx.int64) * n + cols_mx.astype(mx.int64)
        rev_keys = cols_mx.astype(mx.int64) * n + rows_mx.astype(mx.int64)

        # Sort forward keys for binary search
        sort_idx = mx.argsort(fwd_keys)
        sorted_keys = fwd_keys[sort_idx]
        sorted_vals = vals_mx[sort_idx]

        # Find reverse weights via GPU searchsorted
        pos = _searchsorted(sorted_keys, rev_keys)
        pos = mx.minimum(pos, sorted_keys.shape[0] - 1)
        matched = sorted_keys[pos] == rev_keys
        w_rev = mx.where(matched, sorted_vals[pos], 0.0)

        # Symmetrize: P = A + A^T - A * A^T
        w_sym = vals_mx + w_rev - vals_mx * w_rev

        # Prune weak edges and extract (MLX has no boolean indexing)
        threshold = mx.max(w_sym) / self.n_epochs
        mx.eval(w_sym, threshold)
        mask_np = np.array(w_sym >= threshold)
        active = np.nonzero(mask_np)[0].astype(np.int32)
        active_mx = mx.array(active)
        return (
            rows_mx[active_mx],
            cols_mx[active_mx],
            w_sym[active_mx],
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

    def _spectral_init(self, rows, cols, vals, n):
        """Spectral initialization via normalized graph Laplacian on GPU.

        Computes eigenvectors of D^{-1/2} W D^{-1/2} using power iteration.
        Equivalent to smallest eigenvectors of normalized Laplacian L = I - D^{-1/2} W D^{-1/2}.
        Falls back to random initialization on failure.
        """
        try:
            k = self.n_components + 1  # +1 because first eigenvector is trivial

            # Build normalized adjacency: A_norm = D^{-1/2} W D^{-1/2}
            # W is sparse (rows, cols, vals), compute degrees
            mx.eval(rows, cols, vals)
            rows_np = np.array(rows)
            cols_np = np.array(cols)
            vals_np = np.array(vals)

            # Degree vector
            degrees = np.zeros(n, dtype=np.float64)
            np.add.at(degrees, rows_np, vals_np)
            degrees = np.maximum(degrees, 1e-10)
            d_inv_sqrt = (1.0 / np.sqrt(degrees)).astype(np.float32)

            # Normalize edge weights: w_norm = d_inv_sqrt[i] * w * d_inv_sqrt[j]
            w_norm = vals_np * d_inv_sqrt[rows_np] * d_inv_sqrt[cols_np]

            # Sparse matrix-vector multiply on GPU via scatter
            rows_mx = mx.array(rows_np.astype(np.int32))
            cols_mx = mx.array(cols_np.astype(np.int32))
            w_norm_mx = mx.array(w_norm.astype(np.float32))

            def sparse_matvec(x):
                """Compute A_norm @ x using sparse (rows, cols, w_norm)."""
                # Gather: vals * x[cols]
                gathered = w_norm_mx[:, None] * x[cols_mx]
                # Scatter-add into result
                result = mx.zeros_like(x)
                result = result.at[rows_mx].add(gathered)
                return result

            # Power iteration for top-k eigenvectors of A_norm
            # Use random initial vectors
            V = mx.random.normal((n, k))
            mx.eval(V)

            for iteration in range(100):
                # Multiply: V = A_norm @ V
                V = sparse_matvec(V)

                # QR orthogonalization via modified Gram-Schmidt
                for j in range(k):
                    for i in range(j):
                        proj = mx.sum(V[:, j] * V[:, i])
                        V = V.at[:, j].add(-proj * V[:, i])
                    norm = mx.sqrt(mx.sum(V[:, j] * V[:, j]) + 1e-10)
                    V = V.at[:, j].multiply(1.0 / norm)

                # Eval once per iteration (not per column)
                mx.eval(V)

            # Skip first eigenvector (constant), take next n_components
            embedding = V[:, 1:k]

            # Expand to reasonable scale
            mx.eval(embedding)
            embed_np = np.array(embedding)
            noise = np.random.RandomState(self.random_state or 42).normal(
                scale=0.0001, size=embed_np.shape
            ).astype(np.float32)
            embed_np = embed_np / (embed_np.std(axis=0, keepdims=True) + 1e-10) * 0.0001
            embed_np += noise

            if self.verbose:
                print("Using spectral initialization")
            return mx.array(embed_np)

        except Exception as e:
            if self.verbose:
                print(f"Spectral init failed ({e}), using random initialization")
            return mx.random.normal((n, self.n_components)) * (10.0 if n > 10000 else 0.01)

    @staticmethod
    def _sgd_step(Y, ef, et, neg_from_idx, neg_to_idx, alpha_epoch, a, b):
        """Single SGD step -- compiled for GPU fusion."""
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

        # Scatter updates
        Y = Y.at[ef].add(pos_grad)
        Y = Y.at[et].add(-pos_grad)
        Y = Y.at[neg_from_idx].add(neg_grad)
        return Y

    def _optimize(
        self, edge_from: mx.array, edge_to: mx.array, edge_weights: mx.array,
        Y: mx.array, a: float, b: float, n: int
    ) -> mx.array:
        """Pure MLX SGD optimization on Metal GPU with compiled step."""
        # Compute epochs_per_sample (scheduling only, stays in numpy)
        mx.eval(edge_weights)
        max_weight = float(mx.max(edge_weights).item())
        weights_np = np.array(edge_weights)
        n_samples = self.n_epochs * (weights_np / max_weight)
        epochs_per_sample = np.where(n_samples > 0, self.n_epochs / n_samples, -1.0)
        epochs_per_next = epochs_per_sample.copy()

        alpha = self.learning_rate
        # Wrap a/b as mx.array to avoid recompilation on each call
        a_mx = mx.array(a)
        b_mx = mx.array(b)
        # Compile the SGD step for GPU kernel fusion
        compiled_step = mx.compile(self._sgd_step)

        for epoch in range(self.n_epochs):
            alpha_epoch = mx.array(alpha * (1.0 - epoch / self.n_epochs))

            # Active edges (scheduling in numpy)
            active = np.where(epochs_per_next <= epoch)[0]
            if len(active) == 0:
                continue

            active_mx = mx.array(active.astype(np.int32))
            ef = edge_from[active_mx]
            et = edge_to[active_mx]
            n_active = len(active)

            # Negative sampling indices
            n_neg = self.negative_sample_rate * n_active
            neg_from_idx = ef[mx.arange(n_neg) % n_active]
            neg_to_idx = mx.random.randint(0, n, (n_neg,))

            Y = compiled_step(Y, ef, et, neg_from_idx, neg_to_idx, alpha_epoch, a_mx, b_mx)
            mx.eval(Y)

            epochs_per_next[active] += epochs_per_sample[active]

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}")

        return Y
