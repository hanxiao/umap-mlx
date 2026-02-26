"""Pure MLX implementation of UMAP for Apple Silicon."""
import mlx.core as mx
import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import coo_matrix
from typing import Optional


def _find_ab_params(spread: float, min_dist: float):
    """Fit a, b parameters for the differentiable curve used in UMAP."""
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))
    
    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros_like(xv)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    
    params, _ = curve_fit(curve, xv, yv)
    return float(params[0]), float(params[1])


def _smooth_knn_dist(distances, k, n_iter=64, bandwidth=1.0):
    """Compute smooth KNN distances (sigma) via binary search.
    
    For each point, find sigma such that:
        sum_{j in knn} exp(-(d(i,j) - rho_i) / sigma_i) = log2(k)
    
    distances: (n, k) - distances to k nearest neighbors
    Returns: sigma (n,), rho (n,)
    """
    target = np.log2(k) * bandwidth
    n = distances.shape[0]
    
    rho = distances[:, 0]  # distance to nearest neighbor
    sigma = np.ones(n, dtype=np.float64)
    
    for i in range(n):
        lo = 0.0
        hi = np.inf
        mid = 1.0
        
        for _ in range(n_iter):
            psum = 0.0
            for j in range(distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-d / mid)
                else:
                    psum += 1.0
            
            if np.abs(psum - target) < 1e-5:
                break
            
            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == np.inf:
                    mid *= 2.0
                else:
                    mid = (lo + hi) / 2.0
        
        sigma[i] = mid
    
    return sigma, rho


class UMAP:
    """UMAP dimensionality reduction using MLX on Metal GPU.
    
    Parameters:
        n_components: Dimension of embedding space (default: 2)
        n_neighbors: Number of neighbors for manifold approximation (default: 15)
        min_dist: Minimum distance between points in embedding (default: 0.1)
        spread: Effective scale of embedded points (default: 1.0)
        n_epochs: Number of optimization epochs (default: 200)
        learning_rate: SGD learning rate (default: 1.0)
        negative_sample_rate: Negative samples per positive edge (default: 5)
        random_state: Random seed for reproducibility (default: None)
    """
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: int = 500,
        learning_rate: float = 1.0,
        negative_sample_rate: int = 5,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.embedding_ = None
        self._a, self._b = _find_ab_params(spread, min_dist)
        
    def _compute_knn_mlx(self, X: mx.array):
        """Compute KNN graph using MLX pairwise distances on Metal GPU.
        
        Returns numpy arrays for downstream processing.
        """
        n = X.shape[0]
        k = self.n_neighbors
        
        # For large n, compute in chunks to avoid OOM
        chunk_size = min(n, 4096)
        
        knn_indices = np.zeros((n, k), dtype=np.int32)
        knn_dists = np.zeros((n, k), dtype=np.float32)
        
        # Precompute norms
        X_norm = mx.sum(X * X, axis=1)  # (n,)
        mx.eval(X_norm)
        
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            X_chunk = X[start:end]  # (chunk, d)
            
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
            chunk_norm = mx.sum(X_chunk * X_chunk, axis=1, keepdims=True)  # (chunk, 1)
            dist_sq = chunk_norm + X_norm[None, :] - 2 * mx.matmul(X_chunk, X.T)
            dist_sq = mx.maximum(dist_sq, 0)
            
            # Get k+1 nearest (including self)
            # MLX argsort then slice
            sorted_idx = mx.argsort(dist_sq, axis=1)[:, :k+1]
            sorted_dist_sq = mx.take_along_axis(dist_sq, sorted_idx, axis=1)
            mx.eval(sorted_idx, sorted_dist_sq)
            
            sorted_idx_np = np.array(sorted_idx)
            sorted_dist_np = np.sqrt(np.array(sorted_dist_sq))
            
            # Remove self (index 0 if data has no duplicates)
            for i in range(end - start):
                row = sorted_idx_np[i]
                dists_row = sorted_dist_np[i]
                # Find self
                self_pos = np.where(row == (start + i))[0]
                if len(self_pos) > 0:
                    mask = np.ones(len(row), dtype=bool)
                    mask[self_pos[0]] = False
                    knn_indices[start + i] = row[mask][:k]
                    knn_dists[start + i] = dists_row[mask][:k]
                else:
                    knn_indices[start + i] = row[:k]
                    knn_dists[start + i] = dists_row[:k]
        
        return knn_indices, knn_dists
    
    def _fuzzy_simplicial_set(self, knn_indices, knn_dists, n):
        """Construct fuzzy simplicial set from KNN graph.
        
        Returns sparse COO matrix of membership strengths.
        """
        k = self.n_neighbors
        
        # Compute sigma and rho via binary search
        sigma, rho = _smooth_knn_dist(knn_dists, k)
        
        # Build membership strengths
        rows = []
        cols = []
        vals = []
        
        for i in range(n):
            for j in range(k):
                neighbor = knn_indices[i, j]
                d = knn_dists[i, j]
                
                if d - rho[i] <= 0:
                    val = 1.0
                else:
                    val = np.exp(-(d - rho[i]) / sigma[i])
                
                rows.append(i)
                cols.append(neighbor)
                vals.append(val)
        
        # Build sparse matrix
        graph = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        
        # Symmetrize: P_sym = P + P^T - P * P^T
        transpose = graph.T.tocsr()
        product = graph.multiply(transpose)
        graph = graph + transpose - product
        
        return graph.tocoo()
    
    def _optimize_layout(self, graph_coo, n):
        """Optimize low-dimensional embedding via SGD.
        
        Uses epoch-based sampling with per-edge epoch scheduling
        (following the original UMAP implementation).
        """
        rng = np.random.RandomState(self.random_state)
        
        a, b = self._a, self._b
        
        # Spectral initialization via PCA of the graph Laplacian
        # Use sparse eigenvector decomposition for speed
        try:
            from scipy.sparse import diags as sparse_diags
            from scipy.sparse.linalg import eigsh
            
            graph_csr = graph_coo.tocsr()
            degrees = np.array(graph_csr.sum(axis=1)).flatten()
            D = sparse_diags(degrees)
            L = D - graph_csr
            # Normalized Laplacian eigenvectors
            D_inv_sqrt = sparse_diags(1.0 / np.sqrt(np.maximum(degrees, 1e-10)))
            L_norm = D_inv_sqrt @ L @ D_inv_sqrt
            
            # Get smallest non-trivial eigenvectors
            _, eigvecs = eigsh(L_norm, k=self.n_components + 1, which='SM')
            embedding = eigvecs[:, 1:self.n_components + 1].astype(np.float32)
            # Normalize each component to have small std (like umap-learn)
            for d in range(self.n_components):
                std = np.std(embedding[:, d])
                if std > 0:
                    embedding[:, d] = embedding[:, d] / std * 1e-4
                    # Center
                    embedding[:, d] -= np.median(embedding[:, d])
        except Exception:
            embedding = rng.normal(0, 1e-4, size=(n, self.n_components)).astype(np.float32)
        
        # Get edges and weights
        rows = graph_coo.row.astype(np.int32)
        cols = graph_coo.col.astype(np.int32)
        weights = graph_coo.data.astype(np.float32)
        n_edges = len(rows)
        
        # Compute epochs per edge (higher weight = sampled more often)
        # Following original UMAP: edges with weight w are sampled
        # ceil(w/max_w * n_epochs) times total
        max_weight = weights.max()
        n_epochs_per_edge = weights / max_weight * self.n_epochs
        epochs_per_sample = np.where(n_epochs_per_edge > 0, 
                                      self.n_epochs / n_epochs_per_edge, 
                                      float('inf'))
        # Start sampling immediately, staggered
        epoch_of_next_sample = rng.uniform(0, epochs_per_sample)
        
        for epoch in range(self.n_epochs):
            alpha = self.learning_rate * (1.0 - epoch / self.n_epochs)
            
            # Find edges to sample this epoch
            mask = epoch_of_next_sample <= epoch
            active = np.where(mask)[0]
            
            if len(active) == 0:
                continue
            
            # Update next sample time
            epoch_of_next_sample[active] += epochs_per_sample[active]
            
            i_idx = rows[active]
            j_idx = cols[active]
            
            # --- Attractive forces (pull connected points together) ---
            diff = embedding[i_idx] - embedding[j_idx]
            dist_sq = np.sum(diff * diff, axis=1)
            dist_sq = np.maximum(dist_sq, 1e-10)
            
            # Attractive: grad wrt y_i of -log(1/(1+a*d^2b))
            # = -2ab * d^(2b-2) / (1 + a*d^2b) * (y_i - y_j)
            grad_coef = -2.0 * a * b * np.power(dist_sq, b - 1.0)
            grad_coef /= (1.0 + a * np.power(dist_sq, b))
            
            # grad is negative (attractive), so adding it to i moves i toward j
            grad = grad_coef[:, None] * diff
            grad = np.clip(grad, -4.0, 4.0)
            
            embedding[i_idx] += alpha * grad
            embedding[j_idx] -= alpha * grad
            
            # --- Repulsive forces (push non-neighbors apart) ---
            # Vectorized: sample all negative pairs at once
            n_active = len(active)
            n_neg_total = n_active * self.negative_sample_rate
            neg_i = np.repeat(i_idx, self.negative_sample_rate)
            neg_j = rng.randint(0, n, size=n_neg_total)
            
            # Skip self-pairs
            valid = neg_j != neg_i
            neg_i = neg_i[valid]
            neg_j = neg_j[valid]
            
            diff_neg = embedding[neg_i] - embedding[neg_j]
            dist_sq_neg = np.sum(diff_neg * diff_neg, axis=1)
            dist_sq_neg = np.maximum(dist_sq_neg, 1e-10)
            
            grad_coef_neg = 2.0 * b
            grad_coef_neg /= ((0.001 + dist_sq_neg) * (1.0 + a * np.power(dist_sq_neg, b)))
            grad_coef_neg = np.minimum(grad_coef_neg, 4.0)
            
            grad_neg = grad_coef_neg[:, None] * diff_neg
            grad_neg = np.clip(grad_neg, -4.0, 4.0)
            
            embedding[neg_i] += alpha * grad_neg
        
        return embedding
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit UMAP and return transformed data.
        
        Args:
            X: Input data (n_samples, n_features) as numpy array
            
        Returns:
            Embedding (n_samples, n_components) as numpy array
        """
        n = X.shape[0]
        X_mlx = mx.array(X.astype(np.float32))
        
        # Step 1: KNN on Metal GPU
        knn_indices, knn_dists = self._compute_knn_mlx(X_mlx)
        
        # Step 2: Fuzzy simplicial set
        graph = self._fuzzy_simplicial_set(knn_indices, knn_dists, n)
        
        # Step 3: Optimize layout
        embedding = self._optimize_layout(graph, n)
        
        self.embedding_ = embedding
        return self.embedding_
    
    def fit(self, X: np.ndarray):
        """Fit UMAP to data."""
        self.fit_transform(X)
        return self
