"""Verify UMAP-MLX correctness against umap-learn."""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from umap import UMAP as UMAP_learn
from umap_mlx import UMAP as UMAP_mlx


def compare_umaps(n_samples=1797):
    """Compare UMAP-MLX vs umap-learn on full MNIST digits."""
    print(f"Loading data ({n_samples} samples)...")
    digits = load_digits()
    X = digits.data[:n_samples]
    y = digits.target[:n_samples]
    X = StandardScaler().fit_transform(X)
    print(f"Data shape: {X.shape}")
    
    n_epochs = 500
    
    # umap-learn
    print("\nRunning umap-learn...")
    t0 = time.time()
    embedding_orig = UMAP_learn(n_components=2, n_neighbors=15, min_dist=0.1, 
                                 random_state=42, n_epochs=n_epochs).fit_transform(X)
    time_orig = time.time() - t0
    print(f"umap-learn: {time_orig:.2f}s")
    
    # UMAP-MLX
    print("Running UMAP-MLX...")
    t0 = time.time()
    embedding_mlx = UMAP_mlx(n_components=2, n_neighbors=15, min_dist=0.1,
                              random_state=42, n_epochs=n_epochs).fit_transform(X)
    time_mlx = time.time() - t0
    print(f"UMAP-MLX: {time_mlx:.2f}s")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, emb, label, t in [
        (axes[0], embedding_orig, 'umap-learn', time_orig),
        (axes[1], embedding_mlx, 'UMAP-MLX', time_mlx),
    ]:
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
        ax.set_title(f'{label} ({t:.1f}s)', fontsize=14)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_aspect('equal')
    
    plt.colorbar(scatter, ax=axes[1], label='Digit Class')
    plt.suptitle(f'MNIST Digits ({n_samples} samples, {n_epochs} epochs)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison.png")
    
    # Topology check
    total_orig = np.var(embedding_orig, axis=0).mean()
    total_mlx = np.var(embedding_mlx, axis=0).mean()
    cls_orig = np.mean([np.var(embedding_orig[y==d], axis=0).mean() for d in range(10) if (y==d).sum()>1])
    cls_mlx = np.mean([np.var(embedding_mlx[y==d], axis=0).mean() for d in range(10) if (y==d).sum()>1])
    
    print(f"\nClustering quality (intra/total variance, lower=better):")
    print(f"  umap-learn: {cls_orig/total_orig:.3f}")
    print(f"  UMAP-MLX:   {cls_mlx/total_mlx:.3f}")


def benchmark():
    """Benchmark on different data sizes."""
    print("\n" + "="*60)
    print("BENCHMARK")
    print("="*60)
    
    n_epochs = 200
    results = []
    for n in [1000, 10000, 50000]:
        print(f"\n--- {n} samples, 64 dims, {n_epochs} epochs ---")
        np.random.seed(42)
        X = StandardScaler().fit_transform(np.random.randn(n, 64).astype(np.float32))
        
        t0 = time.time()
        UMAP_learn(n_components=2, n_neighbors=15, min_dist=0.1, 
                   random_state=42, n_epochs=n_epochs, verbose=False).fit_transform(X)
        t_orig = time.time() - t0
        
        t0 = time.time()
        UMAP_mlx(n_components=2, n_neighbors=15, min_dist=0.1,
                 random_state=42, n_epochs=n_epochs).fit_transform(X)
        t_mlx = time.time() - t0
        
        speedup = t_orig / t_mlx
        results.append((n, t_orig, t_mlx, speedup))
        print(f"  umap-learn: {t_orig:.2f}s | UMAP-MLX: {t_mlx:.2f}s | {speedup:.2f}x")
    
    print(f"\n{'Samples':<10} {'umap-learn':<12} {'UMAP-MLX':<12} {'Speedup':<10}")
    print("-" * 44)
    for n, t1, t2, s in results:
        print(f"{n:<10} {t1:<12.2f} {t2:<12.2f} {s:<10.2f}x")
    
    return results


if __name__ == '__main__':
    compare_umaps()
    benchmark()
