"""Fashion-MNIST 70K: UMAP-MLX vs umap-learn side-by-side animation.

Both run 500 epochs with per-epoch snapshots.
umap-learn snapshots collected by running at [1,2,...,500] epochs.
Results cached to avoid re-running.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import fetch_openml

LABELS = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
N_EPOCHS = 500
SEED = 42
FPS = 30
CACHE = '/Users/hanxiao/.openclaw/workspace/umap-mlx/ref_snapshots.npz'

print("Loading Fashion-MNIST 70K...")
fm = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
X = fm.data.astype(np.float32) / 255.0
y = fm.target.astype(np.int32)
n = X.shape[0]

np.random.seed(42)
viz_idx = np.random.choice(n, 15000, replace=False)
y_viz = y[viz_idx]
cmap = plt.cm.tab10
colors = cmap(y_viz / 9.0)

# ---- MLX with per-epoch snapshots ----
from umap_mlx import UMAP as UMAP_mlx

mlx_snaps = []
def mlx_cb(epoch, Y_np):
    mlx_snaps.append(Y_np[viz_idx])  # only store viz subset to save memory

print("Running UMAP-MLX (500 epochs)...")
t0 = time.time()
UMAP_mlx(n_components=2, n_neighbors=15, min_dist=0.1,
         random_state=SEED, n_epochs=N_EPOCHS, verbose=True).fit_transform(X, epoch_callback=mlx_cb)
t_mlx = time.time() - t0
print(f"MLX: {t_mlx:.2f}s, {len(mlx_snaps)} snapshots")

# ---- umap-learn: collect snapshots via incremental epochs ----
from umap import UMAP as UMAP_learn

if os.path.exists(CACHE):
    print(f"Loading cached umap-learn snapshots from {CACHE}")
    data = np.load(CACHE)
    ref_snaps = list(data['snapshots'])
    t_ref = float(data['time'])
    print(f"umap-learn: {t_ref:.1f}s (cached), {len(ref_snaps)} snapshots")
else:
    print("Running umap-learn with per-epoch snapshots (this takes a while)...")
    # Use n_epochs as list to get intermediate embeddings
    # umap-learn supports n_epochs=list for epoch checkpoints
    epoch_list = list(range(0, N_EPOCHS + 1))

    t0 = time.time()
    model = UMAP_learn(n_components=2, n_neighbors=15, min_dist=0.1,
                        random_state=SEED, n_epochs=epoch_list)
    result = model.fit_transform(X)
    t_ref = time.time() - t0

    # result should be a list of embeddings when n_epochs is a list
    if isinstance(result, list):
        ref_snaps = [r[viz_idx] for r in result]
    else:
        # Fallback: only got final result, run individually at key epochs
        print("List mode didn't work, running at sampled epochs...")
        sample_epochs = list(range(0, N_EPOCHS + 1, 5))  # every 5 epochs
        if N_EPOCHS not in sample_epochs:
            sample_epochs.append(N_EPOCHS)
        ref_snaps = []
        t0 = time.time()
        for ep in sample_epochs:
            emb = UMAP_learn(n_components=2, n_neighbors=15, min_dist=0.1,
                              random_state=SEED, n_epochs=max(ep, 1)).fit_transform(X)
            ref_snaps.append(emb[viz_idx])
            if len(ref_snaps) % 20 == 0:
                print(f"  {len(ref_snaps)}/{len(sample_epochs)} snapshots...")
        t_ref_total = time.time() - t0
        # Use single-run time for fair comparison
        t0b = time.time()
        UMAP_learn(n_components=2, n_neighbors=15, min_dist=0.1,
                    random_state=SEED, n_epochs=N_EPOCHS).fit_transform(X)
        t_ref = time.time() - t0b

    print(f"umap-learn: {t_ref:.2f}s, {len(ref_snaps)} snapshots")
    np.savez_compressed(CACHE, snapshots=np.array(ref_snaps), time=t_ref)
    print(f"Cached to {CACHE}")

# ---- Align snapshot counts ----
# MLX has 501 (init + 500 epochs), ref may differ
n_mlx = len(mlx_snaps)
n_ref = len(ref_snaps)
n_snap = min(n_mlx, n_ref)

# If ref has fewer, interpolate (nearest neighbor)
if n_ref < n_mlx:
    ref_indices = np.linspace(0, n_ref - 1, n_mlx).astype(int)
    ref_snaps_aligned = [ref_snaps[i] for i in ref_indices]
else:
    ref_snaps_aligned = ref_snaps[:n_snap]
    mlx_snaps = mlx_snaps[:n_snap]

n_snap = min(len(mlx_snaps), len(ref_snaps_aligned))

# ---- Build animation ----
print("Rendering...")

def get_lims(emb, margin=0.1):
    xmin, xmax = emb[:, 0].min(), emb[:, 0].max()
    ymin, ymax = emb[:, 1].min(), emb[:, 1].max()
    xp, yp = (xmax - xmin) * margin, (ymax - ymin) * margin
    return (xmin - xp, xmax + xp), (ymin - yp, ymax + yp)

mlx_xlim, mlx_ylim = get_lims(mlx_snaps[-1])
ref_xlim, ref_ylim = get_lims(ref_snaps_aligned[-1])

fig, (ax_mlx, ax_ref) = plt.subplots(1, 2, figsize=(20, 9))
fig.set_facecolor('black')

for ax, xlim, ylim in [(ax_mlx, mlx_xlim, mlx_ylim), (ax_ref, ref_xlim, ref_ylim)]:
    ax.set_facecolor('black')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')
    for i, label in enumerate(LABELS):
        ax.scatter([], [], c=[cmap(i / 9.0)], s=20, label=label)
    ax.legend(loc='upper left', fontsize=7, framealpha=0.3,
              labelcolor='white', facecolor='black', edgecolor='gray')

scatter_mlx = ax_mlx.scatter([], [], s=1.5, alpha=0.6)
scatter_ref = ax_ref.scatter([], [], s=1.5, alpha=0.6)
title_mlx = ax_mlx.set_title('', color='white', fontsize=14, pad=8)
title_ref = ax_ref.set_title('', color='white', fontsize=14, pad=8)

init_f = 30
hold_f = 90
total_f = init_f + n_snap + hold_f
mlx_tpe = t_mlx / N_EPOCHS
ref_tpe = t_ref / N_EPOCHS

print(f"Frames: init={init_f}, epochs={n_snap}, hold={hold_f}, total={total_f}")

def update(frame):
    if frame < init_f:
        idx = 0
        lm = "UMAP-MLX / Init"
        lr = "umap-learn / Init"
    elif frame < init_f + n_snap:
        idx = frame - init_f
        ep = int(idx * N_EPOCHS / (n_snap - 1)) if n_snap > 1 else N_EPOCHS
        lm = f"UMAP-MLX / Epoch {ep}/{N_EPOCHS} / {ep*mlx_tpe:.1f}s"
        lr = f"umap-learn / Epoch {ep}/{N_EPOCHS} / {ep*ref_tpe:.1f}s"
    else:
        idx = n_snap - 1
        lm = f"UMAP-MLX / {t_mlx:.1f}s ({t_ref/t_mlx:.0f}x faster)"
        lr = f"umap-learn / {t_ref:.1f}s"

    scatter_mlx.set_offsets(mlx_snaps[idx])
    scatter_mlx.set_color(colors)
    scatter_ref.set_offsets(ref_snaps_aligned[idx])
    scatter_ref.set_color(colors)
    title_mlx.set_text(lm)
    title_ref.set_text(lr)
    return scatter_mlx, scatter_ref, title_mlx, title_ref

anim = animation.FuncAnimation(fig, update, frames=total_f, blit=True, interval=1000 // FPS)
outpath = '/Users/hanxiao/.openclaw/workspace/umap-mlx/comparison_video.mp4'
anim.save(outpath, writer=animation.FFMpegWriter(fps=FPS, bitrate=4000))
plt.close()
print(f"Saved {outpath}")
print(f"\nMLX: {t_mlx:.1f}s | umap-learn: {t_ref:.1f}s | speedup: {t_ref/t_mlx:.1f}x")
