"""
=============================================================================
10 - DBSCAN: Complete Implementation
=============================================================================
Covers:
  1. From-scratch DBSCAN (eps-neighborhood, core/border/noise, expansion)
  2. Scikit-learn DBSCAN comparison
  3. k-Distance graph for eps selection
  4. eps parameter impact — same data, different eps
  5. min_samples parameter impact
  6. Non-convex data — moons, circles, complex shapes
  7. Noise detection visualization
  8. DBSCAN vs k-Means side-by-side
  9. Varying density problem — DBSCAN's limitation
  10. Real data clustering — Iris

Datasets:
  - Synthetic: make_blobs, make_moons, make_circles, custom shapes
  - Real:      Iris (150 samples, 4 features, 3 classes)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.datasets import (
    make_blobs, make_moons, make_circles, load_iris
)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score
)

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# SECTION 1: From-Scratch DBSCAN
# =============================================================================
class DBSCANScratch:
    """
    DBSCAN — from scratch.

    Parameters
    ----------
    eps : float — Maximum distance for two points to be neighbors.
    min_samples : int — Minimum points in eps-neighborhood for a core point.
    metric : str — 'euclidean' or 'manhattan'.
    """

    def __init__(self, eps=0.5, min_samples=5, metric="euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
        self.n_clusters_ = 0
        self.n_noise_ = 0

    def _compute_distance(self, a, b):
        if self.metric == "manhattan":
            return np.sum(np.abs(a - b))
        return np.sqrt(np.sum((a - b) ** 2))

    def _get_neighbors(self, X, point_idx):
        """Return indices of all points within eps of point_idx."""
        neighbors = []
        for i in range(len(X)):
            if self._compute_distance(X[point_idx], X[i]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        n_samples = X.shape[0]

        # -1 = unvisited/unassigned, -2 = noise (temporarily)
        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)

        # Precompute all neighborhoods for efficiency
        neighborhoods = [self._get_neighbors(X, i) for i in range(n_samples)]

        # Identify core points
        core_mask = np.array([len(n) >= self.min_samples for n in neighborhoods])
        self.core_sample_indices_ = np.where(core_mask)[0]

        cluster_id = -1

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = neighborhoods[i]

            if len(neighbors) < self.min_samples:
                # Mark as noise (may be reclaimed as border point later)
                labels[i] = -1
                continue

            # New cluster
            cluster_id += 1
            labels[i] = cluster_id

            # Expand cluster — use a queue (seed set)
            seed_set = list(neighbors)
            j = 0
            while j < len(seed_set):
                q = seed_set[j]
                j += 1

                if not visited[q]:
                    visited[q] = True
                    q_neighbors = neighborhoods[q]

                    if len(q_neighbors) >= self.min_samples:
                        # q is a core point — add its neighbors to seed set
                        for nb in q_neighbors:
                            if nb not in seed_set:
                                seed_set.append(nb)

                if labels[q] == -1:
                    # Was noise or unassigned — assign to this cluster
                    labels[q] = cluster_id

        self.labels_ = labels
        self.n_clusters_ = cluster_id + 1
        self.n_noise_ = np.sum(labels == -1)
        return self


# =============================================================================
# SECTION 2: Visualization Helpers
# =============================================================================
def plot_dbscan_result(X, labels, core_indices, title, filename):
    """Plot DBSCAN clusters with core/border/noise distinction."""
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    colors = sns.color_palette("husl", max(n_clusters, 1))

    core_mask = np.zeros(len(labels), dtype=bool)
    if core_indices is not None and len(core_indices) > 0:
        core_mask[core_indices] = True

    for k in sorted(unique_labels):
        if k == -1:
            # Noise
            mask = labels == k
            ax.scatter(X[mask, 0], X[mask, 1], c="gray", marker="x",
                       s=30, alpha=0.5, label=f"Noise ({np.sum(mask)})")
        else:
            mask = labels == k
            # Core points
            core_in_cluster = mask & core_mask
            border_in_cluster = mask & ~core_mask

            ax.scatter(X[core_in_cluster, 0], X[core_in_cluster, 1],
                       c=[colors[k % len(colors)]], s=30, alpha=0.7, edgecolors="black",
                       linewidths=0.5, label=f"Cluster {k} (core)")
            if np.any(border_in_cluster):
                ax.scatter(X[border_in_cluster, 0], X[border_in_cluster, 1],
                           c=[colors[k % len(colors)]], s=30, alpha=0.4,
                           marker="s", edgecolors="black", linewidths=0.3)

    n_noise = np.sum(labels == -1)
    ax.set_title(f"{title}\n{n_clusters} clusters, {n_noise} noise points")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_k_distance(X, k, filename):
    """k-distance plot for eps selection."""
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_dist = distances[:, -1]  # distance to k-th neighbor
    k_dist_sorted = np.sort(k_dist)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_dist_sorted, color="steelblue", linewidth=2)
    ax.set_xlabel("Points (sorted by k-distance)")
    ax.set_ylabel(f"{k}-Distance")
    ax.set_title(f"k-Distance Graph (k={k}) — Look for the Elbow")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")
    return k_dist_sorted


def plot_eps_impact(X, eps_values, min_samples, filename):
    """Show how different eps values affect clustering."""
    n_eps = len(eps_values)
    fig, axes = plt.subplots(1, n_eps, figsize=(5 * n_eps, 4.5))
    if n_eps == 1:
        axes = [axes]

    for ax, eps in zip(axes, eps_values):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        db.fit(X)
        labels = db.labels_
        n_clusters = len(set(labels) - {-1})
        n_noise = np.sum(labels == -1)

        unique_labels = set(labels)
        colors = sns.color_palette("husl", max(n_clusters, 1))

        for k in sorted(unique_labels):
            mask = labels == k
            if k == -1:
                ax.scatter(X[mask, 0], X[mask, 1], c="gray", marker="x",
                           s=20, alpha=0.5)
            else:
                ax.scatter(X[mask, 0], X[mask, 1],
                           c=[colors[k % len(colors)]], s=15, alpha=0.6)
        ax.set_title(f"ε={eps}\n{n_clusters} clusters, {n_noise} noise")

    plt.suptitle(f"eps Impact (min_samples={min_samples})", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_min_samples_impact(X, eps, min_samples_values, filename):
    """Show how different min_samples values affect clustering."""
    n_ms = len(min_samples_values)
    fig, axes = plt.subplots(1, n_ms, figsize=(5 * n_ms, 4.5))
    if n_ms == 1:
        axes = [axes]

    for ax, ms in zip(axes, min_samples_values):
        db = DBSCAN(eps=eps, min_samples=ms)
        db.fit(X)
        labels = db.labels_
        n_clusters = len(set(labels) - {-1})
        n_noise = np.sum(labels == -1)

        unique_labels = set(labels)
        colors = sns.color_palette("husl", max(n_clusters, 1))

        for k in sorted(unique_labels):
            mask = labels == k
            if k == -1:
                ax.scatter(X[mask, 0], X[mask, 1], c="gray", marker="x",
                           s=20, alpha=0.5)
            else:
                ax.scatter(X[mask, 0], X[mask, 1],
                           c=[colors[k % len(colors)]], s=15, alpha=0.6)
        ax.set_title(f"min_samples={ms}\n{n_clusters} clusters, {n_noise} noise")

    plt.suptitle(f"min_samples Impact (ε={eps})", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_dbscan_vs_kmeans(X, y_true, eps, min_samples, k, title_prefix, filename):
    """Side-by-side: DBSCAN vs k-Means on same data."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True labels
    for cls in np.unique(y_true):
        mask = y_true == cls
        axes[0].scatter(X[mask, 0], X[mask, 1], s=20, alpha=0.6)
    axes[0].set_title(f"{title_prefix} — True Labels")

    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    labels_db = db.labels_
    n_clusters_db = len(set(labels_db) - {-1})
    n_noise_db = np.sum(labels_db == -1)
    ari_db = adjusted_rand_score(y_true, labels_db)

    unique_db = set(labels_db)
    colors_db = sns.color_palette("husl", max(n_clusters_db, 1))
    for k_label in sorted(unique_db):
        mask = labels_db == k_label
        if k_label == -1:
            axes[1].scatter(X[mask, 0], X[mask, 1], c="gray", marker="x",
                           s=20, alpha=0.5)
        else:
            axes[1].scatter(X[mask, 0], X[mask, 1],
                           c=[colors_db[k_label % len(colors_db)]], s=20, alpha=0.6)
    axes[1].set_title(f"DBSCAN (ARI={ari_db:.3f})\n{n_clusters_db} clusters, {n_noise_db} noise")

    # k-Means
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    labels_km = km.labels_
    ari_km = adjusted_rand_score(y_true, labels_km)

    colors_km = sns.color_palette("husl", k)
    for j in range(k):
        mask = labels_km == j
        axes[2].scatter(X[mask, 0], X[mask, 1], c=[colors_km[j]], s=20, alpha=0.6)
    axes[2].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                    c="black", marker="X", s=120, edgecolors="white", linewidths=2)
    axes[2].set_title(f"k-Means k={k} (ARI={ari_km:.3f})")

    plt.suptitle(f"DBSCAN vs k-Means — {title_prefix}", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


# =============================================================================
# SECTION 3: Main Execution
# =============================================================================
def main():
    print("=" * 70)
    print("10 — DBSCAN (Density-Based Spatial Clustering)")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    # PART A: BASIC DBSCAN ON BLOBS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART A: BASIC DBSCAN — Synthetic Blobs")
    print("=" * 70)

    X_blobs, y_blobs = make_blobs(n_samples=500, centers=4, cluster_std=0.8,
                                   random_state=42)
    scaler = StandardScaler()
    X_blobs_s = scaler.fit_transform(X_blobs)

    print(f"  Shape: {X_blobs_s.shape}, True clusters: 4")

    # ── k-Distance Graph for eps selection ───────────────────────────────────
    print("\n  --- k-Distance Graph ---")
    k_dist_sorted = plot_k_distance(X_blobs_s, k=5, filename="01_k_distance_blobs.png")

    # From the k-distance plot, choose eps around the elbow
    eps_chosen = 0.5
    print(f"  Chosen eps: {eps_chosen}")

    # ── From-Scratch DBSCAN ──────────────────────────────────────────────────
    print("\n  --- From-Scratch DBSCAN ---")
    scratch_db = DBSCANScratch(eps=eps_chosen, min_samples=5, metric="euclidean")
    scratch_db.fit(X_blobs_s)

    print(f"  Clusters found: {scratch_db.n_clusters_}")
    print(f"  Noise points:   {scratch_db.n_noise_}")
    print(f"  Core points:    {len(scratch_db.core_sample_indices_)}")

    ari_scratch = adjusted_rand_score(y_blobs, scratch_db.labels_)
    print(f"  ARI: {ari_scratch:.4f}")

    plot_dbscan_result(X_blobs_s, scratch_db.labels_,
                       scratch_db.core_sample_indices_,
                       f"Scratch DBSCAN (ε={eps_chosen}, MinPts=5)",
                       "02_scratch_dbscan_blobs.png")

    # ── Sklearn DBSCAN ───────────────────────────────────────────────────────
    print("\n  --- Sklearn DBSCAN ---")
    sk_db = DBSCAN(eps=eps_chosen, min_samples=5)
    sk_db.fit(X_blobs_s)

    n_clusters_sk = len(set(sk_db.labels_) - {-1})
    n_noise_sk = np.sum(sk_db.labels_ == -1)
    ari_sk = adjusted_rand_score(y_blobs, sk_db.labels_)

    print(f"  Clusters found: {n_clusters_sk}")
    print(f"  Noise points:   {n_noise_sk}")
    print(f"  Core points:    {len(sk_db.core_sample_indices_)}")
    print(f"  ARI: {ari_sk:.4f}")

    plot_dbscan_result(X_blobs_s, sk_db.labels_, sk_db.core_sample_indices_,
                       f"Sklearn DBSCAN (ε={eps_chosen}, MinPts=5)",
                       "03_sklearn_dbscan_blobs.png")

    # ── Scratch vs Sklearn ───────────────────────────────────────────────────
    print(f"\n  Scratch clusters: {scratch_db.n_clusters_}, "
          f"Sklearn clusters: {n_clusters_sk}")
    print(f"  Labels match: {np.array_equal(scratch_db.labels_, sk_db.labels_)}")

    # ══════════════════════════════════════════════════════════════════════════
    # PARAMETER IMPACT — eps
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EPS PARAMETER IMPACT")
    print("=" * 70)

    eps_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    for eps in eps_values:
        db_e = DBSCAN(eps=eps, min_samples=5)
        db_e.fit(X_blobs_s)
        nc = len(set(db_e.labels_) - {-1})
        nn = np.sum(db_e.labels_ == -1)
        print(f"  eps={eps:.1f}: {nc} clusters, {nn} noise")

    plot_eps_impact(X_blobs_s, eps_values, min_samples=5, filename="04_eps_impact.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PARAMETER IMPACT — min_samples
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MIN_SAMPLES PARAMETER IMPACT")
    print("=" * 70)

    ms_values = [2, 3, 5, 10, 20]
    for ms in ms_values:
        db_ms = DBSCAN(eps=eps_chosen, min_samples=ms)
        db_ms.fit(X_blobs_s)
        nc = len(set(db_ms.labels_) - {-1})
        nn = np.sum(db_ms.labels_ == -1)
        print(f"  min_samples={ms:2d}: {nc} clusters, {nn} noise")

    plot_min_samples_impact(X_blobs_s, eps=eps_chosen, min_samples_values=ms_values,
                            filename="05_min_samples_impact.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PART B: NON-CONVEX DATA — Where DBSCAN Shines
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART B: NON-CONVEX DATA — DBSCAN vs k-Means")
    print("=" * 70)

    # ── Moons ────────────────────────────────────────────────────────────────
    print("\n  --- Moons ---")
    X_moon, y_moon = make_moons(n_samples=500, noise=0.1, random_state=42)
    X_moon_s = StandardScaler().fit_transform(X_moon)
    plot_dbscan_vs_kmeans(X_moon_s, y_moon, 0.3, 5, 2,
                          "Moons", "06_dbscan_vs_kmeans_moons.png")

    # ── Circles ──────────────────────────────────────────────────────────────
    print("\n  --- Circles ---")
    X_circ, y_circ = make_circles(n_samples=500, noise=0.05, factor=0.4,
                                   random_state=42)
    X_circ_s = StandardScaler().fit_transform(X_circ)
    plot_dbscan_vs_kmeans(X_circ_s, y_circ, 0.3, 5, 2,
                          "Circles", "07_dbscan_vs_kmeans_circles.png")

    # ── Complex shape: interlocking spirals ──────────────────────────────────
    print("\n  --- Spirals ---")
    n_pts = 300
    theta = np.linspace(0, 4 * np.pi, n_pts)
    r = theta
    X_spiral1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    X_spiral2 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
    X_spiral = np.vstack([X_spiral1, X_spiral2])
    X_spiral += np.random.randn(*X_spiral.shape) * 0.5
    y_spiral = np.array([0] * n_pts + [1] * n_pts)

    X_spiral_s = StandardScaler().fit_transform(X_spiral)
    plot_dbscan_vs_kmeans(X_spiral_s, y_spiral, 0.25, 5, 2,
                          "Spirals", "08_dbscan_vs_kmeans_spirals.png")

    # ══════════════════════════════════════════════════════════════════════════
    # NOISE DETECTION DEMO
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("NOISE / OUTLIER DETECTION")
    print("=" * 70)

    # Blobs with added noise
    X_noisy, y_noisy = make_blobs(n_samples=400, centers=3, cluster_std=0.6,
                                   random_state=42)
    # Add 50 random outliers
    n_outliers = 50
    outliers = np.random.uniform(
        low=X_noisy.min(axis=0) - 3,
        high=X_noisy.max(axis=0) + 3,
        size=(n_outliers, 2)
    )
    X_with_noise = np.vstack([X_noisy, outliers])
    y_with_noise = np.concatenate([y_noisy, np.full(n_outliers, -1)])

    X_wn_s = StandardScaler().fit_transform(X_with_noise)

    db_noise = DBSCAN(eps=0.4, min_samples=5)
    db_noise.fit(X_wn_s)

    detected_noise = np.sum(db_noise.labels_ == -1)
    n_clusters_noise = len(set(db_noise.labels_) - {-1})

    print(f"  True outliers added: {n_outliers}")
    print(f"  DBSCAN detected noise: {detected_noise}")
    print(f"  Clusters found: {n_clusters_noise}")

    # Check overlap: how many of the true outliers were labeled as noise?
    true_outlier_mask = np.zeros(len(X_with_noise), dtype=bool)
    true_outlier_mask[-n_outliers:] = True
    detected_as_noise = db_noise.labels_ == -1
    correctly_detected = np.sum(true_outlier_mask & detected_as_noise)
    print(f"  Correctly detected outliers: {correctly_detected}/{n_outliers}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ground truth
    for cls in [0, 1, 2]:
        mask = y_with_noise == cls
        axes[0].scatter(X_wn_s[mask, 0], X_wn_s[mask, 1], s=20, alpha=0.6)
    axes[0].scatter(X_wn_s[true_outlier_mask, 0], X_wn_s[true_outlier_mask, 1],
                    c="red", marker="x", s=40, label=f"True outliers ({n_outliers})")
    axes[0].set_title("Ground Truth (with injected outliers)")
    axes[0].legend(fontsize=8)

    # DBSCAN result
    unique_db = set(db_noise.labels_)
    colors_db = sns.color_palette("husl", max(n_clusters_noise, 1))
    for k in sorted(unique_db):
        mask = db_noise.labels_ == k
        if k == -1:
            axes[1].scatter(X_wn_s[mask, 0], X_wn_s[mask, 1], c="red",
                           marker="x", s=30, alpha=0.6, label=f"Noise ({np.sum(mask)})")
        else:
            axes[1].scatter(X_wn_s[mask, 0], X_wn_s[mask, 1],
                           c=[colors_db[k % len(colors_db)]], s=20, alpha=0.6)
    axes[1].set_title(f"DBSCAN — {n_clusters_noise} clusters, {detected_noise} noise")
    axes[1].legend(fontsize=8)

    plt.suptitle("DBSCAN Noise / Outlier Detection", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_noise_detection.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/09_noise_detection.png")

    # ══════════════════════════════════════════════════════════════════════════
    # VARYING DENSITY PROBLEM
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VARYING DENSITY PROBLEM — DBSCAN's Limitation")
    print("=" * 70)

    # Create clusters with different densities
    X_dense, _ = make_blobs(n_samples=300, centers=[[0, 0]], cluster_std=0.3,
                             random_state=42)
    X_sparse, _ = make_blobs(n_samples=300, centers=[[5, 5]], cluster_std=1.5,
                              random_state=42)
    X_vary = np.vstack([X_dense, X_sparse])
    y_vary = np.array([0] * 300 + [1] * 300)

    X_vary_s = StandardScaler().fit_transform(X_vary)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # True labels
    for cls in [0, 1]:
        mask = y_vary == cls
        axes[0].scatter(X_vary_s[mask, 0], X_vary_s[mask, 1], s=15, alpha=0.6)
    axes[0].set_title("True (Dense + Sparse)")

    eps_trials = [0.2, 0.4, 0.8]
    for ax, eps in zip(axes[1:], eps_trials):
        db_v = DBSCAN(eps=eps, min_samples=5)
        db_v.fit(X_vary_s)
        labels_v = db_v.labels_
        nc = len(set(labels_v) - {-1})
        nn = np.sum(labels_v == -1)
        ari_v = adjusted_rand_score(y_vary, labels_v)

        unique_v = set(labels_v)
        colors_v = sns.color_palette("husl", max(nc, 1))
        for k in sorted(unique_v):
            mask = labels_v == k
            if k == -1:
                ax.scatter(X_vary_s[mask, 0], X_vary_s[mask, 1], c="gray",
                           marker="x", s=15, alpha=0.5)
            else:
                ax.scatter(X_vary_s[mask, 0], X_vary_s[mask, 1],
                           c=[colors_v[k % len(colors_v)]], s=15, alpha=0.6)
        ax.set_title(f"ε={eps} ({nc} cl, {nn} noise)\nARI={ari_v:.3f}")

    plt.suptitle("Varying Density Problem — No single ε works for both clusters", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_varying_density.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/10_varying_density.png")

    # ══════════════════════════════════════════════════════════════════════════
    # REAL DATA — Iris Clustering
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("REAL DATA — Iris Clustering with DBSCAN")
    print("=" * 70)

    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    iris_names = list(iris.target_names)

    scaler_iris = StandardScaler()
    X_iris_s = scaler_iris.fit_transform(X_iris)

    # k-distance plot for Iris
    print("\n  --- k-Distance Graph for Iris ---")
    plot_k_distance(X_iris_s, k=5, filename="11_k_distance_iris.png")

    # Try DBSCAN with a reasonable eps
    db_iris = DBSCAN(eps=0.9, min_samples=5)
    db_iris.fit(X_iris_s)

    n_clusters_iris = len(set(db_iris.labels_) - {-1})
    n_noise_iris = np.sum(db_iris.labels_ == -1)
    ari_iris = adjusted_rand_score(y_iris, db_iris.labels_)
    nmi_iris = normalized_mutual_info_score(y_iris, db_iris.labels_)

    print(f"\n  Clusters found: {n_clusters_iris}")
    print(f"  Noise points:   {n_noise_iris}")
    print(f"  ARI: {ari_iris:.4f}")
    print(f"  NMI: {nmi_iris:.4f}")

    # Compare with k-Means
    km_iris = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_iris.fit(X_iris_s)
    ari_km_iris = adjusted_rand_score(y_iris, km_iris.labels_)

    print(f"\n  k-Means ARI: {ari_km_iris:.4f}")
    print(f"  DBSCAN  ARI: {ari_iris:.4f}")

    # Petal features plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_iris = sns.color_palette("husl", 3)

    for c in range(3):
        mask = y_iris == c
        axes[0].scatter(X_iris[mask, 2], X_iris[mask, 3], c=[colors_iris[c]],
                        label=iris_names[c], s=30, alpha=0.7, edgecolors="gray")
    axes[0].set_title("True Labels")
    axes[0].set_xlabel("Petal Length")
    axes[0].set_ylabel("Petal Width")
    axes[0].legend(fontsize=8)

    # DBSCAN
    unique_iris = set(db_iris.labels_)
    colors_db_iris = sns.color_palette("husl", max(n_clusters_iris, 1))
    for k in sorted(unique_iris):
        mask = db_iris.labels_ == k
        if k == -1:
            axes[1].scatter(X_iris[mask, 2], X_iris[mask, 3], c="gray",
                           marker="x", s=30, alpha=0.6, label="Noise")
        else:
            axes[1].scatter(X_iris[mask, 2], X_iris[mask, 3],
                           c=[colors_db_iris[k % len(colors_db_iris)]],
                           s=30, alpha=0.7, label=f"Cluster {k}")
    axes[1].set_title(f"DBSCAN (ARI={ari_iris:.3f})")
    axes[1].set_xlabel("Petal Length")
    axes[1].legend(fontsize=8)

    # k-Means
    for j in range(3):
        mask = km_iris.labels_ == j
        axes[2].scatter(X_iris[mask, 2], X_iris[mask, 3], c=[colors_iris[j]],
                        s=30, alpha=0.7, label=f"Cluster {j}")
    axes[2].set_title(f"k-Means (ARI={ari_km_iris:.3f})")
    axes[2].set_xlabel("Petal Length")
    axes[2].legend(fontsize=8)

    plt.suptitle("Iris — True vs DBSCAN vs k-Means", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_iris_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/12_iris_comparison.png")

    # ══════════════════════════════════════════════════════════════════════════
    # CORE / BORDER / NOISE BREAKDOWN
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("POINT TYPE BREAKDOWN — Core vs Border vs Noise")
    print("=" * 70)

    # Use moons for a nice illustration
    X_demo, y_demo = make_moons(n_samples=300, noise=0.15, random_state=42)
    X_demo_s = StandardScaler().fit_transform(X_demo)

    db_demo = DBSCAN(eps=0.3, min_samples=5)
    db_demo.fit(X_demo_s)

    core_mask = np.zeros(len(X_demo_s), dtype=bool)
    core_mask[db_demo.core_sample_indices_] = True
    noise_mask = db_demo.labels_ == -1
    border_mask = ~core_mask & ~noise_mask

    print(f"  Core points:   {np.sum(core_mask)}")
    print(f"  Border points: {np.sum(border_mask)}")
    print(f"  Noise points:  {np.sum(noise_mask)}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(X_demo_s[core_mask, 0], X_demo_s[core_mask, 1],
               c="steelblue", s=40, alpha=0.7, edgecolors="black", linewidths=0.5,
               label=f"Core ({np.sum(core_mask)})")
    ax.scatter(X_demo_s[border_mask, 0], X_demo_s[border_mask, 1],
               c="gold", s=60, alpha=0.8, marker="s", edgecolors="black", linewidths=0.5,
               label=f"Border ({np.sum(border_mask)})")
    ax.scatter(X_demo_s[noise_mask, 0], X_demo_s[noise_mask, 1],
               c="red", s=50, alpha=0.7, marker="x", linewidths=2,
               label=f"Noise ({np.sum(noise_mask)})")
    ax.set_title("DBSCAN Point Types — Core / Border / Noise")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_core_border_noise.png", dpi=150)
    plt.close()
    print("  [Saved] plots/13_core_border_noise.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1.  DENSITY-BASED CLUSTERING - Groups dense regions, ignores sparse
    2.  CORE / BORDER / NOISE    - Three point types define cluster structure
    3.  EPS PARAMETER             - Neighborhood radius controls cluster scale
    4.  MIN_SAMPLES PARAMETER    - Density threshold for core points
    5.  k-DISTANCE GRAPH         - Visual method for choosing eps
    6.  NON-CONVEX CLUSTERS      - Moons, circles, spirals - DBSCAN excels
    7.  OUTLIER DETECTION        - Built-in noise labeling (label = -1)
    8.  DBSCAN vs k-MEANS        - DBSCAN wins on complex shapes
    9.  VARYING DENSITY PROBLEM  - Single eps fails for mixed-density data
    10. DETERMINISTIC RESULTS    - Core points always assigned consistently
    11. PARAMETER SENSITIVITY    - Small eps changes -> big clustering changes
    12. REAL DATA APPLICATION     - Iris clustering with noise detection
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
