"""
=============================================================================
09 - K-MEANS CLUSTERING: Complete Implementation
=============================================================================
Covers:
  1. From-scratch k-Means (random init + k-means++ init)
  2. Scikit-learn KMeans comparison
  3. Elbow method — inertia vs k
  4. Silhouette analysis — per-cluster silhouette plots
  5. Initialization comparison — random vs k-means++
  6. Iteration-by-iteration visualization — watch clusters form
  7. Failure cases — non-convex shapes, different sizes/densities
  8. Mini-Batch k-Means — speed comparison
  9. Real data clustering — Iris (with known labels for validation)
  10. Feature scaling impact

Datasets:
  - Synthetic:  make_blobs, make_moons, make_circles
  - Real:       Iris (150 samples, 4 features, 3 classes)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

from sklearn.datasets import (
    make_blobs, make_moons, make_circles, load_iris
)
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    adjusted_rand_score, normalized_mutual_info_score
)

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# SECTION 1: From-Scratch k-Means
# =============================================================================
class KMeansScratch:
    """
    k-Means Clustering — from scratch.

    Supports:
        - Random initialization
        - k-Means++ initialization
        - Multiple restarts (n_init)
        - Inertia (WCSS) tracking

    Parameters
    ----------
    k : int — Number of clusters
    max_iter : int — Maximum iterations per run
    init : str — 'random' or 'kmeans++'
    n_init : int — Number of random restarts (best result kept)
    tol : float — Convergence tolerance for centroid movement
    """

    def __init__(self, k=3, max_iter=300, init="kmeans++", n_init=10, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.centroid_history_ = []  # For animation

    # ── Initialization ───────────────────────────────────────────────────────
    def _init_random(self, X):
        """Pick k random data points as centroids."""
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[indices].copy()

    def _init_kmeans_plus_plus(self, X):
        """k-Means++ initialization: spread centroids apart."""
        n_samples = X.shape[0]
        centroids = np.empty((self.k, X.shape[1]))

        # First centroid: random
        first_idx = np.random.randint(0, n_samples)
        centroids[0] = X[first_idx]

        for j in range(1, self.k):
            # Distance from each point to its nearest existing centroid
            dists = np.min(
                np.array([np.sum((X - centroids[c]) ** 2, axis=1) for c in range(j)]),
                axis=0
            )
            # Probability proportional to distance squared
            probs = dists / dists.sum()
            next_idx = np.random.choice(n_samples, p=probs)
            centroids[j] = X[next_idx]

        return centroids

    # ── Core Algorithm ───────────────────────────────────────────────────────
    def _assign_clusters(self, X, centroids):
        """Assign each point to the nearest centroid."""
        distances = np.array([
            np.sum((X - c) ** 2, axis=1) for c in centroids
        ])  # shape: (k, n_samples)
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        """Recompute centroids as cluster means."""
        new_centroids = np.zeros((self.k, X.shape[1]))
        for j in range(self.k):
            members = X[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                # Empty cluster — reinitialize randomly
                new_centroids[j] = X[np.random.randint(0, X.shape[0])]
        return new_centroids

    def _compute_inertia(self, X, labels, centroids):
        """WCSS: sum of squared distances from each point to its centroid."""
        inertia = 0.0
        for j in range(self.k):
            members = X[labels == j]
            if len(members) > 0:
                inertia += np.sum((members - centroids[j]) ** 2)
        return inertia

    def _single_run(self, X, record_history=False):
        """Run one instance of k-means."""
        # Initialize
        if self.init == "kmeans++":
            centroids = self._init_kmeans_plus_plus(X)
        else:
            centroids = self._init_random(X)

        history = [centroids.copy()] if record_history else []

        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels)

            if record_history:
                history.append(new_centroids.copy())

            # Check convergence
            shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids

            if shift < self.tol:
                break

        inertia = self._compute_inertia(X, labels, centroids)
        return centroids, labels, inertia, iteration + 1, history

    def fit(self, X, record_history=False):
        """Run k-means n_init times, keep best result."""
        X = np.array(X, dtype=np.float64)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_iters = 0
        best_history = []

        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter, history = self._single_run(
                X, record_history=record_history
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_iters = n_iter
                best_history = history

        self.centroids = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iters
        self.centroid_history_ = best_history
        return self

    def predict(self, X):
        """Assign new points to the nearest learned centroid."""
        X = np.array(X, dtype=np.float64)
        return self._assign_clusters(X, self.centroids)


# =============================================================================
# SECTION 2: Visualization Helpers
# =============================================================================
def plot_clusters(X, labels, centroids, title, filename, true_labels=None):
    """Plot 2D clusters with centroids."""
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", len(unique_labels))

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], s=25, alpha=0.6,
                   label=f"Cluster {lab}")

    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
                   s=200, edgecolors="white", linewidths=2, zorder=10,
                   label="Centroids")

    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_elbow(k_range, inertias, filename, silhouettes=None):
    """Elbow method plot — inertia vs k, optionally with silhouette."""
    if silhouettes is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(k_range, inertias, "o-", color="steelblue", linewidth=2.5, markersize=8)
    ax1.set_xlabel("k (Number of Clusters)")
    ax1.set_ylabel("Inertia (WCSS)")
    ax1.set_title("Elbow Method — Inertia vs k")
    ax1.set_xticks(k_range)

    if silhouettes is not None:
        ax2.plot(k_range, silhouettes, "s-", color="coral", linewidth=2.5, markersize=8)
        ax2.set_xlabel("k")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title("Silhouette Score vs k")
        ax2.set_xticks(k_range)
        best_k = k_range[np.argmax(silhouettes)]
        ax2.axvline(x=best_k, color="green", linestyle="--", alpha=0.6,
                     label=f"Best k={best_k}")
        ax2.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_silhouette_diagram(X, labels, k, filename):
    """Silhouette diagram — per-sample silhouette width, grouped by cluster."""
    sil_vals = silhouette_samples(X, labels)
    avg_sil = silhouette_score(X, labels)

    fig, ax = plt.subplots(figsize=(8, 6))

    y_lower = 10
    unique_labels = np.unique(labels)
    colors = sns.color_palette("husl", k)

    for i, lab in enumerate(unique_labels):
        cluster_sil = sil_vals[labels == lab]
        cluster_sil.sort()
        y_upper = y_lower + len(cluster_sil)

        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(cluster_sil), str(lab), fontsize=10)
        y_lower = y_upper + 10

    ax.axvline(x=avg_sil, color="red", linestyle="--", linewidth=2,
               label=f"Avg = {avg_sil:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Silhouette Diagram (k={k})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_iteration_steps(X, centroid_history, title, filename):
    """Show centroid movement over iterations."""
    n_steps = min(len(centroid_history), 6)
    indices = np.linspace(0, len(centroid_history) - 1, n_steps, dtype=int)

    fig, axes = plt.subplots(1, n_steps, figsize=(5 * n_steps, 4.5))
    if n_steps == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        centroids = centroid_history[idx]
        labels = np.argmin(
            np.array([np.sum((X - c) ** 2, axis=1) for c in centroids]),
            axis=0
        )
        colors = sns.color_palette("husl", len(centroids))

        for j in range(len(centroids)):
            mask = labels == j
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[j]], s=15, alpha=0.5)
        ax.scatter(centroids[:, 0], centroids[:, 1], c="black", marker="X",
                   s=150, edgecolors="white", linewidths=2, zorder=10)
        ax.set_title(f"Iter {idx}")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


# =============================================================================
# SECTION 3: Main Execution
# =============================================================================
def main():
    print("=" * 70)
    print("09 — K-MEANS CLUSTERING")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    # PART A: BASIC k-MEANS ON SYNTHETIC BLOBS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART A: k-MEANS ON SYNTHETIC BLOBS")
    print("=" * 70)

    X_blobs, y_blobs = make_blobs(n_samples=500, centers=4, cluster_std=1.0,
                                   random_state=42)
    print(f"  Shape: {X_blobs.shape}, True clusters: 4")

    # ── From-Scratch k-Means ─────────────────────────────────────────────────
    print("\n  --- From-Scratch k-Means ---")
    scratch_km = KMeansScratch(k=4, init="kmeans++", n_init=10, max_iter=300)
    scratch_km.fit(X_blobs, record_history=True)

    print(f"  Inertia: {scratch_km.inertia_:.4f}")
    print(f"  Iterations: {scratch_km.n_iter_}")
    print(f"  Centroids:\n{scratch_km.centroids}")

    sil_scratch = silhouette_score(X_blobs, scratch_km.labels_)
    ari_scratch = adjusted_rand_score(y_blobs, scratch_km.labels_)
    print(f"  Silhouette: {sil_scratch:.4f}")
    print(f"  Adjusted Rand Index: {ari_scratch:.4f}")

    plot_clusters(X_blobs, scratch_km.labels_, scratch_km.centroids,
                  f"Scratch k-Means (k=4, Sil={sil_scratch:.3f})",
                  "01_scratch_kmeans_blobs.png")

    # ── Iteration-by-iteration ───────────────────────────────────────────────
    print("\n  --- Iteration Steps ---")
    plot_iteration_steps(X_blobs, scratch_km.centroid_history_,
                         "k-Means Convergence — Step by Step",
                         "02_iteration_steps.png")

    # ── Sklearn k-Means ──────────────────────────────────────────────────────
    print("\n  --- Sklearn KMeans ---")
    sk_km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
    sk_km.fit(X_blobs)

    print(f"  Inertia: {sk_km.inertia_:.4f}")
    print(f"  Iterations: {sk_km.n_iter_}")

    sil_sk = silhouette_score(X_blobs, sk_km.labels_)
    ari_sk = adjusted_rand_score(y_blobs, sk_km.labels_)
    print(f"  Silhouette: {sil_sk:.4f}")
    print(f"  Adjusted Rand Index: {ari_sk:.4f}")

    plot_clusters(X_blobs, sk_km.labels_, sk_km.cluster_centers_,
                  f"Sklearn KMeans (k=4, Sil={sil_sk:.3f})",
                  "03_sklearn_kmeans_blobs.png")

    # ── Scratch vs Sklearn ───────────────────────────────────────────────────
    print(f"\n  Scratch Inertia: {scratch_km.inertia_:.4f}")
    print(f"  Sklearn Inertia: {sk_km.inertia_:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # ELBOW METHOD + SILHOUETTE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ELBOW METHOD + SILHOUETTE ANALYSIS")
    print("=" * 70)

    k_range = list(range(2, 11))
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km.fit(X_blobs)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_blobs, km.labels_))
        print(f"  k={k}: Inertia={km.inertia_:.2f}, Silhouette={silhouettes[-1]:.4f}")

    plot_elbow(k_range, inertias, "04_elbow_silhouette.png", silhouettes=silhouettes)

    # ── Silhouette Diagrams for k=2,3,4,5 ───────────────────────────────────
    print("\n  --- Silhouette Diagrams ---")
    for k in [2, 3, 4, 5]:
        km_sil = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km_sil.fit(X_blobs)
        plot_silhouette_diagram(X_blobs, km_sil.labels_, k,
                                f"05_silhouette_k{k}.png")

    # ══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION COMPARISON — Random vs k-Means++
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("INITIALIZATION COMPARISON — Random vs k-Means++")
    print("=" * 70)

    # Run 20 random inits and 20 kmeans++ inits, compare best inertia
    random_inertias = []
    kpp_inertias = []

    for trial in range(20):
        km_rand = KMeans(n_clusters=4, init="random", n_init=1,
                         random_state=trial)
        km_rand.fit(X_blobs)
        random_inertias.append(km_rand.inertia_)

        km_kpp = KMeans(n_clusters=4, init="k-means++", n_init=1,
                        random_state=trial)
        km_kpp.fit(X_blobs)
        kpp_inertias.append(km_kpp.inertia_)

    print(f"  Random Init — Mean Inertia: {np.mean(random_inertias):.2f} "
          f"± {np.std(random_inertias):.2f}")
    print(f"  k-Means++   — Mean Inertia: {np.mean(kpp_inertias):.2f} "
          f"± {np.std(kpp_inertias):.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([random_inertias, kpp_inertias],
               labels=["Random", "k-Means++"],
               patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Initialization Comparison — 20 Runs Each")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_init_comparison.png", dpi=150)
    plt.close()
    print("  [Saved] plots/06_init_comparison.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FAILURE CASES — Non-Convex Shapes
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FAILURE CASES — k-Means on Non-Convex Data")
    print("=" * 70)

    datasets = {
        "Moons": make_moons(n_samples=500, noise=0.1, random_state=42),
        "Circles": make_circles(n_samples=500, noise=0.05, factor=0.4, random_state=42),
        "Aniso Blobs": make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 2.5, 0.5],
                                   random_state=170),
        "Unequal Var": make_blobs(n_samples=[100, 400, 50], centers=[[-5, 0], [0, 5], [5, 0]],
                                   cluster_std=1.0, random_state=42),
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for col, (name, (X_d, y_d)) in enumerate(datasets.items()):
        scaler_d = StandardScaler()
        X_d_s = scaler_d.fit_transform(X_d)

        # True labels
        ax_true = axes[0, col]
        for cls in np.unique(y_d):
            mask = y_d == cls
            ax_true.scatter(X_d_s[mask, 0], X_d_s[mask, 1], s=15, alpha=0.6)
        ax_true.set_title(f"{name} — True Labels")

        # k-Means labels
        n_true = len(np.unique(y_d))
        km_fail = KMeans(n_clusters=n_true, random_state=42, n_init=10)
        km_fail.fit(X_d_s)

        ax_km = axes[1, col]
        colors = sns.color_palette("husl", n_true)
        for j in range(n_true):
            mask = km_fail.labels_ == j
            ax_km.scatter(X_d_s[mask, 0], X_d_s[mask, 1], c=[colors[j]], s=15, alpha=0.6)
        ax_km.scatter(km_fail.cluster_centers_[:, 0], km_fail.cluster_centers_[:, 1],
                      c="black", marker="X", s=120, edgecolors="white", linewidths=2)

        ari = adjusted_rand_score(y_d, km_fail.labels_)
        ax_km.set_title(f"k-Means (ARI={ari:.3f})")

    axes[0, 0].set_ylabel("True Labels", fontsize=12)
    axes[1, 0].set_ylabel("k-Means Labels", fontsize=12)

    plt.suptitle("k-Means Failure Cases — Non-Convex & Unbalanced Data", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_failure_cases.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/07_failure_cases.png")

    # ══════════════════════════════════════════════════════════════════════════
    # MINI-BATCH k-MEANS — Speed Comparison
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MINI-BATCH k-MEANS — Speed Comparison")
    print("=" * 70)

    X_large, _ = make_blobs(n_samples=50000, centers=8, random_state=42)

    # Standard KMeans
    t0 = time.time()
    km_std = KMeans(n_clusters=8, init="k-means++", n_init=5, random_state=42)
    km_std.fit(X_large)
    t_std = time.time() - t0

    # Mini-Batch KMeans
    t0 = time.time()
    km_mb = MiniBatchKMeans(n_clusters=8, init="k-means++", n_init=5,
                            batch_size=1000, random_state=42)
    km_mb.fit(X_large)
    t_mb = time.time() - t0

    sil_std = silhouette_score(X_large, km_std.labels_, sample_size=5000)
    sil_mb = silhouette_score(X_large, km_mb.labels_, sample_size=5000)

    print(f"  KMeans:      {t_std:.3f}s, Inertia={km_std.inertia_:.0f}, Sil={sil_std:.4f}")
    print(f"  MiniBatch:   {t_mb:.3f}s,  Inertia={km_mb.inertia_:.0f},  Sil={sil_mb:.4f}")
    print(f"  Speedup:     {t_std / t_mb:.1f}x")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = ["KMeans", "MiniBatch"]
    times = [t_std, t_mb]
    sils = [sil_std, sil_mb]

    bars1 = axes[0].bar(methods, times, color=["steelblue", "coral"], edgecolor="white")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Training Time (50k samples)")
    for bar, val in zip(bars1, times):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}s", ha="center", va="bottom", fontweight="bold")

    bars2 = axes[1].bar(methods, sils, color=["steelblue", "coral"], edgecolor="white")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Clustering Quality")
    for bar, val in zip(bars2, sils):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.suptitle("Standard vs Mini-Batch k-Means", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_minibatch_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/08_minibatch_comparison.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE SCALING IMPACT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FEATURE SCALING IMPACT")
    print("=" * 70)

    # Create data where one feature has a much larger range
    X_scale = np.random.randn(300, 2)
    X_scale[:, 0] *= 100  # Feature 0 in range [-300, 300]
    X_scale[:, 1] *= 1    # Feature 1 in range [-3, 3]
    true_labels_s = np.array([0] * 100 + [1] * 100 + [2] * 100)
    X_scale[:100] += [100, 2]
    X_scale[100:200] += [-100, -2]
    X_scale[200:] += [0, 0]

    # Without scaling
    km_noscale = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_noscale.fit(X_scale)
    ari_noscale = adjusted_rand_score(true_labels_s, km_noscale.labels_)

    # With scaling
    scaler_demo = StandardScaler()
    X_scale_s = scaler_demo.fit_transform(X_scale)
    km_scale = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_scale.fit(X_scale_s)
    ari_scale = adjusted_rand_score(true_labels_s, km_scale.labels_)

    print(f"  Without scaling — ARI: {ari_noscale:.4f}")
    print(f"  With scaling    — ARI: {ari_scale:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_ns = sns.color_palette("husl", 3)

    for j in range(3):
        mask = km_noscale.labels_ == j
        axes[0].scatter(X_scale[mask, 0], X_scale[mask, 1], c=[colors_ns[j]], s=15, alpha=0.6)
    axes[0].scatter(km_noscale.cluster_centers_[:, 0], km_noscale.cluster_centers_[:, 1],
                    c="black", marker="X", s=150, edgecolors="white", linewidths=2)
    axes[0].set_title(f"Without Scaling (ARI={ari_noscale:.3f})")

    for j in range(3):
        mask = km_scale.labels_ == j
        axes[1].scatter(X_scale_s[mask, 0], X_scale_s[mask, 1], c=[colors_ns[j]], s=15, alpha=0.6)
    axes[1].scatter(km_scale.cluster_centers_[:, 0], km_scale.cluster_centers_[:, 1],
                    c="black", marker="X", s=150, edgecolors="white", linewidths=2)
    axes[1].set_title(f"With Scaling (ARI={ari_scale:.3f})")

    plt.suptitle("Feature Scaling Impact on k-Means", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_scaling_impact.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/09_scaling_impact.png")

    # ══════════════════════════════════════════════════════════════════════════
    # REAL DATA — Iris Clustering
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("REAL DATA — Iris Clustering (Unsupervised vs True Labels)")
    print("=" * 70)

    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    iris_names = list(iris.target_names)

    scaler_iris = StandardScaler()
    X_iris_s = scaler_iris.fit_transform(X_iris)

    km_iris = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
    km_iris.fit(X_iris_s)

    ari_iris = adjusted_rand_score(y_iris, km_iris.labels_)
    nmi_iris = normalized_mutual_info_score(y_iris, km_iris.labels_)
    sil_iris = silhouette_score(X_iris_s, km_iris.labels_)

    print(f"  Adjusted Rand Index:         {ari_iris:.4f}")
    print(f"  Normalized Mutual Info:      {nmi_iris:.4f}")
    print(f"  Silhouette Score:            {sil_iris:.4f}")
    print(f"  Inertia:                     {km_iris.inertia_:.4f}")

    # Compare true labels vs k-means on petal features
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors_iris = sns.color_palette("husl", 3)

    for c in range(3):
        mask = y_iris == c
        axes[0].scatter(X_iris[mask, 2], X_iris[mask, 3], c=[colors_iris[c]],
                        label=iris_names[c], s=30, alpha=0.7, edgecolors="gray")
    axes[0].set_xlabel("Petal Length")
    axes[0].set_ylabel("Petal Width")
    axes[0].set_title("True Labels")
    axes[0].legend(fontsize=8)

    for c in range(3):
        mask = km_iris.labels_ == c
        axes[1].scatter(X_iris[mask, 2], X_iris[mask, 3], c=[colors_iris[c]],
                        label=f"Cluster {c}", s=30, alpha=0.7, edgecolors="gray")
    axes[1].set_xlabel("Petal Length")
    axes[1].set_ylabel("Petal Width")
    axes[1].set_title(f"k-Means (ARI={ari_iris:.3f}, Sil={sil_iris:.3f})")
    axes[1].legend(fontsize=8)

    plt.suptitle("Iris — True Labels vs k-Means Clusters", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_iris_clustering.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/10_iris_clustering.png")

    # Silhouette diagram for Iris
    plot_silhouette_diagram(X_iris_s, km_iris.labels_, 3,
                            "11_silhouette_iris.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1.  LLOYD'S ALGORITHM       — Iterative assign-update until convergence
    2.  INERTIA (WCSS)          — Objective function minimized by k-Means
    3.  k-MEANS++ INIT          — Smarter spread-out initialization
    4.  ELBOW METHOD            — Choose k at the inertia "elbow"
    5.  SILHOUETTE SCORE        — Cohesion vs separation per sample
    6.  CONVERGENCE             — Guaranteed but to local minimum
    7.  MULTIPLE RESTARTS       — n_init runs, keep best
    8.  FAILURE CASES           — Non-convex, unbalanced, different densities
    9.  MINI-BATCH k-MEANS      — Faster on large data, slight quality loss
    10. FEATURE SCALING          — Mandatory for distance-based clustering
    11. ARI & NMI               — External cluster evaluation vs true labels
    12. ITERATION VISUALIZATION  — Watch centroids converge step-by-step
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
