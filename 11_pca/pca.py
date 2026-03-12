"""
=============================================================================
11 - PCA (Principal Component Analysis): Complete Implementation
=============================================================================
Covers:
  1. From-scratch PCA (centering, covariance, eigendecomposition, projection)
  2. Scikit-learn PCA comparison
  3. Scree plot — eigenvalues and explained variance ratio
  4. Cumulative explained variance — choosing number of components
  5. 2D & 3D visualization — projecting high-dimensional data
  6. Reconstruction — compress and decompress, visualize error
  7. Feature contribution — loadings heatmap
  8. Scaling impact — PCA with vs without StandardScaler
  9. PCA as preprocessing — classifier accuracy vs n_components
  10. Biplot — samples and feature vectors in PC space

Datasets:
  - Breast Cancer Wisconsin (569, 30) — high-dimensional
  - Iris (150, 4, 3 classes) — classic visualization
  - Digits (1797, 64) — image reconstruction demo
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import load_breast_cancer, load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# SECTION 1: From-Scratch PCA
# =============================================================================
class PCAScratch:
    """
    Principal Component Analysis — from scratch.

    Steps:
        1. Center the data (subtract mean)
        2. Compute covariance matrix
        3. Eigendecomposition
        4. Sort eigenvectors by eigenvalue (descending)
        5. Project onto top-k components

    Parameters
    ----------
    n_components : int or None
        Number of components to keep. If None, keep all.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None       # Principal component directions (k x d)
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_ = None

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        if self.n_components is None:
            self.n_components = n_features

        # Step 1: Center the data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Step 2: Covariance matrix (d x d)
        cov_matrix = np.cov(X_centered, rowvar=False)
        # Equivalent to: (X_centered.T @ X_centered) / (n_samples - 1)

        # Step 3: Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # eigh returns sorted ascending for symmetric matrices

        # Step 4: Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 5: Keep top-k
        self.components_ = eigenvectors[:, :self.n_components].T  # shape (k, d)
        self.explained_variance_ = eigenvalues[:self.n_components]
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

        # Store all eigenvalues for scree plot
        self._all_eigenvalues = eigenvalues
        self._all_explained_variance_ratio = eigenvalues / total_var

        return self

    def transform(self, X):
        """Project data onto principal components."""
        X = np.array(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T  # (n, d) @ (d, k) = (n, k)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        """Reconstruct data from reduced representation."""
        Z = np.array(Z, dtype=np.float64)
        return Z @ self.components_ + self.mean_  # (n, k) @ (k, d) + (d,)

    def get_reconstruction_error(self, X):
        """Mean squared reconstruction error."""
        X = np.array(X, dtype=np.float64)
        Z = self.transform(X)
        X_reconstructed = self.inverse_transform(Z)
        return np.mean((X - X_reconstructed) ** 2)


# =============================================================================
# SECTION 2: Visualization Helpers
# =============================================================================
def plot_scree(evr, filename, title="Scree Plot"):
    """Scree plot: explained variance ratio + cumulative."""
    n = len(evr)
    cumulative = np.cumsum(evr)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = "steelblue"
    ax1.bar(range(1, n + 1), evr, color=color1, alpha=0.7, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.plot(range(1, n + 1), cumulative, "o-", color=color2, linewidth=2.5,
             markersize=6, label="Cumulative")
    ax2.set_ylabel("Cumulative Explained Variance", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(y=0.95, color="green", linestyle="--", alpha=0.6, label="95% threshold")

    # Find number of components for 95%
    n_95 = np.argmax(cumulative >= 0.95) + 1
    ax2.axvline(x=n_95, color="green", linestyle=":", alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax1.set_title(f"{title}\n(95% variance at {n_95} components)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")
    return n_95


def plot_2d_projection(Z, y, class_names, title, filename, evr=None):
    """Plot data projected onto first 2 PCs."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("husl", len(class_names))

    for i, name in enumerate(class_names):
        mask = y == i
        ax.scatter(Z[mask, 0], Z[mask, 1], c=[colors[i]], s=25, alpha=0.6,
                   label=name, edgecolors="gray", linewidths=0.3)

    xlabel = "PC1"
    ylabel = "PC2"
    if evr is not None and len(evr) >= 2:
        xlabel += f" ({evr[0]:.1%})"
        ylabel += f" ({evr[1]:.1%})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_3d_projection(Z, y, class_names, title, filename, evr=None):
    """Plot data projected onto first 3 PCs."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = sns.color_palette("husl", len(class_names))

    for i, name in enumerate(class_names):
        mask = y == i
        ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2], c=[colors[i]],
                   s=15, alpha=0.6, label=name)

    labels = ["PC1", "PC2", "PC3"]
    if evr is not None and len(evr) >= 3:
        labels = [f"PC{j+1} ({evr[j]:.1%})" for j in range(3)]

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_loadings_heatmap(components, feature_names, filename, n_show=None):
    """Heatmap of feature loadings (contributions to each PC)."""
    n_pcs = components.shape[0]
    if n_show is not None:
        # Show only top features by absolute loading magnitude
        importance = np.max(np.abs(components), axis=0)
        top_idx = np.argsort(importance)[-n_show:]
        components = components[:, top_idx]
        feature_names = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(feature_names) * 0.5),
                                     max(4, n_pcs * 0.8)))
    sns.heatmap(components, annot=True, fmt=".2f", cmap="RdBu_r",
                xticklabels=feature_names,
                yticklabels=[f"PC{i+1}" for i in range(n_pcs)],
                ax=ax, center=0, linewidths=0.5)
    ax.set_title("Feature Loadings (Contribution to Each PC)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_biplot(Z, components, y, class_names, feature_names, filename,
                n_features_show=10):
    """Biplot: samples in PC space + feature vectors."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", len(class_names))

    # Plot samples
    for i, name in enumerate(class_names):
        mask = y == i
        ax.scatter(Z[mask, 0], Z[mask, 1], c=[colors[i]], s=15, alpha=0.4,
                   label=name)

    # Plot feature vectors (arrows)
    # Select top features by loading magnitude
    magnitudes = np.sqrt(components[0] ** 2 + components[1] ** 2)
    top_idx = np.argsort(magnitudes)[-n_features_show:]

    scale = max(np.abs(Z[:, :2]).max(), 1) * 0.8
    for idx in top_idx:
        ax.arrow(0, 0,
                 components[0, idx] * scale,
                 components[1, idx] * scale,
                 color="black", alpha=0.7, head_width=0.05 * scale,
                 head_length=0.03 * scale, linewidth=1.2)
        ax.text(components[0, idx] * scale * 1.12,
                components[1, idx] * scale * 1.12,
                feature_names[idx], fontsize=7, ha="center", color="darkred")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Biplot — Samples + Feature Vectors")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_reconstruction_digits(originals, reconstructed, title, filename, n_show=10):
    """Show original vs reconstructed digit images."""
    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))

    for i in range(n_show):
        axes[0, i].imshow(originals[i].reshape(8, 8), cmap="gray")
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10)

        axes[1, i].imshow(reconstructed[i].reshape(8, 8), cmap="gray")
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=10)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


# =============================================================================
# SECTION 3: Main Execution
# =============================================================================
def main():
    print("=" * 70)
    print("11 — PCA (Principal Component Analysis)")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    # PART A: PCA ON BREAST CANCER (30 → k features)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART A: PCA — Breast Cancer Dataset (30 features)")
    print("=" * 70)

    cancer = load_breast_cancer()
    X_bc, y_bc = cancer.data, cancer.target
    feature_names_bc = list(cancer.feature_names)
    class_names_bc = list(cancer.target_names)

    print(f"  Shape: {X_bc.shape}")
    print(f"  Classes: {class_names_bc}")

    # Standardize
    scaler = StandardScaler()
    X_bc_s = scaler.fit_transform(X_bc)

    # ── Feature Scaling Impact ───────────────────────────────────────────────
    print("\n  --- Scaling Impact ---")
    pca_noscale = PCAScratch(n_components=2)
    Z_noscale = pca_noscale.fit_transform(X_bc)

    pca_scaled = PCAScratch(n_components=2)
    Z_scaled = pca_scaled.fit_transform(X_bc_s)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = sns.color_palette("husl", 2)
    for i, name in enumerate(class_names_bc):
        mask = y_bc == i
        axes[0].scatter(Z_noscale[mask, 0], Z_noscale[mask, 1], c=[colors[i]],
                        s=20, alpha=0.5, label=name)
        axes[1].scatter(Z_scaled[mask, 0], Z_scaled[mask, 1], c=[colors[i]],
                        s=20, alpha=0.5, label=name)
    axes[0].set_title(f"Without Scaling\nPC1 var={pca_noscale.explained_variance_ratio_[0]:.1%}")
    axes[1].set_title(f"With Scaling\nPC1 var={pca_scaled.explained_variance_ratio_[0]:.1%}")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
    plt.suptitle("Feature Scaling Impact on PCA", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_scaling_impact.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/01_scaling_impact.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FROM-SCRATCH PCA (all components)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A1: PCA — From Scratch")
    print("=" * 70)

    scratch_pca = PCAScratch(n_components=None)
    scratch_pca.fit(X_bc_s)

    print(f"\n  All eigenvalues (top 10): {scratch_pca.explained_variance_[:10].round(4)}")
    print(f"  Explained variance ratio (top 5): "
          f"{scratch_pca.explained_variance_ratio_[:5].round(4)}")
    print(f"  Cumulative at 2 PCs: "
          f"{scratch_pca.explained_variance_ratio_[:2].sum():.4f}")

    # Scree plot (all 30 components)
    n_95 = plot_scree(scratch_pca._all_explained_variance_ratio,
                      "02_scree_plot.png",
                      title="Scree Plot — Breast Cancer (30 features)")
    print(f"  Components for 95% variance: {n_95}")

    # ── 2D Projection ────────────────────────────────────────────────────────
    scratch_pca_2d = PCAScratch(n_components=2)
    Z_2d = scratch_pca_2d.fit_transform(X_bc_s)

    plot_2d_projection(Z_2d, y_bc, class_names_bc,
                       "Scratch PCA — Breast Cancer (2D)",
                       "03_2d_projection_scratch.png",
                       evr=scratch_pca_2d.explained_variance_ratio_)

    # ── 3D Projection ────────────────────────────────────────────────────────
    scratch_pca_3d = PCAScratch(n_components=3)
    Z_3d = scratch_pca_3d.fit_transform(X_bc_s)

    plot_3d_projection(Z_3d, y_bc, class_names_bc,
                       "Scratch PCA — Breast Cancer (3D)",
                       "04_3d_projection.png",
                       evr=scratch_pca_3d.explained_variance_ratio_)

    # ══════════════════════════════════════════════════════════════════════════
    # SKLEARN PCA + Comparison
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A2: Scikit-learn PCA")
    print("=" * 70)

    sk_pca = PCA(n_components=2)
    Z_sk = sk_pca.fit_transform(X_bc_s)

    print(f"\n  Explained variance ratio: {sk_pca.explained_variance_ratio_.round(4)}")
    print(f"  Cumulative: {sk_pca.explained_variance_ratio_.sum():.4f}")

    plot_2d_projection(Z_sk, y_bc, class_names_bc,
                       "Sklearn PCA — Breast Cancer (2D)",
                       "05_2d_projection_sklearn.png",
                       evr=sk_pca.explained_variance_ratio_)

    # Scratch vs Sklearn
    # Note: signs may be flipped (eigenvectors are unique up to sign)
    print(f"\n  Scratch EVR: {scratch_pca_2d.explained_variance_ratio_.round(6)}")
    print(f"  Sklearn EVR: {sk_pca.explained_variance_ratio_.round(6)}")

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE LOADINGS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FEATURE LOADINGS — Which Features Drive Each PC")
    print("=" * 70)

    sk_pca_5 = PCA(n_components=5)
    sk_pca_5.fit(X_bc_s)

    # Show top 15 most important features
    plot_loadings_heatmap(sk_pca_5.components_, feature_names_bc,
                          "06_loadings_heatmap.png", n_show=15)

    # Top features for PC1
    pc1_loadings = sk_pca_5.components_[0]
    top_pc1 = np.argsort(np.abs(pc1_loadings))[-5:][::-1]
    print("\n  Top 5 features for PC1:")
    for idx in top_pc1:
        print(f"    {feature_names_bc[idx]:25s}  loading={pc1_loadings[idx]:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # BIPLOT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n  --- Biplot ---")
    plot_biplot(Z_sk, sk_pca.components_, y_bc, class_names_bc,
                feature_names_bc, "07_biplot.png", n_features_show=8)

    # ══════════════════════════════════════════════════════════════════════════
    # PART B: PCA ON IRIS (Classic Visualization)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART B: PCA — Iris Dataset (4D → 2D)")
    print("=" * 70)

    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    iris_names = list(iris.target_names)
    iris_feats = list(iris.feature_names)

    X_iris_s = StandardScaler().fit_transform(X_iris)

    pca_iris = PCA()
    pca_iris.fit(X_iris_s)

    print(f"  Explained variance ratio: {pca_iris.explained_variance_ratio_.round(4)}")
    print(f"  Cumulative: {np.cumsum(pca_iris.explained_variance_ratio_).round(4)}")

    Z_iris = pca_iris.transform(X_iris_s)[:, :2]

    plot_2d_projection(Z_iris, y_iris, iris_names,
                       "PCA — Iris (4D → 2D)",
                       "08_iris_2d.png",
                       evr=pca_iris.explained_variance_ratio_)

    # Loadings for Iris
    plot_loadings_heatmap(pca_iris.components_, iris_feats,
                          "09_iris_loadings.png")

    # Biplot for Iris
    plot_biplot(Z_iris, pca_iris.components_[:2], y_iris, iris_names,
                iris_feats, "10_iris_biplot.png", n_features_show=4)

    # ══════════════════════════════════════════════════════════════════════════
    # PART C: RECONSTRUCTION (Digits Dataset)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART C: RECONSTRUCTION — Digits Dataset (64 pixels)")
    print("=" * 70)

    digits = load_digits()
    X_dig, y_dig = digits.data, digits.target
    print(f"  Shape: {X_dig.shape} (8x8 images)")

    X_dig_s = StandardScaler().fit_transform(X_dig)

    # Reconstruction with different numbers of components
    n_comp_list = [2, 5, 10, 20, 40, 64]

    fig, axes = plt.subplots(len(n_comp_list) + 1, 10, figsize=(20, 3 * (len(n_comp_list) + 1)))
    sample_idx = np.random.choice(len(X_dig), 10, replace=False)

    # Original
    for j, idx in enumerate(sample_idx):
        axes[0, j].imshow(X_dig[idx].reshape(8, 8), cmap="gray")
        axes[0, j].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=10, rotation=0, labelpad=60)

    reconstruction_errors = []
    for row, nc in enumerate(n_comp_list, start=1):
        pca_rec = PCA(n_components=nc)
        Z_rec = pca_rec.fit_transform(X_dig_s)
        X_rec = pca_rec.inverse_transform(Z_rec)
        # Inverse-scale for visualization
        X_rec_orig = X_rec * np.sqrt(StandardScaler().fit(X_dig).var_) + StandardScaler().fit(X_dig).mean_

        mse = np.mean((X_dig - X_rec_orig) ** 2)
        reconstruction_errors.append(mse)
        evr_total = pca_rec.explained_variance_ratio_.sum()

        for j, idx in enumerate(sample_idx):
            axes[row, j].imshow(X_rec_orig[idx].reshape(8, 8), cmap="gray")
            axes[row, j].axis("off")
        axes[row, 0].set_ylabel(f"n={nc}\n({evr_total:.0%} var)", fontsize=9,
                                 rotation=0, labelpad=60)

    plt.suptitle("Digit Reconstruction — Varying Number of Components", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_digit_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/11_digit_reconstruction.png")

    # Reconstruction error curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n_comp_list, reconstruction_errors, "o-", color="steelblue",
            linewidth=2.5, markersize=8)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Mean Squared Reconstruction Error")
    ax.set_title("Reconstruction Error vs Number of Components (Digits)")
    for nc, err in zip(n_comp_list, reconstruction_errors):
        ax.annotate(f"{err:.1f}", (nc, err), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_reconstruction_error.png", dpi=150)
    plt.close()
    print("  [Saved] plots/12_reconstruction_error.png")

    # Digits 2D visualization (colored by digit)
    pca_dig_2d = PCA(n_components=2)
    Z_dig_2d = pca_dig_2d.fit_transform(X_dig_s)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(Z_dig_2d[:, 0], Z_dig_2d[:, 1], c=y_dig,
                         cmap="tab10", s=8, alpha=0.6)
    ax.set_xlabel(f"PC1 ({pca_dig_2d.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca_dig_2d.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Digits — PCA 2D Projection (64 → 2)")
    plt.colorbar(scatter, label="Digit", ticks=range(10))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_digits_2d.png", dpi=150)
    plt.close()
    print("  [Saved] plots/13_digits_2d.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PCA AS PREPROCESSING — Classifier Accuracy vs Components
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PCA AS PREPROCESSING — Accuracy vs N Components")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bc_s, y_bc, test_size=0.2, random_state=42, stratify=y_bc
    )

    comp_range = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
    lr_accs = []
    svm_accs = []

    for nc in comp_range:
        pca_pp = PCA(n_components=nc)
        X_tr_pca = pca_pp.fit_transform(X_train)
        X_te_pca = pca_pp.transform(X_test)

        lr = LogisticRegression(max_iter=5000, random_state=42)
        lr.fit(X_tr_pca, y_train)
        lr_accs.append(lr.score(X_te_pca, y_test))

        svm = SVC(kernel="rbf", random_state=42)
        svm.fit(X_tr_pca, y_train)
        svm_accs.append(svm.score(X_te_pca, y_test))

        print(f"  n_components={nc:2d}: LR={lr_accs[-1]:.4f}, SVM={svm_accs[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(comp_range, lr_accs, "o-", color="steelblue", linewidth=2, label="Logistic Reg")
    ax.plot(comp_range, svm_accs, "s-", color="coral", linewidth=2, label="SVM (RBF)")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("PCA as Preprocessing — Breast Cancer Classification")
    ax.legend()
    ax.set_xticks(comp_range)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "14_pca_preprocessing.png", dpi=150)
    plt.close()
    print("  [Saved] plots/14_pca_preprocessing.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1.  DIMENSIONALITY REDUCTION — Fewer features, retain max information
    2.  COVARIANCE MATRIX        — Captures feature relationships
    3.  EIGENDECOMPOSITION       — Finds directions of max variance
    4.  EXPLAINED VARIANCE RATIO — How much info each PC captures
    5.  SCREE PLOT               — Visual guide for choosing n_components
    6.  2D & 3D VISUALIZATION    — Project high-dim data for inspection
    7.  FEATURE LOADINGS         — Which features drive each component
    8.  BIPLOT                   — Samples + feature vectors together
    9.  RECONSTRUCTION           — Compress → decompress with error analysis
    10. FEATURE SCALING          — Mandatory before PCA (variance-based)
    11. PCA AS PREPROCESSING     — Reduce dims → improve classifier speed
    12. DIGITS RECONSTRUCTION    — Visual quality at different compressions
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
