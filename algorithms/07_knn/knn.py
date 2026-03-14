"""
=============================================================================
07 - K-NEAREST NEIGHBORS (KNN): Complete Implementation
=============================================================================
Covers:
  1. From-scratch KNN Classifier (Euclidean, Manhattan, weighted voting)
  2. From-scratch KNN Regressor (mean, distance-weighted mean)
  3. Scikit-learn KNeighborsClassifier / KNeighborsRegressor comparison
  4. Optimal k selection via cross-validated error curves
  5. Distance metric comparison (Euclidean vs Manhattan)
  6. Weighted vs uniform voting
  7. Decision boundary visualization — how k changes the boundary
  8. Curse of dimensionality demo — accuracy vs dimensionality
  9. Feature scaling importance — scaled vs unscaled
  10. KNN vs other classifiers benchmark

Datasets:
  - Classification: Breast Cancer Wisconsin (569, 30)
  - Regression:     California Housing (20640, 8)
  - Multiclass:     Iris (150, 4, 3 classes)
  - Synthetic:      make_classification for curse-of-dim demo
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

from sklearn.datasets import (
    load_breast_cancer, load_iris, fetch_california_housing,
    make_classification
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# SECTION 1: From-Scratch KNN Classifier
# =============================================================================
class KNNClassifierScratch:
    """
    K-Nearest Neighbors Classifier — from scratch.

    Parameters
    ----------
    k : int
        Number of neighbors.
    metric : str
        Distance metric: 'euclidean' or 'manhattan'.
    weights : str
        'uniform' → equal vote; 'distance' → inverse-distance weighting.
    """

    def __init__(self, k=5, metric="euclidean", weights="uniform"):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    # ── Distance Functions ───────────────────────────────────────────────────
    def _euclidean(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def _manhattan(self, a, b):
        return np.sum(np.abs(a - b), axis=1)

    def _compute_distances(self, x):
        """Compute distance from a single point x to all training points."""
        if self.metric == "manhattan":
            return self._manhattan(self.X_train, x)
        return self._euclidean(self.X_train, x)

    # ── Fit / Predict ────────────────────────────────────────────────────────
    def fit(self, X, y):
        """Store the training data (lazy learner — no computation)."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        k_distances = distances[k_indices]

        if self.weights == "distance":
            # Inverse-distance weighting; handle zero distance
            w = np.where(k_distances == 0, 1e10, 1.0 / k_distances)
            class_weights = {}
            for label, weight in zip(k_labels, w):
                class_weights[label] = class_weights.get(label, 0) + weight
            return max(class_weights, key=class_weights.get)
        else:
            # Uniform — majority vote
            counter = Counter(k_labels)
            return counter.most_common(1)[0][0]

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# =============================================================================
# SECTION 2: From-Scratch KNN Regressor
# =============================================================================
class KNNRegressorScratch:
    """
    K-Nearest Neighbors Regressor — from scratch.

    Parameters
    ----------
    k : int
    metric : str — 'euclidean' or 'manhattan'
    weights : str — 'uniform' or 'distance'
    """

    def __init__(self, k=5, metric="euclidean", weights="uniform"):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def _compute_distances(self, x):
        diff = self.X_train - x
        if self.metric == "manhattan":
            return np.sum(np.abs(diff), axis=1)
        return np.sqrt(np.sum(diff ** 2, axis=1))

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[:self.k]
        k_values = self.y_train[k_indices]
        k_distances = distances[k_indices]

        if self.weights == "distance":
            w = np.where(k_distances == 0, 1e10, 1.0 / k_distances)
            return np.average(k_values, weights=w)
        return np.mean(k_values)

    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


# =============================================================================
# SECTION 3: Visualization Helpers
# =============================================================================
def plot_decision_boundary_knn(X, y, model, title, filename, k_val=None):
    """Plot 2D decision boundary for a KNN classifier."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    ax.contour(xx, yy, Z, colors="black", linewidths=0.5, alpha=0.3)

    classes = np.unique(y)
    colors_pts = ["coral", "steelblue", "mediumseagreen", "gold"]
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_pts[i % len(colors_pts)],
                   label=f"Class {cls}", edgecolors="black", s=25, alpha=0.7)

    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_k_vs_error(k_range, train_errors, test_errors, filename,
                    ylabel="Error Rate", title="K vs Error"):
    """Plot train/test error as a function of k."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_range, train_errors, "o-", color="coral", linewidth=2, label="Train Error")
    ax.plot(k_range, test_errors, "s-", color="steelblue", linewidth=2, label="Test Error")
    ax.set_xlabel("k (Number of Neighbors)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_xticks(k_range)
    best_k = k_range[np.argmin(test_errors)]
    ax.axvline(x=best_k, color="green", linestyle="--", alpha=0.5,
               label=f"Best k={best_k}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_bar(names, values, ylabel, title, filename, colors=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    if colors is None:
        colors = sns.color_palette("viridis", len(names))
    bars = ax.bar(names, values, color=colors, edgecolor="white")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_regression_pred_vs_actual(y_true, y_pred, title, filename):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=15, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


# =============================================================================
# SECTION 4: Main Execution
# =============================================================================
def main():
    # ══════════════════════════════════════════════════════════════════════════
    # PART A: KNN CLASSIFICATION (Breast Cancer)
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("07 — K-NEAREST NEIGHBORS (KNN)")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("PART A: KNN CLASSIFICATION — Breast Cancer Dataset")
    print("=" * 70)

    cancer = load_breast_cancer()
    X_cls, y_cls = cancer.data, cancer.target
    class_names = list(cancer.target_names)

    print(f"Shape: {X_cls.shape}")
    print(f"Classes: {class_names}")

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    scaler_c = StandardScaler()
    X_train_cs = scaler_c.fit_transform(X_train_c)
    X_test_cs = scaler_c.transform(X_test_c)

    # ── Feature Scaling Importance ───────────────────────────────────────────
    print("\n  --- Feature Scaling Importance ---")
    knn_unscaled = KNeighborsClassifier(n_neighbors=5)
    knn_unscaled.fit(X_train_c, y_train_c)
    acc_unscaled = knn_unscaled.score(X_test_c, y_test_c)

    knn_scaled = KNeighborsClassifier(n_neighbors=5)
    knn_scaled.fit(X_train_cs, y_train_c)
    acc_scaled = knn_scaled.score(X_test_cs, y_test_c)

    print(f"  WITHOUT scaling: {acc_unscaled:.4f}")
    print(f"  WITH scaling:    {acc_scaled:.4f}")
    plot_bar(["Without Scaling", "With Scaling"],
             [acc_unscaled, acc_scaled],
             "Accuracy", "Feature Scaling Impact on KNN (k=5)",
             "01_scaling_impact.png", colors=["coral", "steelblue"])

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A1: From-Scratch KNN Classifier
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A1: KNN Classifier — From Scratch")
    print("=" * 70)

    scratch_knn = KNNClassifierScratch(k=5, metric="euclidean", weights="uniform")
    scratch_knn.fit(X_train_cs, y_train_c)

    y_pred_scratch = scratch_knn.predict(X_test_cs)
    acc_scratch = accuracy_score(y_test_c, y_pred_scratch)

    print(f"\n  Scratch KNN (k=5, euclidean, uniform)")
    print(f"  Test Accuracy: {acc_scratch:.6f}")

    # Weighted version
    scratch_knn_w = KNNClassifierScratch(k=5, metric="euclidean", weights="distance")
    scratch_knn_w.fit(X_train_cs, y_train_c)
    acc_scratch_w = scratch_knn_w.score(X_test_cs, y_test_c)
    print(f"\n  Scratch KNN (k=5, euclidean, distance-weighted)")
    print(f"  Test Accuracy: {acc_scratch_w:.6f}")

    # Manhattan version
    scratch_knn_m = KNNClassifierScratch(k=5, metric="manhattan", weights="uniform")
    scratch_knn_m.fit(X_train_cs, y_train_c)
    acc_scratch_m = scratch_knn_m.score(X_test_cs, y_test_c)
    print(f"\n  Scratch KNN (k=5, manhattan, uniform)")
    print(f"  Test Accuracy: {acc_scratch_m:.6f}")

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A2: Sklearn KNN Classifier
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A2: Scikit-learn KNeighborsClassifier")
    print("=" * 70)

    sk_knn = KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean")
    sk_knn.fit(X_train_cs, y_train_c)
    y_pred_sk = sk_knn.predict(X_test_cs)
    acc_sk = accuracy_score(y_test_c, y_pred_sk)

    print(f"\n  Sklearn KNN (k=5, euclidean, uniform)")
    print(f"  Test Accuracy: {acc_sk:.6f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_c, y_pred_sk, target_names=class_names))

    plot_confusion_matrix(y_test_c, y_pred_sk, class_names,
                          "KNN (k=5) — Confusion Matrix",
                          "02_confusion_matrix.png")

    # ── Scratch vs Sklearn comparison ────────────────────────────────────────
    print("\n  --- Scratch vs Sklearn ---")
    print(f"  Scratch: {acc_scratch:.6f}")
    print(f"  Sklearn: {acc_sk:.6f}")
    print(f"  Match:   {np.array_equal(y_pred_scratch, y_pred_sk)}")

    # ══════════════════════════════════════════════════════════════════════════
    # OPTIMAL K SELECTION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("OPTIMAL K SELECTION — Error Curves")
    print("=" * 70)

    k_range = list(range(1, 31))
    train_errors = []
    test_errors = []

    for k in k_range:
        knn_k = KNeighborsClassifier(n_neighbors=k)
        knn_k.fit(X_train_cs, y_train_c)
        train_errors.append(1 - knn_k.score(X_train_cs, y_train_c))
        test_errors.append(1 - knn_k.score(X_test_cs, y_test_c))

    best_k = k_range[np.argmin(test_errors)]
    print(f"\n  Best k = {best_k} (test error = {min(test_errors):.4f})")

    plot_k_vs_error(k_range, train_errors, test_errors,
                    "03_k_vs_error.png",
                    ylabel="Error Rate (1 - Accuracy)",
                    title="K vs Classification Error — Breast Cancer")

    # ── Cross-validation for more robust k selection ─────────────────────────
    print("\n  --- Cross-Validated K Selection ---")
    cv_scores_k = []
    for k in k_range:
        knn_cv = KNeighborsClassifier(n_neighbors=k)
        cv = cross_val_score(knn_cv, X_train_cs, y_train_c, cv=5, scoring="accuracy")
        cv_scores_k.append(cv.mean())

    best_k_cv = k_range[np.argmax(cv_scores_k)]
    print(f"  Best k (CV) = {best_k_cv} (CV accuracy = {max(cv_scores_k):.4f})")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(k_range, cv_scores_k, "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("k")
    ax.set_ylabel("5-Fold CV Accuracy")
    ax.set_title("Cross-Validated K Selection")
    ax.axvline(x=best_k_cv, color="green", linestyle="--", alpha=0.6,
               label=f"Best k={best_k_cv}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_cv_k_selection.png", dpi=150)
    plt.close()
    print("  [Saved] plots/04_cv_k_selection.png")

    # ══════════════════════════════════════════════════════════════════════════
    # DISTANCE METRIC COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DISTANCE METRIC COMPARISON")
    print("=" * 70)

    metrics = ["euclidean", "manhattan", "chebyshev", "minkowski"]
    metric_acc = []
    for m in metrics:
        knn_m = KNeighborsClassifier(n_neighbors=best_k_cv, metric=m)
        knn_m.fit(X_train_cs, y_train_c)
        acc = knn_m.score(X_test_cs, y_test_c)
        metric_acc.append(acc)
        print(f"  {m:12s}: {acc:.4f}")

    plot_bar(metrics, metric_acc,
             "Test Accuracy", f"Distance Metric Comparison (k={best_k_cv})",
             "05_metric_comparison.png")

    # ══════════════════════════════════════════════════════════════════════════
    # WEIGHTED vs UNIFORM COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("WEIGHTED vs UNIFORM KNN")
    print("=" * 70)

    weight_names = []
    weight_accs = []
    for w in ["uniform", "distance"]:
        for m in ["euclidean", "manhattan"]:
            knn_wm = KNeighborsClassifier(n_neighbors=best_k_cv, weights=w, metric=m)
            knn_wm.fit(X_train_cs, y_train_c)
            acc = knn_wm.score(X_test_cs, y_test_c)
            label = f"{w}-{m}"
            weight_names.append(label)
            weight_accs.append(acc)
            print(f"  {label:25s}: {acc:.4f}")

    plot_bar(weight_names, weight_accs,
             "Test Accuracy", "Weighted vs Uniform KNN",
             "06_weighted_vs_uniform.png")

    # ══════════════════════════════════════════════════════════════════════════
    # DECISION BOUNDARY — Different K Values
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DECISION BOUNDARIES — Different K Values (Iris 2D)")
    print("=" * 70)

    iris = load_iris()
    X_iris = iris.data[:, 2:4]  # petal length & width
    y_iris = iris.target
    iris_names = list(iris.target_names)

    scaler_iris = StandardScaler()
    X_iris_s = scaler_iris.fit_transform(X_iris)

    k_vals = [1, 3, 7, 15, 25]
    fig, axes = plt.subplots(1, len(k_vals), figsize=(5 * len(k_vals), 4.5))

    for ax, kv in zip(axes, k_vals):
        knn_b = KNeighborsClassifier(n_neighbors=kv)
        knn_b.fit(X_iris_s, y_iris)
        acc_b = knn_b.score(X_iris_s, y_iris)

        x_min, x_max = X_iris_s[:, 0].min() - 0.5, X_iris_s[:, 0].max() + 0.5
        y_min, y_max = X_iris_s[:, 1].min() - 0.5, X_iris_s[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = knn_b.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")

        colors_pts = ["coral", "steelblue", "mediumseagreen"]
        for i, cls in enumerate(np.unique(y_iris)):
            mask = y_iris == cls
            ax.scatter(X_iris_s[mask, 0], X_iris_s[mask, 1],
                       c=colors_pts[i], edgecolors="black", s=20, alpha=0.7)
        ax.set_title(f"k={kv} (Acc={acc_b:.3f})")

    plt.suptitle("Decision Boundary Changes with K (Iris — Petal Features)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_boundary_k_values.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/07_boundary_k_values.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PART B: KNN REGRESSION (California Housing)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART B: KNN REGRESSION — California Housing")
    print("=" * 70)

    housing = fetch_california_housing()
    X_reg, y_reg = housing.data, housing.target
    feature_names_reg = list(housing.feature_names)
    print(f"Shape: {X_reg.shape}")

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    scaler_r = StandardScaler()
    X_train_rs = scaler_r.fit_transform(X_train_r)
    X_test_rs = scaler_r.transform(X_test_r)

    # ── From-scratch KNN Regressor ───────────────────────────────────────────
    print("\n  --- From-Scratch KNN Regressor ---")
    # Use subset for speed (full dataset too slow for brute-force scratch)
    n_sub = 2000
    idx = np.random.choice(len(X_train_rs), n_sub, replace=False)
    X_train_sub = X_train_rs[idx]
    y_train_sub = y_train_r[idx]

    idx_test = np.random.choice(len(X_test_rs), 500, replace=False)
    X_test_sub = X_test_rs[idx_test]
    y_test_sub = y_test_r[idx_test]

    scratch_reg = KNNRegressorScratch(k=5, metric="euclidean", weights="uniform")
    scratch_reg.fit(X_train_sub, y_train_sub)
    y_pred_sr = scratch_reg.predict(X_test_sub)
    r2_scratch = r2_score(y_test_sub, y_pred_sr)
    rmse_scratch = np.sqrt(mean_squared_error(y_test_sub, y_pred_sr))
    print(f"  Scratch KNN Regressor (k=5, n_sub={n_sub})")
    print(f"  R² Score: {r2_scratch:.6f}")
    print(f"  RMSE:     {rmse_scratch:.6f}")

    # Weighted
    scratch_reg_w = KNNRegressorScratch(k=5, metric="euclidean", weights="distance")
    scratch_reg_w.fit(X_train_sub, y_train_sub)
    y_pred_srw = scratch_reg_w.predict(X_test_sub)
    r2_scratch_w = r2_score(y_test_sub, y_pred_srw)
    print(f"\n  Scratch KNN Regressor (k=5, distance-weighted)")
    print(f"  R² Score: {r2_scratch_w:.6f}")

    # ── Sklearn KNN Regressor (full dataset) ─────────────────────────────────
    print("\n  --- Sklearn KNN Regressor ---")
    sk_reg = KNeighborsRegressor(n_neighbors=5, weights="uniform")
    sk_reg.fit(X_train_rs, y_train_r)
    y_pred_skr = sk_reg.predict(X_test_rs)
    r2_sk = r2_score(y_test_r, y_pred_skr)
    rmse_sk = np.sqrt(mean_squared_error(y_test_r, y_pred_skr))
    print(f"  R² Score: {r2_sk:.6f}")
    print(f"  RMSE:     {rmse_sk:.6f}")

    sk_reg_w = KNeighborsRegressor(n_neighbors=5, weights="distance")
    sk_reg_w.fit(X_train_rs, y_train_r)
    y_pred_skrw = sk_reg_w.predict(X_test_rs)
    r2_sk_w = r2_score(y_test_r, y_pred_skrw)
    print(f"\n  Sklearn KNN (distance-weighted)")
    print(f"  R² Score: {r2_sk_w:.6f}")

    plot_regression_pred_vs_actual(
        y_test_r, y_pred_skr,
        f"KNN Regressor (k=5) — R²={r2_sk:.4f}",
        "08_regression_pred_vs_actual.png"
    )

    # ── Optimal K for regression ─────────────────────────────────────────────
    print("\n  --- K vs RMSE (Regression) ---")
    k_range_reg = list(range(1, 26))
    train_rmse = []
    test_rmse = []

    for k in k_range_reg:
        knn_rk = KNeighborsRegressor(n_neighbors=k)
        knn_rk.fit(X_train_rs, y_train_r)
        train_rmse.append(np.sqrt(mean_squared_error(y_train_r, knn_rk.predict(X_train_rs))))
        test_rmse.append(np.sqrt(mean_squared_error(y_test_r, knn_rk.predict(X_test_rs))))

    best_k_reg = k_range_reg[np.argmin(test_rmse)]
    print(f"  Best k (regression) = {best_k_reg} (RMSE = {min(test_rmse):.4f})")

    plot_k_vs_error(k_range_reg, train_rmse, test_rmse,
                    "09_k_vs_rmse_regression.png",
                    ylabel="RMSE", title="K vs RMSE — California Housing")

    # ══════════════════════════════════════════════════════════════════════════
    # CURSE OF DIMENSIONALITY DEMO
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("CURSE OF DIMENSIONALITY DEMO")
    print("=" * 70)

    dim_list = [2, 5, 10, 20, 50, 100, 200]
    knn_acc_by_dim = []

    for d in dim_list:
        X_hd, y_hd = make_classification(
            n_samples=500, n_features=d, n_informative=min(5, d),
            n_redundant=0, n_classes=2, random_state=42
        )
        scaler_hd = StandardScaler()
        X_hd = scaler_hd.fit_transform(X_hd)

        X_tr, X_te, y_tr, y_te = train_test_split(X_hd, y_hd, test_size=0.3,
                                                    random_state=42)
        knn_hd = KNeighborsClassifier(n_neighbors=5)
        knn_hd.fit(X_tr, y_tr)
        acc_hd = knn_hd.score(X_te, y_te)
        knn_acc_by_dim.append(acc_hd)
        print(f"  d={d:4d}: accuracy={acc_hd:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(dim_list, knn_acc_by_dim, "o-", color="steelblue", linewidth=2.5,
            markersize=8)
    ax.set_xlabel("Number of Dimensions")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Curse of Dimensionality — KNN (k=5, 500 samples, 5 informative)")
    ax.set_xscale("log")
    for d, a in zip(dim_list, knn_acc_by_dim):
        ax.annotate(f"{a:.3f}", (d, a), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_curse_of_dimensionality.png", dpi=150)
    plt.close()
    print("  [Saved] plots/10_curse_of_dimensionality.png")

    # ══════════════════════════════════════════════════════════════════════════
    # KNN vs OTHER CLASSIFIERS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KNN vs OTHER CLASSIFIERS (Breast Cancer)")
    print("=" * 70)

    classifiers = {
        f"KNN (k={best_k_cv})": KNeighborsClassifier(n_neighbors=best_k_cv),
        "KNN Weighted": KNeighborsClassifier(n_neighbors=best_k_cv, weights="distance"),
        "Logistic Reg": LogisticRegression(max_iter=5000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Grad Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", random_state=42),
    }

    names_comp = []
    scores_comp = []

    for name, clf in classifiers.items():
        clf.fit(X_train_cs, y_train_c)
        acc = clf.score(X_test_cs, y_test_c)
        names_comp.append(name)
        scores_comp.append(acc)
        print(f"  {name:20s}: {acc:.4f}")

    plot_bar(names_comp, scores_comp,
             "Test Accuracy", "KNN vs Other Classifiers",
             "11_knn_vs_others.png")

    # ══════════════════════════════════════════════════════════════════════════
    # MULTICLASS — Iris
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MULTICLASS KNN — Iris Dataset")
    print("=" * 70)

    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    scaler_i = StandardScaler()
    X_train_is = scaler_i.fit_transform(X_train_i)
    X_test_is = scaler_i.transform(X_test_i)

    knn_iris = KNeighborsClassifier(n_neighbors=5)
    knn_iris.fit(X_train_is, y_train_i)
    y_pred_iris = knn_iris.predict(X_test_is)
    acc_iris = accuracy_score(y_test_i, y_pred_iris)

    print(f"\n  Test Accuracy: {acc_iris:.6f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_i, y_pred_iris, target_names=iris_names))

    plot_confusion_matrix(y_test_i, y_pred_iris, iris_names,
                          "KNN (k=5) — Iris Confusion Matrix",
                          "12_confusion_matrix_iris.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1.  LAZY LEARNING           — No training phase; all work at prediction
    2.  K HYPERPARAMETER        — Controls bias-variance tradeoff (small k=overfit)
    3.  DISTANCE METRICS        — Euclidean, Manhattan, Chebyshev, Minkowski
    4.  WEIGHTED VOTING         — Closer neighbors count more (inverse distance)
    5.  FEATURE SCALING         — Mandatory for distance-based methods
    6.  OPTIMAL K SELECTION     — Error curves + cross-validation
    7.  DECISION BOUNDARIES     — Smoothness increases with k
    8.  KNN FOR REGRESSION      — Averages neighbor values instead of voting
    9.  CURSE OF DIMENSIONALITY — Performance degrades in high dimensions
    10. KNN vs ENSEMBLES/SVM    — Simple but competitive on clean datasets
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
