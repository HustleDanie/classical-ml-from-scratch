"""
=============================================================================
06 - SUPPORT VECTOR MACHINES (SVM): Complete Implementation
=============================================================================
Covers:
  1. From-scratch Linear SVM (hinge loss + sub-gradient descent)
  2. Scikit-learn SVM with multiple kernels (linear, RBF, poly)
  3. Kernel comparison — same data, different boundaries
  4. C parameter impact — margin width vs accuracy
  5. Gamma parameter impact — RBF complexity
  6. Support vector visualization
  7. Decision boundary comparison (linear vs RBF vs poly)
  8. Feature scaling importance
  9. Multiclass SVM (One-vs-One on Iris)
  10. SVM vs other classifiers comparison

Datasets:
  - Binary:     Breast Cancer Wisconsin (569 samples, 30 features)
  - Multiclass: Iris (150 samples, 4 features, 3 classes)
  - Synthetic:  make_moons & make_circles (for kernel demos)
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
    load_breast_cancer, load_iris, make_moons, make_circles
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# SECTION 1: From-Scratch Linear SVM (Hinge Loss + SGD)
# =============================================================================
class LinearSVMScratch:
    """
    Linear SVM using sub-gradient descent on Hinge Loss.
    
    Loss function:
        L = (1/n) Σ max(0, 1 - y_i * (w^T x_i + b)) + (λ/2) ||w||²
    
    Sub-gradients:
        If y_i * (w^T x_i + b) >= 1  (correctly classified with margin):
            dw = λ * w
            db = 0
        Else (inside margin or misclassified):
            dw = λ * w - y_i * x_i
            db = -y_i
    
    Note: y must be in {-1, +1} for SVM.
    
    Parameters
    ----------
    learning_rate : float
    lambda_param : float — Regularization strength (1/C essentially)
    n_iterations : int
    """
    
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """
        Train Linear SVM using sub-gradient descent.
        Labels must be {-1, +1}.
        """
        n_samples, n_features = X.shape
        
        # Convert labels to {-1, +1} if needed
        y_svm = np.where(y <= 0, -1, 1)
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        
        for epoch in range(self.n_iterations):
            total_loss = 0.0
            
            for i in range(n_samples):
                # Compute decision value
                decision = y_svm[i] * (X[i].dot(self.weights) + self.bias)
                
                if decision >= 1:
                    # Correctly classified with sufficient margin
                    # Only regularization gradient
                    self.weights -= self.learning_rate * (self.lambda_param * self.weights)
                else:
                    # Inside margin or misclassified
                    self.weights -= self.learning_rate * (
                        self.lambda_param * self.weights - y_svm[i] * X[i]
                    )
                    self.bias -= self.learning_rate * (-y_svm[i])
                    total_loss += 1 - decision
            
            # Average hinge loss + regularization
            avg_loss = total_loss / n_samples + 0.5 * self.lambda_param * np.dot(self.weights, self.weights)
            self.loss_history.append(avg_loss)
        
        return self
    
    def decision_function(self, X):
        """Raw decision values: w^T x + b."""
        return X.dot(self.weights) + self.bias
    
    def predict(self, X):
        """Predict class labels {0, 1}."""
        return (self.decision_function(X) >= 0).astype(int)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# =============================================================================
# SECTION 2: Visualization Functions
# =============================================================================
def plot_decision_boundary(X, y, model, title, filename, feature_names=None,
                           show_support_vectors=False):
    """Plot 2D decision boundary with optional support vectors."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    if hasattr(model, "decision_function"):
        Z = model.decision_function(grid)
        if Z.ndim > 1:
            Z = Z.argmax(axis=1)
        else:
            Z = (Z >= 0).astype(int)
    else:
        Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    ax.contour(xx, yy, Z, colors="black", linewidths=0.5, alpha=0.4)
    
    # Plot margin boundaries for SVM
    if hasattr(model, "decision_function") and not isinstance(Z.ravel()[0], np.integer):
        try:
            Z_dec = model.decision_function(grid).reshape(xx.shape)
            if Z_dec.ndim == 2 and Z_dec.shape[-1] == 1:
                Z_dec = Z_dec.ravel().reshape(xx.shape)
            if Z_dec.ndim == 2:
                pass  # skip margin lines for multiclass
            else:
                ax.contour(xx, yy, Z_dec, levels=[-1, 0, 1],
                           colors=["coral", "black", "steelblue"],
                           linestyles=["--", "-", "--"], linewidths=[1.5, 2, 1.5])
        except Exception:
            pass
    
    classes = np.unique(y)
    colors_pts = ["coral", "steelblue", "mediumseagreen", "gold"]
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_pts[i % len(colors_pts)],
                   label=f"Class {cls}", edgecolors="black", s=30, alpha=0.7)
    
    # Highlight support vectors
    if show_support_vectors and hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=120, facecolors="none",
                   edgecolors="black", linewidths=2, label=f"Support Vectors ({len(sv)})")
    
    ax.set_xlabel(feature_names[0] if feature_names else "Feature 1")
    ax.set_ylabel(feature_names[1] if feature_names else "Feature 2")
    ax.set_title(title)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_kernel_comparison(X_train, y_train, X_test, y_test, filename):
    """Train SVM with 4 kernels and show decision boundaries side-by-side."""
    kernels = [
        ("Linear", {"kernel": "linear", "C": 1.0}),
        ("RBF", {"kernel": "rbf", "C": 1.0, "gamma": "scale"}),
        ("Poly (d=3)", {"kernel": "poly", "C": 1.0, "degree": 3}),
        ("Sigmoid", {"kernel": "sigmoid", "C": 1.0, "gamma": "scale"}),
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    for ax, (name, params) in zip(axes, kernels):
        svm = SVC(**params)
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
        
        for cls in np.unique(y_train):
            mask = y_train == cls
            ax.scatter(X_train[mask, 0], X_train[mask, 1],
                       c="coral" if cls == 0 else "steelblue",
                       edgecolors="black", s=20, alpha=0.7)
        
        # Support vectors
        sv = svm.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=80, facecolors="none",
                   edgecolors="black", linewidths=1.5)
        
        ax.set_title(f"{name}\nAcc={acc:.4f}, SVs={len(sv)}")
    
    plt.suptitle("Kernel Comparison — Same Data, Different Boundaries", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_c_parameter_impact(X_train, y_train, X_test, y_test, filename):
    """Show how C affects the margin and decision boundary."""
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    fig, axes = plt.subplots(1, len(C_values), figsize=(5 * len(C_values), 4.5))
    
    for ax, C in zip(axes, C_values):
        svm = SVC(kernel="rbf", C=C, gamma="scale")
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
        
        for cls in np.unique(y_train):
            mask = y_train == cls
            ax.scatter(X_train[mask, 0], X_train[mask, 1],
                       c="coral" if cls == 0 else "steelblue",
                       edgecolors="black", s=20, alpha=0.7)
        
        sv = svm.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=80, facecolors="none",
                   edgecolors="black", linewidths=1.5)
        ax.set_title(f"C={C}\nAcc={acc:.3f}, SVs={len(sv)}")
    
    plt.suptitle("C Parameter Impact (RBF Kernel)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_gamma_impact(X_train, y_train, X_test, y_test, filename):
    """Show how gamma affects the RBF decision boundary."""
    gamma_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    fig, axes = plt.subplots(1, len(gamma_values), figsize=(5 * len(gamma_values), 4.5))
    
    for ax, g in zip(axes, gamma_values):
        svm = SVC(kernel="rbf", C=1.0, gamma=g)
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
        
        for cls in np.unique(y_train):
            mask = y_train == cls
            ax.scatter(X_train[mask, 0], X_train[mask, 1],
                       c="coral" if cls == 0 else "steelblue",
                       edgecolors="black", s=20, alpha=0.7)
        
        sv = svm.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=80, facecolors="none",
                   edgecolors="black", linewidths=1.5)
        ax.set_title(f"γ={g}\nAcc={acc:.3f}, SVs={len(sv)}")
    
    plt.suptitle("Gamma Parameter Impact (RBF Kernel, C=1.0)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_hinge_loss():
    """Visualize hinge loss vs other loss functions."""
    z = np.linspace(-3, 3, 300)
    
    # Hinge: max(0, 1 - y*f) where y=1
    hinge = np.maximum(0, 1 - z)
    # Log loss: log(1 + exp(-y*f))
    log_loss_vals = np.log(1 + np.exp(-z))
    # 0-1 loss
    zero_one = (z < 0).astype(float)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, hinge, color="steelblue", linewidth=2.5, label="Hinge Loss (SVM)")
    ax.plot(z, log_loss_vals, color="coral", linewidth=2.5, label="Log Loss (Logistic Reg)")
    ax.plot(z, zero_one, color="mediumseagreen", linewidth=2.5, linestyle="--",
            label="0-1 Loss (ideal)")
    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=1, color="steelblue", linestyle=":", alpha=0.5, label="Margin (z=1)")
    ax.set_xlabel("y · f(x)  (margin)")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Functions Comparison")
    ax.legend()
    ax.set_ylim(-0.1, 4)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_hinge_loss.png", dpi=150)
    plt.close()
    print("  [Saved] plots/01_hinge_loss.png")


def plot_scaling_impact(acc_unscaled, acc_scaled, filename):
    """Show feature scaling impact on SVM."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Without Scaling", "With Scaling"],
                  [acc_unscaled, acc_scaled],
                  color=["coral", "steelblue"], edgecolor="white", width=0.5)
    ax.set_ylabel("Accuracy")
    ax.set_title("Feature Scaling Impact on SVM")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, [acc_unscaled, acc_scaled]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
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


def plot_model_comparison(names, scores, filename):
    """Bar chart comparing classifiers."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(names))
    bars = ax.bar(names, scores, color=colors, edgecolor="white")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("SVM vs Other Classifiers")
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_loss_convergence(loss_history, filename):
    """Plot hinge loss convergence for scratch SVM."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hinge Loss + Regularization")
    ax.set_title("Scratch Linear SVM — Loss Convergence")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_support_vector_count(C_values, sv_counts, acc_values, filename):
    """Show how C affects number of support vectors."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    color1 = "steelblue"
    ax1.plot(C_values, sv_counts, "o-", color=color1, linewidth=2, markersize=6)
    ax1.set_xlabel("C")
    ax1.set_ylabel("Number of Support Vectors", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")
    
    ax2 = ax1.twinx()
    color2 = "coral"
    ax2.plot(C_values, acc_values, "s--", color=color2, linewidth=2, markersize=6)
    ax2.set_ylabel("Test Accuracy", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    ax1.set_title("C vs Support Vectors & Accuracy")
    fig.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


# =============================================================================
# SECTION 3: Main Execution
# =============================================================================
def main():
    # ══════════════════════════════════════════════════════════════════════════
    # PREAMBLE: Hinge Loss Visualization
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("06 — SUPPORT VECTOR MACHINES (SVM)")
    print("=" * 70)
    
    print("\n--- Hinge Loss Visualization ---")
    plot_hinge_loss()
    
    # ══════════════════════════════════════════════════════════════════════════
    # PART A: BINARY CLASSIFICATION (Breast Cancer)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART A: BINARY SVM — Breast Cancer Dataset")
    print("=" * 70)
    
    cancer = load_breast_cancer()
    X_bin, y_bin = cancer.data, cancer.target
    feature_names_bin = list(cancer.feature_names)
    class_names_bin = list(cancer.target_names)
    
    print(f"Shape: {X_bin.shape}")
    print(f"Classes: {class_names_bin}")
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    scaler_b = StandardScaler()
    X_train_bs = scaler_b.fit_transform(X_train_b)
    X_test_bs = scaler_b.transform(X_test_b)
    
    # ── Feature Scaling Impact ───────────────────────────────────────────────
    print("\n  --- Feature Scaling Impact ---")
    svm_unscaled = SVC(kernel="rbf", random_state=42)
    svm_unscaled.fit(X_train_b, y_train_b)
    acc_unscaled = svm_unscaled.score(X_test_b, y_test_b)
    
    svm_scaled = SVC(kernel="rbf", random_state=42)
    svm_scaled.fit(X_train_bs, y_train_bs)
    acc_scaled = svm_scaled.score(X_test_bs, y_test_bs)
    
    print(f"  WITHOUT scaling: {acc_unscaled:.4f}")
    print(f"  WITH scaling:    {acc_scaled:.4f}")
    plot_scaling_impact(acc_unscaled, acc_scaled, "02_scaling_impact.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A1: From-Scratch Linear SVM
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A1: Linear SVM — From Scratch (Hinge Loss + SGD)")
    print("=" * 70)
    
    scratch_svm = LinearSVMScratch(learning_rate=0.001, lambda_param=0.01, n_iterations=500)
    scratch_svm.fit(X_train_bs, y_train_b)
    
    y_pred_ss = scratch_svm.predict(X_test_bs)
    acc_ss = accuracy_score(y_test_b, y_pred_ss)
    
    print(f"\n  Test Accuracy: {acc_ss:.6f}")
    print(f"  Weights shape: {scratch_svm.weights.shape}")
    print(f"  Final loss: {scratch_svm.loss_history[-1]:.6f}")
    
    plot_loss_convergence(scratch_svm.loss_history, "03_loss_convergence.png")
    
    # Decision boundary (first 2 features)
    scratch_2d = LinearSVMScratch(learning_rate=0.001, lambda_param=0.01, n_iterations=500)
    scratch_2d.fit(X_train_bs[:, :2], y_train_b)
    plot_decision_boundary(
        X_test_bs[:, :2], y_test_b, scratch_2d,
        f"Scratch Linear SVM (Acc={scratch_2d.score(X_test_bs[:, :2], y_test_b):.3f})",
        "04_boundary_scratch_linear.png",
        feature_names=["Feature 1 (scaled)", "Feature 2 (scaled)"]
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A2: Sklearn SVM (RBF) on Breast Cancer
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A2: Scikit-learn SVM (RBF Kernel)")
    print("=" * 70)
    
    sklearn_svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    sklearn_svm.fit(X_train_bs, y_train_b)
    
    y_pred_sk = sklearn_svm.predict(X_test_bs)
    acc_sk = accuracy_score(y_test_b, y_pred_sk)
    
    print(f"\n  Test Accuracy: {acc_sk:.6f}")
    print(f"  Support Vectors: {sklearn_svm.n_support_} (total={sum(sklearn_svm.n_support_)})")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_b, y_pred_sk, target_names=class_names_bin))
    
    plot_confusion_matrix(y_test_b, y_pred_sk, class_names_bin,
                          "SVM (RBF) — Confusion Matrix", "05_confusion_matrix.png")
    
    # Decision boundary with support vectors
    sk_2d = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    sk_2d.fit(X_train_bs[:, :2], y_train_b)
    plot_decision_boundary(
        X_train_bs[:, :2], y_train_b, sk_2d,
        f"SVM RBF — Decision Boundary (SVs={sum(sk_2d.n_support_)})",
        "06_boundary_rbf_support_vectors.png",
        feature_names=["Feature 1", "Feature 2"],
        show_support_vectors=True
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # C PARAMETER IMPACT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("C PARAMETER IMPACT")
    print("=" * 70)
    
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    sv_counts = []
    acc_values_c = []
    
    for C in C_values:
        svm_c = SVC(kernel="rbf", C=C, gamma="scale", random_state=42)
        svm_c.fit(X_train_bs, y_train_b)
        acc = svm_c.score(X_test_bs, y_test_b)
        n_sv = sum(svm_c.n_support_)
        sv_counts.append(n_sv)
        acc_values_c.append(acc)
        print(f"  C={C:8.3f}: Acc={acc:.4f}, Support Vectors={n_sv}")
    
    plot_support_vector_count(C_values, sv_counts, acc_values_c,
                              "07_c_vs_sv_count.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PART B: KERNEL DEMO (Synthetic Data — Moons & Circles)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART B: KERNEL COMPARISON — Synthetic Data")
    print("=" * 70)
    
    # ── Moons dataset ────────────────────────────────────────────────────────
    print("\n  --- Moons Dataset ---")
    X_moon, y_moon = make_moons(n_samples=500, noise=0.2, random_state=42)
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_moon, y_moon, test_size=0.2, random_state=42
    )
    
    scaler_m = StandardScaler()
    X_train_ms = scaler_m.fit_transform(X_train_m)
    X_test_ms = scaler_m.transform(X_test_m)
    
    plot_kernel_comparison(X_train_ms, y_train_m, X_test_ms, y_test_m,
                           "08_kernel_comparison_moons.png")
    
    # ── Circles dataset ──────────────────────────────────────────────────────
    print("\n  --- Circles Dataset ---")
    X_circ, y_circ = make_circles(n_samples=500, noise=0.1, factor=0.4, random_state=42)
    X_train_ci, X_test_ci, y_train_ci, y_test_ci = train_test_split(
        X_circ, y_circ, test_size=0.2, random_state=42
    )
    
    scaler_ci = StandardScaler()
    X_train_cis = scaler_ci.fit_transform(X_train_ci)
    X_test_cis = scaler_ci.transform(X_test_ci)
    
    plot_kernel_comparison(X_train_cis, y_train_ci, X_test_cis, y_test_ci,
                           "09_kernel_comparison_circles.png")
    
    # ── C parameter impact on moons ──────────────────────────────────────────
    print("\n  --- C Parameter Impact (Moons) ---")
    plot_c_parameter_impact(X_train_ms, y_train_m, X_test_ms, y_test_m,
                            "10_c_impact_moons.png")
    
    # ── Gamma impact on moons ────────────────────────────────────────────────
    print("\n  --- Gamma Parameter Impact (Moons) ---")
    plot_gamma_impact(X_train_ms, y_train_m, X_test_ms, y_test_m,
                      "11_gamma_impact_moons.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PART C: MULTICLASS SVM (Iris)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART C: MULTICLASS SVM — Iris Dataset")
    print("=" * 70)
    
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    class_names_iris = list(iris.target_names)
    feature_names_iris = list(iris.feature_names)
    
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
    )
    
    scaler_i = StandardScaler()
    X_train_is = scaler_i.fit_transform(X_train_i)
    X_test_is = scaler_i.transform(X_test_i)
    
    svm_multi = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm_multi.fit(X_train_is, y_train_i)
    
    y_pred_multi = svm_multi.predict(X_test_is)
    acc_multi = accuracy_score(y_test_i, y_pred_multi)
    
    print(f"\n  Test Accuracy: {acc_multi:.6f}")
    print(f"  Support Vectors per class: {svm_multi.n_support_}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_i, y_pred_multi, target_names=class_names_iris))
    
    plot_confusion_matrix(y_test_i, y_pred_multi, class_names_iris,
                          "Multiclass SVM (Iris) — Confusion Matrix",
                          "12_confusion_matrix_iris.png")
    
    # Decision boundary (petal features)
    svm_iris_2d = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    svm_iris_2d.fit(X_train_is[:, 2:4], y_train_i)
    plot_decision_boundary(
        X_test_is[:, 2:4], y_test_i, svm_iris_2d,
        "Multiclass SVM (RBF) — Iris Decision Boundary",
        "13_boundary_iris_multiclass.png",
        feature_names=feature_names_iris[2:4],
        show_support_vectors=True
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # PART D: SVM vs OTHER CLASSIFIERS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SVM vs OTHER CLASSIFIERS (Breast Cancer)")
    print("=" * 70)
    
    classifiers = {
        "Logistic Reg": LogisticRegression(max_iter=5000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Grad Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM Linear": SVC(kernel="linear", C=1.0, random_state=42),
        "SVM RBF": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42),
    }
    
    names_comp = []
    scores_comp = []
    
    for name, clf in classifiers.items():
        clf.fit(X_train_bs, y_train_b)
        acc = clf.score(X_test_bs, y_test_b)
        names_comp.append(name)
        scores_comp.append(acc)
        print(f"  {name:16s}: {acc:.4f}")
    
    plot_model_comparison(names_comp, scores_comp, "14_svm_vs_others.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1.  MAXIMUM MARGIN          — Finding the widest separating hyperplane
    2.  SUPPORT VECTORS         — Only boundary points define the model
    3.  HINGE LOSS              — SVM's loss function (max(0, 1-y·f(x)))
    4.  SOFT MARGIN (C param)   — Allowing misclassifications for robustness
    5.  KERNEL TRICK            — Nonlinear boundaries without explicit mapping
    6.  RBF KERNEL              — Projects to infinite dimensions (most versatile)
    7.  GAMMA PARAMETER         — Controls RBF "reach" (complexity)
    8.  C vs SUPPORT VECTORS    — Higher C → fewer SVs, tighter boundary
    9.  FEATURE SCALING         — Mandatory for SVM (distance-based)
    10. MULTICLASS (OvO)        — Pairwise binary classifiers combined
    11. LINEAR vs RBF vs POLY   — Kernel comparison on same data
    12. SVM vs ENSEMBLES        — Competitive on small, high-dim datasets
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
