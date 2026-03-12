"""
=============================================================================
02 - LOGISTIC REGRESSION: Complete Implementation
=============================================================================
Covers:
  1. From-scratch binary logistic regression (gradient descent + sigmoid)
  2. From-scratch multiclass logistic regression (softmax + cross-entropy)
  3. Scikit-learn comparison (binary & multiclass)
  4. Decision boundary visualization (2D)
  5. Threshold tuning & precision-recall tradeoff
  6. ROC curve & AUC
  7. Confusion matrix heatmap
  8. Regularization comparison (L1 vs L2 vs None)
  9. Feature scaling impact

Datasets:
  - Binary:     Breast Cancer Wisconsin (569 samples, 30 features)
  - Multiclass: Iris (150 samples, 4 features, 3 classes)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, log_loss
)

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# SECTION 1: Sigmoid & Softmax Functions
# =============================================================================
def sigmoid(z):
    """
    Sigmoid (logistic) function: σ(z) = 1 / (1 + e^(-z))
    Maps any real number to (0, 1).
    Clips z to avoid overflow in exp.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    """
    Softmax function for multiclass.
    σ(z_k) = e^(z_k) / Σ e^(z_j)
    
    Input:  z of shape (n_samples, n_classes)
    Output: probabilities of shape (n_samples, n_classes), rows sum to 1
    """
    # Subtract max for numerical stability (prevent overflow)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# =============================================================================
# SECTION 2: From-Scratch Binary Logistic Regression
# =============================================================================
class LogisticRegressionBinaryScratch:
    """
    Binary Logistic Regression using gradient descent.
    
    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent.
    n_iterations : int
        Number of gradient descent iterations.
    
    Attributes
    ----------
    weights : np.ndarray of shape (n_features,)
    bias : float
    cost_history : list of log-loss values per iteration
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train using batch gradient descent.
        
        Cost function (Binary Cross-Entropy / Log Loss):
            J = -(1/n) Σ [ y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i) ]
        
        Gradients:
            dw = (1/n) * X^T . (ŷ - y)
            db = (1/n) * Σ(ŷ - y)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []
        
        for i in range(self.n_iterations):
            # Forward pass
            z = X.dot(self.weights) + self.bias
            y_pred = sigmoid(z)
            
            # Compute log loss (clip to avoid log(0))
            eps = 1e-15
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            cost = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.cost_history.append(cost)
            
            # Compute gradients
            error = y_pred - y
            dw = (1 / n_samples) * X.T.dot(error)
            db = (1 / n_samples) * np.sum(error)
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X):
        """Return probability of class 1."""
        z = X.dot(self.weights) + self.bias
        return sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Return binary predictions using threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X, y):
        """Return accuracy."""
        return accuracy_score(y, self.predict(X))


# =============================================================================
# SECTION 3: From-Scratch Multiclass Logistic Regression (Softmax)
# =============================================================================
class LogisticRegressionMulticlassScratch:
    """
    Multiclass Logistic Regression using softmax + categorical cross-entropy.
    
    Parameters
    ----------
    learning_rate : float
    n_iterations : int
    
    Attributes
    ----------
    weights : np.ndarray of shape (n_features, n_classes)
    bias : np.ndarray of shape (n_classes,)
    cost_history : list
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Train using softmax + categorical cross-entropy.
        
        Cost:
            J = -(1/n) Σ_i Σ_k y_ik * log(ŷ_ik)
        
        Gradients:
            dW = (1/n) * X^T . (Ŷ - Y_onehot)
            db = (1/n) * Σ (Ŷ - Y_onehot)   (sum over samples)
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # One-hot encode y
        Y_onehot = np.zeros((n_samples, n_classes))
        for i, cls in enumerate(self.classes_):
            Y_onehot[y == cls, i] = 1.0
        
        # Initialize parameters
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        self.cost_history = []
        
        for iteration in range(self.n_iterations):
            # Forward pass
            z = X.dot(self.weights) + self.bias  # (n_samples, n_classes)
            Y_pred = softmax(z)                   # (n_samples, n_classes)
            
            # Compute categorical cross-entropy
            eps = 1e-15
            Y_pred_clipped = np.clip(Y_pred, eps, 1 - eps)
            cost = -np.mean(np.sum(Y_onehot * np.log(Y_pred_clipped), axis=1))
            self.cost_history.append(cost)
            
            # Compute gradients
            error = Y_pred - Y_onehot  # (n_samples, n_classes)
            dW = (1 / n_samples) * X.T.dot(error)      # (n_features, n_classes)
            db = (1 / n_samples) * np.sum(error, axis=0) # (n_classes,)
            
            # Update
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X):
        """Return class probabilities (n_samples, n_classes)."""
        z = X.dot(self.weights) + self.bias
        return softmax(z)
    
    def predict(self, X):
        """Return class predictions."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# =============================================================================
# SECTION 4: Evaluation Utilities
# =============================================================================
def print_classification_metrics(name, y_true, y_pred, y_proba=None):
    """Print full classification report."""
    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"{'─' * 55}")
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.6f}")
    
    is_binary = len(np.unique(y_true)) == 2
    avg = "binary" if is_binary else "weighted"
    
    print(f"  Precision : {precision_score(y_true, y_pred, average=avg, zero_division=0):.6f}")
    print(f"  Recall    : {recall_score(y_true, y_pred, average=avg, zero_division=0):.6f}")
    print(f"  F1 Score  : {f1_score(y_true, y_pred, average=avg, zero_division=0):.6f}")
    
    if y_proba is not None and is_binary:
        print(f"  Log Loss  : {log_loss(y_true, y_proba):.6f}")
    
    return {
        "name": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
    }


# =============================================================================
# SECTION 5: Visualization Functions
# =============================================================================
def plot_cost_convergence(cost_history, title="Log Loss Convergence"):
    """Plot log loss over iterations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(cost_history, color="steelblue", linewidth=1.5)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Log Loss")
    axes[0].set_title(f"{title} — Full")
    
    start = len(cost_history) // 5
    axes[1].plot(range(start, len(cost_history)), cost_history[start:],
                 color="coral", linewidth=1.5)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Log Loss")
    axes[1].set_title(f"{title} — Zoomed (last 80%)")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_cost_convergence.png", dpi=150)
    plt.close()
    print("  [Saved] plots/01_cost_convergence.png")


def plot_sigmoid_function():
    """Visualize the sigmoid function and its properties."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    z = np.linspace(-8, 8, 300)
    
    # Sigmoid
    axes[0].plot(z, sigmoid(z), color="steelblue", linewidth=2.5)
    axes[0].axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Threshold = 0.5")
    axes[0].axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    axes[0].fill_between(z, sigmoid(z), 0.5, where=(sigmoid(z) >= 0.5),
                         alpha=0.15, color="green", label="Predict class 1")
    axes[0].fill_between(z, sigmoid(z), 0.5, where=(sigmoid(z) < 0.5),
                         alpha=0.15, color="red", label="Predict class 0")
    axes[0].set_xlabel("z = w^T x + b")
    axes[0].set_ylabel("σ(z)")
    axes[0].set_title("Sigmoid Function")
    axes[0].legend(loc="upper left")
    axes[0].set_ylim(-0.05, 1.05)
    
    # Log loss components
    y_hat = np.linspace(0.01, 0.99, 200)
    loss_y1 = -np.log(y_hat)       # Loss when y=1
    loss_y0 = -np.log(1 - y_hat)   # Loss when y=0
    
    axes[1].plot(y_hat, loss_y1, color="steelblue", linewidth=2.5, label="y=1: -log(ŷ)")
    axes[1].plot(y_hat, loss_y0, color="coral", linewidth=2.5, label="y=0: -log(1-ŷ)")
    axes[1].set_xlabel("Predicted Probability (ŷ)")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Binary Cross-Entropy Loss Components")
    axes[1].legend()
    axes[1].set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_sigmoid_and_loss.png", dpi=150)
    plt.close()
    print("  [Saved] plots/02_sigmoid_and_loss.png")


def plot_decision_boundary(X, y, model, title, filename, feature_names=None):
    """
    Plot 2D decision boundary using the first 2 features.
    Works for both scratch and sklearn models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    if hasattr(model, 'predict_proba') and callable(model.predict_proba):
        try:
            proba = model.predict_proba(grid)
            if proba.ndim == 1:
                Z = (proba >= 0.5).astype(int)
            else:
                Z = np.argmax(proba, axis=1)
        except Exception:
            Z = model.predict(grid)
    else:
        Z = model.predict(grid)
    
    if hasattr(Z, 'values'):
        Z = Z.values
    Z = Z.reshape(xx.shape)
    
    # Plot boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    ax.contour(xx, yy, Z, colors="black", linewidths=0.5, alpha=0.5)
    
    # Plot data points
    classes = np.unique(y)
    colors = ["coral", "steelblue", "mediumseagreen", "gold"]
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i % len(colors)],
                   label=f"Class {cls}", edgecolors="black", s=30, alpha=0.7)
    
    ax.set_xlabel(feature_names[0] if feature_names else "Feature 1")
    ax.set_ylabel(feature_names[1] if feature_names else "Feature 2")
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Heatmap of confusion matrix."""
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


def plot_roc_curve(y_true, y_proba, title, filename):
    """Plot ROC curve with AUC score."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="steelblue")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_precision_recall_vs_threshold(y_true, y_proba, filename):
    """Plot precision and recall as a function of threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions[:-1], color="steelblue", linewidth=2, label="Precision")
    ax.plot(thresholds, recalls[:-1], color="coral", linewidth=2, label="Recall")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Default threshold (0.5)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall vs Threshold")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_regularization_comparison(results_list, filename):
    """Bar chart comparing metrics across regularization variants."""
    names = [r["name"] for r in results_list]
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    
    x = np.arange(len(names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["steelblue", "coral", "mediumseagreen", "gold"]
    
    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results_list]
        ax.bar(x + i * width, values, width, label=metric, color=colors[i])
    
    ax.set_ylabel("Score")
    ax.set_title("Regularization Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_multiclass_cost(cost_history, filename):
    """Plot softmax cross-entropy convergence."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Categorical Cross-Entropy")
    ax.set_title("Multiclass Softmax — Cost Convergence")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


# =============================================================================
# SECTION 6: Main Execution
# =============================================================================
def main():
    # ══════════════════════════════════════════════════════════════════════════
    # PART A: BINARY CLASSIFICATION (Breast Cancer Dataset)
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PART A: BINARY CLASSIFICATION — Breast Cancer Dataset")
    print("=" * 70)
    
    # ── Load Data ────────────────────────────────────────────────────────────
    cancer = load_breast_cancer()
    X_bin, y_bin = cancer.data, cancer.target
    feature_names_bin = cancer.feature_names
    class_names_bin = cancer.target_names  # ['malignant', 'benign']
    
    print(f"Shape: {X_bin.shape}")
    print(f"Classes: {class_names_bin} (0=malignant, 1=benign)")
    print(f"Class distribution: {np.bincount(y_bin)}")
    
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )
    
    scaler_b = StandardScaler()
    X_train_bs = scaler_b.fit_transform(X_train_b)
    X_test_bs = scaler_b.transform(X_test_b)
    
    # ── Plot Sigmoid ─────────────────────────────────────────────────────────
    print("\n--- Sigmoid Function & Log Loss Visualization ---")
    plot_sigmoid_function()
    
    binary_results = []
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A1: From-Scratch Binary Logistic Regression
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A1: Binary Logistic Regression — From Scratch")
    print("=" * 70)
    
    scratch_bin = LogisticRegressionBinaryScratch(learning_rate=0.1, n_iterations=2000)
    scratch_bin.fit(X_train_bs, y_train_b)
    
    y_pred_sb = scratch_bin.predict(X_test_bs)
    y_proba_sb = scratch_bin.predict_proba(X_test_bs)
    
    res_sb = print_classification_metrics("Scratch Binary", y_test_b, y_pred_sb, y_proba_sb)
    binary_results.append(res_sb)
    
    print(f"\n  Final log loss: {scratch_bin.cost_history[-1]:.6f}")
    print(f"  Weights shape: {scratch_bin.weights.shape}")
    
    plot_cost_convergence(scratch_bin.cost_history, "Binary LR — Gradient Descent")
    
    # ── Decision Boundary (using first 2 features) ──────────────────────────
    scratch_2d = LogisticRegressionBinaryScratch(learning_rate=0.1, n_iterations=2000)
    scratch_2d.fit(X_train_bs[:, :2], y_train_b)
    plot_decision_boundary(
        X_test_bs[:, :2], y_test_b, scratch_2d,
        "Scratch Binary — Decision Boundary (2 features)",
        "03_decision_boundary_scratch.png",
        feature_names=["Feature 1 (scaled)", "Feature 2 (scaled)"]
    )
    
    # ── ROC Curve ────────────────────────────────────────────────────────────
    plot_roc_curve(y_test_b, y_proba_sb, "Scratch Binary — ROC Curve",
                   "04_roc_curve_scratch.png")
    
    # ── Precision-Recall vs Threshold ────────────────────────────────────────
    plot_precision_recall_vs_threshold(y_test_b, y_proba_sb,
                                       "05_precision_recall_threshold.png")
    
    # ── Confusion Matrix ─────────────────────────────────────────────────────
    plot_confusion_matrix(y_test_b, y_pred_sb, class_names_bin,
                          "Scratch Binary — Confusion Matrix",
                          "06_confusion_matrix_scratch.png")
    
    # ── Threshold Tuning Demo ────────────────────────────────────────────────
    print("\n  --- Threshold Tuning Demo ---")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_t = (y_proba_sb >= threshold).astype(int)
        p = precision_score(y_test_b, y_t, zero_division=0)
        r = recall_score(y_test_b, y_t, zero_division=0)
        f = f1_score(y_test_b, y_t, zero_division=0)
        print(f"    Threshold {threshold:.1f} → Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A2: Scikit-learn Logistic Regression (Binary)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A2: Scikit-learn Logistic Regression (Binary)")
    print("=" * 70)
    
    sklearn_bin = LogisticRegression(max_iter=2000, random_state=42)
    sklearn_bin.fit(X_train_bs, y_train_b)
    
    y_pred_skb = sklearn_bin.predict(X_test_bs)
    y_proba_skb = sklearn_bin.predict_proba(X_test_bs)[:, 1]
    
    res_skb = print_classification_metrics("Sklearn Binary", y_test_b, y_pred_skb, y_proba_skb)
    binary_results.append(res_skb)
    
    print(f"\n  Coefficients shape: {sklearn_bin.coef_.shape}")
    print(f"  Intercept: {sklearn_bin.intercept_[0]:.4f}")
    
    # Decision boundary for sklearn
    sklearn_2d = LogisticRegression(max_iter=2000, random_state=42)
    sklearn_2d.fit(X_train_bs[:, :2], y_train_b)
    plot_decision_boundary(
        X_test_bs[:, :2], y_test_b, sklearn_2d,
        "Sklearn Binary — Decision Boundary",
        "07_decision_boundary_sklearn.png",
        feature_names=["Feature 1 (scaled)", "Feature 2 (scaled)"]
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # REGULARIZATION COMPARISON (Binary)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("REGULARIZATION COMPARISON (Binary)")
    print("=" * 70)
    
    reg_results = []
    
    for name, params in [
        ("No Reg", {"penalty": None, "solver": "lbfgs"}),
        ("L2 (C=1)", {"penalty": "l2", "C": 1.0, "solver": "lbfgs"}),
        ("L2 (C=0.1)", {"penalty": "l2", "C": 0.1, "solver": "lbfgs"}),
        ("L1 (C=1)", {"penalty": "l1", "C": 1.0, "solver": "saga"}),
        ("L1 (C=0.1)", {"penalty": "l1", "C": 0.1, "solver": "saga"}),
    ]:
        model = LogisticRegression(max_iter=5000, random_state=42, **params)
        model.fit(X_train_bs, y_train_b)
        y_p = model.predict(X_test_bs)
        res = print_classification_metrics(name, y_test_b, y_p)
        reg_results.append(res)
        
        n_zero = np.sum(np.abs(model.coef_) < 1e-6)
        print(f"  Coefficients near zero: {n_zero}/{model.coef_.size}")
    
    plot_regularization_comparison(reg_results, "08_regularization_comparison.png")
    
    # ── Feature Scaling Impact ───────────────────────────────────────────────
    print("\n  --- Feature Scaling Impact ---")
    lr_unscaled = LogisticRegression(max_iter=5000, random_state=42)
    lr_unscaled.fit(X_train_b, y_train_b)
    acc_unscaled = lr_unscaled.score(X_test_b, y_test_b)
    
    lr_scaled = LogisticRegression(max_iter=5000, random_state=42)
    lr_scaled.fit(X_train_bs, y_train_b)
    acc_scaled = lr_scaled.score(X_test_bs, y_test_b)
    
    print(f"  Accuracy WITHOUT scaling: {acc_unscaled:.6f}")
    print(f"  Accuracy WITH scaling:    {acc_scaled:.6f}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PART B: MULTICLASS CLASSIFICATION (Iris Dataset)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART B: MULTICLASS CLASSIFICATION — Iris Dataset")
    print("=" * 70)
    
    iris = load_iris()
    X_multi, y_multi = iris.data, iris.target
    class_names_multi = iris.target_names
    feature_names_multi = iris.feature_names
    
    print(f"Shape: {X_multi.shape}")
    print(f"Classes: {class_names_multi}")
    print(f"Class distribution: {np.bincount(y_multi)}")
    
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    
    scaler_m = StandardScaler()
    X_train_ms = scaler_m.fit_transform(X_train_m)
    X_test_ms = scaler_m.transform(X_test_m)
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL B1: From-Scratch Multiclass (Softmax)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL B1: Multiclass Logistic Regression — From Scratch (Softmax)")
    print("=" * 70)
    
    scratch_multi = LogisticRegressionMulticlassScratch(learning_rate=0.1, n_iterations=2000)
    scratch_multi.fit(X_train_ms, y_train_m)
    
    y_pred_sm = scratch_multi.predict(X_test_ms)
    
    res_sm = print_classification_metrics("Scratch Multiclass", y_test_m, y_pred_sm)
    
    print(f"\n  Weights shape: {scratch_multi.weights.shape}")
    print(f"  Final cost: {scratch_multi.cost_history[-1]:.6f}")
    
    # Classification report
    print(f"\n  Full Classification Report:")
    print(classification_report(y_test_m, y_pred_sm, target_names=class_names_multi))
    
    plot_multiclass_cost(scratch_multi.cost_history, "09_multiclass_cost.png")
    
    # Decision boundary (first 2 features)
    scratch_multi_2d = LogisticRegressionMulticlassScratch(learning_rate=0.1, n_iterations=2000)
    scratch_multi_2d.fit(X_train_ms[:, :2], y_train_m)
    plot_decision_boundary(
        X_test_ms[:, :2], y_test_m, scratch_multi_2d,
        "Scratch Softmax — Decision Boundary (Iris)",
        "10_decision_boundary_multiclass_scratch.png",
        feature_names=feature_names_multi[:2]
    )
    
    # Confusion Matrix
    plot_confusion_matrix(y_test_m, y_pred_sm, class_names_multi,
                          "Scratch Multiclass — Confusion Matrix",
                          "11_confusion_matrix_multiclass.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL B2: Scikit-learn Multiclass
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL B2: Scikit-learn Logistic Regression (Multiclass)")
    print("=" * 70)
    
    sklearn_multi = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=2000, random_state=42
    )
    sklearn_multi.fit(X_train_ms, y_train_m)
    
    y_pred_skm = sklearn_multi.predict(X_test_ms)
    
    res_skm = print_classification_metrics("Sklearn Multiclass", y_test_m, y_pred_skm)
    
    print(f"\n  Coefficients shape: {sklearn_multi.coef_.shape}")
    print(f"\n  Full Classification Report:")
    print(classification_report(y_test_m, y_pred_skm, target_names=class_names_multi))
    
    # Decision boundary
    sklearn_multi_2d = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=2000, random_state=42
    )
    sklearn_multi_2d.fit(X_train_ms[:, :2], y_train_m)
    plot_decision_boundary(
        X_test_ms[:, :2], y_test_m, sklearn_multi_2d,
        "Sklearn Multinomial — Decision Boundary (Iris)",
        "12_decision_boundary_multiclass_sklearn.png",
        feature_names=feature_names_multi[:2]
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — All Models")
    print("=" * 70)
    
    all_results = binary_results + [res_sm, res_skm]
    summary_df = pd.DataFrame(all_results)
    print(f"\n{summary_df.to_string(index=False)}\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # KEY CONCEPTS
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1.  SIGMOID FUNCTION       — Squashes linear output to probability [0,1]
    2.  LOG LOSS               — Convex cost for classification (not MSE!)
    3.  GRADIENT DESCENT       — Same gradient form as linear regression (with sigmoid)
    4.  SOFTMAX                — Generalizes sigmoid to K classes
    5.  DECISION BOUNDARY      — Linear boundary in feature space
    6.  THRESHOLD TUNING       — Adjusting precision-recall tradeoff
    7.  ROC CURVE & AUC        — Model discrimination performance
    8.  CONFUSION MATRIX       — TP, TN, FP, FN breakdown
    9.  REGULARIZATION (L1/L2) — Prevent overfitting, L1 for feature selection
    10. FEATURE SCALING        — Important for convergence speed & regularization
    11. MULTICLASS (SOFTMAX)   — One-hot encoding + categorical cross-entropy
    12. PRECISION / RECALL / F1 — Metrics beyond accuracy for imbalanced data
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
