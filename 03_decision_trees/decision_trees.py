"""
=============================================================================
03 - DECISION TREES: Complete Implementation
=============================================================================
Covers:
  1. From-scratch Decision Tree Classifier (Gini & Entropy)
  2. From-scratch Decision Tree Regressor (MSE)
  3. Scikit-learn comparison (classification & regression)
  4. Tree visualization (text-based + graphical)
  5. Gini vs Entropy comparison
  6. Overfitting demo (unrestricted vs pruned)
  7. Cost-complexity pruning (ccp_alpha)
  8. Feature importance
  9. Decision boundary visualization (2D)
  10. Regression tree on continuous target

Datasets:
  - Classification: Iris (150 samples, 4 features, 3 classes)
  - Regression:     California Housing (20,640 samples, 8 features)
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

from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# SECTION 1: From-Scratch Decision Tree Node
# =============================================================================
class TreeNode:
    """A single node in the decision tree."""
    
    def __init__(self, feature_index=None, threshold=None, left=None, right=None,
                 value=None, impurity=None, n_samples=None):
        # Decision node attributes
        self.feature_index = feature_index  # Which feature to split on
        self.threshold = threshold          # Threshold value for the split
        self.left = left                    # Left subtree (feature <= threshold)
        self.right = right                  # Right subtree (feature > threshold)
        
        # Leaf node attributes
        self.value = value                  # Prediction (class label or mean)
        
        # Info for visualization
        self.impurity = impurity
        self.n_samples = n_samples
    
    def is_leaf(self):
        return self.value is not None


# =============================================================================
# SECTION 2: From-Scratch Decision Tree Classifier
# =============================================================================
class DecisionTreeClassifierScratch:
    """
    Decision Tree Classifier built from scratch using recursive splitting.
    
    Parameters
    ----------
    max_depth : int or None
        Maximum depth of the tree. None = unlimited.
    min_samples_split : int
        Minimum samples required to split a node.
    min_samples_leaf : int
        Minimum samples required in a leaf.
    criterion : str
        'gini' or 'entropy'
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.root = None
        self.n_classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y):
        """Build the tree recursively."""
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root = self._build_tree(X, y, depth=0)
        
        # Normalize feature importances
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        
        return self
    
    def _compute_impurity(self, y):
        """
        Compute impurity of a node.
        
        Gini:    1 - Σ p_k²
        Entropy: -Σ p_k * log2(p_k)
        """
        n = len(y)
        if n == 0:
            return 0.0
        
        counts = np.bincount(y)
        probabilities = counts / n
        probabilities = probabilities[probabilities > 0]  # Remove zeros
        
        if self.criterion == "gini":
            return 1.0 - np.sum(probabilities ** 2)
        elif self.criterion == "entropy":
            return -np.sum(probabilities * np.log2(probabilities))
    
    def _information_gain(self, parent_y, left_y, right_y):
        """
        Compute weighted impurity reduction from a split.
        
        IG = impurity(parent) - [n_left/n * impurity(left) + n_right/n * impurity(right)]
        """
        n = len(parent_y)
        n_left, n_right = len(left_y), len(right_y)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        parent_impurity = self._compute_impurity(parent_y)
        child_impurity = (
            (n_left / n) * self._compute_impurity(left_y) +
            (n_right / n) * self._compute_impurity(right_y)
        )
        
        return parent_impurity - child_impurity
    
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split on.
        
        Iterates over every feature and every unique threshold,
        picking the one with maximum information gain.
        """
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            # Get unique values as candidate thresholds
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                # Split the data
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                # Check min_samples_leaf constraint
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                # Compute information gain
                gain = self._information_gain(y, left_y, right_y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth):
        """Recursively build the decision tree."""
        n_samples = len(y)
        n_classes_in_node = len(np.unique(y))
        
        # ── Stopping conditions ──────────────────────────────────────────────
        # 1. Pure node (all same class)
        if n_classes_in_node == 1:
            return TreeNode(value=y[0], impurity=0.0, n_samples=n_samples)
        
        # 2. Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            majority_class = Counter(y).most_common(1)[0][0]
            return TreeNode(value=majority_class,
                          impurity=self._compute_impurity(y),
                          n_samples=n_samples)
        
        # 3. Not enough samples to split
        if n_samples < self.min_samples_split:
            majority_class = Counter(y).most_common(1)[0][0]
            return TreeNode(value=majority_class,
                          impurity=self._compute_impurity(y),
                          n_samples=n_samples)
        
        # ── Find best split ──────────────────────────────────────────────────
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)
        
        if best_gain <= 0 or best_feature is None:
            majority_class = Counter(y).most_common(1)[0][0]
            return TreeNode(value=majority_class,
                          impurity=self._compute_impurity(y),
                          n_samples=n_samples)
        
        # ── Record feature importance ────────────────────────────────────────
        self.feature_importances_[best_feature] += best_gain * n_samples
        
        # ── Split and recurse ────────────────────────────────────────────────
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            impurity=self._compute_impurity(y),
            n_samples=n_samples
        )
    
    def predict(self, X):
        """Predict class for each sample."""
        return np.array([self._traverse(x, self.root) for x in X])
    
    def _traverse(self, x, node):
        """Traverse the tree to make a prediction for a single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    
    def print_tree(self, feature_names=None, node=None, indent=""):
        """Print a text representation of the tree."""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            print(f"{indent}└── PREDICT: class {node.value} "
                  f"(impurity={node.impurity:.4f}, samples={node.n_samples})")
            return
        
        fname = feature_names[node.feature_index] if feature_names else f"X[{node.feature_index}]"
        print(f"{indent}├── {fname} <= {node.threshold:.4f}? "
              f"(impurity={node.impurity:.4f}, samples={node.n_samples})")
        
        print(f"{indent}│   YES:")
        self.print_tree(feature_names, node.left, indent + "│   ")
        print(f"{indent}│   NO:")
        self.print_tree(feature_names, node.right, indent + "│   ")


# =============================================================================
# SECTION 3: From-Scratch Decision Tree Regressor
# =============================================================================
class DecisionTreeRegressorScratch:
    """
    Decision Tree Regressor built from scratch.
    
    Splits are chosen to minimize MSE (variance) in child nodes.
    Leaf predictions are the mean of the target values at that leaf.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.n_features_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root = self._build_tree(X, y, depth=0)
        
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        
        return self
    
    def _compute_mse(self, y):
        """Compute MSE (variance) of a node."""
        if len(y) == 0:
            return 0.0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _mse_reduction(self, parent_y, left_y, right_y):
        """Compute weighted MSE reduction from a split."""
        n = len(parent_y)
        n_left, n_right = len(left_y), len(right_y)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        parent_mse = self._compute_mse(parent_y)
        child_mse = (
            (n_left / n) * self._compute_mse(left_y) +
            (n_right / n) * self._compute_mse(right_y)
        )
        return parent_mse - child_mse
    
    def _find_best_split(self, X, y):
        """Find feature & threshold that maximizes MSE reduction."""
        n_samples, n_features = X.shape
        best_reduction = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            # Use midpoints between sorted unique values as thresholds
            sorted_unique = np.unique(feature_values)
            if len(sorted_unique) <= 1:
                continue
            thresholds = (sorted_unique[:-1] + sorted_unique[1:]) / 2.0
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                
                reduction = self._mse_reduction(y, left_y, right_y)
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_reduction
    
    def _build_tree(self, X, y, depth):
        """Recursively build the regression tree."""
        n_samples = len(y)
        
        # Stopping conditions
        if n_samples < self.min_samples_split:
            return TreeNode(value=np.mean(y), impurity=self._compute_mse(y),
                          n_samples=n_samples)
        
        if self.max_depth is not None and depth >= self.max_depth:
            return TreeNode(value=np.mean(y), impurity=self._compute_mse(y),
                          n_samples=n_samples)
        
        if self._compute_mse(y) == 0:
            return TreeNode(value=np.mean(y), impurity=0.0, n_samples=n_samples)
        
        # Find best split
        best_feature, best_threshold, best_reduction = self._find_best_split(X, y)
        
        if best_reduction <= 0 or best_feature is None:
            return TreeNode(value=np.mean(y), impurity=self._compute_mse(y),
                          n_samples=n_samples)
        
        # Record feature importance
        self.feature_importances_[best_feature] += best_reduction * n_samples
        
        # Split and recurse
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return TreeNode(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            impurity=self._compute_mse(y),
            n_samples=n_samples
        )
    
    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])
    
    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


# =============================================================================
# SECTION 4: Visualization Functions
# =============================================================================
def plot_impurity_functions():
    """Visualize Gini impurity and Entropy as a function of p (binary case)."""
    p = np.linspace(0.001, 0.999, 300)
    gini = 2 * p * (1 - p)  # Binary Gini = 1 - p² - (1-p)² = 2p(1-p)
    entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    misclass = 1 - np.maximum(p, 1 - p)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p, gini, color="steelblue", linewidth=2.5, label="Gini Impurity")
    ax.plot(p, entropy, color="coral", linewidth=2.5, label="Entropy")
    ax.plot(p, misclass, color="mediumseagreen", linewidth=2.5,
            linestyle="--", label="Misclassification Error")
    ax.set_xlabel("Proportion of Class 1 (p)")
    ax.set_ylabel("Impurity")
    ax.set_title("Impurity Measures (Binary Classification)")
    ax.legend()
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_impurity_functions.png", dpi=150)
    plt.close()
    print("  [Saved] plots/01_impurity_functions.png")


def plot_decision_boundary_tree(X, y, model, title, filename, feature_names=None):
    """Plot 2D decision boundary for a tree model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    ax.contour(xx, yy, Z, colors="black", linewidths=0.5, alpha=0.5)
    
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


def plot_sklearn_tree(model, feature_names, class_names, title, filename):
    """Visualize sklearn decision tree graphically."""
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=class_names,
              filled=True, rounded=True, ax=ax, fontsize=8, impurity=True)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_feature_importance(feature_names, importances, title, filename):
    """Bar chart of feature importances."""
    sorted_idx = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx],
            color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_overfitting_demo(depths, train_scores, test_scores, metric_name, filename):
    """Plot train vs test score across different tree depths."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(depths, train_scores, "o-", color="steelblue", linewidth=2, label=f"Train {metric_name}")
    ax.plot(depths, test_scores, "o-", color="coral", linewidth=2, label=f"Test {metric_name}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Overfitting Demo — {metric_name} vs Tree Depth")
    ax.legend()
    ax.set_xticks(depths)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_pruning_path(ccp_alphas, train_scores, test_scores, filename):
    """Plot accuracy vs ccp_alpha for cost-complexity pruning."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ccp_alphas, train_scores, "o-", color="steelblue", linewidth=1.5,
            markersize=3, label="Train Accuracy")
    ax.plot(ccp_alphas, test_scores, "o-", color="coral", linewidth=1.5,
            markersize=3, label="Test Accuracy")
    ax.set_xlabel("Cost-Complexity Pruning Alpha (ccp_alpha)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cost-Complexity Pruning Path")
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


def plot_regression_tree_predictions(y_true, y_pred, title, filename):
    """Scatter of predicted vs actual for regression."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
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


def plot_gini_vs_entropy(gini_results, entropy_results, filename):
    """Compare Gini and Entropy side by side."""
    metrics = ["Accuracy", "Train Acc"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics))
    width = 0.3
    
    ax.bar(x - width/2, [gini_results[m] for m in metrics], width,
           label="Gini", color="steelblue")
    ax.bar(x + width/2, [entropy_results[m] for m in metrics], width,
           label="Entropy", color="coral")
    
    ax.set_ylabel("Score")
    ax.set_title("Gini vs Entropy Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    for i, m in enumerate(metrics):
        ax.text(i - width/2, gini_results[m] + 0.02, f"{gini_results[m]:.4f}",
                ha="center", fontsize=9)
        ax.text(i + width/2, entropy_results[m] + 0.02, f"{entropy_results[m]:.4f}",
                ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


# =============================================================================
# SECTION 5: Main Execution
# =============================================================================
def main():
    # ══════════════════════════════════════════════════════════════════════════
    # PART A: CLASSIFICATION (Iris Dataset)
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PART A: DECISION TREE CLASSIFICATION — Iris Dataset")
    print("=" * 70)
    
    iris = load_iris()
    X_cls, y_cls = iris.data, iris.target
    feature_names_cls = iris.feature_names
    class_names_cls = list(iris.target_names)
    
    print(f"Shape: {X_cls.shape}")
    print(f"Classes: {class_names_cls}")
    print(f"Class distribution: {np.bincount(y_cls)}")
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    print(f"Train: {X_train_c.shape}, Test: {X_test_c.shape}\n")
    
    # ── Plot Impurity Functions ──────────────────────────────────────────────
    print("--- Impurity Functions Visualization ---")
    plot_impurity_functions()
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A1: From-Scratch Classifier (Gini)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A1: From-Scratch Decision Tree Classifier (Gini)")
    print("=" * 70)
    
    scratch_gini = DecisionTreeClassifierScratch(max_depth=5, criterion="gini")
    scratch_gini.fit(X_train_c, y_train_c)
    
    y_pred_sg = scratch_gini.predict(X_test_c)
    acc_sg = accuracy_score(y_test_c, y_pred_sg)
    train_acc_sg = accuracy_score(y_train_c, scratch_gini.predict(X_train_c))
    
    print(f"\n  Train Accuracy: {train_acc_sg:.6f}")
    print(f"  Test Accuracy:  {acc_sg:.6f}")
    
    print("\n  Tree Structure (first few levels):")
    scratch_gini.print_tree(feature_names=feature_names_cls)
    
    gini_results = {"Accuracy": acc_sg, "Train Acc": train_acc_sg}
    
    # Decision boundary (using features 2 & 3: petal length & width)
    scratch_gini_2d = DecisionTreeClassifierScratch(max_depth=5, criterion="gini")
    scratch_gini_2d.fit(X_train_c[:, 2:4], y_train_c)
    plot_decision_boundary_tree(
        X_test_c[:, 2:4], y_test_c, scratch_gini_2d,
        "Scratch DT (Gini) — Decision Boundary",
        "02_decision_boundary_scratch_gini.png",
        feature_names=feature_names_cls[2:4]
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A2: From-Scratch Classifier (Entropy)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A2: From-Scratch Decision Tree Classifier (Entropy)")
    print("=" * 70)
    
    scratch_ent = DecisionTreeClassifierScratch(max_depth=5, criterion="entropy")
    scratch_ent.fit(X_train_c, y_train_c)
    
    y_pred_se = scratch_ent.predict(X_test_c)
    acc_se = accuracy_score(y_test_c, y_pred_se)
    train_acc_se = accuracy_score(y_train_c, scratch_ent.predict(X_train_c))
    
    print(f"\n  Train Accuracy: {train_acc_se:.6f}")
    print(f"  Test Accuracy:  {acc_se:.6f}")
    
    entropy_results = {"Accuracy": acc_se, "Train Acc": train_acc_se}
    
    # Gini vs Entropy comparison
    plot_gini_vs_entropy(gini_results, entropy_results, "03_gini_vs_entropy.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A3: Scikit-learn Decision Tree Classifier
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A3: Scikit-learn Decision Tree Classifier")
    print("=" * 70)
    
    sklearn_cls = DecisionTreeClassifier(max_depth=5, random_state=42)
    sklearn_cls.fit(X_train_c, y_train_c)
    
    y_pred_skc = sklearn_cls.predict(X_test_c)
    acc_skc = accuracy_score(y_test_c, y_pred_skc)
    
    print(f"\n  Test Accuracy:  {acc_skc:.6f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_c, y_pred_skc, target_names=class_names_cls))
    
    # Tree text representation
    print("  Tree structure (sklearn text):")
    print(export_text(sklearn_cls, feature_names=feature_names_cls, max_depth=3))
    
    # Graphical tree visualization
    plot_sklearn_tree(sklearn_cls, feature_names_cls, class_names_cls,
                      "Sklearn Decision Tree (depth=5)", "04_sklearn_tree.png")
    
    # Feature importance
    plot_feature_importance(feature_names_cls, sklearn_cls.feature_importances_,
                           "Sklearn DT — Feature Importance", "05_feature_importance_cls.png")
    
    # Decision boundary
    sklearn_cls_2d = DecisionTreeClassifier(max_depth=5, random_state=42)
    sklearn_cls_2d.fit(X_train_c[:, 2:4], y_train_c)
    plot_decision_boundary_tree(
        X_test_c[:, 2:4], y_test_c, sklearn_cls_2d,
        "Sklearn DT — Decision Boundary",
        "06_decision_boundary_sklearn.png",
        feature_names=feature_names_cls[2:4]
    )
    
    # Confusion matrix
    plot_confusion_matrix(y_test_c, y_pred_skc, class_names_cls,
                          "Sklearn DT — Confusion Matrix",
                          "07_confusion_matrix.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # OVERFITTING DEMONSTRATION
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("OVERFITTING DEMONSTRATION — Accuracy vs Tree Depth")
    print("=" * 70)
    
    depths = list(range(1, 16))
    train_accs = []
    test_accs = []
    
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train_c, y_train_c)
        train_accs.append(dt.score(X_train_c, y_train_c))
        test_accs.append(dt.score(X_test_c, y_test_c))
        print(f"  depth={d:2d}: Train={train_accs[-1]:.4f}, Test={test_accs[-1]:.4f}")
    
    plot_overfitting_demo(depths, train_accs, test_accs, "Accuracy",
                          "08_overfitting_depth.png")
    
    # ── Unrestricted tree (no depth limit) ───────────────────────────────────
    print("\n  --- Unrestricted Tree ---")
    dt_full = DecisionTreeClassifier(random_state=42)
    dt_full.fit(X_train_c, y_train_c)
    print(f"  Unrestricted: depth={dt_full.get_depth()}, "
          f"leaves={dt_full.get_n_leaves()}, "
          f"Train={dt_full.score(X_train_c, y_train_c):.4f}, "
          f"Test={dt_full.score(X_test_c, y_test_c):.4f}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # COST-COMPLEXITY PRUNING
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COST-COMPLEXITY PRUNING (ccp_alpha)")
    print("=" * 70)
    
    # Get the pruning path
    path = dt_full.cost_complexity_pruning_path(X_train_c, y_train_c)
    ccp_alphas = path.ccp_alphas
    
    pruned_train = []
    pruned_test = []
    
    for alpha in ccp_alphas:
        dt_pruned = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
        dt_pruned.fit(X_train_c, y_train_c)
        pruned_train.append(dt_pruned.score(X_train_c, y_train_c))
        pruned_test.append(dt_pruned.score(X_test_c, y_test_c))
    
    best_idx = np.argmax(pruned_test)
    best_alpha = ccp_alphas[best_idx]
    print(f"\n  Best ccp_alpha: {best_alpha:.6f}")
    print(f"  Best test accuracy: {pruned_test[best_idx]:.4f}")
    print(f"  Total alphas tested: {len(ccp_alphas)}")
    
    plot_pruning_path(ccp_alphas, pruned_train, pruned_test,
                      "09_pruning_path.png")
    
    # Optimal pruned tree
    dt_optimal = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
    dt_optimal.fit(X_train_c, y_train_c)
    print(f"\n  Optimal tree: depth={dt_optimal.get_depth()}, "
          f"leaves={dt_optimal.get_n_leaves()}")
    
    plot_sklearn_tree(dt_optimal, feature_names_cls, class_names_cls,
                      f"Optimally Pruned Tree (alpha={best_alpha:.4f})",
                      "10_pruned_tree.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PART B: REGRESSION (California Housing — subsample for speed)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART B: DECISION TREE REGRESSION — California Housing")
    print("=" * 70)
    
    housing = fetch_california_housing(as_frame=True)
    X_reg = housing.data.values
    y_reg = housing.target.values
    feature_names_reg = list(housing.feature_names)
    
    # Subsample for faster scratch implementation
    idx = np.random.choice(len(X_reg), size=3000, replace=False)
    X_reg_sub, y_reg_sub = X_reg[idx], y_reg[idx]
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg_sub, y_reg_sub, test_size=0.2, random_state=42
    )
    print(f"Shape (subsampled): {X_reg_sub.shape}")
    print(f"Train: {X_train_r.shape}, Test: {X_test_r.shape}\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL B1: From-Scratch Regression Tree
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("MODEL B1: From-Scratch Decision Tree Regressor")
    print("=" * 70)
    
    scratch_reg = DecisionTreeRegressorScratch(max_depth=8, min_samples_leaf=5)
    scratch_reg.fit(X_train_r, y_train_r)
    
    y_pred_sr = scratch_reg.predict(X_test_r)
    mse_sr = mean_squared_error(y_test_r, y_pred_sr)
    r2_sr = r2_score(y_test_r, y_pred_sr)
    
    print(f"\n  MSE :  {mse_sr:.6f}")
    print(f"  RMSE:  {np.sqrt(mse_sr):.6f}")
    print(f"  R²  :  {r2_sr:.6f}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL B2: Scikit-learn Regression Tree
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL B2: Scikit-learn Decision Tree Regressor")
    print("=" * 70)
    
    sklearn_reg = DecisionTreeRegressor(max_depth=8, min_samples_leaf=5, random_state=42)
    sklearn_reg.fit(X_train_r, y_train_r)
    
    y_pred_skr = sklearn_reg.predict(X_test_r)
    mse_skr = mean_squared_error(y_test_r, y_pred_skr)
    r2_skr = r2_score(y_test_r, y_pred_skr)
    
    print(f"\n  MSE :  {mse_skr:.6f}")
    print(f"  RMSE:  {np.sqrt(mse_skr):.6f}")
    print(f"  R²  :  {r2_skr:.6f}")
    
    # Regression predictions plot
    plot_regression_tree_predictions(y_test_r, y_pred_skr,
                                     "Sklearn DT Regressor — Predicted vs Actual",
                                     "11_regression_predictions.png")
    
    # Feature importance for regression
    plot_feature_importance(feature_names_reg, sklearn_reg.feature_importances_,
                           "Sklearn DT Regressor — Feature Importance",
                           "12_feature_importance_reg.png")
    
    # ── Regression overfitting demo ──────────────────────────────────────────
    print("\n  --- Regression Overfitting Demo ---")
    depths_reg = list(range(1, 21))
    train_r2s = []
    test_r2s = []
    
    for d in depths_reg:
        dt_r = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt_r.fit(X_train_r, y_train_r)
        train_r2s.append(dt_r.score(X_train_r, y_train_r))
        test_r2s.append(dt_r.score(X_test_r, y_test_r))
    
    best_depth = depths_reg[np.argmax(test_r2s)]
    print(f"  Best depth for regression: {best_depth} "
          f"(Test R²={max(test_r2s):.4f})")
    
    plot_overfitting_demo(depths_reg, train_r2s, test_r2s, "R² Score",
                          "13_regression_overfitting.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"""
    CLASSIFICATION (Iris):
      Scratch (Gini, depth=5):    Test Acc = {acc_sg:.4f}
      Scratch (Entropy, depth=5): Test Acc = {acc_se:.4f}
      Sklearn (depth=5):          Test Acc = {acc_skc:.4f}
      Sklearn (pruned):           Test Acc = {pruned_test[best_idx]:.4f}

    REGRESSION (California Housing subsample):
      Scratch (depth=8):          Test R² = {r2_sr:.4f}
      Sklearn (depth=8):          Test R² = {r2_skr:.4f}
    """)
    
    # ══════════════════════════════════════════════════════════════════════════
    # KEY CONCEPTS
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1.  RECURSIVE SPLITTING    — Binary partitioning of feature space
    2.  GINI IMPURITY          — 1 - Σ p_k² (fast, default)
    3.  ENTROPY / INFO GAIN    — -Σ p_k log2(p_k) (information-theoretic)
    4.  GINI vs ENTROPY        — Nearly identical in practice
    5.  MSE FOR REGRESSION     — Variance reduction at each split
    6.  STOPPING CRITERIA      — max_depth, min_samples_split, min_samples_leaf
    7.  OVERFITTING            — Unrestricted trees memorize training data
    8.  COST-COMPLEXITY PRUNING— ccp_alpha controls tree complexity post-hoc
    9.  FEATURE IMPORTANCE     — Based on total impurity reduction
    10. DECISION BOUNDARIES    — Axis-aligned rectangular regions
    11. NO SCALING NEEDED      — Trees only care about feature ordering
    12. BUILDING BLOCK         — Foundation for Random Forest & Gradient Boosting
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
