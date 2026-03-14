"""
=============================================================================
04 - RANDOM FOREST: Complete Implementation
=============================================================================
Covers:
  1. From-scratch Random Forest Classifier (bootstrap + feature subsampling)
  2. From-scratch Random Forest Regressor
  3. Scikit-learn comparison (classification & regression)
  4. OOB score demonstration
  5. Number of trees vs performance (diminishing returns)
  6. Feature importance (impurity-based + permutation)
  7. Single tree vs Forest comparison
  8. max_features impact analysis
  9. Decision boundary: single tree vs forest
  10. Variance reduction visualization

Datasets:
  - Classification: Breast Cancer Wisconsin (569 samples, 30 features)
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

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score
)

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# SECTION 1: Bootstrap Sampling Utility
# =============================================================================
def bootstrap_sample(X, y):
    """
    Draw a bootstrap sample (sampling with replacement).
    
    Returns a sample of size n (same as original) where:
      - ~63.2% of unique samples are included
      - ~36.8% are left out (Out-of-Bag)
    """
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    oob_indices = np.array(list(set(range(n_samples)) - set(indices)))
    return X[indices], y[indices], oob_indices


# =============================================================================
# SECTION 2: From-Scratch Random Forest Classifier
# =============================================================================
class RandomForestClassifierScratch:
    """
    Random Forest Classifier from scratch.
    
    Combines bagging (bootstrap sampling) with feature randomness
    (random subspace at each split) using sklearn's DecisionTreeClassifier
    as the base learner.
    
    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Max depth for each tree.
    max_features : str or int
        Features considered per split: 'sqrt', 'log2', int, or None (all).
    min_samples_split : int
        Min samples to split an internal node.
    min_samples_leaf : int
        Min samples in a leaf node.
    bootstrap : bool
        Whether to use bootstrap sampling.
    oob_score : bool
        Whether to compute Out-of-Bag score.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, max_features="sqrt",
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                 oob_score=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score_flag = oob_score
        
        self.trees = []
        self.oob_score_ = None
        self.feature_importances_ = None
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Build the forest:
        1. For each tree, draw a bootstrap sample
        2. Train a decision tree on that sample with random feature subsets
        3. Optionally compute OOB score
        """
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.trees = []
        self.feature_importances_ = np.zeros(n_features)
        
        # Store OOB predictions for OOB score
        oob_predictions = np.zeros((n_samples, len(self.classes_)))
        oob_count = np.zeros(n_samples)
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            if self.bootstrap:
                X_boot, y_boot, oob_idx = bootstrap_sample(X, y)
            else:
                X_boot, y_boot, oob_idx = X, y, np.array([])
            
            # Train a decision tree with feature randomness
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=i
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            # Accumulate feature importances
            self.feature_importances_ += tree.feature_importances_
            
            # OOB predictions
            if self.oob_score_flag and len(oob_idx) > 0:
                oob_pred = tree.predict(X[oob_idx])
                for j, idx in enumerate(oob_idx):
                    cls_idx = np.where(self.classes_ == oob_pred[j])[0][0]
                    oob_predictions[idx, cls_idx] += 1
                    oob_count[idx] += 1
        
        # Normalize feature importances
        self.feature_importances_ /= self.n_estimators
        
        # Compute OOB score
        if self.oob_score_flag:
            valid = oob_count > 0
            oob_final = self.classes_[np.argmax(oob_predictions[valid], axis=1)]
            self.oob_score_ = accuracy_score(y[valid], oob_final)
        
        return self
    
    def predict(self, X):
        """
        Predict by majority vote across all trees.
        Each tree casts one vote → most common class wins.
        """
        # Collect predictions from all trees
        all_preds = np.array([tree.predict(X) for tree in self.trees])  # (n_trees, n_samples)
        
        # Majority vote for each sample
        y_pred = np.zeros(X.shape[0], dtype=self.classes_.dtype)
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            y_pred[i] = Counter(votes).most_common(1)[0][0]
        
        return y_pred
    
    def predict_proba(self, X):
        """Average predicted probabilities across all trees."""
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(all_proba, axis=0)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# =============================================================================
# SECTION 3: From-Scratch Random Forest Regressor
# =============================================================================
class RandomForestRegressorScratch:
    """
    Random Forest Regressor from scratch.
    
    Same as classifier but:
    - Leaf predictions are means (not class labels)
    - Final prediction = average of all tree predictions
    - max_features defaults to n/3 for regression
    """
    
    def __init__(self, n_estimators=100, max_depth=None, max_features=0.33,
                 min_samples_split=2, min_samples_leaf=1, bootstrap=True,
                 oob_score=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.oob_score_flag = oob_score
        
        self.trees = []
        self.oob_score_ = None
        self.feature_importances_ = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.trees = []
        self.feature_importances_ = np.zeros(n_features)
        
        oob_predictions = np.zeros(n_samples)
        oob_count = np.zeros(n_samples)
        
        for i in range(self.n_estimators):
            if self.bootstrap:
                X_boot, y_boot, oob_idx = bootstrap_sample(X, y)
            else:
                X_boot, y_boot, oob_idx = X, y, np.array([])
            
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=i
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
            self.feature_importances_ += tree.feature_importances_
            
            if self.oob_score_flag and len(oob_idx) > 0:
                oob_pred = tree.predict(X[oob_idx])
                oob_predictions[oob_idx] += oob_pred
                oob_count[oob_idx] += 1
        
        self.feature_importances_ /= self.n_estimators
        
        if self.oob_score_flag:
            valid = oob_count > 0
            oob_final = oob_predictions[valid] / oob_count[valid]
            self.oob_score_ = r2_score(y[valid], oob_final)
        
        return self
    
    def predict(self, X):
        """Average predictions from all trees."""
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(all_preds, axis=0)
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))


# =============================================================================
# SECTION 4: Visualization Functions
# =============================================================================
def plot_n_trees_vs_performance(n_range, train_scores, test_scores, oob_scores,
                                metric, filename):
    """Show how performance changes with number of trees."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_range, train_scores, "o-", color="steelblue", linewidth=2,
            markersize=4, label=f"Train {metric}")
    ax.plot(n_range, test_scores, "o-", color="coral", linewidth=2,
            markersize=4, label=f"Test {metric}")
    if oob_scores:
        ax.plot(n_range, oob_scores, "o--", color="mediumseagreen", linewidth=2,
                markersize=4, label=f"OOB {metric}")
    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel(metric)
    ax.set_title(f"Random Forest — {metric} vs Number of Trees")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_single_tree_vs_forest(X, y, tree_model, forest_model, feature_names, filename):
    """Side-by-side decision boundary: single tree vs random forest."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = [("Single Decision Tree", tree_model), ("Random Forest", forest_model)]
    
    for ax, (title, model) in zip(axes, models):
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid).reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
        ax.contour(xx, yy, Z, colors="black", linewidths=0.3, alpha=0.4)
        
        classes = np.unique(y)
        colors = ["coral", "steelblue", "mediumseagreen"]
        for i, cls in enumerate(classes):
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], c=colors[i % len(colors)],
                       label=f"Class {cls}", edgecolors="black", s=25, alpha=0.7)
        
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.set_title(title)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_feature_importance_comparison(feature_names, imp_scratch, imp_sklearn,
                                       perm_imp, filename):
    """Compare three types of feature importance."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    data = [
        ("Scratch (Impurity)", imp_scratch),
        ("Sklearn (Impurity)", imp_sklearn),
        ("Permutation Importance", perm_imp),
    ]
    
    for ax, (title, importances) in zip(axes, data):
        sorted_idx = np.argsort(importances)
        top_n = min(15, len(importances))
        idx = sorted_idx[-top_n:]
        
        ax.barh(np.array(feature_names)[idx], importances[idx], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_max_features_impact(max_features_values, test_scores, filename):
    """Show how max_features affects performance."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    labels = [str(v) for v in max_features_values]
    colors = sns.color_palette("viridis", len(labels))
    bars = ax.bar(labels, test_scores, color=colors, edgecolor="white")
    
    ax.set_xlabel("max_features")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Impact of max_features on Random Forest Performance")
    
    for bar, val in zip(bars, test_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_variance_reduction(X, y, filename):
    """
    Demonstrate variance reduction: train many single trees and a forest,
    compare prediction variance.
    """
    n_trials = 50
    single_preds = []
    forest_preds = []
    
    for i in range(n_trials):
        # Single tree with different bootstrap
        X_boot, y_boot, _ = bootstrap_sample(X, y)
        
        dt = DecisionTreeClassifier(max_depth=5, random_state=i)
        dt.fit(X_boot, y_boot)
        single_preds.append(dt.predict(X[:100]))
        
    # One forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, y)
    forest_pred = rf.predict(X[:100])
    
    single_preds = np.array(single_preds)  # (n_trials, 100)
    
    # Compute variance per sample
    single_variance = np.var(single_preds, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribution of per-sample variance for single trees
    axes[0].hist(single_variance, bins=30, color="coral", edgecolor="white", alpha=0.8)
    axes[0].axvline(x=np.mean(single_variance), color="red", linestyle="--",
                    linewidth=2, label=f"Mean var={np.mean(single_variance):.4f}")
    axes[0].set_xlabel("Prediction Variance")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Single Tree — Per-Sample Prediction Variance")
    axes[0].legend()
    
    # Compare agreement across single trees vs forest stability
    agreement = np.mean(single_preds == single_preds[0], axis=0)
    axes[1].hist(agreement, bins=20, color="steelblue", edgecolor="white", alpha=0.8,
                 label="Single trees agreement")
    axes[1].axvline(x=np.mean(agreement), color="blue", linestyle="--", linewidth=2,
                    label=f"Mean agreement={np.mean(agreement):.2f}")
    axes[1].set_xlabel("Fraction of Trees Agreeing")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Prediction Stability Across Bootstrap Samples")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_oob_vs_test(n_range, oob_scores, test_scores, filename):
    """Show OOB score closely tracks test score."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n_range, oob_scores, "o-", color="mediumseagreen", linewidth=2,
            markersize=4, label="OOB Score")
    ax.plot(n_range, test_scores, "o-", color="coral", linewidth=2,
            markersize=4, label="Test Score")
    ax.set_xlabel("Number of Trees")
    ax.set_ylabel("Accuracy")
    ax.set_title("OOB Score vs Test Score — Free Validation!")
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


def plot_regression_comparison(y_true, preds_dict, filename):
    """Predicted vs actual for multiple regression models."""
    n = len(preds_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="steelblue")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2, label="Perfect")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(name)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


# =============================================================================
# SECTION 5: Main Execution
# =============================================================================
def main():
    # ══════════════════════════════════════════════════════════════════════════
    # PART A: CLASSIFICATION (Breast Cancer)
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PART A: RANDOM FOREST CLASSIFICATION — Breast Cancer Dataset")
    print("=" * 70)
    
    cancer = load_breast_cancer()
    X_cls, y_cls = cancer.data, cancer.target
    feature_names_cls = list(cancer.feature_names)
    class_names_cls = list(cancer.target_names)
    
    print(f"Shape: {X_cls.shape}")
    print(f"Classes: {class_names_cls}")
    print(f"Distribution: {np.bincount(y_cls)}")
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    print(f"Train: {X_train_c.shape}, Test: {X_test_c.shape}\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Variance Reduction Visualization
    # ══════════════════════════════════════════════════════════════════════════
    print("--- Variance Reduction Demo ---")
    plot_variance_reduction(X_cls, y_cls, "01_variance_reduction.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A1: From-Scratch Random Forest Classifier
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A1: Random Forest Classifier — From Scratch")
    print("=" * 70)
    
    scratch_rf = RandomForestClassifierScratch(
        n_estimators=100, max_depth=10, max_features="sqrt",
        oob_score=True
    )
    scratch_rf.fit(X_train_c, y_train_c)
    
    y_pred_sc = scratch_rf.predict(X_test_c)
    acc_sc = accuracy_score(y_test_c, y_pred_sc)
    
    print(f"\n  Test Accuracy: {acc_sc:.6f}")
    print(f"  OOB Score:     {scratch_rf.oob_score_:.6f}")
    print(f"  Trees:         {len(scratch_rf.trees)}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A2: Scikit-learn Random Forest Classifier
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A2: Scikit-learn Random Forest Classifier")
    print("=" * 70)
    
    sklearn_rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, max_features="sqrt",
        oob_score=True, random_state=42, n_jobs=-1
    )
    sklearn_rf.fit(X_train_c, y_train_c)
    
    y_pred_skc = sklearn_rf.predict(X_test_c)
    acc_skc = accuracy_score(y_test_c, y_pred_skc)
    
    print(f"\n  Test Accuracy: {acc_skc:.6f}")
    print(f"  OOB Score:     {sklearn_rf.oob_score_:.6f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_c, y_pred_skc, target_names=class_names_cls))
    
    plot_confusion_matrix(y_test_c, y_pred_skc, class_names_cls,
                          "Random Forest — Confusion Matrix",
                          "02_confusion_matrix.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Single Tree vs Forest Comparison
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SINGLE TREE vs RANDOM FOREST")
    print("=" * 70)
    
    dt_single = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_single.fit(X_train_c, y_train_c)
    acc_dt = dt_single.score(X_test_c, y_test_c)
    
    print(f"\n  Single Tree Accuracy:  {acc_dt:.6f}")
    print(f"  Random Forest Accuracy: {acc_skc:.6f}")
    print(f"  Improvement:            +{(acc_skc - acc_dt):.6f}")
    
    # Decision boundary comparison (first 2 features)
    dt_2d = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_2d.fit(X_train_c[:, :2], y_train_c)
    
    rf_2d = RandomForestClassifier(n_estimators=100, max_depth=10,
                                    random_state=42, n_jobs=-1)
    rf_2d.fit(X_train_c[:, :2], y_train_c)
    
    plot_single_tree_vs_forest(
        X_test_c[:, :2], y_test_c, dt_2d, rf_2d,
        [feature_names_cls[0], feature_names_cls[1]],
        "03_tree_vs_forest_boundary.png"
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # N_ESTIMATORS vs Performance
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("NUMBER OF TREES vs PERFORMANCE")
    print("=" * 70)
    
    n_range = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300]
    train_accs = []
    test_accs = []
    oob_accs = []
    
    for n in n_range:
        rf = RandomForestClassifier(n_estimators=n, max_depth=10,
                                     oob_score=True, random_state=42, n_jobs=-1)
        rf.fit(X_train_c, y_train_c)
        train_accs.append(rf.score(X_train_c, y_train_c))
        test_accs.append(rf.score(X_test_c, y_test_c))
        oob_accs.append(rf.oob_score_)
        print(f"  n={n:3d}: Train={train_accs[-1]:.4f}, "
              f"Test={test_accs[-1]:.4f}, OOB={oob_accs[-1]:.4f}")
    
    plot_n_trees_vs_performance(n_range, train_accs, test_accs, oob_accs,
                                "Accuracy", "04_n_trees_vs_accuracy.png")
    
    # OOB vs Test score tracking
    plot_oob_vs_test(n_range, oob_accs, test_accs, "05_oob_vs_test.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MAX_FEATURES Impact
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MAX_FEATURES IMPACT")
    print("=" * 70)
    
    mf_values = [1, 2, 5, "sqrt", "log2", 10, 20, None]
    mf_test_scores = []
    
    for mf in mf_values:
        rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                     max_features=mf, random_state=42, n_jobs=-1)
        rf.fit(X_train_c, y_train_c)
        score = rf.score(X_test_c, y_test_c)
        mf_test_scores.append(score)
        print(f"  max_features={str(mf):6s}: Test Accuracy={score:.4f}")
    
    plot_max_features_impact(mf_values, mf_test_scores,
                             "06_max_features_impact.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Feature Importance (3 types)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    
    # Scratch impurity-based
    imp_scratch = scratch_rf.feature_importances_
    
    # Sklearn impurity-based
    imp_sklearn = sklearn_rf.feature_importances_
    
    # Permutation importance
    perm_result = permutation_importance(sklearn_rf, X_test_c, y_test_c,
                                          n_repeats=10, random_state=42, n_jobs=-1)
    imp_perm = perm_result.importances_mean
    
    print("\n  Top 5 features (sklearn impurity-based):")
    top5 = np.argsort(imp_sklearn)[-5:][::-1]
    for idx in top5:
        print(f"    {feature_names_cls[idx]:25s}: {imp_sklearn[idx]:.4f}")
    
    print("\n  Top 5 features (permutation):")
    top5_perm = np.argsort(imp_perm)[-5:][::-1]
    for idx in top5_perm:
        print(f"    {feature_names_cls[idx]:25s}: {imp_perm[idx]:.4f}")
    
    plot_feature_importance_comparison(
        feature_names_cls, imp_scratch, imp_sklearn, imp_perm,
        "07_feature_importance_comparison.png"
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # PART B: REGRESSION (California Housing)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART B: RANDOM FOREST REGRESSION — California Housing")
    print("=" * 70)
    
    housing = fetch_california_housing(as_frame=True)
    X_reg, y_reg = housing.data.values, housing.target.values
    feature_names_reg = list(housing.feature_names)
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    print(f"Shape: {X_reg.shape}")
    print(f"Train: {X_train_r.shape}, Test: {X_test_r.shape}\n")
    
    # ── Single Tree Regressor ────────────────────────────────────────────────
    dt_reg = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt_reg.fit(X_train_r, y_train_r)
    y_pred_dt_r = dt_reg.predict(X_test_r)
    
    # ── Scratch RF Regressor ─────────────────────────────────────────────────
    print("=" * 70)
    print("MODEL B1: From-Scratch Random Forest Regressor")
    print("=" * 70)
    
    scratch_rfr = RandomForestRegressorScratch(
        n_estimators=100, max_depth=10, max_features=0.33, oob_score=True
    )
    scratch_rfr.fit(X_train_r, y_train_r)
    y_pred_sr = scratch_rfr.predict(X_test_r)
    
    mse_sr = mean_squared_error(y_test_r, y_pred_sr)
    r2_sr = r2_score(y_test_r, y_pred_sr)
    
    print(f"\n  RMSE:      {np.sqrt(mse_sr):.6f}")
    print(f"  R²:        {r2_sr:.6f}")
    print(f"  OOB R²:    {scratch_rfr.oob_score_:.6f}")
    
    # ── Sklearn RF Regressor ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL B2: Scikit-learn Random Forest Regressor")
    print("=" * 70)
    
    sklearn_rfr = RandomForestRegressor(
        n_estimators=100, max_depth=10, oob_score=True,
        random_state=42, n_jobs=-1
    )
    sklearn_rfr.fit(X_train_r, y_train_r)
    y_pred_skr = sklearn_rfr.predict(X_test_r)
    
    mse_skr = mean_squared_error(y_test_r, y_pred_skr)
    r2_skr = r2_score(y_test_r, y_pred_skr)
    
    print(f"\n  RMSE:      {np.sqrt(mse_skr):.6f}")
    print(f"  R²:        {r2_skr:.6f}")
    print(f"  OOB R²:    {sklearn_rfr.oob_score_:.6f}")
    
    # Comparison
    print(f"\n  --- Single Tree vs Forest (Regression) ---")
    print(f"  Single Tree R²:  {r2_score(y_test_r, y_pred_dt_r):.4f}")
    print(f"  Scratch RF R²:   {r2_sr:.4f}")
    print(f"  Sklearn RF R²:   {r2_skr:.4f}")
    
    plot_regression_comparison(y_test_r, {
        "Single Tree": y_pred_dt_r,
        "Scratch RF": y_pred_sr,
        "Sklearn RF": y_pred_skr,
    }, "08_regression_comparison.png")
    
    # Regression n_trees curve
    print("\n  --- N_trees vs R² (Regression) ---")
    n_range_r = [1, 5, 10, 25, 50, 100, 200]
    train_r2s = []
    test_r2s = []
    
    for n in n_range_r:
        rf_r = RandomForestRegressor(n_estimators=n, max_depth=10,
                                      random_state=42, n_jobs=-1)
        rf_r.fit(X_train_r, y_train_r)
        train_r2s.append(rf_r.score(X_train_r, y_train_r))
        test_r2s.append(rf_r.score(X_test_r, y_test_r))
        print(f"    n={n:3d}: Train R²={train_r2s[-1]:.4f}, Test R²={test_r2s[-1]:.4f}")
    
    plot_n_trees_vs_performance(n_range_r, train_r2s, test_r2s, [],
                                "R²", "09_n_trees_vs_r2_regression.png")
    
    # Feature importance for regression
    sorted_idx = np.argsort(sklearn_rfr.feature_importances_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_names_reg)[sorted_idx],
            sklearn_rfr.feature_importances_[sorted_idx], color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest Regressor — Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_feature_importance_regression.png", dpi=150)
    plt.close()
    print("  [Saved] plots/10_feature_importance_regression.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"""
    CLASSIFICATION (Breast Cancer):
      Single Tree (depth=10):      Test Acc = {acc_dt:.4f}
      Scratch RF  (100 trees):     Test Acc = {acc_sc:.4f}, OOB = {scratch_rf.oob_score_:.4f}
      Sklearn RF  (100 trees):     Test Acc = {acc_skc:.4f}, OOB = {sklearn_rf.oob_score_:.4f}

    REGRESSION (California Housing):
      Single Tree (depth=10):      Test R² = {r2_score(y_test_r, y_pred_dt_r):.4f}
      Scratch RF  (100 trees):     Test R² = {r2_sr:.4f}, OOB R² = {scratch_rfr.oob_score_:.4f}
      Sklearn RF  (100 trees):     Test R² = {r2_skr:.4f}, OOB R² = {sklearn_rfr.oob_score_:.4f}
    """)
    
    print("=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1.  BAGGING (BOOTSTRAP)    — Training each tree on a random resample
    2.  FEATURE RANDOMNESS     — Random subset of features at each split
    3.  MAJORITY VOTE          — Combining tree predictions (classification)
    4.  AVERAGING              — Combining tree predictions (regression)
    5.  OOB SCORE              — Free validation from ~36.8% unseen samples
    6.  N_ESTIMATORS CURVE     — More trees → better, with diminishing returns
    7.  MAX_FEATURES IMPACT    — sqrt(p) for classification, p/3 for regression
    8.  VARIANCE REDUCTION     — Ensemble smooths out individual tree instability
    9.  SINGLE TREE vs FOREST  — Forest consistently outperforms single tree
    10. FEATURE IMPORTANCE     — Impurity-based vs permutation-based
    11. DECISION BOUNDARIES    — Forest produces smoother boundaries than single tree
    12. NO FEATURE SCALING     — Trees don't need standardization
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
