"""
=============================================================================
05 - GRADIENT BOOSTING: Complete Implementation
=============================================================================
Covers:
  1. From-scratch Gradient Boosting Regressor (MSE loss + residual fitting)
  2. From-scratch Gradient Boosting Classifier (log loss + sigmoid)
  3. Scikit-learn comparison (classification & regression)
  4. Learning rate vs n_estimators tradeoff
  5. Staged predictions — model improving round by round
  6. Early stopping demonstration
  7. Subsample impact (stochastic gradient boosting)
  8. Feature importance
  9. Residual evolution over boosting rounds
  10. Comparison: Single Tree vs Random Forest vs Gradient Boosting

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

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, log_loss
)

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# SECTION 1: Sigmoid Utility
# =============================================================================
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# =============================================================================
# SECTION 2: From-Scratch Gradient Boosting Regressor
# =============================================================================
class GradientBoostingRegressorScratch:
    """
    Gradient Boosting for Regression (MSE loss) from scratch.
    
    Algorithm:
        1. F_0(x) = mean(y)
        2. For m = 1..M:
           a. Compute residuals: r_i = y_i - F_{m-1}(x_i)
           b. Fit tree h_m to residuals
           c. Update: F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
    
    Parameters
    ----------
    n_estimators : int          — Number of boosting rounds
    learning_rate : float       — Shrinkage (η)
    max_depth : int             — Depth of each weak learner
    subsample : float           — Fraction of data per round (1.0 = no subsampling)
    min_samples_leaf : int      — Minimum samples in leaf
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 subsample=1.0, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        
        self.trees = []
        self.initial_prediction = None
        self.train_losses = []         # MSE at each round
        self.feature_importances_ = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)
        self.trees = []
        self.train_losses = []
        
        # Step 1: Initialize with mean
        self.initial_prediction = np.mean(y)
        F = np.full(n_samples, self.initial_prediction)
        
        for m in range(self.n_estimators):
            # Step 2a: Compute pseudo-residuals (negative gradient of MSE)
            # For MSE: -∂L/∂F = y - F (simply the residuals)
            residuals = y - F
            
            # Record training loss
            mse = np.mean(residuals ** 2)
            self.train_losses.append(mse)
            
            # Subsample
            if self.subsample < 1.0:
                n_sub = max(1, int(n_samples * self.subsample))
                idx = np.random.choice(n_samples, size=n_sub, replace=False)
                X_sub, r_sub = X[idx], residuals[idx]
            else:
                X_sub, r_sub = X, residuals
            
            # Step 2b: Fit a weak learner (shallow tree) to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=m
            )
            tree.fit(X_sub, r_sub)
            self.trees.append(tree)
            
            # Accumulate feature importances
            self.feature_importances_ += tree.feature_importances_
            
            # Step 2c: Update prediction with shrinkage
            F += self.learning_rate * tree.predict(X)
        
        # Normalize feature importances
        self.feature_importances_ /= self.n_estimators
        
        return self
    
    def predict(self, X):
        """Sum initial prediction + all tree contributions."""
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F
    
    def staged_predict(self, X):
        """Yield predictions after adding each tree (for visualization)."""
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
            yield F.copy()
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))


# =============================================================================
# SECTION 3: From-Scratch Gradient Boosting Classifier
# =============================================================================
class GradientBoostingClassifierScratch:
    """
    Gradient Boosting for Binary Classification (log loss) from scratch.
    
    Algorithm:
        1. F_0(x) = log(p / (1-p))  where p = mean(y)  (log-odds)
        2. For m = 1..M:
           a. p_i = sigmoid(F_{m-1}(x_i))
           b. Pseudo-residuals: r_i = y_i - p_i
           c. Fit tree h_m to pseudo-residuals
           d. Update: F_m(x) = F_{m-1}(x) + learning_rate * h_m(x)
        3. Final prob: sigmoid(F_M(x))
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 subsample=1.0, min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        
        self.trees = []
        self.initial_prediction = None
        self.train_losses = []
        self.feature_importances_ = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros(n_features)
        self.trees = []
        self.train_losses = []
        
        # Step 1: Initialize with log-odds
        p = np.mean(y)
        self.initial_prediction = np.log(p / (1 - p + 1e-15))
        F = np.full(n_samples, self.initial_prediction)
        
        for m in range(self.n_estimators):
            # Step 2a: Convert to probabilities
            proba = sigmoid(F)
            
            # Record log loss
            eps = 1e-15
            proba_clipped = np.clip(proba, eps, 1 - eps)
            loss = -np.mean(y * np.log(proba_clipped) + (1 - y) * np.log(1 - proba_clipped))
            self.train_losses.append(loss)
            
            # Step 2b: Pseudo-residuals (negative gradient of log loss)
            # -∂L/∂F = y - sigmoid(F) = y - p
            residuals = y - proba
            
            # Subsample
            if self.subsample < 1.0:
                n_sub = max(1, int(n_samples * self.subsample))
                idx = np.random.choice(n_samples, size=n_sub, replace=False)
                X_sub, r_sub = X[idx], residuals[idx]
            else:
                X_sub, r_sub = X, residuals
            
            # Step 2c: Fit tree to pseudo-residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=m
            )
            tree.fit(X_sub, r_sub)
            self.trees.append(tree)
            
            self.feature_importances_ += tree.feature_importances_
            
            # Step 2d: Update
            F += self.learning_rate * tree.predict(X)
        
        self.feature_importances_ /= self.n_estimators
        return self
    
    def predict_proba(self, X):
        """Return probability of class 1."""
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return sigmoid(F)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def staged_predict_proba(self, X):
        """Yield probabilities after each boosting round."""
        F = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            F = F + self.learning_rate * tree.predict(X)
            yield sigmoid(F.copy())
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# =============================================================================
# SECTION 4: Visualization Functions
# =============================================================================
def plot_loss_convergence(train_losses, title, filename):
    """Plot training loss over boosting rounds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_losses, color="steelblue", linewidth=1.5)
    axes[0].set_xlabel("Boosting Round")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Full")
    
    start = len(train_losses) // 5
    axes[1].plot(range(start, len(train_losses)), train_losses[start:],
                 color="coral", linewidth=1.5)
    axes[1].set_xlabel("Boosting Round")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{title} — Zoomed")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_staged_performance(n_rounds, train_metric, test_metric, metric_name,
                            title, filename, val_metric=None):
    """Plot train/test metric evolving over boosting rounds."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_rounds, train_metric, color="steelblue", linewidth=2, label=f"Train {metric_name}")
    ax.plot(n_rounds, test_metric, color="coral", linewidth=2, label=f"Test {metric_name}")
    if val_metric is not None:
        ax.plot(n_rounds, val_metric, color="mediumseagreen", linewidth=2,
                linestyle="--", label=f"Validation {metric_name}")
    ax.set_xlabel("Number of Boosting Rounds")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_learning_rate_comparison(results, filename):
    """Compare different learning rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = sns.color_palette("viridis", len(results))
    
    for i, (lr, train_losses, test_score) in enumerate(results):
        axes[0].plot(train_losses, color=colors[i], linewidth=1.5, label=f"lr={lr}")
    axes[0].set_xlabel("Boosting Round")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Convergence")
    axes[0].legend()
    
    lrs = [r[0] for r in results]
    scores = [r[2] for r in results]
    bars = axes[1].bar([str(lr) for lr in lrs], scores, color=colors, edgecolor="white")
    axes[1].set_xlabel("Learning Rate")
    axes[1].set_ylabel("Test Score")
    axes[1].set_title("Test Score vs Learning Rate")
    for bar, val in zip(bars, scores):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_residual_evolution(X, y, model, rounds_to_show, filename):
    """Show how residuals shrink over boosting rounds."""
    staged = list(model.staged_predict(X))
    
    n_plots = len(rounds_to_show)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    
    for ax, rnd in zip(axes, rounds_to_show):
        if rnd > len(staged):
            rnd = len(staged)
        residuals = y - staged[rnd - 1]
        ax.hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        ax.set_title(f"Round {rnd}\nStd={np.std(residuals):.4f}")
    
    plt.suptitle("Residual Distribution Over Boosting Rounds", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_subsample_comparison(subsample_vals, test_scores, filename):
    """Show impact of subsampling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(subsample_vals))
    bars = ax.bar([str(s) for s in subsample_vals], test_scores, color=colors, edgecolor="white")
    ax.set_xlabel("subsample")
    ax.set_ylabel("Test Score")
    ax.set_title("Stochastic Gradient Boosting — Subsample Impact")
    for bar, val in zip(bars, test_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_model_comparison_bar(names, scores, metric_name, filename):
    """Bar chart comparing models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["coral", "steelblue", "mediumseagreen", "gold", "plum"]
    bars = ax.bar(names, scores, color=colors[:len(names)], edgecolor="white")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Model Comparison — {metric_name}")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_feature_importance(feature_names, importances, title, filename):
    sorted_idx = np.argsort(importances)
    top_n = min(15, len(importances))
    idx = sorted_idx[-top_n:]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[idx], importances[idx], color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_early_stopping(n_range, train_scores, val_scores, best_n, metric, filename):
    """Show early stopping point."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_range, train_scores, color="steelblue", linewidth=2, label=f"Train {metric}")
    ax.plot(n_range, val_scores, color="coral", linewidth=2, label=f"Validation {metric}")
    ax.axvline(x=best_n, color="red", linestyle="--", linewidth=2,
               label=f"Best = {best_n} rounds")
    ax.set_xlabel("Number of Boosting Rounds")
    ax.set_ylabel(metric)
    ax.set_title("Early Stopping — When to Stop Adding Trees")
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


# =============================================================================
# SECTION 5: Main Execution
# =============================================================================
def main():
    # ══════════════════════════════════════════════════════════════════════════
    # PART A: REGRESSION (California Housing)
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PART A: GRADIENT BOOSTING REGRESSION — California Housing")
    print("=" * 70)
    
    housing = fetch_california_housing(as_frame=True)
    X_reg, y_reg = housing.data.values, housing.target.values
    feature_names_reg = list(housing.feature_names)
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    # Validation set for early stopping demo
    X_tr_r, X_val_r, y_tr_r, y_val_r = train_test_split(
        X_train_r, y_train_r, test_size=0.15, random_state=42
    )
    
    print(f"Shape: {X_reg.shape}")
    print(f"Train: {X_tr_r.shape}, Val: {X_val_r.shape}, Test: {X_test_r.shape}\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A1: From-Scratch GB Regressor
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("MODEL A1: Gradient Boosting Regressor — From Scratch")
    print("=" * 70)
    
    scratch_gbr = GradientBoostingRegressorScratch(
        n_estimators=200, learning_rate=0.1, max_depth=3
    )
    scratch_gbr.fit(X_train_r, y_train_r)
    
    y_pred_sr = scratch_gbr.predict(X_test_r)
    r2_sr = r2_score(y_test_r, y_pred_sr)
    rmse_sr = np.sqrt(mean_squared_error(y_test_r, y_pred_sr))
    
    print(f"\n  R²:   {r2_sr:.6f}")
    print(f"  RMSE: {rmse_sr:.6f}")
    print(f"  Initial prediction (mean): {scratch_gbr.initial_prediction:.4f}")
    print(f"  Final train MSE: {scratch_gbr.train_losses[-1]:.6f}")
    
    plot_loss_convergence(scratch_gbr.train_losses,
                          "Scratch GB Regressor — MSE", "01_loss_convergence_reg.png")
    
    # Residual evolution
    plot_residual_evolution(X_test_r, y_test_r, scratch_gbr,
                           [1, 10, 50, 200], "02_residual_evolution.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A2: Scikit-learn GB Regressor
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A2: Scikit-learn GradientBoostingRegressor")
    print("=" * 70)
    
    sklearn_gbr = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
    )
    sklearn_gbr.fit(X_train_r, y_train_r)
    
    y_pred_skr = sklearn_gbr.predict(X_test_r)
    r2_skr = r2_score(y_test_r, y_pred_skr)
    rmse_skr = np.sqrt(mean_squared_error(y_test_r, y_pred_skr))
    
    print(f"\n  R²:   {r2_skr:.6f}")
    print(f"  RMSE: {rmse_skr:.6f}")
    
    # Staged predictions (train vs test R²)
    print("\n  --- Staged Performance (round by round) ---")
    staged_train = list(sklearn_gbr.staged_predict(X_train_r))
    staged_test = list(sklearn_gbr.staged_predict(X_test_r))
    
    rounds = list(range(1, len(staged_train) + 1))
    train_r2_staged = [r2_score(y_train_r, p) for p in staged_train]
    test_r2_staged = [r2_score(y_test_r, p) for p in staged_test]
    
    for rnd in [1, 10, 50, 100, 200]:
        if rnd <= len(train_r2_staged):
            print(f"    Round {rnd:3d}: Train R²={train_r2_staged[rnd-1]:.4f}, "
                  f"Test R²={test_r2_staged[rnd-1]:.4f}")
    
    plot_staged_performance(rounds, train_r2_staged, test_r2_staged,
                            "R²", "Staged Performance — Regression",
                            "03_staged_r2_regression.png")
    
    # Feature importance
    plot_feature_importance(feature_names_reg, sklearn_gbr.feature_importances_,
                           "GB Regressor — Feature Importance",
                           "04_feature_importance_reg.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # LEARNING RATE vs N_ESTIMATORS Tradeoff (Regression)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("LEARNING RATE vs N_ESTIMATORS TRADEOFF")
    print("=" * 70)
    
    lr_results = []
    for lr in [0.01, 0.05, 0.1, 0.2, 0.5]:
        gb = GradientBoostingRegressorScratch(
            n_estimators=300, learning_rate=lr, max_depth=3
        )
        gb.fit(X_train_r, y_train_r)
        test_r2 = gb.score(X_test_r, y_test_r)
        lr_results.append((lr, gb.train_losses, test_r2))
        print(f"  lr={lr:.2f}: Test R²={test_r2:.4f}, Final MSE={gb.train_losses[-1]:.4f}")
    
    plot_learning_rate_comparison(lr_results, "05_learning_rate_comparison.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # EARLY STOPPING (Regression)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EARLY STOPPING DEMONSTRATION")
    print("=" * 70)
    
    gb_es = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.1, max_depth=3, random_state=42
    )
    gb_es.fit(X_tr_r, y_tr_r)
    
    staged_tr = list(gb_es.staged_predict(X_tr_r))
    staged_val = list(gb_es.staged_predict(X_val_r))
    
    rounds_es = list(range(1, 501))
    train_r2_es = [r2_score(y_tr_r, p) for p in staged_tr]
    val_r2_es = [r2_score(y_val_r, p) for p in staged_val]
    
    best_n = rounds_es[np.argmax(val_r2_es)]
    print(f"\n  Best validation R²: {max(val_r2_es):.4f} at round {best_n}")
    print(f"  R² at round 500:    {val_r2_es[-1]:.4f}")
    print(f"  Overfitting after round {best_n}!")
    
    plot_early_stopping(rounds_es, train_r2_es, val_r2_es, best_n, "R²",
                        "06_early_stopping.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # SUBSAMPLE IMPACT (Stochastic GB)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STOCHASTIC GRADIENT BOOSTING — SUBSAMPLE IMPACT")
    print("=" * 70)
    
    subsample_vals = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    sub_scores = []
    
    for ss in subsample_vals:
        gb_ss = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=3,
            subsample=ss, random_state=42
        )
        gb_ss.fit(X_train_r, y_train_r)
        score = gb_ss.score(X_test_r, y_test_r)
        sub_scores.append(score)
        print(f"  subsample={ss:.1f}: Test R²={score:.4f}")
    
    plot_subsample_comparison(subsample_vals, sub_scores, "07_subsample_impact.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # PART B: CLASSIFICATION (Breast Cancer)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART B: GRADIENT BOOSTING CLASSIFICATION — Breast Cancer")
    print("=" * 70)
    
    cancer = load_breast_cancer()
    X_cls, y_cls = cancer.data, cancer.target
    feature_names_cls = list(cancer.feature_names)
    class_names_cls = list(cancer.target_names)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    print(f"Shape: {X_cls.shape}")
    print(f"Classes: {class_names_cls}")
    print(f"Train: {X_train_c.shape}, Test: {X_test_c.shape}\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL B1: From-Scratch GB Classifier
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("MODEL B1: Gradient Boosting Classifier — From Scratch")
    print("=" * 70)
    
    scratch_gbc = GradientBoostingClassifierScratch(
        n_estimators=200, learning_rate=0.1, max_depth=3
    )
    scratch_gbc.fit(X_train_c, y_train_c)
    
    y_pred_sc = scratch_gbc.predict(X_test_c)
    acc_sc = accuracy_score(y_test_c, y_pred_sc)
    
    print(f"\n  Test Accuracy: {acc_sc:.6f}")
    print(f"  Final train log loss: {scratch_gbc.train_losses[-1]:.6f}")
    
    plot_loss_convergence(scratch_gbc.train_losses,
                          "Scratch GB Classifier — Log Loss", "08_loss_convergence_cls.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL B2: Scikit-learn GB Classifier
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL B2: Scikit-learn GradientBoostingClassifier")
    print("=" * 70)
    
    sklearn_gbc = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
    )
    sklearn_gbc.fit(X_train_c, y_train_c)
    
    y_pred_skc = sklearn_gbc.predict(X_test_c)
    acc_skc = accuracy_score(y_test_c, y_pred_skc)
    
    print(f"\n  Test Accuracy: {acc_skc:.6f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_c, y_pred_skc, target_names=class_names_cls))
    
    plot_confusion_matrix(y_test_c, y_pred_skc, class_names_cls,
                          "GB Classifier — Confusion Matrix",
                          "09_confusion_matrix_cls.png")
    
    # Staged accuracy (classification)
    staged_train_c = list(sklearn_gbc.staged_predict(X_train_c))
    staged_test_c = list(sklearn_gbc.staged_predict(X_test_c))
    
    rounds_c = list(range(1, 201))
    train_acc_staged = [accuracy_score(y_train_c, p) for p in staged_train_c]
    test_acc_staged = [accuracy_score(y_test_c, p) for p in staged_test_c]
    
    plot_staged_performance(rounds_c, train_acc_staged, test_acc_staged,
                            "Accuracy", "Staged Performance — Classification",
                            "10_staged_accuracy_cls.png")
    
    # Feature importance
    plot_feature_importance(feature_names_cls, sklearn_gbc.feature_importances_,
                           "GB Classifier — Feature Importance",
                           "11_feature_importance_cls.png")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL COMPARISON: Single Tree vs RF vs Gradient Boosting
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARISON: Single Tree vs Random Forest vs Gradient Boosting")
    print("=" * 70)
    
    # Classification
    dt_cls = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_cls.fit(X_train_c, y_train_c)
    acc_dt = dt_cls.score(X_test_c, y_test_c)
    
    rf_cls = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_cls.fit(X_train_c, y_train_c)
    acc_rf = rf_cls.score(X_test_c, y_test_c)
    
    print(f"\n  CLASSIFICATION (Breast Cancer):")
    print(f"    Single Tree:      {acc_dt:.4f}")
    print(f"    Random Forest:    {acc_rf:.4f}")
    print(f"    Gradient Boosting: {acc_skc:.4f}")
    
    plot_model_comparison_bar(
        ["Single Tree", "Random Forest", "Scratch GB", "Sklearn GB"],
        [acc_dt, acc_rf, acc_sc, acc_skc],
        "Accuracy (Classification)",
        "12_model_comparison_cls.png"
    )
    
    # Regression
    dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_reg.fit(X_train_r, y_train_r)
    r2_dt = dt_reg.score(X_test_r, y_test_r)
    
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_r, y_train_r)
    r2_rf = rf_reg.score(X_test_r, y_test_r)
    
    print(f"\n  REGRESSION (California Housing):")
    print(f"    Single Tree:      R²={r2_dt:.4f}")
    print(f"    Random Forest:    R²={r2_rf:.4f}")
    print(f"    Scratch GB:       R²={r2_sr:.4f}")
    print(f"    Sklearn GB:       R²={r2_skr:.4f}")
    
    plot_model_comparison_bar(
        ["Single Tree", "Random Forest", "Scratch GB", "Sklearn GB"],
        [r2_dt, r2_rf, r2_sr, r2_skr],
        "R² Score (Regression)",
        "13_model_comparison_reg.png"
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1.  SEQUENTIAL BOOSTING     — Each tree corrects previous errors
    2.  PSEUDO-RESIDUALS        — Negative gradient of the loss function
    3.  MSE LOSS (REGRESSION)   — Residuals = y - F(x)
    4.  LOG LOSS (CLASSIFICATION) — Residuals = y - sigmoid(F(x))
    5.  LEARNING RATE           — Shrinkage controls each tree's contribution
    6.  LR vs N_ESTIMATORS      — Lower lr + more trees = better generalization
    7.  STAGED PREDICTIONS      — Watch model improve round by round
    8.  EARLY STOPPING          — Stop when validation score plateaus
    9.  STOCHASTIC GB           — Subsample adds randomness (regularization)
    10. RESIDUAL EVOLUTION      — Residuals shrink over boosting rounds
    11. FEATURE IMPORTANCE      — Based on impurity reduction across all trees
    12. DT vs RF vs GB          — Gradient Boosting often wins on accuracy
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
