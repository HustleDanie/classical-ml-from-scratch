"""
=============================================================================
01 - LINEAR REGRESSION: Complete Implementation
=============================================================================
Covers:
  1. From-scratch implementation using Gradient Descent
  2. Normal Equation (closed-form solution)
  3. Scikit-learn LinearRegression comparison
  4. Regularization (Ridge, Lasso, ElasticNet)
  5. Polynomial Regression (extending linear to nonlinear)
  6. Feature scaling impact demonstration
  7. Full evaluation metrics & visualizations

Dataset: California Housing (sklearn built-in)
  - 20,640 samples, 8 features
  - Target: median house value (in $100k)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# scikit-learn imports
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# SECTION 1: Load & Explore the Dataset
# =============================================================================
def load_data():
    """Load California Housing dataset and return DataFrame + target."""
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame  # includes target column 'MedHouseVal'
    print("=" * 70)
    print("DATASET: California Housing")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"\nFeature names: {housing.feature_names}")
    print(f"Target: MedHouseVal (median house value in $100k)\n")
    print(df.describe().round(3))
    print()
    return housing.data, housing.target, housing.feature_names


# =============================================================================
# SECTION 2: From-Scratch Linear Regression (Gradient Descent)
# =============================================================================
class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using batch gradient descent.
    
    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent (alpha).
    n_iterations : int
        Number of gradient descent iterations.
    
    Attributes
    ----------
    weights : np.ndarray
        Learned weight vector (n_features,).
    bias : float
        Learned bias (intercept) term.
    cost_history : list
        MSE at each iteration (for convergence plotting).
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """
        Train the model using batch gradient descent.
        
        The update rules:
            w := w - alpha * (1/n) * X^T . (X.w + b - y)
            b := b - alpha * (1/n) * sum(X.w + b - y)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []
        
        for i in range(self.n_iterations):
            # Forward pass: predictions
            y_pred = X.dot(self.weights) + self.bias
            
            # Compute error
            error = y_pred - y  # shape (n_samples,)
            
            # Compute gradients
            dw = (1 / n_samples) * X.T.dot(error)     # (n_features,)
            db = (1 / n_samples) * np.sum(error)       # scalar
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Record cost (MSE)
            cost = np.mean(error ** 2)
            self.cost_history.append(cost)
        
        return self
    
    def predict(self, X):
        """Generate predictions: y_hat = X.w + b"""
        return X.dot(self.weights) + self.bias
    
    def score(self, X, y):
        """Return R² score."""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


# =============================================================================
# SECTION 3: Normal Equation (Closed-Form Solution)
# =============================================================================
class LinearRegressionNormalEq:
    """
    Linear Regression using the Normal Equation.
    
    w = (X^T X)^(-1) X^T y
    
    No iterations, no learning rate — direct analytical solution.
    """
    
    def __init__(self):
        self.weights = None  # includes bias as weights[0]
    
    def fit(self, X, y):
        """
        Compute weights using the normal equation.
        We prepend a column of 1s to X for the bias term.
        """
        # Add bias column (column of ones)
        X_b = np.c_[np.ones(X.shape[0]), X]
        
        # Normal equation: w = (X^T X)^-1 X^T y
        self.weights = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b.dot(self.weights)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


# =============================================================================
# SECTION 4: Evaluation Utilities
# =============================================================================
def evaluate_model(name, y_true, y_pred):
    """Compute and print all regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  MSE  : {mse:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    print(f"  R²   : {r2:.6f}")
    
    return {"name": name, "MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}


# =============================================================================
# SECTION 5: Visualization Functions
# =============================================================================
def plot_cost_convergence(cost_history, title="Gradient Descent Convergence"):
    """Plot MSE over gradient descent iterations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full history
    axes[0].plot(cost_history, color="steelblue", linewidth=1.5)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("MSE (Cost)")
    axes[0].set_title(f"{title} — Full")
    
    # Last 80% (zoom into convergence region)
    start = len(cost_history) // 5
    axes[1].plot(range(start, len(cost_history)), cost_history[start:],
                 color="coral", linewidth=1.5)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("MSE (Cost)")
    axes[1].set_title(f"{title} — Zoomed (last 80%)")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_cost_convergence.png", dpi=150)
    plt.close()
    print("  [Saved] plots/01_cost_convergence.png")


def plot_predictions_vs_actual(y_true, predictions_dict):
    """Scatter plot of predicted vs actual for multiple models."""
    n_models = len(predictions_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(name)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_predictions_vs_actual.png", dpi=150)
    plt.close()
    print("  [Saved] plots/02_predictions_vs_actual.png")


def plot_residuals(y_true, y_pred, name="Model"):
    """Plot residuals distribution and residuals vs predicted."""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color="steelblue")
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title(f"{name} — Residuals vs Predicted")
    
    # Residuals Distribution
    axes[1].hist(residuals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Residual Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{name} — Residual Distribution")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_residuals.png", dpi=150)
    plt.close()
    print("  [Saved] plots/03_residuals.png")


def plot_feature_importance(feature_names, coefficients):
    """Bar chart of feature coefficients (importance)."""
    sorted_idx = np.argsort(np.abs(coefficients))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["coral" if c < 0 else "steelblue" for c in coefficients[sorted_idx]]
    ax.barh(np.array(feature_names)[sorted_idx], coefficients[sorted_idx], color=colors)
    ax.set_xlabel("Coefficient Value")
    ax.set_title("Feature Coefficients (Importance)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_feature_importance.png", dpi=150)
    plt.close()
    print("  [Saved] plots/04_feature_importance.png")


def plot_regularization_comparison(results_list):
    """Bar chart comparing R² across all models."""
    names = [r["name"] for r in results_list]
    r2_scores = [r["R²"] for r in results_list]
    rmse_scores = [r["RMSE"] for r in results_list]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² comparison
    bars1 = axes[0].bar(names, r2_scores, color="steelblue", edgecolor="white")
    axes[0].set_ylabel("R² Score")
    axes[0].set_title("Model Comparison — R²")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=30)
    for bar, val in zip(bars1, r2_scores):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    
    # RMSE comparison
    bars2 = axes[1].bar(names, rmse_scores, color="coral", edgecolor="white")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Model Comparison — RMSE")
    axes[1].tick_params(axis="x", rotation=30)
    for bar, val in zip(bars2, rmse_scores):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_model_comparison.png", dpi=150)
    plt.close()
    print("  [Saved] plots/05_model_comparison.png")


def plot_scaling_impact(r2_unscaled, r2_scaled):
    """Show the impact of feature scaling on gradient descent performance."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Without Scaling", "With Scaling"],
                  [r2_unscaled, r2_scaled],
                  color=["coral", "steelblue"], edgecolor="white", width=0.5)
    ax.set_ylabel("R² Score")
    ax.set_title("Impact of Feature Scaling on Gradient Descent")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, [r2_unscaled, r2_scaled]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_scaling_impact.png", dpi=150)
    plt.close()
    print("  [Saved] plots/06_scaling_impact.png")


def plot_polynomial_comparison(degrees, train_scores, test_scores):
    """Show train vs test R² for different polynomial degrees (overfitting demo)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(degrees, train_scores, "o-", color="steelblue", linewidth=2, label="Train R²")
    ax.plot(degrees, test_scores, "o-", color="coral", linewidth=2, label="Test R²")
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("R² Score")
    ax.set_title("Polynomial Regression — Bias-Variance Tradeoff")
    ax.legend()
    ax.set_xticks(degrees)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_polynomial_comparison.png", dpi=150)
    plt.close()
    print("  [Saved] plots/07_polynomial_comparison.png")


# =============================================================================
# SECTION 6: Main Execution
# =============================================================================
def main():
    # ── 1. Load Data ─────────────────────────────────────────────────────────
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")
    
    # ── 2. Feature Scaling ───────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A: From-Scratch Gradient Descent (with scaling)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A: Linear Regression from Scratch (Gradient Descent)")
    print("=" * 70)
    
    scratch_model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
    scratch_model.fit(X_train_scaled, y_train.values)
    y_pred_scratch = scratch_model.predict(X_test_scaled)
    
    res_scratch = evaluate_model("Scratch (GD)", y_test.values, y_pred_scratch)
    results.append(res_scratch)
    
    print(f"\n  Learned weights: {scratch_model.weights.round(4)}")
    print(f"  Learned bias:    {scratch_model.bias:.4f}")
    print(f"  Final cost:      {scratch_model.cost_history[-1]:.6f}")
    
    # Plot convergence
    plot_cost_convergence(scratch_model.cost_history)
    
    # ── Demonstrate scaling impact ───────────────────────────────────────────
    print("\n  --- Scaling Impact Demo ---")
    scratch_unscaled = LinearRegressionScratch(learning_rate=0.0000001, n_iterations=1000)
    scratch_unscaled.fit(X_train.values, y_train.values)
    r2_unscaled = scratch_unscaled.score(X_test.values, y_test.values)
    r2_scaled = scratch_model.score(X_test_scaled, y_test.values)
    print(f"  R² WITHOUT scaling: {r2_unscaled:.6f}")
    print(f"  R² WITH scaling:    {r2_scaled:.6f}")
    plot_scaling_impact(r2_unscaled, r2_scaled)
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL B: Normal Equation
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL B: Linear Regression — Normal Equation")
    print("=" * 70)
    
    normal_model = LinearRegressionNormalEq()
    normal_model.fit(X_train.values, y_train.values)
    y_pred_normal = normal_model.predict(X_test.values)
    
    res_normal = evaluate_model("Normal Equation", y_test.values, y_pred_normal)
    results.append(res_normal)
    
    print(f"\n  Weights (including bias): {normal_model.weights.round(4)}")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL C: Scikit-learn LinearRegression
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL C: Scikit-learn LinearRegression")
    print("=" * 70)
    
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)
    
    res_sklearn = evaluate_model("Sklearn LR", y_test.values, y_pred_sklearn)
    results.append(res_sklearn)
    
    print(f"\n  Coefficients: {sklearn_model.coef_.round(4)}")
    print(f"  Intercept:    {sklearn_model.intercept_:.4f}")
    
    # ── Feature importance ───────────────────────────────────────────────────
    plot_feature_importance(feature_names, sklearn_model.coef_)
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL D: Ridge Regression (L2 Regularization)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL D: Ridge Regression (L2 Regularization)")
    print("=" * 70)
    
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge_model.predict(X_test_scaled)
    
    res_ridge = evaluate_model("Ridge (L2)", y_test.values, y_pred_ridge)
    results.append(res_ridge)
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL E: Lasso Regression (L1 Regularization)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL E: Lasso Regression (L1 Regularization)")
    print("=" * 70)
    
    lasso_model = Lasso(alpha=0.01)
    lasso_model.fit(X_train_scaled, y_train)
    y_pred_lasso = lasso_model.predict(X_test_scaled)
    
    res_lasso = evaluate_model("Lasso (L1)", y_test.values, y_pred_lasso)
    results.append(res_lasso)
    
    # Show which features Lasso zeroed out
    print(f"\n  Lasso coefficients: {lasso_model.coef_.round(4)}")
    zero_features = [f for f, c in zip(feature_names, lasso_model.coef_) if abs(c) < 1e-6]
    if zero_features:
        print(f"  Features eliminated by Lasso: {zero_features}")
    else:
        print("  Lasso kept all features (alpha may be too small to eliminate any).")
    
    # ══════════════════════════════════════════════════════════════════════════
    # MODEL F: ElasticNet (L1 + L2)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL F: ElasticNet (L1 + L2 Regularization)")
    print("=" * 70)
    
    enet_model = ElasticNet(alpha=0.01, l1_ratio=0.5)
    enet_model.fit(X_train_scaled, y_train)
    y_pred_enet = enet_model.predict(X_test_scaled)
    
    res_enet = evaluate_model("ElasticNet", y_test.values, y_pred_enet)
    results.append(res_enet)
    
    # ══════════════════════════════════════════════════════════════════════════
    # Predictions vs Actual Plot (top 3 models)
    # ══════════════════════════════════════════════════════════════════════════
    plot_predictions_vs_actual(y_test.values, {
        "Scratch (GD)": y_pred_scratch,
        "Sklearn LR": y_pred_sklearn,
        "Ridge (L2)": y_pred_ridge,
    })
    
    # Residuals for sklearn model
    plot_residuals(y_test.values, y_pred_sklearn, name="Sklearn LR")
    
    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: Polynomial Regression (Overfitting Demo)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("POLYNOMIAL REGRESSION — Bias-Variance Tradeoff")
    print("=" * 70)
    
    # Use only 2 features for polynomial demo (to keep it tractable)
    X_poly_train = X_train_scaled[:, :2]
    X_poly_test = X_test_scaled[:, :2]
    
    degrees = [1, 2, 3, 4, 5]
    train_scores = []
    test_scores = []
    
    for d in degrees:
        poly = PolynomialFeatures(degree=d, include_bias=False)
        X_tr_poly = poly.fit_transform(X_poly_train)
        X_te_poly = poly.transform(X_poly_test)
        
        lr = LinearRegression()
        lr.fit(X_tr_poly, y_train)
        
        train_r2 = lr.score(X_tr_poly, y_train)
        test_r2 = lr.score(X_te_poly, y_test)
        train_scores.append(train_r2)
        test_scores.append(test_r2)
        
        print(f"  Degree {d}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f}, "
              f"n_features={X_tr_poly.shape[1]}")
    
    plot_polynomial_comparison(degrees, train_scores, test_scores)
    
    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY: Compare All Models
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FINAL COMPARISON — All Models")
    print("=" * 70)
    
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values("R²", ascending=False).reset_index(drop=True)
    print(f"\n{summary_df.to_string(index=False)}\n")
    
    plot_regularization_comparison(results)
    
    # ══════════════════════════════════════════════════════════════════════════
    # LEARNING SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1. GRADIENT DESCENT      — Iterative optimization with learning rate & convergence
    2. NORMAL EQUATION        — Exact closed-form solution (no hyperparams)
    3. FEATURE SCALING        — Critical for gradient descent convergence
    4. REGULARIZATION         — Ridge (L2), Lasso (L1), ElasticNet (L1+L2)
    5. FEATURE SELECTION      — Lasso can zero out irrelevant features
    6. POLYNOMIAL REGRESSION  — Extending linear models to capture nonlinearity
    7. BIAS-VARIANCE TRADEOFF — Higher degree → overfit (train ↑, test ↓)
    8. EVALUATION METRICS     — MSE, RMSE, MAE, R² and when to use each
    9. RESIDUAL ANALYSIS      — Checking model assumptions visually
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
