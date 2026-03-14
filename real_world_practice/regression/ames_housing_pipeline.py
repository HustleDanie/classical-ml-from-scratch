"""
Real-World Regression Pipeline -- Ames Housing Price Prediction
================================================================

Dataset: Ames Housing (via sklearn/openml) -- 1,460 houses, 80+ features
Task:    Predict house sale price (continuous regression)

Real-world skills practiced:
  1.  Data exploration on a large-feature dataset (80 features!)
  2.  Data cleaning (missing values in many columns, mixed types)
  3.  Feature engineering (polynomial, interaction, domain-driven)
  4.  Handling skewed target variable (log transformation)
  5.  Encoding ordinal + nominal categoricals properly
  6.  Building robust preprocessing pipelines
  7.  Training & comparing ALL regression models
  8.  Unsupervised methods: PCA feature reduction, K-Means segments
  9.  Feature selection (Mutual Information, RFE)
  10. Hyperparameter tuning with cross-validation
  11. Stacking ensemble (meta-learner combining multiple regressors)
  12. SHAP explainability (why the model predicts what it predicts)
  13. Learning curves for overfitting detection
  14. Residual analysis to diagnose model quality
  15. Model persistence & inference demo

Dataset challenges:
  - 80 features (numeric + categorical)
  - Heavy missing values in multiple columns
  - Ordinal features (quality ratings like 'Ex', 'Gd', 'TA', 'Fa', 'Po')
  - Skewed target (SalePrice) -- needs log transform
  - Multicollinearity (GarageArea ~ GarageCars, TotalBsmtSF ~ 1stFlrSF)
  - Outliers in GrLivArea, LotArea
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import time

from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.feature_selection import (
    mutual_info_regression, RFE, SelectKBest
)
from sklearn.ensemble import StackingRegressor

# ---- All regression-capable classical ML algorithms ----
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
import lightgbm as lgb

# Unsupervised
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

import joblib

warnings.filterwarnings("ignore")

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def print_section(title, level=1):
    if level == 1:
        print("\n\n" + "=" * 70)
        print(title)
        print("=" * 70)
    else:
        print(f"\n  --- {title} ---")


def print_reg_metrics(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"    RMSE: {rmse:>12,.0f}  |  MAE:  {mae:>12,.0f}")
    print(f"    R2:   {r2:>12.4f}  |  MSE:  {mse:>12,.0f}")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def print_log_metrics(name, y_true_log, y_pred_log):
    """Print metrics in log-space and convert back to dollar-space."""
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2_log = r2_score(y_true_log, y_pred_log)

    # Convert back to actual prices
    y_true_actual = np.expm1(y_true_log)
    y_pred_actual = np.expm1(y_pred_log)
    rmse_dollar = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
    mae_dollar = mean_absolute_error(y_true_actual, y_pred_actual)
    r2_dollar = r2_score(y_true_actual, y_pred_actual)

    print(f"    Log-space   -- RMSE: {rmse_log:.4f}, R2: {r2_log:.4f}")
    print(f"    Dollar-space-- RMSE: ${rmse_dollar:>10,.0f}, MAE: ${mae_dollar:>10,.0f}, R2: {r2_dollar:.4f}")
    return {"RMSE_log": rmse_log, "R2_log": r2_log,
            "RMSE_$": rmse_dollar, "MAE_$": mae_dollar, "R2_$": r2_dollar}


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    print("=" * 70)
    print("REAL-WORLD REGRESSION PIPELINE")
    print("Dataset: Ames Housing Price Prediction")
    print("=" * 70)

    # ==================================================================
    # STEP 1: LOAD & EXPLORE RAW DATA
    # ==================================================================
    print_section("STEP 1: LOAD & EXPLORE RAW DATA")

    from sklearn.datasets import fetch_openml
    housing = fetch_openml(name="house_prices", as_frame=True, parser="auto")
    df_raw = housing.frame.copy()

    print(f"\n  Shape: {df_raw.shape} ({df_raw.shape[0]} houses, {df_raw.shape[1]} features)")

    # Separate numeric and categorical
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"  Numeric features: {len(num_cols)}")
    print(f"  Categorical features: {len(cat_cols)}")

    # Missing values
    missing = df_raw.isnull().sum()
    missing_pct = (missing / len(df_raw) * 100).round(1)
    missing_df = pd.DataFrame({"Count": missing, "Pct": missing_pct})
    missing_df = missing_df[missing_df["Count"] > 0].sort_values("Pct", ascending=False)
    print(f"\n  Columns with missing values ({len(missing_df)}):")
    for col, row in missing_df.head(15).iterrows():
        dtype = "num" if col in num_cols else "cat"
        print(f"    {col:20s} {int(row['Count']):4d} ({row['Pct']:5.1f}%) [{dtype}]")
    if len(missing_df) > 15:
        print(f"    ... and {len(missing_df) - 15} more columns with missing values")

    # Target variable
    target = "SalePrice"
    print(f"\n  Target: {target}")
    print(f"    Mean:   ${df_raw[target].mean():>12,.0f}")
    print(f"    Median: ${df_raw[target].median():>12,.0f}")
    print(f"    Std:    ${df_raw[target].std():>12,.0f}")
    print(f"    Range:  ${df_raw[target].min():>12,.0f} - ${df_raw[target].max():>12,.0f}")
    skewness = df_raw[target].skew()
    print(f"    Skewness: {skewness:.4f} {'(heavy right skew -- needs log transform!)' if skewness > 1 else ''}")

    # Plot: data overview
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    df_raw[target].hist(bins=50, ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("SalePrice Distribution (Original)")
    axes[0].set_xlabel("Price ($)")
    np.log1p(df_raw[target]).hist(bins=50, ax=axes[1], color="#2ecc71", edgecolor="black")
    axes[1].set_title("SalePrice Distribution (Log Transformed)")
    axes[1].set_xlabel("log(Price + 1)")
    missing_counts = df_raw.isnull().sum().sort_values(ascending=False)[:20]
    missing_counts[missing_counts > 0].plot(kind="barh", ax=axes[2], color="#e74c3c")
    axes[2].set_title("Top Missing Value Columns")
    axes[2].set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_data_overview.png", dpi=150)
    plt.close()
    print("  [Saved] plots/01_data_overview.png")

    # ==================================================================
    # STEP 2: DATA CLEANING
    # ==================================================================
    print_section("STEP 2: DATA CLEANING")

    df = df_raw.copy()

    # 2a. Drop columns with too many missing values (>40%)
    high_missing = [c for c in df.columns if df[c].isnull().mean() > 0.40]
    df = df.drop(columns=high_missing)
    print(f"\n  [Drop] Columns with >40% missing ({len(high_missing)}): {high_missing}")

    # 2b. Drop ID column
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
        print(f"  [Drop] Id column (not a feature)")

    # 2c. Handle missing values context-aware
    # For many housing features, NaN means "none" (no garage, no basement, etc.)
    fill_none_cols = [c for c in df.columns if df[c].dtype == "object" and df[c].isnull().sum() > 0]
    for col in fill_none_cols:
        df[col] = df[col].fillna("None")
    print(f"  [Fill] {len(fill_none_cols)} categorical columns: NaN -> 'None' (means feature absent)")

    fill_zero_cols = ["MasVnrArea", "GarageYrBlt"]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    print(f"  [Fill] Numeric NaN -> 0: {fill_zero_cols}")

    # Remaining numeric nulls: median imputation
    num_null_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                     if df[c].isnull().sum() > 0 and c != target]
    for col in num_null_cols:
        df[col] = df[col].fillna(df[col].median())
    print(f"  [Fill] Remaining numeric NaN -> median: {num_null_cols}")

    # 2d. Outlier handling
    print(f"\n  [Outliers]")
    # Known outliers: GrLivArea > 4000 with low price
    outlier_mask = (df["GrLivArea"] > 4000) & (df[target] < 300000)
    n_outliers = outlier_mask.sum()
    df = df[~outlier_mask]
    print(f"    Removed {n_outliers} extreme outliers (GrLivArea > 4000 sqft + low price)")

    # Cap LotArea at a reasonable threshold
    lot_upper = df["LotArea"].quantile(0.99)
    n_capped = (df["LotArea"] > lot_upper).sum()
    df.loc[df["LotArea"] > lot_upper, "LotArea"] = lot_upper
    print(f"    Capped {n_capped} LotArea outliers at {lot_upper:,.0f} sqft")

    print(f"\n  Cleaned shape: {df.shape}")
    print(f"  Remaining nulls (excl target): {df.drop(columns=[target]).isnull().sum().sum()}")

    # ==================================================================
    # STEP 3: FEATURE ENGINEERING
    # ==================================================================
    print_section("STEP 3: FEATURE ENGINEERING")

    # Total square footage
    df["TotalSF"] = df.get("TotalBsmtSF", 0) + df.get("1stFlrSF", 0) + df.get("2ndFlrSF", 0)
    print(f"\n  [+] TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF")

    # Total bathrooms
    df["TotalBath"] = (df.get("FullBath", 0) + df.get("HalfBath", 0) * 0.5 +
                       df.get("BsmtFullBath", 0) + df.get("BsmtHalfBath", 0) * 0.5)
    print(f"  [+] TotalBath = FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath")

    # Age features
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    print(f"  [+] HouseAge = YrSold - YearBuilt")
    print(f"  [+] RemodAge = YrSold - YearRemodAdd")

    # Has features (binary)
    df["HasGarage"] = (df.get("GarageArea", 0) > 0).astype(int)
    df["HasBsmt"] = (df.get("TotalBsmtSF", 0) > 0).astype(int)
    df["HasPool"] = (df.get("PoolArea", 0) > 0).astype(int) if "PoolArea" in df.columns else 0
    df["HasFireplace"] = (df.get("Fireplaces", 0) > 0).astype(int)
    print(f"  [+] Binary flags: HasGarage, HasBsmt, HasPool, HasFireplace")

    # Quality score (ordinal encoding for quality features)
    quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    quality_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
                    "HeatingQC", "KitchenQual", "GarageQual", "GarageCond"]
    for col in quality_cols:
        if col in df.columns:
            df[col + "_Num"] = df[col].map(quality_map).fillna(0).astype(int)
    print(f"  [+] Quality columns mapped to ordinal numbers: {len(quality_cols)} features")

    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]
    print(f"  [+] OverallScore = OverallQual * OverallCond")

    print(f"\n  Engineered features: {df.shape[1]} total columns")

    # ==================================================================
    # STEP 4: LOG-TRANSFORM TARGET
    # ==================================================================
    print_section("STEP 4: LOG-TRANSFORM TARGET VARIABLE")

    print(f"\n  Original SalePrice:")
    print(f"    Skewness: {df[target].skew():.4f}")
    print(f"    Range: ${df[target].min():,.0f} - ${df[target].max():,.0f}")

    df["SalePrice_Log"] = np.log1p(df[target])
    print(f"\n  After log1p transform:")
    print(f"    Skewness: {df['SalePrice_Log'].skew():.4f}")
    print(f"    Range: {df['SalePrice_Log'].min():.4f} - {df['SalePrice_Log'].max():.4f}")
    print(f"    --> Much closer to normal distribution!")

    # ==================================================================
    # STEP 5: EDA
    # ==================================================================
    print_section("STEP 5: EXPLORATORY DATA ANALYSIS (EDA)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top correlated features with SalePrice
    numeric_df = df.select_dtypes(include=[np.number])
    corr_with_price = numeric_df.corr()["SalePrice_Log"].drop(
        ["SalePrice_Log", target]).abs().sort_values(ascending=False)
    top_features = corr_with_price.head(10)

    top_features.plot(kind="barh", ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Top 10 Features Correlated with Price")
    axes[0, 0].set_xlabel("Absolute Correlation")

    # Scatter: top feature vs price
    top_feat = corr_with_price.index[0]
    axes[0, 1].scatter(df[top_feat], df[target], alpha=0.3, s=10, color="steelblue")
    axes[0, 1].set_xlabel(top_feat)
    axes[0, 1].set_ylabel("SalePrice ($)")
    axes[0, 1].set_title(f"{top_feat} vs SalePrice (r={numeric_df[top_feat].corr(df[target]):.3f})")

    # GrLivArea vs Price
    axes[0, 2].scatter(df["GrLivArea"], df[target], alpha=0.3, s=10, color="#e74c3c")
    axes[0, 2].set_xlabel("GrLivArea (sqft)")
    axes[0, 2].set_ylabel("SalePrice ($)")
    axes[0, 2].set_title(f"GrLivArea vs SalePrice")

    # SalePrice by OverallQual
    df.boxplot(column=target, by="OverallQual", ax=axes[1, 0])
    axes[1, 0].set_title("SalePrice by Overall Quality")
    axes[1, 0].set_xlabel("Overall Quality (1-10)")
    axes[1, 0].set_ylabel("SalePrice ($)")
    plt.sca(axes[1, 0])
    plt.xticks(rotation=0)

    # Heatmap of top features
    top_cols = list(corr_with_price.head(6).index) + ["SalePrice_Log"]
    corr_matrix = numeric_df[top_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=axes[1, 1], square=True)
    axes[1, 1].set_title("Correlation Heatmap (Top Features)")

    # Year built vs price
    axes[1, 2].scatter(df["YearBuilt"], df[target], alpha=0.3, s=10, color="#9b59b6")
    axes[1, 2].set_xlabel("Year Built")
    axes[1, 2].set_ylabel("SalePrice ($)")
    axes[1, 2].set_title("Year Built vs SalePrice")

    plt.suptitle("Ames Housing -- EDA", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_eda.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/02_eda.png")

    print(f"\n  Top 10 features correlated with SalePrice (log):")
    for feat, corr in top_features.items():
        print(f"    {feat:25s}  r = {corr:.4f}")

    # ==================================================================
    # STEP 6: PREPARE FEATURES & PREPROCESSING PIPELINE
    # ==================================================================
    print_section("STEP 6: PREPROCESSING PIPELINE")

    # Use log-transformed target
    y = df["SalePrice_Log"]
    drop_cols = [target, "SalePrice_Log"]
    X = df.drop(columns=drop_cols)

    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"\n  Features: {X.shape[1]} total")
    print(f"    Numeric:     {len(numeric_features)}")
    print(f"    Categorical: {len(categorical_features)}")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=10))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    print(f"  Pipeline: Numeric  -> Imputer(median) -> StandardScaler")
    print(f"           Categorical -> Imputer('None') -> OneHotEncoder(max_cat=10)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Train: {X_train.shape},  Test: {X_test.shape}")

    # Convert y back to dollar amounts for display
    y_test_dollar = np.expm1(y_test)

    # ==================================================================
    # STEP 7: TRAIN & COMPARE ALL REGRESSION MODELS
    # ==================================================================
    print_section("STEP 7: TRAIN & COMPARE ALL 10 REGRESSORS (5-fold CV)")

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (L2)": Ridge(alpha=1.0, random_state=42),
        "Lasso (L1)": Lasso(alpha=0.001, random_state=42),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "SVR (RBF)": SVR(kernel="rbf", C=10),
        "KNN (k=5)": KNeighborsRegressor(n_neighbors=5),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        t0 = time.time()

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                     scoring="neg_root_mean_squared_error")
        pipe.fit(X_train, y_train)
        y_pred_log = pipe.predict(X_test)
        elapsed = time.time() - t0

        y_pred_dollar = np.expm1(y_pred_log)
        rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
        r2_log = r2_score(y_test, y_pred_log)
        rmse_dollar = np.sqrt(mean_squared_error(y_test_dollar, y_pred_dollar))
        mae_dollar = mean_absolute_error(y_test_dollar, y_pred_dollar)
        r2_dollar = r2_score(y_test_dollar, y_pred_dollar)

        results.append({
            "Model": name,
            "CV RMSE (log)": -cv_scores.mean(),
            "CV Std": cv_scores.std(),
            "RMSE_log": rmse_log,
            "R2_log": r2_log,
            "RMSE_$": rmse_dollar,
            "MAE_$": mae_dollar,
            "R2_$": r2_dollar,
            "Time": elapsed,
        })
        pipelines[name] = pipe

        print(f"\n  {name}:")
        print(f"    5-Fold CV RMSE (log): {-cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"    Test -- RMSE_log: {rmse_log:.4f}, R2: {r2_log:.4f}")
        print(f"    Test -- RMSE: ${rmse_dollar:>10,.0f}, MAE: ${mae_dollar:>10,.0f}")
        print(f"    Time: {elapsed:.3f}s")

    # Results table
    results_df = pd.DataFrame(results).sort_values("R2_$", ascending=False)
    print(f"\n{'='*70}")
    print("  MODEL COMPARISON (sorted by R2)")
    print(f"{'='*70}")
    display_cols = ["Model", "CV RMSE (log)", "RMSE_$", "MAE_$", "R2_$", "Time"]
    for _, row in results_df.iterrows():
        print(f"  {row['Model']:<22s}  CV_RMSE={row['CV RMSE (log)']:.4f}  "
              f"RMSE=${row['RMSE_$']:>10,.0f}  MAE=${row['MAE_$']:>10,.0f}  "
              f"R2={row['R2_$']:.4f}  {row['Time']:.1f}s")

    # Plot: model comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sorted_res = results_df.sort_values("R2_$", ascending=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_res)))
    axes[0].barh(sorted_res["Model"], sorted_res["R2_$"], color=colors)
    axes[0].set_xlabel("R2 Score")
    axes[0].set_title("Model Comparison (R2)")
    for i, (_, row) in enumerate(sorted_res.iterrows()):
        axes[0].text(row["R2_$"] + 0.005, i, f"{row['R2_$']:.3f}", va="center", fontsize=8)

    axes[1].barh(sorted_res["Model"], sorted_res["RMSE_$"], color=colors)
    axes[1].set_xlabel("RMSE ($)")
    axes[1].set_title("Model Comparison (RMSE in $)")
    for i, (_, row) in enumerate(sorted_res.iterrows()):
        axes[1].text(row["RMSE_$"] + 500, i, f"${row['RMSE_$']:,.0f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_model_comparison.png", dpi=150)
    plt.close()
    print("  [Saved] plots/03_model_comparison.png")

    # ==================================================================
    # STEP 8: RESIDUAL ANALYSIS (best model)
    # ==================================================================
    print_section("STEP 8: RESIDUAL ANALYSIS")

    best_name = results_df.iloc[0]["Model"]
    best_pipe = pipelines[best_name]
    y_pred_log = best_pipe.predict(X_test)
    y_pred_dollar = np.expm1(y_pred_log)
    residuals = y_test_dollar - y_pred_dollar
    residuals_log = y_test - y_pred_log

    print(f"\n  Best model: {best_name}")
    print(f"  Residual stats (dollar):")
    print(f"    Mean:   ${residuals.mean():>12,.0f} (should be ~0)")
    print(f"    Std:    ${residuals.std():>12,.0f}")
    print(f"    Median: ${residuals.median():>12,.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Predicted vs actual
    axes[0, 0].scatter(y_test_dollar, y_pred_dollar, alpha=0.4, s=15, color="steelblue")
    lims = [min(y_test_dollar.min(), y_pred_dollar.min()),
            max(y_test_dollar.max(), y_pred_dollar.max())]
    axes[0, 0].plot(lims, lims, "r--", alpha=0.8)
    axes[0, 0].set_xlabel("Actual Price ($)")
    axes[0, 0].set_ylabel("Predicted Price ($)")
    axes[0, 0].set_title(f"Predicted vs Actual -- {best_name}")

    # Residual distribution
    axes[0, 1].hist(residuals, bins=40, color="steelblue", edgecolor="black")
    axes[0, 1].axvline(0, color="red", linestyle="--")
    axes[0, 1].set_title("Residual Distribution")
    axes[0, 1].set_xlabel("Residual ($)")

    # Residuals vs predicted
    axes[1, 0].scatter(y_pred_dollar, residuals, alpha=0.4, s=15, color="#e74c3c")
    axes[1, 0].axhline(0, color="black", linestyle="--")
    axes[1, 0].set_xlabel("Predicted Price ($)")
    axes[1, 0].set_ylabel("Residual ($)")
    axes[1, 0].set_title("Residuals vs Predicted (check for patterns)")

    # Residuals in log space (better view)
    axes[1, 1].scatter(y_pred_log, residuals_log, alpha=0.4, s=15, color="#9b59b6")
    axes[1, 1].axhline(0, color="black", linestyle="--")
    axes[1, 1].set_xlabel("Predicted (log)")
    axes[1, 1].set_ylabel("Residual (log)")
    axes[1, 1].set_title("Residuals in Log-Space")

    plt.suptitle(f"Residual Analysis -- {best_name}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/04_residual_analysis.png")

    print(f"\n  How to read residual plots:")
    print(f"    - Random scatter around 0 = good model")
    print(f"    - Funnel shape = heteroscedasticity (prediction variance changes)")
    print(f"    - Patterns = model is systematically wrong")

    # ==================================================================
    # STEP 9: UNSUPERVISED METHODS (PCA, K-Means, DBSCAN)
    # ==================================================================
    print_section("STEP 9: UNSUPERVISED METHODS (PCA, K-Means, DBSCAN)")

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # --- PCA ---
    print(f"\n  [PCA] Dimensionality reduction")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_proc)
    X_test_pca = pca.transform(X_test_proc)
    print(f"    Original features: {X_train_proc.shape[1]}")
    print(f"    PCA components (95% variance): {X_train_pca.shape[1]}")
    print(f"    Compression ratio: {X_train_pca.shape[1]/X_train_proc.shape[1]*100:.1f}%")

    # Compare: all features vs PCA
    ridge_all = Ridge(alpha=1.0).fit(X_train_proc, y_train)
    ridge_pca = Ridge(alpha=1.0).fit(X_train_pca, y_train)
    r2_all = r2_score(y_test, ridge_all.predict(X_test_proc))
    r2_pca = r2_score(y_test, ridge_pca.predict(X_test_pca))
    print(f"    Ridge on all features: R2={r2_all:.4f}")
    print(f"    Ridge on PCA features: R2={r2_pca:.4f}")

    # PCA visualization
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_train_proc)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap="YlOrRd",
                         alpha=0.5, s=15)
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA -- Ames Housing (2D Projection, color=log price)")
    plt.colorbar(scatter, label="Log(SalePrice)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_pca_visualization.png", dpi=150)
    plt.close()
    print("  [Saved] plots/05_pca_visualization.png")

    # --- K-Means ---
    print(f"\n  [K-Means] Housing market segments")
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_proc)
    test_clusters = kmeans.predict(X_test_proc)
    print(f"    Clusters: {np.unique(train_clusters)}")
    for c in np.unique(train_clusters):
        mask = train_clusters == c
        median_price = np.expm1(y_train.values[mask]).mean()
        print(f"    Cluster {c}: {mask.sum()} houses, avg price=${median_price:,.0f}")

    X_train_clust = np.column_stack([X_train_proc, train_clusters])
    X_test_clust = np.column_stack([X_test_proc, test_clusters])
    ridge_clust = Ridge(alpha=1.0).fit(X_train_clust, y_train)
    r2_clust = r2_score(y_test, ridge_clust.predict(X_test_clust))
    print(f"    Ridge with cluster feature: R2={r2_clust:.4f} (vs {r2_all:.4f} without)")

    # --- DBSCAN ---
    print(f"\n  [DBSCAN] Outlier detection in feature space")
    dbscan = DBSCAN(eps=5.0, min_samples=5)
    db_labels = dbscan.fit_predict(X_train_proc)
    n_outliers = (db_labels == -1).sum()
    n_clusters_found = len(set(db_labels) - {-1})
    print(f"    Clusters found: {n_clusters_found}")
    print(f"    Outliers: {n_outliers} ({n_outliers/len(db_labels)*100:.1f}%)")
    if n_outliers > 0:
        outlier_prices = np.expm1(y_train.values[db_labels == -1]).mean()
        normal_prices = np.expm1(y_train.values[db_labels != -1]).mean()
        print(f"    Avg price -- outliers: ${outlier_prices:,.0f}, normal: ${normal_prices:,.0f}")

    # ==================================================================
    # STEP 10: FEATURE SELECTION (Mutual Information + RFE)
    # ==================================================================
    print_section("STEP 10: FEATURE SELECTION")

    print(f"\n  Why feature selection matters (especially with 80+ features):")
    print(f"    - Many features are noisy or redundant")
    print(f"    - Reduces overfitting (curse of dimensionality)")
    print(f"    - Faster training + simpler model")
    print(f"    - Key for interpretability")

    # Already have preprocessed data from step 9
    try:
        feat_names_all = list(preprocessor.get_feature_names_out())
    except Exception:
        feat_names_all = [f"feature_{i}" for i in range(X_train_proc.shape[1])]

    # Method 1: Mutual Information
    print(f"\n  [Method 1] Mutual Information (captures non-linear relationships)")
    mi_scores = mutual_info_regression(X_train_proc, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=feat_names_all).sort_values(ascending=False)
    print(f"    Top 15 features by MI score:")
    for feat, score in mi_series.head(15).items():
        print(f"      {feat:40s}  MI = {score:.4f}")

    # Method 2: RFE with Random Forest
    print(f"\n  [Method 2] RFE (Recursive Feature Elimination) with Random Forest")
    rfe_model = RandomForestRegressor(n_estimators=50, random_state=42)
    n_select = min(20, X_train_proc.shape[1])
    rfe = RFE(rfe_model, n_features_to_select=n_select, step=5)
    rfe.fit(X_train_proc, y_train)
    rfe_selected = [feat_names_all[i] for i in range(len(feat_names_all)) if rfe.support_[i]]
    print(f"    Selected {n_select} features out of {X_train_proc.shape[1]}")
    print(f"    Top selected: {rfe_selected[:10]}...")

    # Compare: all features vs selected
    ridge_all_fs = Ridge(alpha=1.0).fit(X_train_proc, y_train)
    r2_all_fs = r2_score(y_test, ridge_all_fs.predict(X_test_proc))

    X_train_rfe = X_train_proc[:, rfe.support_]
    X_test_rfe = X_test_proc[:, rfe.support_]
    ridge_rfe = Ridge(alpha=1.0).fit(X_train_rfe, y_train)
    r2_rfe = r2_score(y_test, ridge_rfe.predict(X_test_rfe))

    print(f"\n    Ridge on ALL features ({X_train_proc.shape[1]}):    R2 = {r2_all_fs:.4f}")
    print(f"    Ridge on RFE features ({n_select}):     R2 = {r2_rfe:.4f}")
    print(f"    -> {'RFE competitive with fewer features!' if r2_rfe > r2_all_fs - 0.01 else 'All features help here (complex dataset)'}")

    # SelectKBest with MI
    print(f"\n  [Method 3] SelectKBest (top-K by Mutual Information)")
    selector = SelectKBest(mutual_info_regression, k=n_select)
    X_train_kb = selector.fit_transform(X_train_proc, y_train)
    X_test_kb = selector.transform(X_test_proc)
    ridge_kb = Ridge(alpha=1.0).fit(X_train_kb, y_train)
    r2_kb = r2_score(y_test, ridge_kb.predict(X_test_kb))
    print(f"    Ridge on SelectKBest ({n_select}):     R2 = {r2_kb:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    mi_series.head(15).plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("Mutual Information Scores (Top 15)")
    axes[0].set_xlabel("MI Score")
    axes[0].invert_yaxis()

    rfe_ranking = pd.Series(rfe.ranking_, index=feat_names_all).sort_values()
    rfe_ranking.head(15).plot(kind="barh", ax=axes[1], color="#e74c3c")
    axes[1].set_title("RFE Ranking (1 = selected)")
    axes[1].set_xlabel("Rank")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_feature_selection.png", dpi=150)
    plt.close()
    print("  [Saved] plots/08_feature_selection.png")

    # ==================================================================
    # STEP 11: HYPERPARAMETER TUNING
    # ==================================================================
    print_section("STEP 11: HYPERPARAMETER TUNING")

    # Tune Gradient Boosting
    print(f"\n  Tuning Gradient Boosting with GridSearchCV...")
    gb_param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.05, 0.1, 0.2],
        "model__subsample": [0.8, 1.0],
    }
    gb_pipe = Pipeline([("preprocessor", preprocessor),
                         ("model", GradientBoostingRegressor(random_state=42))])
    gb_grid = GridSearchCV(gb_pipe, gb_param_grid, cv=cv,
                            scoring="neg_root_mean_squared_error",
                            n_jobs=-1, verbose=0)
    t0 = time.time()
    gb_grid.fit(X_train, y_train)
    gb_time = time.time() - t0
    y_pred_gb = gb_grid.predict(X_test)
    r2_gb = r2_score(y_test_dollar, np.expm1(y_pred_gb))
    rmse_gb = np.sqrt(mean_squared_error(y_test_dollar, np.expm1(y_pred_gb)))
    print(f"    Best params: {gb_grid.best_params_}")
    print(f"    Best CV RMSE (log): {-gb_grid.best_score_:.4f}")
    print(f"    Test R2: {r2_gb:.4f}, RMSE: ${rmse_gb:,.0f}")
    print(f"    Time: {gb_time:.1f}s")

    # Tune XGBoost with RandomizedSearchCV
    print(f"\n  Tuning XGBoost with RandomizedSearchCV...")
    xgb_param_dist = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 5, 7, 9],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__reg_alpha": [0, 0.1, 1.0],
        "model__reg_lambda": [1.0, 2.0, 5.0],
    }
    xgb_pipe = Pipeline([("preprocessor", preprocessor),
                          ("model", xgb.XGBRegressor(random_state=42, verbosity=0))])
    xgb_search = RandomizedSearchCV(xgb_pipe, xgb_param_dist, n_iter=40,
                                     cv=cv, scoring="neg_root_mean_squared_error",
                                     random_state=42, n_jobs=-1, verbose=0)
    t0 = time.time()
    xgb_search.fit(X_train, y_train)
    xgb_time = time.time() - t0
    y_pred_xgb = xgb_search.predict(X_test)
    r2_xgb = r2_score(y_test_dollar, np.expm1(y_pred_xgb))
    rmse_xgb = np.sqrt(mean_squared_error(y_test_dollar, np.expm1(y_pred_xgb)))
    print(f"    Best params: {xgb_search.best_params_}")
    print(f"    Best CV RMSE (log): {-xgb_search.best_score_:.4f}")
    print(f"    Test R2: {r2_xgb:.4f}, RMSE: ${rmse_xgb:,.0f}")
    print(f"    Time: {xgb_time:.1f}s")

    # Compare default vs tuned
    gb_default_r2 = results_df[results_df["Model"] == "Gradient Boosting"]["R2_$"].values[0]
    xgb_default_r2 = results_df[results_df["Model"] == "XGBoost"]["R2_$"].values[0]
    print(f"\n  Tuning Impact (R2):")
    print(f"    Gradient Boosting: {gb_default_r2:.4f} -> {r2_gb:.4f}")
    print(f"    XGBoost:           {xgb_default_r2:.4f} -> {r2_xgb:.4f}")

    # ==================================================================
    # STEP 12: STACKING ENSEMBLE
    # ==================================================================
    print_section("STEP 12: STACKING ENSEMBLE (Meta-Learner)")

    print(f"\n  Why stacking?")
    print(f"    - Combines strengths of diverse models")
    print(f"    - Base models predict -> meta-model learns from their outputs")
    print(f"    - Key: use diverse base learners (linear + tree + instance-based)")

    base_estimators = [
        ("ridge", Ridge(alpha=1.0, random_state=42)),
        ("lasso", Lasso(alpha=0.001, random_state=42)),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("svr", SVR(kernel="rbf", C=10)),
        ("xgb", xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)),
    ]
    stacking_reg = StackingRegressor(
        estimators=base_estimators,
        final_estimator=Ridge(alpha=1.0, random_state=42),
        cv=5,
        passthrough=False
    )
    stacking_pipe = Pipeline([("preprocessor", preprocessor), ("model", stacking_reg)])

    print(f"\n  Base models: {[name for name, _ in base_estimators]}")
    print(f"  Meta-learner: Ridge")
    print(f"\n  Training stacking ensemble...")
    t0 = time.time()
    stacking_pipe.fit(X_train, y_train)
    stack_time = time.time() - t0
    y_pred_stack_log = stacking_pipe.predict(X_test)
    y_pred_stack_dollar = np.expm1(y_pred_stack_log)

    r2_stack = r2_score(y_test_dollar, y_pred_stack_dollar)
    rmse_stack = np.sqrt(mean_squared_error(y_test_dollar, y_pred_stack_dollar))
    mae_stack = mean_absolute_error(y_test_dollar, y_pred_stack_dollar)
    print(f"    R2:   {r2_stack:.4f}")
    print(f"    RMSE: ${rmse_stack:,.0f}")
    print(f"    MAE:  ${mae_stack:,.0f}")
    print(f"    Time: {stack_time:.1f}s")

    # Compare stacking vs best single model
    best_single_r2 = results_df.iloc[0]["R2_$"]
    best_single_name = results_df.iloc[0]["Model"]
    print(f"\n  Stacking vs Best Single Model:")
    print(f"    {best_single_name}: R2={best_single_r2:.4f}")
    print(f"    Stacking Ensemble:  R2={r2_stack:.4f}")
    print(f"    -> {'Stacking wins!' if r2_stack > best_single_r2 else 'Close -- stacking shines more with complex data patterns'}")

    # ==================================================================
    # STEP 13: SHAP EXPLAINABILITY
    # ==================================================================
    print_section("STEP 13: SHAP EXPLAINABILITY")

    print(f"\n  Why SHAP for housing prices?")
    print(f"    - 'Why is this house predicted at $350K?'")
    print(f"    - Which features drive prices up/down?")
    print(f"    - Required for fair lending / real estate models")
    print(f"    - Interview gold: shows you understand model interpretability")

    try:
        import shap

        # Use XGBoost for fast TreeExplainer
        xgb_pipe_for_shap = pipelines.get("XGBoost", pipelines.get("Random Forest"))
        X_test_shap = preprocessor.transform(X_test)
        try:
            shap_feat_names = list(preprocessor.get_feature_names_out())
        except Exception:
            shap_feat_names = [f"feature_{i}" for i in range(X_test_shap.shape[1])]

        model_for_shap = xgb_pipe_for_shap.named_steps["model"]
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(X_test_shap)

        # Global feature importance
        print(f"\n  [SHAP] Global feature importance:")
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        shap_imp = pd.Series(shap_abs_mean, index=shap_feat_names).sort_values(ascending=False)
        for feat, val in shap_imp.head(10).items():
            print(f"    {feat:40s}  mean|SHAP| = {val:.4f}")

        # SHAP summary plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_values, X_test_shap, feature_names=shap_feat_names,
                          show=False, max_display=15)
        plt.title("SHAP Summary -- Feature Impact on House Price")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "09_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [Saved] plots/09_shap_summary.png")

        # Individual prediction explanations
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        # Expensive house
        expensive_idx = np.argmax(y_test.values)
        cheap_idx = np.argmin(y_test.values)
        for idx, (ax, sample_idx, label) in enumerate(zip(
            axes, [expensive_idx, cheap_idx], ["Expensive", "Cheap"])):
            shap_vals_single = shap_values[sample_idx]
            actual_price = np.expm1(y_test.values[sample_idx])
            top_k = 10
            top_feat_idx = np.argsort(np.abs(shap_vals_single))[-top_k:]
            ax.barh([shap_feat_names[i] for i in top_feat_idx],
                    shap_vals_single[top_feat_idx],
                    color=["#e74c3c" if v < 0 else "#2ecc71" for v in shap_vals_single[top_feat_idx]])
            ax.set_title(f"SHAP -- {label} house (actual: ${actual_price:,.0f})")
            ax.set_xlabel("SHAP value (impact on log-price)")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "10_shap_individual.png", dpi=150)
        plt.close()
        print("  [Saved] plots/10_shap_individual.png")

        print(f"\n  SHAP Interpretation Guide:")
        print(f"    - Positive SHAP -> pushes price UP")
        print(f"    - Negative SHAP -> pushes price DOWN")
        print(f"    - Magnitude = strength of influence")
        print(f"    - Summary plot: red=high feature value, blue=low")

    except ImportError:
        print(f"\n  [SKIP] shap not installed. Install with: pip install shap")
        print(f"  SHAP would provide:")
        print(f"    - Global feature importance ranking")
        print(f"    - 'Why is this house $350K?' explanations")
        print(f"    - Feature interaction effects")
    except Exception as e:
        print(f"\n  [SHAP Error] {e}")
        print(f"  (SHAP analysis skipped due to compatibility issue)")

    # ==================================================================
    # STEP 14: LEARNING CURVES
    # ==================================================================
    print_section("STEP 14: LEARNING CURVES")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    lc_models = [
        ("Ridge", Ridge(alpha=1.0, random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("XGBoost", xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)),
    ]
    for ax, (name, model) in zip(axes, lc_models):
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        train_sizes, train_scores, val_scores = learning_curve(
            pipe, X_train, y_train, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 8),
            scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        train_rmse = -train_scores.mean(axis=1)
        val_rmse = -val_scores.mean(axis=1)
        ax.plot(train_sizes, train_rmse, "o-", label="Train", color="#2ecc71")
        ax.fill_between(train_sizes,
                         train_rmse - train_scores.std(axis=1),
                         train_rmse + train_scores.std(axis=1),
                         alpha=0.1, color="#2ecc71")
        ax.plot(train_sizes, val_rmse, "o-", label="Validation", color="#e74c3c")
        ax.fill_between(train_sizes,
                         val_rmse - val_scores.std(axis=1),
                         val_rmse + val_scores.std(axis=1),
                         alpha=0.1, color="#e74c3c")
        ax.set_title(f"Learning Curve -- {name}")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("RMSE (log)")
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_learning_curves.png", dpi=150)
    plt.close()
    print("  [Saved] plots/11_learning_curves.png")

    # ==================================================================
    # STEP 15: FINAL MODEL & PERSISTENCE
    # ==================================================================
    print_section("STEP 15: FINAL MODEL & PERSISTENCE")

    if xgb_search.best_score_ > gb_grid.best_score_:
        # Higher (less negative) is better for neg_rmse
        final_model = xgb_search.best_estimator_
        final_name = "XGBoost (tuned)"
    else:
        final_model = gb_grid.best_estimator_
        final_name = "Gradient Boosting (tuned)"

    y_pred_final_log = final_model.predict(X_test)
    y_pred_final = np.expm1(y_pred_final_log)

    print(f"\n  Best model: {final_name}")
    print(f"  Final Test Metrics:")
    print_log_metrics(final_name, y_test, y_pred_final_log)

    # Feature importance (if available)
    try:
        model_step = final_model.named_steps["model"]
        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
            # Get feature names from preprocessor
            try:
                feature_names = final_model.named_steps["preprocessor"].get_feature_names_out()
            except Exception:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
            top_n = min(15, len(feat_imp))
            fig, ax = plt.subplots(figsize=(10, 6))
            feat_imp.head(top_n).plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title(f"Feature Importance -- {final_name}")
            ax.set_xlabel("Importance")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "12_feature_importance.png", dpi=150)
            plt.close()
            print("  [Saved] plots/12_feature_importance.png")
            print(f"\n  Top 10 Features:")
            for feat, imp in feat_imp.head(10).items():
                print(f"    {feat:40s}  {imp:.4f}")
    except Exception as e:
        print(f"  (Feature importance not available: {e})")

    # Save model
    model_path = PLOTS_DIR.parent / "best_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"\n  Model saved to: {model_path}")

    # Inference demo
    loaded_model = joblib.load(model_path)
    sample = X_test.iloc[:5]
    preds_log = loaded_model.predict(sample)
    preds = np.expm1(preds_log)
    actuals = np.expm1(y_test.iloc[:5].values)
    print(f"\n  Inference on 5 test samples:")
    print(f"  {'Predicted':>12s}  {'Actual':>12s}  {'Error':>12s}")
    for p, a in zip(preds, actuals):
        err = p - a
        print(f"  ${p:>11,.0f}  ${a:>11,.0f}  ${err:>+11,.0f}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print_section("FINAL SUMMARY")
    print("""
    REAL-WORLD SKILLS PRACTICED IN THIS PIPELINE:
    -----------------------------------------------
    1.  DATA EXPLORATION       -- 80 features, mixed types, missing patterns
    2.  DATA CLEANING          -- Drop >40% missing, NaN->'None', outliers
    3.  FEATURE ENGINEERING    -- TotalSF, TotalBath, HouseAge, quality scores
    4.  TARGET TRANSFORMATION  -- Log transform for skewed SalePrice
    5.  EDA VISUALIZATIONS     -- Correlations, distributions, scatterplots
    6.  PREPROCESSING PIPELINE -- ColumnTransformer + Pipeline (reproducible)
    7.  MODEL COMPARISON       -- 11 regressors with 5-fold CV
    8.  RESIDUAL ANALYSIS      -- Predicted vs actual, residual patterns
    9.  UNSUPERVISED FEATURES  -- PCA compression, K-Means segments, DBSCAN outliers
    10. FEATURE SELECTION      -- Mutual Information, RFE, SelectKBest
    11. HYPERPARAMETER TUNING  -- GridSearchCV, RandomizedSearchCV
    12. STACKING ENSEMBLE      -- Meta-learner combining diverse regressors
    13. SHAP EXPLAINABILITY    -- Global + local model explanations
    14. LEARNING CURVES        -- Diagnose overfitting vs underfitting
    15. MODEL PERSISTENCE      -- Save/load with joblib + inference demo

    ALGORITHMS USED:
    -----------------------------------------------
    REGRESSION:   LinReg, Ridge, Lasso, ElasticNet, DecTree, RF, GB, SVR, KNN, XGB, LGBM
    UNSUPERVISED: PCA (dim reduction), K-Means (segments), DBSCAN (outliers)
    ADVANCED:     Stacking (meta-learning), SHAP (explainability)
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
