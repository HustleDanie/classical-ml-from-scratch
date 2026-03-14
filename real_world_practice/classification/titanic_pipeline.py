"""
Real-World Classification Pipeline -- Titanic Survival Prediction
=================================================================

Dataset: Titanic (seaborn built-in) -- 891 passengers, 15 features
Task:    Predict passenger survival (binary classification)

Real-world skills practiced:
  1.  Data exploration & understanding
  2.  Data cleaning (missing values, duplicates, outliers)
  3.  Feature engineering (create new informative features)
  4.  Exploratory data analysis (EDA) with visualizations
  5.  Building preprocessing pipelines (ColumnTransformer)
  6.  Training & comparing ALL classical ML classifiers
  7.  Handling class imbalance (class_weight, SMOTE)
  8.  Using unsupervised ML (PCA, K-Means, DBSCAN) as features
  9.  Feature selection (Mutual Information, RFE)
  10. Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
  11. Stacking ensemble (meta-learner combining multiple models)
  12. SHAP explainability (why the model predicts what it predicts)
  13. Learning curves & overfitting diagnosis
  14. Final model selection & evaluation
  15. Model persistence (saving/loading with joblib)

Dataset challenges:
  - Missing values: age (~20%), deck (~77%), embarked (<1%)
  - Mixed types: numeric + categorical features
  - Class imbalance: ~38% survived vs ~62% died
  - Redundant features: alive/survived, class/pclass, embark_town/embarked
  - Feature engineering opportunities: family size, age groups, fare groups
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
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV, learning_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.feature_selection import (
    mutual_info_classif, RFE, SelectKBest
)
from sklearn.ensemble import StackingClassifier

# ---- All 12 Classical ML Algorithms ----
# Supervised classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb

# Unsupervised (used as feature engineering / preprocessing)
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
    """Print a formatted section header."""
    if level == 1:
        print("\n\n" + "=" * 70)
        print(title)
        print("=" * 70)
    else:
        print(f"\n  --- {title} ---")


def print_metrics(name, y_true, y_pred, y_proba=None):
    """Print classification metrics for a model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0
    print(f"    Accuracy:  {acc:.4f}  |  Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}  |  F1:        {f1:.4f}")
    if y_proba is not None:
        print(f"    AUC-ROC:   {auc:.4f}")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    print("=" * 70)
    print("REAL-WORLD CLASSIFICATION PIPELINE")
    print("Dataset: Titanic Survival Prediction")
    print("=" * 70)

    # ==================================================================
    # STEP 1: LOAD & EXPLORE RAW DATA
    # ==================================================================
    print_section("STEP 1: LOAD & EXPLORE RAW DATA")

    df_raw = sns.load_dataset("titanic")
    print(f"\n  Shape: {df_raw.shape} ({df_raw.shape[0]} passengers, {df_raw.shape[1]} features)")
    print(f"  Columns: {list(df_raw.columns)}")

    print(f"\n  Data types:")
    for col in df_raw.columns:
        print(f"    {col:15s}  {str(df_raw[col].dtype):10s}  nulls={df_raw[col].isnull().sum()}")

    print(f"\n  First 5 rows:")
    print(df_raw.head().to_string(max_cols=10))

    # Missing values summary
    missing = df_raw.isnull().sum()
    missing_pct = (missing / len(df_raw) * 100).round(1)
    missing_df = pd.DataFrame({"Count": missing, "Pct": missing_pct})
    missing_df = missing_df[missing_df["Count"] > 0].sort_values("Pct", ascending=False)
    print(f"\n  Missing values:")
    for col, row in missing_df.iterrows():
        print(f"    {col:15s}  {int(row['Count']):4d} ({row['Pct']:.1f}%)")

    # Target distribution
    surv_counts = df_raw["survived"].value_counts()
    print(f"\n  Target distribution (class imbalance check):")
    print(f"    Died (0):     {surv_counts[0]} ({surv_counts[0]/len(df_raw)*100:.1f}%)")
    print(f"    Survived (1): {surv_counts[1]} ({surv_counts[1]/len(df_raw)*100:.1f}%)")
    print(f"    --> Imbalanced: minority class is ~{surv_counts[1]/len(df_raw)*100:.0f}%")

    # Plot: data overview
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.heatmap(df_raw.isnull(), cbar=True, yticklabels=False, ax=axes[0], cmap="viridis")
    axes[0].set_title("Missing Values Pattern")
    df_raw["survived"].value_counts().plot(kind="bar", ax=axes[1],
                                           color=["#e74c3c", "#2ecc71"])
    axes[1].set_title("Target Distribution")
    axes[1].set_xticklabels(["Died (0)", "Survived (1)"], rotation=0)
    for i, v in enumerate(df_raw["survived"].value_counts().sort_index()):
        axes[1].text(i, v + 10, str(v), ha="center", fontweight="bold")
    df_raw["age"].hist(bins=30, ax=axes[2], color="steelblue", edgecolor="black")
    axes[2].set_title("Age Distribution (with gaps = missing)")
    axes[2].set_xlabel("Age")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_data_overview.png", dpi=150)
    plt.close()
    print("  [Saved] plots/01_data_overview.png")

    # ==================================================================
    # STEP 2: DATA CLEANING
    # ==================================================================
    print_section("STEP 2: DATA CLEANING")

    df = df_raw.copy()

    # 2a. Drop redundant columns
    redundant = ["alive", "class", "embark_town", "adult_male"]
    df = df.drop(columns=redundant)
    print(f"\n  [Drop] Redundant columns removed: {redundant}")
    print(f"    'alive' = inverse of 'survived' (target leak)")
    print(f"    'class' = string version of 'pclass'")
    print(f"    'embark_town' = full name of 'embarked'")
    print(f"    'adult_male' = derivable from 'sex' + 'age'")

    # 2b. Check duplicates
    n_dups = df.duplicated().sum()
    print(f"\n  [Duplicates] Found: {n_dups}")
    if n_dups > 0:
        df = df.drop_duplicates()
        print(f"    Removed {n_dups} duplicates. New shape: {df.shape}")

    # 2c. Handle missing values
    print(f"\n  [Missing Values]")

    # Age: grouped median imputation (more accurate than global median)
    print(f"    Age: {df['age'].isnull().sum()} missing ({df['age'].isnull().mean()*100:.1f}%)")
    print(f"    Strategy: Impute with median age by (pclass, sex) groups")
    age_medians = df.groupby(["pclass", "sex"])["age"].median()
    for (pc, sx), med in age_medians.items():
        mask = df["age"].isnull() & (df["pclass"] == pc) & (df["sex"] == sx)
        df.loc[mask, "age"] = med
    print(f"    After imputation: {df['age'].isnull().sum()} missing")

    # Embarked: mode imputation
    emb_missing = df["embarked"].isnull().sum()
    emb_mode = df["embarked"].mode()[0]
    df["embarked"] = df["embarked"].fillna(emb_mode)
    print(f"    Embarked: {emb_missing} missing -> filled with mode '{emb_mode}'")

    # Deck: 77% missing -> create binary flag, then drop original
    deck_miss = df["deck"].isnull().sum()
    print(f"    Deck: {deck_miss} missing ({deck_miss/len(df)*100:.0f}%)")
    print(f"    Strategy: Create 'has_deck' flag (1=known, 0=unknown), drop 'deck'")
    df["has_deck"] = df["deck"].notna().astype(int)
    df = df.drop(columns=["deck"])

    # 2d. Outlier handling (fare)
    Q1 = df["fare"].quantile(0.25)
    Q3 = df["fare"].quantile(0.75)
    IQR = Q3 - Q1
    upper = Q3 + 3 * IQR
    n_outliers = (df["fare"] > upper).sum()
    print(f"\n  [Outliers] Fare: Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}")
    print(f"    Upper bound (Q3 + 3*IQR): {upper:.1f}")
    print(f"    Outliers capped: {n_outliers}")
    df.loc[df["fare"] > upper, "fare"] = upper

    print(f"\n  Cleaned shape: {df.shape}")
    print(f"  Remaining nulls: {df.isnull().sum().sum()}")

    # ==================================================================
    # STEP 3: FEATURE ENGINEERING
    # ==================================================================
    print_section("STEP 3: FEATURE ENGINEERING")

    df["family_size"] = df["sibsp"] + df["parch"] + 1
    print(f"\n  [+] family_size = sibsp + parch + 1")
    print(f"      Range: {df['family_size'].min()} - {df['family_size'].max()}")

    df["is_alone"] = (df["family_size"] == 1).astype(int)
    print(f"  [+] is_alone = (family_size == 1)")
    print(f"      Alone: {df['is_alone'].sum()} ({df['is_alone'].mean()*100:.1f}%)")

    df["fare_per_person"] = df["fare"] / df["family_size"]
    print(f"  [+] fare_per_person = fare / family_size")

    df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 35, 60, 100],
                              labels=["Child", "Teen", "Adult", "Middle", "Senior"])
    print(f"  [+] age_group = binned age (Child/Teen/Adult/Middle/Senior)")

    print(f"\n  Final columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")

    # ==================================================================
    # STEP 4: EDA VISUALIZATIONS
    # ==================================================================
    print_section("STEP 4: EXPLORATORY DATA ANALYSIS (EDA)")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Survival by sex
    df.groupby("sex")["survived"].mean().plot(kind="bar", ax=axes[0, 0],
                                               color=["#3498db", "#e74c3c"])
    axes[0, 0].set_title("Survival Rate by Sex")
    axes[0, 0].set_ylabel("Survival Rate")
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=0)

    # Survival by pclass
    df.groupby("pclass")["survived"].mean().plot(kind="bar", ax=axes[0, 1],
                                                  color=["#2ecc71", "#f39c12", "#e74c3c"])
    axes[0, 1].set_title("Survival Rate by Class")
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=0)

    # Survival by embarked
    df.groupby("embarked")["survived"].mean().plot(kind="bar", ax=axes[0, 2],
                                                    color="steelblue")
    axes[0, 2].set_title("Survival Rate by Port")
    axes[0, 2].set_xticklabels(axes[0, 2].get_xticklabels(), rotation=0)

    # Age distribution by survival
    axes[1, 0].hist([df[df["survived"] == 0]["age"], df[df["survived"] == 1]["age"]],
                     bins=20, label=["Died", "Survived"],
                     color=["#e74c3c", "#2ecc71"], alpha=0.7)
    axes[1, 0].set_title("Age Distribution by Survival")
    axes[1, 0].legend()

    # Fare distribution by survival
    axes[1, 1].hist([df[df["survived"] == 0]["fare"], df[df["survived"] == 1]["fare"]],
                     bins=20, label=["Died", "Survived"],
                     color=["#e74c3c", "#2ecc71"], alpha=0.7)
    axes[1, 1].set_title("Fare Distribution by Survival")
    axes[1, 1].legend()

    # Family size vs survival
    df.groupby("family_size")["survived"].mean().plot(kind="bar", ax=axes[1, 2],
                                                       color="steelblue")
    axes[1, 2].set_title("Survival Rate by Family Size")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_eda_survival_rates.png", dpi=150)
    plt.close()
    print("  [Saved] plots/02_eda_survival_rates.png")

    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_correlation_heatmap.png", dpi=150)
    plt.close()
    print("  [Saved] plots/03_correlation_heatmap.png")

    # ==================================================================
    # STEP 5: BUILD PREPROCESSING PIPELINE + TRAIN-TEST SPLIT
    # ==================================================================
    print_section("STEP 5: PREPROCESSING PIPELINE + TRAIN-TEST SPLIT")

    # Separate features and target
    drop_for_model = ["survived", "who", "alone", "age_group"]
    # 'who' correlates with sex+age, 'alone' with is_alone, 'age_group' with age
    X = df.drop(columns=drop_for_model)
    y = df["survived"]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"\n  Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    print(f"\n  Pipeline: Numeric  -> Imputer(median) -> StandardScaler")
    print(f"           Categorical -> Imputer(mode) -> OneHotEncoder")
    print(f"           Combined via ColumnTransformer")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {X_train.shape},  Test: {X_test.shape}")
    print(f"  Train target: {dict(y_train.value_counts().sort_index())}")
    print(f"  Test target:  {dict(y_test.value_counts().sort_index())}")

    # ==================================================================
    # STEP 6: TRAIN & COMPARE ALL CLASSICAL ML CLASSIFIERS
    # ==================================================================
    print_section("STEP 6: TRAIN & COMPARE ALL 9 CLASSIFIERS (5-fold CV)")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, eval_metric="logloss",
                                       random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        t0 = time.time()

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

        elapsed = time.time() - t0
        metrics = print_metrics(name, y_test, y_pred, y_proba)

        results.append({
            "Model": name,
            "CV Mean": cv_scores.mean(),
            "CV Std": cv_scores.std(),
            **metrics,
            "Time": elapsed,
        })
        pipelines[name] = pipe

        print(f"\n  {name}:")
        print(f"    5-Fold CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print_metrics(name, y_test, y_pred, y_proba)
        print(f"    Time: {elapsed:.3f}s")

    # Results table
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    print(f"\n{'='*70}")
    print("  MODEL COMPARISON (sorted by F1)")
    print(f"{'='*70}")
    display_cols = ["Model", "CV Mean", "CV Std", "Accuracy", "Precision", "Recall", "F1", "AUC"]
    print(results_df[display_cols].to_string(index=False))

    # Plot: model comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sorted_res = results_df.sort_values("F1", ascending=True)
    axes[0].barh(sorted_res["Model"], sorted_res["F1"], color="steelblue")
    axes[0].set_xlabel("F1 Score")
    axes[0].set_title("Model Comparison (F1 Score)")
    axes[0].set_xlim(0, 1)
    for i, (_, row) in enumerate(sorted_res.iterrows()):
        axes[0].text(row["F1"] + 0.01, i, f"{row['F1']:.3f}", va="center", fontsize=9)

    # ROC curves for top models
    for _, row in results_df.head(5).iterrows():
        name = row["Model"]
        pipe = pipelines[name]
        try:
            y_proba = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            axes[1].plot(fpr, tpr, label=f"{name} (AUC={row['AUC']:.3f})")
        except Exception:
            pass
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curves (Top 5 Models)")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_model_comparison.png", dpi=150)
    plt.close()
    print("  [Saved] plots/04_model_comparison.png")

    # Confusion matrix for best model
    best_name = results_df.iloc[0]["Model"]
    best_pipe = pipelines[best_name]
    y_pred_best = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {best_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_confusion_matrix_best.png", dpi=150)
    plt.close()
    print("  [Saved] plots/05_confusion_matrix_best.png")
    print(f"\n  Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred_best, target_names=["Died", "Survived"]))

    # ==================================================================
    # STEP 7: HANDLING CLASS IMBALANCE
    # ==================================================================
    print_section("STEP 7: HANDLING CLASS IMBALANCE")
    print(f"\n  Class distribution: {dict(y_train.value_counts().sort_index())}")
    print(f"  Minority class (survived): {y_train.mean()*100:.1f}%")

    # Method 1: class_weight='balanced'
    print(f"\n  Method 1: class_weight='balanced' (built into sklearn)")
    imb_models = {
        "LR (default)": LogisticRegression(max_iter=1000, random_state=42),
        "LR (balanced)": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "RF (default)": RandomForestClassifier(n_estimators=100, random_state=42),
        "RF (balanced)": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    }

    print(f"\n  {'Model':<20s} {'Accuracy':>10s} {'Recall':>10s} {'F1':>10s}")
    print(f"  {'-'*50}")
    for name, model in imb_models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"  {name:<20s} {acc:>10.4f} {rec:>10.4f} {f1:>10.4f}")

    print(f"\n  -> 'balanced' increases recall (catches more survivors)")
    print(f"     at the cost of slightly lower precision")
    print(f"     Use this when false negatives are costly!")

    # Method 2: SMOTE (if available)
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        print(f"\n  Method 2: SMOTE oversampling (imblearn)")
        X_train_processed = preprocessor.fit_transform(X_train)
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)
        print(f"    Before SMOTE: {dict(pd.Series(y_train).value_counts().sort_index())}")
        print(f"    After SMOTE:  {dict(pd.Series(y_resampled).value_counts().sort_index())}")

        lr_smote = LogisticRegression(max_iter=1000, random_state=42)
        lr_smote.fit(X_resampled, y_resampled)
        X_test_processed = preprocessor.transform(X_test)
        y_pred_smote = lr_smote.predict(X_test_processed)
        print(f"    LR + SMOTE: Acc={accuracy_score(y_test, y_pred_smote):.4f}, "
              f"Recall={recall_score(y_test, y_pred_smote):.4f}, "
              f"F1={f1_score(y_test, y_pred_smote):.4f}")
    except ImportError:
        print(f"\n  Method 2: SMOTE (imblearn not installed -- skipping)")
        print(f"    Install with: pip install imbalanced-learn")

    # ==================================================================
    # STEP 8: UNSUPERVISED ML AS FEATURES (PCA, K-Means, DBSCAN)
    # ==================================================================
    print_section("STEP 8: UNSUPERVISED ML AS FEATURES (PCA, K-Means, DBSCAN)")
    print(f"\n  All 12 algorithms contribute to real-world pipelines:")
    print(f"    - 9 supervised classifiers (Step 6)")
    print(f"    - PCA: dimensionality reduction / feature extraction")
    print(f"    - K-Means: cluster labels as new features")
    print(f"    - DBSCAN: outlier detection")

    # Preprocess data for unsupervised methods
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # --- PCA ---
    print(f"\n  [PCA] Dimensionality reduction")
    pca = PCA(n_components=0.95, random_state=42)  # Keep 95% variance
    X_train_pca = pca.fit_transform(X_train_proc)
    X_test_pca = pca.transform(X_test_proc)
    print(f"    Original features: {X_train_proc.shape[1]}")
    print(f"    PCA components (95% variance): {X_train_pca.shape[1]}")
    print(f"    Variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    # Train LR on PCA features
    lr_pca = LogisticRegression(max_iter=1000, random_state=42)
    lr_pca.fit(X_train_pca, y_train)
    acc_pca = accuracy_score(y_test, lr_pca.predict(X_test_pca))
    print(f"    LR on PCA features: Acc={acc_pca:.4f}")
    print(f"    LR on all features: Acc={results_df[results_df['Model']=='Logistic Regression']['Accuracy'].values[0]:.4f}")

    # PCA visualization
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X_train_proc)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap="RdYlGn",
                         alpha=0.5, s=20, edgecolors="none")
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA -- Titanic Passengers (2D Projection)")
    plt.colorbar(scatter, label="Survived")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_pca_visualization.png", dpi=150)
    plt.close()
    print("  [Saved] plots/06_pca_visualization.png")

    # --- K-Means cluster features ---
    print(f"\n  [K-Means] Cluster labels as new features")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_proc)
    test_clusters = kmeans.predict(X_test_proc)
    print(f"    Clusters: {np.unique(train_clusters)}")
    for c in np.unique(train_clusters):
        surv_rate = y_train.values[train_clusters == c].mean()
        print(f"    Cluster {c}: {(train_clusters == c).sum()} samples, survival rate={surv_rate:.3f}")

    # Add cluster as feature
    X_train_with_cluster = np.column_stack([X_train_proc, train_clusters])
    X_test_with_cluster = np.column_stack([X_test_proc, test_clusters])
    lr_cluster = LogisticRegression(max_iter=1000, random_state=42)
    lr_cluster.fit(X_train_with_cluster, y_train)
    acc_cluster = accuracy_score(y_test, lr_cluster.predict(X_test_with_cluster))
    print(f"    LR with cluster feature: Acc={acc_cluster:.4f}")

    # --- DBSCAN for outlier detection ---
    print(f"\n  [DBSCAN] Outlier detection")
    dbscan = DBSCAN(eps=3.0, min_samples=5)
    db_labels = dbscan.fit_predict(X_train_proc)
    n_outliers = (db_labels == -1).sum()
    n_clusters_found = len(set(db_labels) - {-1})
    print(f"    Clusters found: {n_clusters_found}")
    print(f"    Outliers (noise): {n_outliers} ({n_outliers/len(db_labels)*100:.1f}%)")
    if n_outliers > 0:
        outlier_surv = y_train.values[db_labels == -1].mean()
        normal_surv = y_train.values[db_labels != -1].mean()
        print(f"    Survival rate -- outliers: {outlier_surv:.3f}, normal: {normal_surv:.3f}")

    # ==================================================================
    # STEP 9: FEATURE SELECTION (Mutual Information + RFE)
    # ==================================================================
    print_section("STEP 9: FEATURE SELECTION")

    print(f"\n  Why feature selection matters:")
    print(f"    - Removes noisy/irrelevant features -> better generalization")
    print(f"    - Reduces overfitting risk")
    print(f"    - Faster training and inference")
    print(f"    - Easier to interpret and explain")

    # Preprocess training data for feature selection
    X_train_fs = preprocessor.fit_transform(X_train)
    X_test_fs = preprocessor.transform(X_test)
    try:
        feat_names_fs = preprocessor.get_feature_names_out()
    except Exception:
        feat_names_fs = [f"feature_{i}" for i in range(X_train_fs.shape[1])]

    # Method 1: Mutual Information
    print(f"\n  [Method 1] Mutual Information (non-linear dependency measure)")
    mi_scores = mutual_info_classif(X_train_fs, y_train, random_state=42)
    mi_series = pd.Series(mi_scores, index=feat_names_fs).sort_values(ascending=False)
    print(f"    Top 10 features by MI score:")
    for feat, score in mi_series.head(10).items():
        print(f"      {feat:30s}  MI = {score:.4f}")

    # Method 2: Recursive Feature Elimination (RFE)
    print(f"\n  [Method 2] RFE (Recursive Feature Elimination) with Random Forest")
    rfe_model = RandomForestClassifier(n_estimators=50, random_state=42)
    n_select = min(8, X_train_fs.shape[1])
    rfe = RFE(rfe_model, n_features_to_select=n_select, step=1)
    rfe.fit(X_train_fs, y_train)
    rfe_selected = [feat_names_fs[i] for i in range(len(feat_names_fs)) if rfe.support_[i]]
    print(f"    Selected {n_select} features: {rfe_selected}")

    # Compare: all features vs selected features
    lr_all = LogisticRegression(max_iter=1000, random_state=42)
    lr_all.fit(X_train_fs, y_train)
    acc_all = accuracy_score(y_test, lr_all.predict(X_test_fs))

    X_train_rfe = X_train_fs[:, rfe.support_]
    X_test_rfe = X_test_fs[:, rfe.support_]
    lr_rfe = LogisticRegression(max_iter=1000, random_state=42)
    lr_rfe.fit(X_train_rfe, y_train)
    acc_rfe = accuracy_score(y_test, lr_rfe.predict(X_test_rfe))
    print(f"\n    LogReg on ALL features ({X_train_fs.shape[1]}):     Acc = {acc_all:.4f}")
    print(f"    LogReg on RFE features ({n_select}):      Acc = {acc_rfe:.4f}")
    print(f"    -> {'RFE helped!' if acc_rfe >= acc_all else 'All features slightly better (more data might help)'}")

    # SelectKBest with MI
    print(f"\n  [Method 3] SelectKBest (top-K by Mutual Information)")
    selector = SelectKBest(mutual_info_classif, k=n_select)
    X_train_kb = selector.fit_transform(X_train_fs, y_train)
    X_test_kb = selector.transform(X_test_fs)
    lr_kb = LogisticRegression(max_iter=1000, random_state=42)
    lr_kb.fit(X_train_kb, y_train)
    acc_kb = accuracy_score(y_test, lr_kb.predict(X_test_kb))
    selected_kb = [feat_names_fs[i] for i in selector.get_support(indices=True)]
    print(f"    Selected: {selected_kb}")
    print(f"    LogReg on SelectKBest ({n_select}):   Acc = {acc_kb:.4f}")

    # Plot feature importance comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mi_series.head(12).plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("Mutual Information Scores")
    axes[0].set_xlabel("MI Score")
    axes[0].invert_yaxis()

    rfe_ranking = pd.Series(rfe.ranking_, index=feat_names_fs).sort_values()
    rfe_ranking.head(12).plot(kind="barh", ax=axes[1], color="#e74c3c")
    axes[1].set_title("RFE Ranking (1 = selected)")
    axes[1].set_xlabel("Rank")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_feature_selection.png", dpi=150)
    plt.close()
    print("  [Saved] plots/08_feature_selection.png")

    # ==================================================================
    # STEP 10: HYPERPARAMETER TUNING
    # ==================================================================
    print_section("STEP 10: HYPERPARAMETER TUNING (GridSearchCV)")

    # Tune Random Forest
    print(f"\n  Tuning Random Forest with GridSearchCV...")
    rf_param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10],
    }
    rf_pipe = Pipeline([("preprocessor", preprocessor),
                         ("model", RandomForestClassifier(random_state=42))])
    rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=cv, scoring="f1",
                            n_jobs=-1, verbose=0)
    t0 = time.time()
    rf_grid.fit(X_train, y_train)
    rf_time = time.time() - t0
    y_pred_rf_tuned = rf_grid.predict(X_test)
    print(f"    Best params: {rf_grid.best_params_}")
    print(f"    Best CV F1: {rf_grid.best_score_:.4f}")
    print(f"    Test F1: {f1_score(y_test, y_pred_rf_tuned):.4f}")
    print(f"    Time: {rf_time:.1f}s")

    # Tune XGBoost with RandomizedSearchCV
    print(f"\n  Tuning XGBoost with RandomizedSearchCV...")
    xgb_param_dist = {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [3, 5, 7, 9],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }
    xgb_pipe = Pipeline([("preprocessor", preprocessor),
                          ("model", xgb.XGBClassifier(eval_metric="logloss",
                                                       random_state=42, verbosity=0))])
    xgb_search = RandomizedSearchCV(xgb_pipe, xgb_param_dist, n_iter=30,
                                     cv=cv, scoring="f1", random_state=42,
                                     n_jobs=-1, verbose=0)
    t0 = time.time()
    xgb_search.fit(X_train, y_train)
    xgb_time = time.time() - t0
    y_pred_xgb_tuned = xgb_search.predict(X_test)
    print(f"    Best params: {xgb_search.best_params_}")
    print(f"    Best CV F1: {xgb_search.best_score_:.4f}")
    print(f"    Test F1: {f1_score(y_test, y_pred_xgb_tuned):.4f}")
    print(f"    Time: {xgb_time:.1f}s")

    # Compare: default vs tuned
    print(f"\n  Tuning Impact:")
    rf_default_f1 = results_df[results_df["Model"] == "Random Forest"]["F1"].values[0]
    xgb_default_f1 = results_df[results_df["Model"] == "XGBoost"]["F1"].values[0]
    print(f"    Random Forest: {rf_default_f1:.4f} -> {f1_score(y_test, y_pred_rf_tuned):.4f}")
    print(f"    XGBoost:       {xgb_default_f1:.4f} -> {f1_score(y_test, y_pred_xgb_tuned):.4f}")

    # ==================================================================
    # STEP 11: STACKING ENSEMBLE
    # ==================================================================
    print_section("STEP 11: STACKING ENSEMBLE (Meta-Learner)")

    print(f"\n  Why stacking?")
    print(f"    - Combines diverse models into a 'meta-learner'")
    print(f"    - Base models make predictions -> meta-model learns from them")
    print(f"    - Often beats any single model")
    print(f"    - Key: use diverse base models (linear + tree + instance-based)")

    base_estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("xgb", xgb.XGBClassifier(n_estimators=100, eval_metric="logloss",
                                    random_state=42, verbosity=0)),
    ]
    # Meta-learner: Logistic Regression combines base predictions
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        passthrough=False  # only base model predictions as meta-features
    )
    stacking_pipe = Pipeline([("preprocessor", preprocessor), ("model", stacking_clf)])

    print(f"\n  Base models: {[name for name, _ in base_estimators]}")
    print(f"  Meta-learner: LogisticRegression")
    print(f"\n  Training stacking ensemble...")
    t0 = time.time()
    stacking_pipe.fit(X_train, y_train)
    stack_time = time.time() - t0
    y_pred_stack = stacking_pipe.predict(X_test)
    y_proba_stack = stacking_pipe.predict_proba(X_test)[:, 1]

    stack_acc = accuracy_score(y_test, y_pred_stack)
    stack_f1 = f1_score(y_test, y_pred_stack)
    stack_auc = roc_auc_score(y_test, y_proba_stack)
    print(f"    Accuracy: {stack_acc:.4f}")
    print(f"    F1:       {stack_f1:.4f}")
    print(f"    AUC-ROC:  {stack_auc:.4f}")
    print(f"    Time:     {stack_time:.1f}s")

    # Compare stacking vs best single model
    best_single_f1 = results_df.iloc[0]["F1"]
    best_single_name = results_df.iloc[0]["Model"]
    print(f"\n  Stacking vs Best Single Model:")
    print(f"    {best_single_name}: F1={best_single_f1:.4f}")
    print(f"    Stacking Ensemble:  F1={stack_f1:.4f}")
    print(f"    -> {'Stacking wins!' if stack_f1 > best_single_f1 else 'Single model competitive (stacking shines with more data)'}")

    # ==================================================================
    # STEP 12: SHAP EXPLAINABILITY
    # ==================================================================
    print_section("STEP 12: SHAP EXPLAINABILITY")

    print(f"\n  Why SHAP?")
    print(f"    - Answers 'why did the model predict THIS?'")
    print(f"    - Based on Shapley values from game theory")
    print(f"    - Model-agnostic (works with any model)")
    print(f"    - Used in interviews + real production systems")

    try:
        import shap

        # Use the best tree-based model for SHAP (fast TreeExplainer)
        xgb_pipe_for_shap = pipelines.get("XGBoost", pipelines.get("Random Forest"))
        X_test_shap = preprocessor.transform(X_test)
        try:
            shap_feat_names = list(preprocessor.get_feature_names_out())
        except Exception:
            shap_feat_names = [f"feature_{i}" for i in range(X_test_shap.shape[1])]

        # Get the underlying model from the pipeline
        model_for_shap = xgb_pipe_for_shap.named_steps["model"]

        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(X_test_shap)

        # Summary plot: global feature importance via SHAP
        print(f"\n  [SHAP Summary] Global feature importance:")
        shap_abs_mean = np.abs(shap_values).mean(axis=0)
        shap_imp = pd.Series(shap_abs_mean, index=shap_feat_names).sort_values(ascending=False)
        for feat, val in shap_imp.head(10).items():
            print(f"    {feat:30s}  mean|SHAP| = {val:.4f}")

        # SHAP summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_shap, feature_names=shap_feat_names,
                          show=False, max_display=15)
        plt.title("SHAP Summary Plot -- Feature Impact on Survival")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "09_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [Saved] plots/09_shap_summary.png")

        # SHAP waterfall for a single prediction
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for idx, (ax, label) in enumerate(zip(axes, ["Survived", "Died"])):
            # Find a sample with that prediction
            pred_class = 1 if label == "Survived" else 0
            sample_idx = np.where(xgb_pipe_for_shap.predict(X_test) == pred_class)[0][0]
            shap_vals_single = shap_values[sample_idx]
            top_k = 10
            top_idx = np.argsort(np.abs(shap_vals_single))[-top_k:]
            ax.barh([shap_feat_names[i] for i in top_idx],
                    shap_vals_single[top_idx],
                    color=["#e74c3c" if v < 0 else "#2ecc71" for v in shap_vals_single[top_idx]])
            ax.set_title(f"SHAP -- Why model predicted '{label}' (sample {sample_idx})")
            ax.set_xlabel("SHAP value (impact on prediction)")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "10_shap_individual.png", dpi=150)
        plt.close()
        print("  [Saved] plots/10_shap_individual.png")

        print(f"\n  SHAP Interpretation Guide:")
        print(f"    - Positive SHAP value -> pushes prediction toward Survived")
        print(f"    - Negative SHAP value -> pushes prediction toward Died")
        print(f"    - Magnitude = strength of that feature's influence")
        print(f"    - Red/blue dots in summary = high/low feature values")

    except ImportError:
        print(f"\n  [SKIP] shap not installed. Install with: pip install shap")
        print(f"  SHAP would provide:")
        print(f"    - Global feature importance (which features matter most)")
        print(f"    - Local explanations (why THIS prediction was made)")
        print(f"    - Interaction effects (how features work together)")
    except Exception as e:
        print(f"\n  [SHAP Error] {e}")
        print(f"  (SHAP analysis skipped due to compatibility issue)")

    # ==================================================================
    # STEP 13: LEARNING CURVES
    # ==================================================================
    print_section("STEP 13: LEARNING CURVES")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    lc_models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("XGBoost", xgb.XGBClassifier(n_estimators=100, eval_metric="logloss",
                                        random_state=42, verbosity=0)),
    ]
    for ax, (name, model) in zip(axes, lc_models):
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        train_sizes, train_scores, val_scores = learning_curve(
            pipe, X_train, y_train, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10), scoring="accuracy",
            n_jobs=-1
        )
        ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train", color="#2ecc71")
        ax.fill_between(train_sizes,
                         train_scores.mean(axis=1) - train_scores.std(axis=1),
                         train_scores.mean(axis=1) + train_scores.std(axis=1),
                         alpha=0.1, color="#2ecc71")
        ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation", color="#e74c3c")
        ax.fill_between(train_sizes,
                         val_scores.mean(axis=1) - val_scores.std(axis=1),
                         val_scores.mean(axis=1) + val_scores.std(axis=1),
                         alpha=0.1, color="#e74c3c")
        ax.set_title(f"Learning Curve -- {name}")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="lower right")
        ax.set_ylim(0.6, 1.05)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_learning_curves.png", dpi=150)
    plt.close()
    print("  [Saved] plots/11_learning_curves.png")
    print(f"\n  How to read learning curves:")
    print(f"    - Train >> Val = overfitting (need more data or simpler model)")
    print(f"    - Train ~= Val (both low) = underfitting (need more complex model)")
    print(f"    - Train ~= Val (both high) = good fit!")

    # ==================================================================
    # STEP 14: FINAL MODEL & PERSISTENCE
    # ==================================================================
    print_section("STEP 14: FINAL MODEL SELECTION & PERSISTENCE")

    # Select best model (tuned XGBoost or RF)
    final_model = xgb_search.best_estimator_ if xgb_search.best_score_ > rf_grid.best_score_ \
        else rf_grid.best_estimator_
    final_name = "XGBoost (tuned)" if xgb_search.best_score_ > rf_grid.best_score_ \
        else "Random Forest (tuned)"

    y_pred_final = final_model.predict(X_test)
    y_proba_final = final_model.predict_proba(X_test)[:, 1]

    print(f"\n  Best model: {final_name}")
    print(f"  Final Test Metrics:")
    print_metrics(final_name, y_test, y_pred_final, y_proba_final)

    # Save model
    model_path = PLOTS_DIR.parent / "best_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"\n  Model saved to: {model_path}")

    # Demonstrate loading
    loaded_model = joblib.load(model_path)
    sample = X_test.iloc[:3]
    predictions = loaded_model.predict(sample)
    probas = loaded_model.predict_proba(sample)[:, 1]
    print(f"\n  Loading saved model and predicting on 3 samples:")
    for i in range(3):
        print(f"    Sample {i+1}: predicted={'Survived' if predictions[i] else 'Died'}, "
              f"probability={probas[i]:.3f}, actual={'Survived' if y_test.iloc[i] else 'Died'}")

    # ==================================================================
    # STEP 15: SUMMARY
    # ==================================================================
    print_section("FINAL SUMMARY")
    print("""
    REAL-WORLD SKILLS PRACTICED IN THIS PIPELINE:
    -----------------------------------------------
    1.  DATA EXPLORATION      -- Shape, types, distributions, missing patterns
    2.  DATA CLEANING         -- Drop redundant, impute missing, cap outliers
    3.  FEATURE ENGINEERING   -- family_size, is_alone, fare_per_person, has_deck
    4.  EDA VISUALIZATIONS    -- Survival rates, correlations, distributions
    5.  PREPROCESSING PIPELINE-- ColumnTransformer + Pipeline (reproducible!)
    6.  MODEL COMPARISON      -- 9 classifiers with 5-fold CV
    7.  CLASS IMBALANCE       -- class_weight='balanced', SMOTE
    8.  UNSUPERVISED FEATURES -- PCA reduction, K-Means clusters, DBSCAN outliers
    9.  FEATURE SELECTION     -- Mutual Information, RFE, SelectKBest
    10. HYPERPARAMETER TUNING -- GridSearchCV, RandomizedSearchCV
    11. STACKING ENSEMBLE     -- Meta-learner combining diverse models
    12. SHAP EXPLAINABILITY   -- Global + local model explanations
    13. LEARNING CURVES       -- Diagnose overfitting vs underfitting
    14. MODEL PERSISTENCE     -- Save/load with joblib
    15. ALL 12 ALGORITHMS     -- Every classical ML algo contributed!

    ALGORITHMS USED:
    -----------------------------------------------
    SUPERVISED:  LogReg, DecTree, RF, GradBoost, SVM, KNN, NaiveBayes, XGB, LGBM
    UNSUPERVISED: PCA (dim reduction), K-Means (clusters), DBSCAN (outliers)
    ADVANCED:    Stacking (meta-learning), SHAP (explainability)
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
