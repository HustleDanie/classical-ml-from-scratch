# The Complete Classical ML Pipeline Guide

> **The full end-to-end pipeline to follow every time you're given a classical ML task.**

> **Each phase below has its own deep-dive page.** This guide is the overview — once you've read it, walk through [Phase 1 — Understand the Problem](/learn/04_phase_1_understand_problem) and continue to Phase 7. Each phase page expands the steps with code, decision trees, common mistakes, and worked examples from the scenario catalog.

---

## Phase 1: Understand the Problem (Before Writing Code)

### Step 1: Define the Problem Type

- What are you predicting? (number = regression, category = classification, groups = clustering)
- What metric matters? (accuracy? precision? recall? RMSE? R²?)
- What's the business context? (false positives vs false negatives — which is worse?)

### Step 2: Understand the Data

- How many rows? (hundreds → simple models, millions → scalable models)
- How many features? (10 → keep all, 500 → need feature selection)
- What types? (numeric, categorical, text, dates, mixed?)
- Is the target balanced? (50/50 → fine, 99/1 → need special handling)

---

## Phase 2: Data Exploration & Cleaning

### Step 3: Exploratory Data Analysis (EDA)

```
- df.shape, df.info(), df.describe()
- Check target distribution (df['target'].value_counts() or .hist())
- Check missing values (df.isnull().sum())
- Check duplicates (df.duplicated().sum())
- Visualize: histograms for numeric, bar charts for categorical
- Correlation heatmap (numeric features vs target)
```

### Step 4: Data Cleaning

| Issue | Action |
|-------|--------|
| Missing values < 5% | Drop rows OR fill with median/mode |
| Missing values 5-30% | Impute (median for numeric, mode for categorical) |
| Missing values > 30% | Drop the column (unless domain-critical) |
| Duplicates | Drop exact duplicates |
| Outliers | Cap at 1st/99th percentile, or use robust models (trees) |
| Wrong data types | Convert (e.g., "123" string → 123 int) |
| Leaker columns | Remove (anything that wouldn't be available at prediction time) |

### Step 5: Feature Engineering

```
- Create ratios/interactions (price_per_sqft = price / sqft)
- Extract from dates (day_of_week, month, is_weekend, days_since_X)
- Extract from text (sentiment, length, keyword counts, TF-IDF)
- Bin continuous variables if needed (age → age_group)
- Domain-specific transforms (BMI, debt_to_income, log transforms)
```

---

## Phase 3: Feature Selection & Preprocessing

### Step 6: Feature Selection (3 Stages)

```
Stage 1 — Common Sense: Drop IDs, names, constants, leakers
Stage 2 — Statistical: Correlation, MI, chi-square, F-test
Stage 3 — Automated: SelectKBest, RFE, Lasso zeroing
```

> See [Phase 3 — Feature Selection & Preprocessing](/learn/06_phase_3_feature_selection_preprocessing) for detailed methods, thresholds, and worked examples.

### Step 7: Preprocessing

| Feature Type | Transform |
|-------------|-----------|
| Numeric (normal-ish) | StandardScaler |
| Numeric (skewed/outliers) | RobustScaler or log transform |
| Numeric (bounded, e.g. 0-100) | MinMaxScaler |
| Categorical (2-5 values) | OneHotEncoder |
| Categorical (10+ values) | Target encoding or frequency encoding |
| Ordinal (low/med/high) | OrdinalEncoder |
| Text | TF-IDF or CountVectorizer |

### Step 8: Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# For classification: add stratify=y
# NEVER preprocess BEFORE splitting (data leakage!)
# Fit scaler/encoder on X_train only, then transform both
```

---

## Phase 4: Model Selection & Training

### Step 9: Baseline Model

```
- Classification: DummyClassifier (most_frequent) → gives you the "floor"
- Regression: DummyRegressor (mean) → gives you the floor
- If your real model can't beat this, your features are bad
```

### Step 10: Try Multiple Models (5-Fold CV)

Quick-test at least 4-5 models with DEFAULT hyperparameters:

| Classification | Regression |
|---------------|------------|
| Logistic Regression | Linear Regression / Ridge / Lasso |
| Random Forest | Random Forest |
| XGBoost / LightGBM | XGBoost / LightGBM |
| SVM | SVR |
| KNN | KNN |

Compare cross-validation scores side by side. Pick the top 2-3 performers.

> See [Phase 4 — Model Selection & Training](/learn/07_phase_4_model_selection_training) for detailed model comparison tables and worked examples.

### Step 11: Handle Class Imbalance (Classification Only)

| Imbalance Ratio | Strategy |
|----------------|----------|
| 60/40 to 70/30 | Usually fine, no action needed |
| 80/20 to 90/10 | Use `class_weight='balanced'` or SMOTE |
| 95/5 or worse | SMOTE + undersample majority + use F1/AUC, NOT accuracy |

---

## Phase 5: Optimization

### Step 12: Hyperparameter Tuning

```
- Use GridSearchCV (< 1000 rows) or RandomizedSearchCV (> 1000 rows)
- Tune top 2-3 models only (don't waste time on poor performers)
- Key params to tune:
  - Random Forest: n_estimators, max_depth, min_samples_split
  - XGBoost: learning_rate, max_depth, n_estimators, subsample
  - Logistic Reg: C, penalty
  - SVM: C, kernel, gamma
```

### Step 13: Ensemble / Stacking (Optional but Powerful)

```
- If top 2-3 models are DIFFERENT types (e.g., LR + RF + XGB) → Stack them
- Use StackingClassifier/StackingRegressor with a simple meta-learner
- Typically gives 1-3% improvement over best single model
```

---

## Phase 6: Evaluation & Validation

### Step 14: Final Evaluation on Test Set

| Classification | Regression |
|---------------|------------|
| Accuracy | R² score |
| Precision / Recall / F1 | RMSE / MAE |
| Confusion Matrix | Residual plot |
| ROC curve + AUC | Predicted vs Actual plot |
| Classification Report | |

### Step 15: Explainability

```
- Feature importances (RF/XGBoost built-in)
- Permutation importance (model-agnostic)
- SHAP values (detailed per-prediction explanations)
- Partial dependence plots (how one feature affects predictions)
```

---

## Phase 7: Deployment

### Step 16: Save & Deploy

```python
import joblib
joblib.dump(model, 'model.joblib')                # Save model
joblib.dump(preprocessor, 'preprocessor.joblib')   # Save preprocessor too!
# Deploy as REST API (Flask/FastAPI) or batch predictions
```

---

## The Pipeline as a Flowchart

```
PROBLEM DEFINITION
       |
       v
  EDA & CLEANING ──> Missing values? Outliers? Duplicates?
       |
       v
  FEATURE ENGINEERING ──> Create new features from existing data
       |
       v
  FEATURE SELECTION ──> Remove useless/redundant features
       |
       v
  TRAIN/TEST SPLIT ──> 80/20 split (stratified for classification)
       |
       v
  PREPROCESSING ──> Scale numeric, encode categorical (fit on train only!)
       |
       v
  BASELINE ──> DummyClassifier/DummyRegressor (your floor)
       |
       v
  MODEL COMPARISON ──> 5+ models with 5-fold CV, default params
       |
       v
  TOP 2-3 MODELS ──> Hyperparameter tuning (GridSearch/RandomSearch)
       |
       v
  ENSEMBLE/STACK ──> Combine best models for extra 1-3%
       |
       v
  FINAL EVALUATION ──> Test set metrics, confusion matrix, residuals
       |
       v
  EXPLAINABILITY ──> SHAP, feature importances, partial dependence
       |
       v
  SAVE & DEPLOY ──> joblib.dump() + REST API
```

---

## Common Mistakes to Avoid

| Mistake | Why It's Bad | Fix |
|---------|-------------|-----|
| Preprocessing before train/test split | Data leakage — test info leaks into training | Always split first, then fit on train |
| Using accuracy for imbalanced data | 95% accuracy when 95% is one class = useless | Use F1, AUC, precision, recall |
| Tuning on test set | Overfitting to test set, fake good results | Use CV for tuning, test set only ONCE at the end |
| Only trying one model | Might miss a much better model | Always try 4-5 model types |
| Ignoring feature engineering | Raw features are rarely optimal | Spend 60% of time on features, 20% on models |
| Not checking for leakers | Artificially perfect results that fail in production | Ask "would I have this feature at prediction time?" |
| Scaling after combining train+test | Subtle leakage from test statistics | `scaler.fit(X_train)` then `scaler.transform(X_test)` |

---

## Time Allocation (Real-World Rule of Thumb)

```
|████████████████████████████████████| 40% — EDA + Cleaning + Feature Engineering
|██████████████████                  | 25% — Feature Selection + Preprocessing
|████████████                        | 15% — Model Selection + Tuning
|████████                            | 10% — Evaluation + Explainability
|██████                              |  5% — Deployment
|████                                |  5% — Problem Definition + Planning
```

Most beginners spend 80% on model tuning. Experts spend 65% on data/features.

---

## Quick Reference: Minimum Viable Pipeline

If you're short on time, here's the absolute minimum you should always do:

```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # or Regressor
from sklearn.metrics import classification_report     # or r2_score, mean_squared_error

# 1. Load & explore
df = pd.read_csv('data.csv')
print(df.shape, df.dtypes, df.isnull().sum())

# 2. Clean
df = df.dropna()  # or impute
df = df.drop(columns=['id', 'name'])  # drop useless columns

# 3. Split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train & evaluate
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Score: {scores.mean():.4f} +/- {scores.std():.4f}")

# 6. Final test
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

This gets you 80% of the way. The remaining 20% (feature engineering, tuning, stacking, SHAP) is what separates good from great.

---

## Next: deep-dive each phase

Each phase below has its own page with code, decision trees, common mistakes, and worked examples:

- [Phase 1 — Understand the Problem](/learn/04_phase_1_understand_problem) — read the brief, inventory the data, pick the metric.
- [Phase 2 — Data Exploration & Cleaning](/learn/05_phase_2_data_exploration_cleaning) — EDA + cleaning + feature engineering. The 40%-of-time phase.
- [Phase 3 — Feature Selection & Preprocessing](/learn/06_phase_3_feature_selection_preprocessing) — narrow features, transform, split.
- [Phase 4 — Model Selection & Training](/learn/07_phase_4_model_selection_training) — baseline, bake-off, model selection, imbalance, target transformation.
- [Phase 5 — Optimization](/learn/08_phase_5_optimization) — hyperparameter tuning + ensembling.
- [Phase 6 — Evaluation & Validation](/learn/09_phase_6_evaluation_validation) — held-out test, threshold, calibration, explainability, fairness.
- [Phase 7 — Deployment](/learn/10_phase_7_deployment) — persist artifacts; address the 8 production concerns.
