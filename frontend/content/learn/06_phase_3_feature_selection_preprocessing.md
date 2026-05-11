# Phase 3: Feature Selection & Preprocessing

> Now narrow the feature set, transform each column appropriately, and split into train/test. **Order matters**: split FIRST, then fit preprocessing on train only. Mixing this up is the single most common data-leakage mistake in classical ML.

---

## What this phase is for

Phase 2 produced a clean DataFrame with potentially hundreds of features. Phase 3 prepares it for modeling:

1. **Feature selection** — keep what helps, drop the rest. Three stages: common-sense drops → statistical filters → automated methods.
2. **Preprocessing** — scale numerics, encode categoricals, vectorise text. Each feature type gets a different transformation.
3. **Train/test split** — divide into train, validation, and test. The split strategy depends on data structure (random / stratified / time-based / group-based / walk-forward / nested).

By the end of Phase 3 you have `X_train`, `X_val`, `X_test` arrays — preprocessed, with the right features, ready for modeling.

The cardinal rule: **train/test split happens BEFORE any preprocessing is fit**. Fitting a StandardScaler on the full dataset and then splitting leaks test-set statistics (mean, std) into training. This is a real, silent, ~5-10% metric inflation that fails in production.

---

## Where you are in the pipeline

```
   PHASE 1 — Understand the Problem
   PHASE 2 — Data Exploration & Cleaning
►► PHASE 3 — Feature Selection & Preprocessing   ◄◄  (you are here)
   PHASE 4 — Model Selection & Training
   PHASE 5 — Optimization
   PHASE 6 — Evaluation & Validation
   PHASE 7 — Deployment
```

## How much time to spend here

Roughly **25% of total project time**. Feature selection is usually fast (an hour or two — most of the work is just running the methods and reading the output). Preprocessing setup is structured (ColumnTransformer). Train/test split is a one-line decision, but a critical one.

---

## Step 3.1 — Feature Selection (3 stages)

### What it is

Reducing the feature set from "all engineered features" to "features that meaningfully contribute". Done in three stages, in order:

1. **Common-sense drops** — IDs, names, constants, leakers.
2. **Statistical filters** — mutual information, correlation, chi-square.
3. **Automated methods** — RFE, L1 regularisation, tree-based importance.

Each stage is faster + simpler than the next. Stop at the earliest stage that's enough.

### Stage 1 — Common Sense (free, instant)

Drop anything that obviously shouldn't be a feature:

```python
# IDs and unique identifiers
drop_cols = []
for c in df.columns:
    if df[c].nunique() == len(df):
        drop_cols.append(c)        # every row unique → ID

# Constants
for c in df.columns:
    if df[c].nunique() == 1:
        drop_cols.append(c)        # never varies → no signal

# Names / text fields with no structure
drop_cols += ['customer_name', 'address', 'description']  # unless you've engineered features from them

# Confirmed leakers (re-audit after Phase 2)
drop_cols += ['discharge_diagnosis', 'chargeback_amount', 'inspector_id']

df = df.drop(columns=[c for c in drop_cols if c in df.columns])
```

Stage 1 typically removes 5-30% of columns and costs nothing. Do it.

### Stage 2 — Statistical Filters (cheap, model-agnostic)

Rank features by their relationship with the target. The two workhorse methods:

#### Mutual Information

Captures both linear and non-linear relationships. Robust to monotonic transformations. Use this as your default.

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# For classification
mi = mutual_info_classif(X_train, y_train, random_state=42)

# For regression
mi = mutual_info_regression(X_train, y_train, random_state=42)

import pandas as pd
ranking = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
print(ranking.head(20))
```

**Thresholds (rule of thumb):**
- MI > 0.05 → strong predictor; keep.
- MI 0.01-0.05 → moderate; keep, but expect modest contribution.
- MI 0.001-0.01 → borderline; keep cheap features (small dimensionality cost), drop expensive ones.
- MI < 0.001 → drop.

#### Correlation (numeric features only)

Drop features highly correlated with another feature (multicollinearity).

```python
import numpy as np

corr = X_train.select_dtypes(include='number').corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
print(f'Dropping {len(to_drop)} features with > 0.9 correlation with another feature')
X_train = X_train.drop(columns=to_drop)
```

**Threshold:** correlation > 0.9 with another feature → drop one. The choice of which to drop should be informed by domain knowledge (keep the simpler / more interpretable one).

#### Chi-square (categorical features and classification)

```python
from sklearn.feature_selection import chi2

# Categorical features only — must be non-negative
chi_scores, p_values = chi2(X_train_cat, y_train)
```

Less commonly used today (MI handles categoricals well too). Mostly relevant for legacy code or text classification with raw counts.

### Stage 3 — Automated Methods (expensive, model-aware)

Use only if Stages 1-2 left you with too many features (> 100) and you need to narrow further.

#### Tree importance (the practical default)

Train a fast forest, read its feature importances. Captures non-linear interactions.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

importance = pd.Series(rf.feature_importances_, index=X_train.columns)\
    .sort_values(ascending=False)
print(importance.head(20))

# Drop the bottom 30%
keep = importance[importance > importance.quantile(0.3)].index
X_train = X_train[keep]
```

**Watch out.** Tree importance is biased toward high-cardinality features. Sanity-check with permutation importance:

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(rf, X_val, y_val, n_repeats=10, n_jobs=-1, random_state=42)
perm_imp = pd.Series(result.importances_mean, index=X_val.columns).sort_values(ascending=False)
print(perm_imp.head(20))
```

#### L1 regularisation (Lasso zeroing)

Logistic Regression or Lasso with L1 penalty sets unimportant coefficients to exactly 0 — automatic selection.

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
lr.fit(X_train, y_train)
selected = X_train.columns[lr.coef_[0] != 0]
print(f'L1 kept {len(selected)} of {X_train.shape[1]} features')
```

Best for linear models specifically. Don't use it to select features for tree models — they value different things.

#### Recursive Feature Elimination (RFE)

Iteratively trains the model, drops the weakest feature, repeats. Slow but principled.

```python
from sklearn.feature_selection import RFE
rfe = RFE(RandomForestClassifier(n_estimators=100), n_features_to_select=20)
rfe.fit(X_train, y_train)
print('Selected:', X_train.columns[rfe.support_].tolist())
```

Use sparingly — slow on > 1K rows or > 100 features. Stage 2 + tree importance usually gets the same answer faster.

### Decision tree for feature selection

```
Start with all features after Phase 2.

  Stage 1 (free):
    Drop IDs, constants, names, confirmed leakers.

  Stage 2 (cheap):
    Compute MI for every feature.
    Drop features with MI < 0.001.
    Drop one of each pair with correlation > 0.9.

  After Stage 2, still > 100 features?
    Yes → Stage 3 (expensive): tree importance OR L1 → keep top 50-100
    No  → done; proceed to preprocessing.
```

### Watch out for…

- **Feature selection on the full dataset** leaks the target into the selection. Always do it on training data only, using the SAME folds you'll use for CV.
- **Different selections per CV fold** is more rigorous but rarely worth the engineering effort. Pick once on train, use everywhere.
- **Feature selection is not free**: each removed feature is a piece of information you can't get back. Err on the side of keeping if the cost is small.

---

## Step 3.2 — Train/Test Split

### What it is

Dividing the data into train, validation, and test sets. The split decides what your metrics *mean*. Six strategies cover essentially every real ML problem:

### The 6 strategies

| # | Strategy | Use when |
|---|---|---|
| A | Stratified random | Cross-sectional, no time/group structure; imbalanced classes |
| B | Time-based | Production predicts the future |
| C | Group-based | Each entity has many rows; production sees new entities |
| D | Group-and-time | Both apply (most production cases) |
| E | Walk-forward (TimeSeriesSplit) | Periodic retraining; need cross-period perf estimate |
| F | Nested CV | Hyperparameter tuning + unbiased final metric |

### Why split happens HERE (and not after preprocessing)

**The cardinal rule of Phase 3**: split data BEFORE fitting any preprocessor. Otherwise statistics from the test set (mean, std, encoder mapping) leak into training.

Wrong:
```python
# ❌ DON'T DO THIS — leaks test stats into train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # mean and std computed using TEST data
X_train, X_test = train_test_split(X_scaled)
```

Right:
```python
# ✓ DO THIS — fit scaler on train ONLY
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # transform with train's params
```

This is why the order matters and why Phase 3 sequences as **selection → split → preprocessing**.

### Strategy A — Stratified random split

The floor for any classification problem; everything else builds on top.

```python
from sklearn.model_selection import train_test_split, StratifiedKFold

# 60/20/20 split with class stratification
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# Stratified K-fold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

For **regression with long-tail targets**, stratify by target quartile:

```python
import pandas as pd
y_quartile = pd.qcut(y, 4, labels=False)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_quartile, random_state=42
)
```

### Strategy B — Time-based split

If production predicts the future, the test set must be the future. **Never shuffle.**

```python
df = df.sort_values('date').reset_index(drop=True)

train_end = '2024-06-30'
val_end   = '2024-09-30'

train = df[df['date'] <= train_end]
val   = df[(df['date'] > train_end) & (df['date'] <= val_end)]
test  = df[df['date'] > val_end]
```

**Watch out.**
- Velocity / rolling-window features (Pattern A in Phase 2) make this MANDATORY — those features are computed up to time T, so test must be after T.
- Pick a clean cutoff (end of day, end of week). Don't split inside a day.

### Strategy C — Group-based split

Each entity (`patient_id`, `user_id`, `device_id`) appears in only one of train, val, or test.

```python
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold

# Single split: 80/20 by patient
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['patient_id']))

# K-fold: each patient in exactly one fold
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=df['patient_id']):
    ...

# Stratified + group (best of both)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

**Watch out.** Random splits on multi-visit-per-patient data leak the patient's pattern between train and test — your test metric is a per-patient overfit estimate.

### Strategy D — Group-and-time split (production mirror)

When you have both time and group structure: different groups in train vs test AND test in the future.

```python
import pandas as pd

train_end = '2024-06-30'
groups_seen_in_train = df[df['date'] <= train_end]['user_id'].unique()
groups_for_test = pd.Series(groups_seen_in_train).sample(frac=0.2, random_state=42)

train = df[(df['user_id'].isin(set(groups_seen_in_train) - set(groups_for_test)))
           & (df['date'] <= train_end)]
test  = df[(df['user_id'].isin(groups_for_test)) | (df['date'] > train_end)]

# Verify:
assert len(set(train['user_id']) & set(test['user_id'])) == 0
assert train['date'].max() < test['date'].min()
```

### Strategy E — Walk-forward (TimeSeriesSplit)

Multiple time-based folds, each training on data up to T_i and testing on the period right after. Mirrors a production system that retrains periodically.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f'Fold {fold}: train={len(train_idx)}, val={len(val_idx)}')
```

**Use when** the brief mentions "we retrain weekly / monthly / quarterly", or for honest performance-over-time estimation.

### Strategy F — Nested CV

Outer loop for unbiased test estimation, inner loop for hyperparameter tuning. Eliminates the bias from tuning and reporting on the same data.

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

gs = GridSearchCV(model, param_grid, cv=inner_cv, scoring='average_precision')
nested_scores = cross_val_score(gs, X, y, cv=outer_cv, scoring='average_precision')
print(f'Nested PR-AUC: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}')
```

**Use when** dataset is small and you can't afford a separate held-out test set. Computationally expensive — use on the final candidate, not during exploratory tuning.

### Decision flowchart for split strategy

```
Does data have a TIME dimension AND production predicts the future?
  ├── Yes → Time-based (B).
  │         Does it ALSO have a GROUP dimension?
  │         └── Yes → Group-and-time (D)
  │         Production retrains periodically?
  │         └── Yes → Walk-forward (E) for honest cross-period estimate
  └── No  → continue

Do entities span multiple rows AND production sees new entities?
  ├── Yes → Group-based (C).
  │         Imbalanced classes? → StratifiedGroupKFold.
  └── No  → continue

Cross-sectional, no time, no groups, just rows.
  └── Stratified random (A). Always stratify on classification.

Need hyperparameter tuning AND unbiased final metrics?
  └── Wrap any of the above in nested CV (F).
```

### Combinations per scenario

| Brief | Split |
|---|---|
| Fraud (6 months of transactions) | Time-based + stratify within month for imbalance |
| Customer churn (multi-year accounts) | Group-and-time (D) |
| House price (5 years of sales) | Time-based + stratify by price quartile |
| Hospital readmission (multi-visit, multi-year) | Group-and-time + stratify |
| A/B test outcome | Stratified random by treatment arm |
| Cross-sectional survey | Stratified random by primary covariate |

---

## Step 3.3 — Preprocessing

### What it is

Transforming each column into a numeric form the model can ingest. Different feature types need different transformations.

### The decision table

| Feature type | Transformation |
|---|---|
| Numeric, normal-ish | `StandardScaler` |
| Numeric, skewed / outliers | `RobustScaler` (uses median + IQR) |
| Numeric, bounded (0-100) | `MinMaxScaler` |
| Numeric, very skewed | `np.log1p` (in Phase 2 feature engineering) |
| Categorical, 2-5 values | `OneHotEncoder(handle_unknown='ignore')` |
| Categorical, 6-50 values | `OneHotEncoder` or target encoding |
| Categorical, 50+ values | Target encoding (done in Phase 2 — Pattern B) |
| Ordinal (Poor/Good/Excellent) | `OrdinalEncoder` with explicit order |
| Text | `TfidfVectorizer` (done in Phase 2 — Pattern E) |

### Tree-based vs linear models

**Tree-based models** (LightGBM, XGBoost, RandomForest) don't need scaling. They're scale-invariant. You can pass raw numerics to them.

**Linear models** (LogReg, SVM, Ridge, Lasso, KNN) MUST be scaled. Without scaling, a feature ranging 0-1M dominates the loss compared to a feature ranging 0-1.

The pragmatic choice: scale everything anyway. It's harmless to scale features that don't need it, and it lets you swap models without reworking preprocessing.

### ColumnTransformer — the workhorse

`ColumnTransformer` applies different transformations to different columns in one step. It's the canonical preprocessing pattern in sklearn.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

numeric_cols     = ['age', 'income', 'tenure', 'amount']
categorical_cols = ['cuisine_type', 'ownership_type', 'county']
ordinal_cols     = ['quality']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=50), categorical_cols),
    ('ord', OrdinalEncoder(categories=[['Poor', 'Fair', 'Good', 'Excellent']]), ordinal_cols),
], remainder='passthrough')   # passthrough = keep other columns as-is

# Fit on train, transform train AND test
X_train_t = preprocessor.fit_transform(X_train)
X_test_t  = preprocessor.transform(X_test)
```

### Why fit on train only

```python
# ❌ DON'T
scaler.fit(X)              # uses test data — leakage
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)

# ✓ DO
scaler.fit(X_train)         # train stats only
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)
```

The `fit_transform` / `transform` distinction in sklearn enforces this — `fit_transform(X_train)` computes statistics AND applies them; `transform(X_test)` reuses the trained statistics. If you write `scaler.fit_transform(X)` on the whole dataset and split afterwards, you've leaked.

### Pipeline — chain it all together

`Pipeline` wraps preprocessing + model into a single object you can pass to CV or save with joblib. Always wrap your full preprocessor + model in a pipeline.

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=400, n_jobs=-1)),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

In Phase 4 you'll fit several different models — all of them benefit from this Pipeline pattern.

### Watch out for…

- **`handle_unknown='ignore'`** on OneHotEncoder: required for production robustness. Without it, an unseen category at test time crashes the pipeline.
- **`min_frequency`** parameter: collapses rare categories into "other". Critical for high-cardinality columns to avoid massive sparse matrices.
- **Ordinal feature treated as nominal**: one-hot encoding "Poor", "Fair", "Good", "Excellent" loses the ordering. Use OrdinalEncoder with explicit `categories=[order]`.
- **Saving the preprocessor**: the trained `preprocessor` (or full `Pipeline`) is one of your production artifacts. Save it with joblib alongside the model.

---

## Decision tree for this phase

```
Step 3.1 — Feature Selection
  Stage 1: Drop IDs, constants, names, confirmed leakers.
  Stage 2: Drop features with MI < 0.001; drop one of each pair with corr > 0.9.
  Stage 3 (if still > 100 features): tree importance OR L1, keep top 50-100.

Step 3.2 — Train/Test Split (BEFORE preprocessing)
  Time + group? → Group-and-time
  Time only?    → Time-based
  Group only?   → Group-based (StratifiedGroupKFold for imbalanced)
  Neither?      → Stratified random
  Tuning + unbiased final? → wrap in Nested CV.

Step 3.3 — Preprocessing (fit on train only)
  ColumnTransformer with:
    Numeric → StandardScaler or RobustScaler
    Cat 2-5 → OneHotEncoder
    Cat 50+ → Target encoding (already in Phase 2)
    Ordinal → OrdinalEncoder with explicit categories
    Text    → TfidfVectorizer (already in Phase 2)
  Wrap everything in Pipeline(preprocessor + model).
```

---

## How much time to spend here

| Sub-step | Time |
|---|---|
| Feature selection (3 stages) | 1-3 hours |
| Train/test split decision | 5 minutes — but verify with `assert` checks |
| Preprocessing setup (ColumnTransformer) | 1-2 hours |
| **Total** | **~half a day** |

Most of the work is setup and audit. Once Phase 3 is done, you can iterate quickly through models in Phase 4.

---

## Common mistakes in this phase

| Mistake | Why bad | Fix |
|---|---|---|
| Preprocessing BEFORE train/test split | Test statistics leak into train; metrics inflate | Split first, fit on train only |
| Random split when data is temporal | Future leaks into past; 5-20% metric inflation | Time-based split |
| Random split when entities span rows | Per-entity overfit | Group-based split |
| Not stratifying on imbalanced classification | Some folds may have zero positives | `stratify=y` on every split |
| One-hot encoding 14,000 merchants | Sparse matrix dominates feature space | Target encoding (Phase 2) |
| Forgetting `handle_unknown='ignore'` | Test-time unseen category crashes pipeline | Set it on every OneHotEncoder |
| Skipping feature selection | Hundreds of noise features hurt linear models | At minimum: MI filter + correlation filter |
| Selecting features on the full dataset | Leakage; selection itself uses test data | Select on train only, with the same CV folds |
| Saving model without preprocessor | Production can't reproduce the inputs | Always save Pipeline, not just model |

---

## Worked example from the catalog

[`/scenarios/classification/01_fraud_detection`](/scenarios/classification/01_fraud_detection) demonstrates clean Phase 3 sequencing:

1. **Feature selection** — Stage 1 dropped `inspector_id` (a confounder, not a feature). Stage 2: MI ranking kept top-20 by mutual information; correlation > 0.9 collapsed `total_amount_30d` and `avg_amount_30d` (kept the average).
2. **Train/test split** — Time-based: 5 months train, 1 month test. Stratified within the test month to confirm class balance.
3. **Preprocessing** — `ColumnTransformer` with StandardScaler on numerics (for the LogReg challenger model) and OneHotEncoder on low-cardinality categoricals; target-encoded merchant rates pass through unscaled (already engineered in Phase 2).

[`/scenarios/regression/04_house_pricing`](/scenarios/regression/04_house_pricing) shows the regression analog: stratified by price quartile so train and test both have representation in the long tail.

---

## Cheat sheet

```
ORDER (cardinal rule):
  Select features → split → preprocess
  Never preprocess before split.

FEATURE SELECTION:
  Stage 1: Drop IDs, constants, names, leakers
  Stage 2: MI ranking + correlation > 0.9 collapse
  Stage 3 (if needed): tree importance OR L1, keep top 50-100

SPLIT:
  Time + group → Group-and-time (D)
  Time only    → Time-based (B)
  Group only   → Group-based (C); use StratifiedGroupKFold for imbalance
  Neither      → Stratified random (A)
  Tuning + unbiased report → wrap in Nested CV (F)

  ALWAYS stratify on imbalanced classification.

PREPROCESSING:
  ColumnTransformer with:
    Numeric              → StandardScaler (or RobustScaler if skewed)
    Categorical, 2-50    → OneHotEncoder(handle_unknown='ignore')
    Categorical, 50+     → Target encoding (Phase 2)
    Ordinal              → OrdinalEncoder with explicit order
  Wrap full pipeline in Pipeline().
  Fit on TRAIN ONLY; transform train + test.
  Save the Pipeline, not just the model.
```

---

## What comes next

You have `X_train`, `X_val`, `X_test` arrays, preprocessed, with the right features. **Phase 4** is the model bake-off: try 5+ models with default hyperparameters, pick the top 2-3, and handle class imbalance or target transformation. This is also where you'll absorb the model-selection deep-dives — different problem shapes need different model families.
