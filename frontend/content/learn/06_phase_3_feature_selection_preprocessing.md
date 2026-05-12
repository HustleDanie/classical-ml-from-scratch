# Phase 3: Feature Selection & Preprocessing

> Narrow features, transform each column, **split before preprocessing**. Order matters.

---

## Step 3.1 — Feature selection (3 stages)

1. **Common sense** — drop IDs, constants, names, confirmed leakers.
2. **Statistical filter** — mutual information + correlation collapse.

   ```python
   from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
   mi = mutual_info_classif(X_train, y_train, random_state=42)
   ranking = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
   # Drop MI < 0.001. For correlated pairs (>0.9) drop one.
   ```

3. **Automated** — only if > 100 features remain. Tree importance via fast RandomForest, **or** L1 (Lasso) for linear-model selection.

---

## Step 3.2 — Train/test split (do this BEFORE preprocessing)

| Data shape | Strategy |
|---|---|
| Cross-sectional, no time, no groups | Stratified random (`train_test_split(stratify=y)`) |
| Production predicts the future | Time-based (sort by date, split at a cutoff) |
| Entities span many rows; production sees new entities | Group-based (`GroupKFold` / `StratifiedGroupKFold`) |
| Both time AND groups | Group-and-time (different groups in test AND test in the future) |
| Production retrains periodically | Walk-forward (`TimeSeriesSplit`) |
| Tuning + unbiased final metric | Nested CV (outer + inner loops) |

**Always stratify on imbalanced classification.** Velocity / rolling features force a time-based split.

---

## Step 3.3 — Preprocessing

Use `ColumnTransformer` + `Pipeline`. **Fit on train only**, transform both:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=50), cat_cols),
])

pipe = Pipeline([('pre', pre), ('clf', model)])
pipe.fit(X_train, y_train)        # fit on train
y_pred = pipe.predict(X_test)     # transform via train's params
```

| Feature type | Transform |
|---|---|
| Numeric, normal-ish | `StandardScaler` |
| Numeric, skewed | `RobustScaler` or `log1p` (Phase 2) |
| Categorical 2–50 | `OneHotEncoder(handle_unknown='ignore')` |
| Categorical > 50 | Target encoding (Phase 2) |
| Ordinal | `OrdinalEncoder` with explicit `categories=[…]` |
| Text | `TfidfVectorizer` (Phase 2) |

Tree models don't need scaling, but scaling them is harmless and lets you swap models without rework.

---

## Common mistakes

| Mistake | Fix |
|---|---|
| Preprocessing before train/test split | **Split first, then fit on train only** |
| Random split on temporal data | Time-based — inflates metric by 5–20pp otherwise |
| Random split with multi-row entities | Group-based — otherwise per-entity overfit |
| `OneHotEncoder` without `handle_unknown='ignore'` | Test-time unseen category crashes the pipeline |
| Saving just the model, not the preprocessor | Save the full `Pipeline` — production can't reproduce inputs otherwise |
