# Phase 4: Model Selection & Training

> Baseline → bake-off → choose the family → handle imbalance / target shape.

---

## Step 4.1 — Baseline

Two baselines, always:

```python
from sklearn.dummy import DummyClassifier
DummyClassifier(strategy='most_frequent').fit(X_train, y_train)   # the floor

# + a simple rule from domain knowledge
rule = ((X_test['amount'] > 1000) | (X_test['merchant_age_days'] < 30)).astype(int)
```

If your model can't beat the rule, **ship the rule**.

---

## Step 4.2 — Bake-off (5–6 models, defaults, CV)

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
import lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = {
    'logreg': LogisticRegression(class_weight='balanced', max_iter=2000),
    'rf':     RandomForestClassifier(n_estimators=400, class_weight='balanced', n_jobs=-1),
    'lgbm':   lgb.LGBMClassifier(n_estimators=500, scale_pos_weight=spw),
    'xgb':    xgb.XGBClassifier(n_estimators=500, scale_pos_weight=spw, eval_metric='aucpr'),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in models.items():
    scores = cross_val_score(Pipeline([('pre', pre), ('clf', m)]), X, y, cv=cv, scoring='average_precision')
    print(f'{name:6s}  {scores.mean():.3f} ± {scores.std():.3f}')
```

Pick **top 2–3** by mean − 1 σ. Prefer the simpler family when within ~5% (e.g. LogReg over LGBM if interpretable). LGBM/XGB typically win tabular.

---

## Step 4.3 — Model family quick guide

**Classification:**

| Condition | Best models |
|---|---|
| < 500 rows | LogReg, Naive Bayes, Decision Tree |
| > 10K rows | Drop KNN + SVM (too slow) |
| Features ≫ rows (text, TF-IDF) | Lasso LogReg, Naive Bayes, Linear SVM |
| Need explainability | LogReg, Decision Tree (or tree + SHAP) |
| Real-time < 1 ms | LogReg, Naive Bayes |
| Default winner | LightGBM / XGBoost |

**Regression:**

| Condition | Best models |
|---|---|
| Linear relationship | Linear / Ridge / Lasso / ElasticNet |
| Non-linear | RF / LightGBM / XGBoost |
| Outliers in target | Quantile regression or Huber loss |
| Need explainability | Ridge / Lasso (coefficients) |

---

## Step 4.4 — Handle imbalance (classification)

```python
n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
spw = n_neg / n_pos    # e.g. 13 at 7% positive
```

| Positive rate | Strategy |
|---|---|
| > 30% | Nothing |
| 5–30% | `class_weight='balanced'` (LogReg/RF) **or** `scale_pos_weight=spw` (LGBM/XGB) + threshold-tune (Phase 6) |
| 0.5–5% | Same as above. Calibrate if probabilities are consumed. |
| < 0.5% | + focal loss, or reframe as anomaly detection |

**SMOTE** only if positives are small in absolute count AND features are numeric only. Usually `scale_pos_weight + threshold tune` wins.

---

## Step 4.5 — Target transformation (regression)

Plot the target. Then:

| Target shape | Transform |
|---|---|
| Symmetric, no zeros | None — raw RMSE |
| Right-skewed, strictly positive | `np.log1p` |
| > 30% zeros (zero-inflated) | Two-stage (classifier × regressor) **or** Tweedie (`objective='tweedie'`) |
| Bounded 0–1 | Logit transform |
| Asymmetric cost | Quantile regression at relevant quantile |
| High-stakes tail rows | Sample weighting (`fit(..., sample_weight=…)`) |

Report metrics on **both** transformed and raw scale.

---

## Common mistakes

| Mistake | Fix |
|---|---|
| Skipping baseline | DummyClassifier + rule baseline every time |
| Tuning one model before bake-off | Bake-off first, then tune the winner |
| `StratifiedKFold` missing on imbalanced data | Folds with zero positives = undefined metric |
| RMSE on a skewed target | `log1p` the target; report MAE on raw scale too |
| Reaching for SMOTE first | `class_weight` / `scale_pos_weight` is usually enough |
| Picking XGBoost because Kaggle | Justify via the rows/features/explainability/speed signals |
