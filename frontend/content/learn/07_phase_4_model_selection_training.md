# Phase 4: Model Selection & Training

> Pick the model family, train candidates, handle imbalance or target shape. This is the phase beginners over-invest in — but it's structured and quick once Phases 1-3 are done right. Most of the work is bake-offs, not deep hyperparameter tuning.

---

## What this phase is for

You have clean features, a split, and a metric. Phase 4 picks the model family that fits your data and trains the candidates. Steps in order:

1. **Baseline** — DummyClassifier or DummyRegressor. The floor your model must beat.
2. **Bake-off** — 4-6 different model families with default hyperparameters, evaluated with cross-validation.
3. **Model selection** — narrow to the top 2-3 using the *N questions* framework for your problem type.
4. **Handle imbalance / target transformation** — pick the right tool for the data's shape.

By the end of Phase 4 you have 2-3 candidate models that beat baseline, with their CV scores recorded. Optimisation (hyperparameter tuning, ensembling) happens in Phase 5.

The rule of thumb: **15% of project time**. The bake-off is fast (5 models × CV takes minutes-to-hours). Most beginner mistakes happen here because they skip baseline and skip multi-model comparison — going straight to "I'll use XGBoost" without justifying the choice.

---

## Where you are in the pipeline

```
   PHASE 1 — Understand the Problem
   PHASE 2 — Data Exploration & Cleaning
   PHASE 3 — Feature Selection & Preprocessing
►► PHASE 4 — Model Selection & Training   ◄◄  (you are here)
   PHASE 5 — Optimization
   PHASE 6 — Evaluation & Validation
   PHASE 7 — Deployment
```

## How much time to spend here

Roughly **15% of total project time**. About a day on a normal project — running the bake-off, reading CV scores, picking candidates, applying imbalance / target-transformation strategies.

---

## Step 4.1 — Baseline

### What it is

The trivial model. Predicts the most-frequent class (classification) or the mean (regression). Gives you the **floor** — what your "real" model must beat to justify its existence.

If your sophisticated XGBoost model can't beat a `DummyClassifier`, you don't have a problem with the model — you have a problem with your features (Phase 2) or your target (Phase 1).

```python
from sklearn.dummy import DummyClassifier, DummyRegressor

# Classification baselines
dc_majority = DummyClassifier(strategy='most_frequent')
dc_stratified = DummyClassifier(strategy='stratified')   # samples from class distribution

dc_majority.fit(X_train, y_train)
print(f'Majority-class baseline accuracy: {dc_majority.score(X_test, y_test):.3f}')

# Regression baselines
dr_mean = DummyRegressor(strategy='mean')
dr_median = DummyRegressor(strategy='median')

dr_mean.fit(X_train, y_train)
print(f'Mean baseline RMSE: {mean_squared_error(y_test, dr_mean.predict(X_test), squared=False):.3f}')
```

### Beyond Dummy — the rule-based baseline

For most production-quality projects, also include a **simple rule** as a second baseline:

```python
# Fraud: rule baseline = "flag if amount > $1000 OR new merchant"
rule_pred = ((X_test['amount'] > 1000) | (X_test['merchant_age_days'] < 30)).astype(int)

# House price: rule baseline = "average price per square foot × house's sqft"
avg_psf = (y_train / X_train['total_sf']).median()
rule_pred = X_train['total_sf'] * avg_psf
```

Your ML model must beat the rule baseline by a meaningful margin. If it doesn't, **ship the rule** — it's simpler, faster, and easier to debug.

### Watch out for…

- **Reporting a model "beats baseline by 5%" without reporting the baseline itself.** Always show both numbers.
- **Using accuracy as the baseline metric for imbalanced classification.** The majority-class baseline can hit 99% accuracy on extreme imbalance. Use PR-AUC or recall@k from the start.

---

## Step 4.2 — Multi-Model Bake-Off

### What it is

Try **4-6 model families** at default hyperparameters with cross-validation. Pick the top 2-3 to tune in Phase 5.

This is the most important step you'll do in Phase 4. Beginners often pick one model (usually XGBoost) and tune it heavily. Experts try several first and discover the right family — sometimes the answer is a humble Logistic Regression, sometimes it's an ensemble of unrelated models.

### The classification menu

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

models = {
    'logreg': LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000),
    'rf':     RandomForestClassifier(n_estimators=400, class_weight='balanced', n_jobs=-1),
    'gbm':    GradientBoostingClassifier(n_estimators=300),
    'lgbm':   LGBMClassifier(n_estimators=500, scale_pos_weight=13.0),
    'xgb':    XGBClassifier(n_estimators=500, scale_pos_weight=13.0, eval_metric='aucpr'),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    pipe = Pipeline([('pre', preprocessor), ('clf', model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='average_precision', n_jobs=-1)
    print(f'{name:6s} PR-AUC = {scores.mean():.3f} ± {scores.std():.3f}')
```

### The regression menu

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

models = {
    'ridge':       Ridge(alpha=1.0),
    'lasso':       Lasso(alpha=0.01),
    'elasticnet':  ElasticNet(alpha=0.01, l1_ratio=0.5),
    'rf':          RandomForestRegressor(n_estimators=400, n_jobs=-1),
    'lgbm':        LGBMRegressor(n_estimators=500),
    'xgb':         XGBRegressor(n_estimators=500),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    pipe = Pipeline([('pre', preprocessor), ('reg', model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    print(f'{name:10s} MAE = {-scores.mean():.3f} ± {scores.std():.3f}')
```

### Reading the bake-off

Look for:

- **The leader** — the model with the best mean score. Tune this one in Phase 5.
- **Close seconds** — anything within 1-2% of the leader. These become candidates for ensembling.
- **Stability** — low standard deviation means the model is stable; high std means it's overfitting some folds. Prefer stable models.
- **Linear vs tree** — if a regularised LogReg / Ridge is competitive (within 5%) with LightGBM, prefer the linear model for explainability and latency.

### Watch out for…

- **CV with the wrong split**: use `StratifiedKFold` for classification, `TimeSeriesSplit` for temporal data, `GroupKFold` if entities span rows. Plain `KFold` on imbalanced classification often gives folds with zero positives.
- **Comparing with the wrong metric**: optimise for what the brief asks for (PR-AUC, MAE, recall@k), not for accuracy or RMSE.
- **Defaults that already include class weighting** (LGBM `scale_pos_weight`, LogReg `class_weight='balanced'`): set them in the bake-off too, not just after.

---

## Step 4.3 — Choosing a Classification Model (the 8 questions)

When the bake-off is close or you want to justify the model family choice, use this framework.

### Q1. How many rows?

| Model | Minimum rows | Why |
|---|---|---|
| Logistic Regression (+ L1/L2/EN) | 30+ | Simple model, fast convergence |
| Decision Tree | 100+ | Can memorise small datasets |
| KNN | 50+, slow above 10K | Distance computation is expensive |
| SVM | 50-10K | Training time explodes on large data |
| Naive Bayes | 30+ | Gold standard for text |
| Random Forest | 500+ | Needs data to build diverse trees |
| XGBoost / LightGBM | 500-1K+ | Designed for medium-large data |

**Rule:** < 500 rows → eliminate Random Forest / boosters. > 10K rows → eliminate KNN / SVM.

### Q2. How many features?

| Features | Best models |
|---|---|
| Few (< 10) | All work fine — start simple (LogReg) |
| Medium (10-50) | Tree-based, regularised LogReg |
| Many (50-500) | Lasso LogReg (auto-selects), LightGBM, Naive Bayes |
| Very many (500+, e.g. TF-IDF) | Multinomial NB, Lasso LogReg, Linear SVM, ElasticNet |

**Rule:** If features > rows (e.g., 5K TF-IDF terms, 1K rows), MUST use regularised linear or Naive Bayes. Tree models and plain LogReg will overfit.

### Q3. Linearly separable?

| Boundary | Best models |
|---|---|
| Linearly separable (low/high spend by income) | LogReg, Linear SVM, Naive Bayes |
| Non-linear (interactions, complex patterns) | Decision Tree, RF, XGBoost, LightGBM, KNN, RBF SVM |
| Don't know | Try both — start LogReg, then a tree model |

**Quick check:** plot pairs of features as scatter plots coloured by class. If a straight line separates them, it's linear.

### Q4. Noisy data, outliers, mislabeled examples?

| Situation | Avoid | Use |
|---|---|---|
| Feature outliers | KNN (distance distorted) | Trees, Naive Bayes |
| Mislabeled examples | Decision Tree, KNN | RF, LogReg + L2, XGBoost + early stopping |
| Very noisy features | Decision Tree | RF, regularised LogReg |
| Hard / overlapping classes | Hard-margin SVM | LogReg, GBM with calibration |

### Q5. Need to explain?

| Audience | Use | Avoid |
|---|---|---|
| Non-technical / client | LogReg, Decision Tree | XGBoost, RF (without SHAP) |
| Regulated (banking, healthcare, lending) | LogReg, Decision Tree (+ SHAP if needed) | Pure black-box |
| Just need accuracy (Kaggle, internal tool) | XGBoost, LightGBM, Stacking | — |

### Q6. How fast does prediction need to be?

| Speed | Use | Avoid |
|---|---|---|
| Real-time (< 1ms) | LogReg, Naive Bayes, small Decision Tree | KNN, large RF |
| Batch (seconds OK) | Any | — |
| Fast training | LogReg, Naive Bayes, LightGBM | SVM, GBM |

### Q7. Class balance?

| Imbalance | Strategy | Models |
|---|---|---|
| Balanced (50/50 - 60/40) | None needed | Any |
| Mild (70/30 - 80/20) | F1 / ROC-AUC, stratify | Any with stratified CV |
| Moderate (85/15 - 90/10) | class_weight, threshold tune | LogReg, RF, XGBoost |
| Severe (95/5) | SMOTE / scale_pos_weight; PR-AUC | XGBoost, LightGBM |
| Extreme (99/1+) | scale_pos_weight + threshold + calibration | XGBoost, LightGBM, calibrated LogReg |

Detailed handling in Step 4.6 below.

### Q8. Binary or multiclass?

| Model | Binary | Multiclass |
|---|---|---|
| LogReg | native | native (`multinomial`) |
| Decision Tree | native | native |
| RF / XGBoost / LightGBM | native | native |
| SVM | native | One-vs-Rest or One-vs-One |
| KNN | native | native |
| Naive Bayes | native | native |

For multiclass with > 10 classes, prefer `multinomial` LogReg over OvR (one model vs many).

---

## Step 4.4 — Choosing a Regression Model (the 6 questions)

### Q1. How many rows?

| Model | Minimum rows |
|---|---|
| Linear Regression / Ridge / Lasso / ElasticNet | 30+ |
| Decision Tree | 100+ |
| KNN | 50+, slow > 10K |
| SVR | 50-10K |
| RandomForest | 500+ |
| GradientBoosting / XGBoost / LightGBM | 500-1K+ |

### Q2. How many features?

| Features | Best models |
|---|---|
| Few (< 10) | Linear, Ridge — easy to interpret |
| Medium (10-50) | Ridge, RF, LightGBM |
| Many (50-500) | Lasso (auto-select), ElasticNet, LightGBM |
| Very many (500+) | Lasso, ElasticNet, Linear SVR |

### Q3. Linear or non-linear relationship?

| Relationship | Best models |
|---|---|
| Linear (price vs sqft) | Linear / Ridge / Lasso / ElasticNet |
| Non-linear (complex interactions) | RF, XGBoost, LightGBM, KNN |
| Don't know | Try both — Ridge first, then LightGBM |

### Q4. Outliers in target or features?

| Situation | Use | Avoid |
|---|---|---|
| Outliers in target | Quantile regression, Huber loss, MAE-objective tree | RMSE-objective; outliers dominate |
| Outliers in features | Trees, robust scaling | KNN, SVR |
| Mostly clean | Any | — |

### Q5. Need to explain?

Same as classification Q5. Linear / Ridge / Lasso give interpretable coefficients.

### Q6. Speed?

| Speed | Use |
|---|---|
| Real-time | Linear, small tree |
| Batch | Any |
| Big data | LightGBM, Linear SGD |

---

## Step 4.5 — Choosing a Clustering Model

Clustering is different — no labels, so no metric in the usual sense. The first question is "should I cluster at all?":

```
Are you trying to:
  - Segment customers for marketing             → cluster (or rule-based segmentation)
  - Detect anomalies                            → anomaly detection, not clustering
  - Find structure in unlabelled data            → cluster
  - Compress features                            → dimensionality reduction (PCA), not clustering
  - Predict labels                               → classification, not clustering
```

If clustering is genuinely the answer:

| Question | If yes | If no |
|---|---|---|
| Do you know K (number of clusters)? | KMeans, MiniBatchKMeans | DBSCAN (auto-finds K), HDBSCAN |
| Are clusters roughly spherical? | KMeans | DBSCAN (any shape) |
| Different density clusters? | HDBSCAN | DBSCAN |
| Hierarchical structure useful? | Agglomerative | KMeans / DBSCAN |
| Need probabilistic assignment? | Gaussian Mixture (GMM) | KMeans (hard assignment) |

**Validation:** silhouette score, Calinski-Harabasz, Davies-Bouldin, OR domain expert validates the segments visually.

---

## Step 4.6 — Handling Class Imbalance

### When to reach for it — the signals

You need imbalance handling when **all three** are true:

- Minority class < ~30%.
- The minority class is the one you care about.
- Errors are not symmetric.

### The 5 main tools

#### A. `class_weight='balanced'` (Linear models, RF)

Tells the loss function to weight rare-class examples higher.

```python
LogisticRegression(class_weight='balanced', max_iter=2000)
RandomForestClassifier(class_weight='balanced', n_estimators=400)
# Custom:
RandomForestClassifier(class_weight={0: 1, 1: 5})
```

**Use when:** moderate imbalance (5-30%); first line of defense; works without breaking anything.

**Watch out:** changes the probability the model outputs — they're no longer well-calibrated. Follow up with `CalibratedClassifierCV` if you ship probabilities.

#### B. `scale_pos_weight` (XGBoost, LightGBM)

The gradient-boosting equivalent of `class_weight`. Set to `negatives / positives`.

```python
import lightgbm as lgb
n_pos = (y_train == 1).sum()
n_neg = (y_train == 0).sum()
spw = n_neg / n_pos       # e.g., 13.3 at 7% prevalence

lgb.LGBMClassifier(scale_pos_weight=spw, n_estimators=500)
```

**Use when:** moderate to severe imbalance with tree boosters (which usually win tabular bake-offs).

**Watch out:** LightGBM with `scale_pos_weight` is almost always over-confident — wrap with `CalibratedClassifierCV(method='isotonic')` if you ship probabilities.

#### C. Threshold tuning

Don't change the loss — change the threshold at which you call something "positive". Default 0.5 is rarely right under imbalance.

```python
import numpy as np
probs = model.predict_proba(X_val)[:, 1]
k = 3500   # team capacity
threshold = np.sort(probs)[-k]     # threshold yielding exactly k positives
```

**Use when:** ALWAYS, for any imbalanced classification with cost asymmetry or fixed budget.

Detail covered in Phase 6.

#### D. SMOTE

Generate synthetic minority-class samples by interpolating between real ones.

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

pipe = Pipeline([
    ('smote', SMOTE(sampling_strategy=0.1, random_state=42)),
    ('model', LogisticRegression()),
])
```

**Use when:** very small absolute counts of positives (few hundred); numeric features only.

**Watch out:**
- Apply SMOTE only to training data, never to validation/test (use `imblearn.pipeline.Pipeline`, NOT sklearn's).
- Synthetic samples on high-cardinality categoricals (cuisine, ZIP) create incoherent rows — drop or target-encode first.
- Almost never the best answer; class_weight + threshold tuning usually wins.

#### E. Focal loss

Down-weights easy-to-classify examples, up-weights hard ones. Extreme imbalance only.

```python
import lightgbm as lgb
# Custom objective; see lightgbm-focal-loss for full implementation
model = lgb.LGBMClassifier(objective=focal_obj)
```

**Use when:** < 0.5% prevalence after `scale_pos_weight` isn't enough.

### Decision flowchart for imbalance

```
What's the positive-class rate?

  > 30%       → No handling. Default settings.
  5-30%       → class_weight='balanced' (LogReg/RF) OR
                scale_pos_weight = neg/pos (LGB/XGB)
                + threshold-tune
                + calibrate if probs consumed
  0.5-5%      → Same as above. Verify class_weight is having effect.
  < 0.5%      → class_weight + threshold + calibrate.
                If validation PR-AUC still low: consider anomaly framing
                (Isolation Forest) or focal loss.
```

### Common combinations

The standard pattern for an imbalanced classifier:

```python
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import numpy as np

# 1. Class weighting
spw = (y_train == 0).sum() / (y_train == 1).sum()
base = lgb.LGBMClassifier(scale_pos_weight=spw, n_estimators=500)
base.fit(X_train, y_train)

# 2. Calibrate
model = CalibratedClassifierCV(base, cv='prefit', method='isotonic')
model.fit(X_val, y_val)

# 3. Threshold-tune
probs = model.predict_proba(X_val)[:, 1]
threshold = np.sort(probs)[-3500]
```

**scale_pos_weight + threshold tune** is the right answer 80% of the time.

---

## Step 4.7 — Target Transformation (Regression)

### When to reach for it — the signals

Plot the target. Then ask:

- **Skew > 1, strictly positive**: right-skewed → log1p or Box-Cox.
- **> 30% zeros**: zero-inflated → two-stage or Tweedie.
- **P99 / P50 > 5**: long-tailed → log transform, or quantile loss for asymmetric cost.
- **Asymmetric cost**: under-predicting costs differently than over → quantile loss.
- **High-stakes tail rows**: sample weighting.

```python
print(f'Skew: {y.skew():.2f}  Zero fraction: {(y == 0).mean():.2%}')
print(f'P99/P50: {y.quantile(0.99) / y.quantile(0.5):.1f}x')
```

### The 6 strategies

#### A. Log transform (`np.log1p`)

```python
import numpy as np
y_train_log = np.log1p(y_train)
model.fit(X_train, y_train_log)

# Predict and invert
y_pred = np.expm1(model.predict(X_test))

# Report metrics on log AND raw scale
rmse_log = mean_squared_error(np.log1p(y_test), model.predict(X_test), squared=False)
mae_raw  = mean_absolute_error(y_test, y_pred)
```

**Use when:** right-skewed, strictly positive — house prices, salaries, claim amounts.

**Watch out:** `log1p` adds 1 inside so 0 maps to 0. Predictions on log scale invert to *median* not mean.

#### B. Box-Cox / Yeo-Johnson

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
y_train_t = pt.fit_transform(y_train.values.reshape(-1, 1)).ravel()
```

**Use when:** linear models specifically. Yeo-Johnson handles any sign; Box-Cox is strictly positive only.

#### C. Two-stage modelling (zero-inflated)

```python
# Stage 1: P(y > 0)
clf_base = lgb.LGBMClassifier(scale_pos_weight=4.0)
clf_base.fit(X_train, (y_train > 0).astype(int))
clf = CalibratedClassifierCV(clf_base, cv='prefit', method='isotonic')
clf.fit(X_val, (y_val > 0).astype(int))

# Stage 2: E[y | y > 0] on positive rows
positive = y_train > 0
reg = lgb.LGBMRegressor(objective='regression_l1')
reg.fit(X_train[positive], np.log1p(y_train[positive]))

# Combine
p_positive = clf.predict_proba(X_test)[:, 1]
y_given_positive = np.expm1(reg.predict(X_test))
y_pred = p_positive * y_given_positive
```

**Use when:** > 30% zeros — insurance claims, ad clicks, equipment failures.

**Watch out:** Stage 1 calibration is mandatory. Verify positive count ≥ 1,000 before committing to Stage 2.

#### D. Tweedie regression

Single-model alternative to two-stage. Interpolates between Poisson (count-like) and Gamma (continuous-positive).

```python
model = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.5,    # try 1.3, 1.5, 1.7
    n_estimators=400,
)
```

**Use when:** zero-inflated + light-positive tail. Insurance pure premium, ad CPC.

**Watch out:** can underestimate the heavy tail. Often blended with two-stage for the best of both.

#### E. Sample weighting

```python
import numpy as np
weights = np.where(y_train > 14, 3.0, 1.0)
model.fit(X_train, y_train, sample_weight=weights)
```

**Use when:** target tail is operationally important — under-predicting long hospital stays, peak demand, etc.

**Watch out:** 3-5× max weight. Validation must use the same weights for comparable metrics.

#### F. Quantile regression

```python
quantiles = [0.1, 0.5, 0.9]
models = {q: lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=400).fit(X_train, y_train)
          for q in quantiles}

y_low  = models[0.1].predict(X_test)
y_med  = models[0.5].predict(X_test)
y_high = models[0.9].predict(X_test)
```

**Use when:** need uncertainty bands; asymmetric cost; capacity / inventory planning.

**Watch out:** median (q=0.5) is not the mean. Quantile predictions can "cross" — fit with monotonic constraints if you see that.

### Decision flowchart for target transformation

```
Plot the target. Then:

  Symmetric, no zeros          → no transform; raw RMSE
  Right-skewed positive        → log1p (default), Box-Cox (linear models)
  Zero-inflated (> 30% zeros)  → Two-stage, Tweedie, or blend
  Bounded 0-1                  → logit transform
  Asymmetric cost              → quantile regression
  Need uncertainty band        → multi-quantile regression
  High-stakes tail             → sample weighting (combine with another transform)
  Censored time-to-event       → survival models (out of scope)
```

---

## Decision tree for this phase

```
Step 4.1 — Baseline
  Run DummyClassifier(most_frequent) / DummyRegressor(mean).
  Run a simple rule-based baseline too.
  Record both scores.

Step 4.2 — Bake-Off
  Try 5-6 models with defaults + stratified/time/group CV.
  Use the metric from Phase 1 (PR-AUC, MAE, etc.).
  Identify top 2-3 by mean score, prefer lower std.

Step 4.3-4.5 — Justify model family choice
  Use the N-questions framework (8 for classification, 6 for regression).
  If linear + tree-based are within 5%, prefer linear for explainability.

Step 4.6 — Handle imbalance (classification)
  Imbalance < 30%? scale_pos_weight or class_weight + threshold tune.
  Imbalance < 0.5%? Add anomaly framing or focal as fallback.

Step 4.7 — Target transformation (regression)
  Skewed positive? log1p.
  Zero-inflated? Two-stage or Tweedie.
  Asymmetric cost? Quantile loss.
```

---

## Common mistakes in this phase

| Mistake | Why bad | Fix |
|---|---|---|
| Skipping baseline | Can't tell if model adds value | DummyClassifier + rule baseline every time |
| Trying one model only | Miss a better family | At least 4-5 in the bake-off |
| Tuning before bake-off | Wastes tuning budget on the wrong model | Bake-off first, tune the winner |
| Reaching for SMOTE before class_weight | More complex, often worse | class_weight first; SMOTE only if needed |
| Forgetting `stratify` on imbalanced CV | Folds with zero positives → undefined metric | `StratifiedKFold` mandatory |
| RMSE on heavily-skewed regression target | Top 1% dominates training | log1p the target; report MAE on raw scale |
| Optimising on training data | Inflated; the metric you report is fake | CV with held-out folds |
| Choosing XGBoost just because Kaggle | Linear might be better + more explainable | Justify with the 8 questions / 6 questions |

---

## Worked example from the catalog

[`/scenarios/classification/01_fraud_detection`](/scenarios/classification/01_fraud_detection):

1. **Baseline** — DummyClassifier (always 0): accuracy 99.83%, recall 0%. Rule baseline (`amount > $1000 OR new merchant`): recall 0.42, precision 0.03.
2. **Bake-off**:
   ```
   logreg    PR-AUC = 0.34 ± 0.02
   rf        PR-AUC = 0.42 ± 0.03
   gbm       PR-AUC = 0.45 ± 0.02
   xgb       PR-AUC = 0.47 ± 0.02
   lgbm      PR-AUC = 0.48 ± 0.01   ← winner
   ```
3. **Model family choice** — Q1 (1.5M rows: keep boosters); Q7 (extreme imbalance: scale_pos_weight); Q6 (real-time <100ms: single model, not stacking). LightGBM wins.
4. **Imbalance** — `scale_pos_weight = 577` (1/0.0017). SMOTE tried, abandoned (merchant ID interpolation incoherent).
5. **Threshold** — tuned in Phase 6 (cost-driven: 0.5 → 0.15).

[`/scenarios/regression/04_house_pricing`](/scenarios/regression/04_house_pricing):

1. **Baseline** — Mean ($180K predicted): MAE = $54K. Rule baseline (avg-$/sqft × sqft): MAE = $34K.
2. **Bake-off**:
   ```
   ridge       MAE = 0.130 (log) → ~$22K raw
   lasso       MAE = 0.131
   elasticnet  MAE = 0.126        ← winner among linear
   rf          MAE = 0.135
   lgbm        MAE = 0.121        ← winner overall
   ```
3. **Target transformation** — log1p (skew = 1.6). Report MAE on raw scale ($14K) AND MAPE (5%).
4. **Model family choice** — ElasticNet very close to LightGBM; chose stacking (Phase 5) to combine.

---

## Cheat sheet

```
Step 4.1 — Baseline
  DummyClassifier / DummyRegressor + rule baseline. Beat both.

Step 4.2 — Bake-off
  5-6 models, defaults, CV with the right metric.
  Pick top 2-3 by mean - 1σ.

Step 4.3-5 — Justify
  Classification: 8 questions (rows, features, linearity, noise,
                    explainability, speed, imbalance, # classes)
  Regression:     6 questions (rows, features, linearity, outliers,
                    explainability, speed)
  Clustering:     "Should I cluster at all?" + K-shape, density, hierarchy

Step 4.6 — Imbalance
  First: class_weight (LogReg/RF) OR scale_pos_weight (LGB/XGB)
  Always: threshold tuning (Phase 6)
  Probs consumed: + isotonic calibration
  Stratified K-fold mandatory
  SMOTE only if positives small + numeric only
  Focal only at <0.5%

Step 4.7 — Target transformation
  Skewed positive → log1p
  Zero-inflated → two-stage OR Tweedie
  Asymmetric cost → quantile loss
  High-stakes tail → sample weighting
```

---

## What comes next

You have 2-3 candidate models, all beating baseline, with imbalance / target shape addressed. **Phase 5** is optimisation: tune the hyperparameters of the top model, maybe ensemble two of them, and decide whether the gain is worth the complexity. Most beginners spend too long here; the rule of thumb is **10% of project time** on Phase 5 combined.
