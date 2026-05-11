# Phase 5: Optimization

> Tune the winner and consider an ensemble. The phase beginners over-invest in. Most of the gains came from Phases 2-4; Phase 5 polishes. Diminishing returns are real — set a budget and stick to it.

---

## What this phase is for

You have 2-3 candidate models from Phase 4, each beating baseline. Phase 5:

1. **Hyperparameter tuning** — squeeze 1-5% more out of the leader by tuning its key parameters.
2. **Ensemble / stacking** — combine the top 2-3 models for an additional 0.5-2% if it's worth the complexity cost.

The honest truth: in modern classical ML with strong defaults (LightGBM, XGBoost, sklearn ensembles), Phase 5 rarely delivers > 5% improvement. The exceptions are when you find a hyperparameter that's wildly off (`learning_rate=0.1` should have been `0.01` and converged differently) or your training data is small and regularisation matters a lot.

**Beginner trap:** spending 60% of project time here because tuning feels like "the ML part". It's not. Phase 2 (features) was the ML part. Phase 5 is the bow on top.

---

## Where you are in the pipeline

```
   PHASE 1 — Understand the Problem
   PHASE 2 — Data Exploration & Cleaning
   PHASE 3 — Feature Selection & Preprocessing
   PHASE 4 — Model Selection & Training
►► PHASE 5 — Optimization                    ◄◄  (you are here)
   PHASE 6 — Evaluation & Validation
   PHASE 7 — Deployment
```

## How much time to spend here

Roughly **10% of total project time**. Less than Phase 4. Often a few hours of compute + an hour or two of looking at results.

If you're spending more than 10% here and gains are < 2%, the problem isn't tuning — it's features or target framing. Go back to Phase 2.

---

## Step 5.1 — Hyperparameter Tuning

### What it is

Searching the hyperparameter space of a model to find the configuration that maximises validation performance. Three main strategies, in order of cost:

- **GridSearchCV** — exhaustive over a grid. Use for < 1K rows or small param grids.
- **RandomizedSearchCV** — random sampling from distributions. Use as the default.
- **Optuna (Bayesian)** — model-guided search. Use for large search spaces or production tuning.

### Which strategy to pick

| Budget | Rows | Param grid size | Use |
|---|---|---|---|
| < 30 min compute | Any | Small (< 20 combos) | GridSearchCV |
| 1-2 hours compute | Any | Medium (20-200 combos) | RandomizedSearchCV |
| > 2 hours / production | Any | Large (> 200 combos) | Optuna |
| Need reproducibility above all | Any | Small | GridSearchCV |

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear'],
}

gs = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=2000),
    param_grid,
    cv=5,
    scoring='average_precision',
    n_jobs=-1,
    verbose=1,
)
gs.fit(X_train, y_train)
print(f'Best: {gs.best_params_}')
print(f'Best score: {gs.best_score_:.3f}')
```

**Watch out:** grid size is multiplicative. 5 × 2 × 1 = 10 combos × 5 folds = 50 fits. With 10 hyperparameters and 3 values each, that's 5,000 fits — infeasible. Use Random or Optuna at scale.

### RandomizedSearchCV — the default

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
import lightgbm as lgb

param_dist = {
    'n_estimators':      randint(300, 1500),
    'learning_rate':     loguniform(0.01, 0.2),
    'num_leaves':        randint(15, 128),
    'max_depth':         randint(3, 12),
    'min_child_samples': randint(20, 300),
    'reg_alpha':         loguniform(1e-3, 10),
    'reg_lambda':        loguniform(1e-3, 10),
    'subsample':         [0.7, 0.8, 0.9, 1.0],
}

rs = RandomizedSearchCV(
    lgb.LGBMClassifier(scale_pos_weight=13.0, n_jobs=-1),
    param_dist,
    n_iter=80,
    cv=5,
    scoring='average_precision',
    n_jobs=1,           # n_jobs=-1 inside LightGBM, so 1 outside
    verbose=1,
    random_state=42,
)
rs.fit(X_train, y_train)
print(f'Best: {rs.best_params_}')
print(f'Best score: {rs.best_score_:.3f}')
```

Random search beats Grid search on the same compute budget in almost every case — most hyperparameters don't matter, so spending budget on a grid over all of them wastes most of your tries.

### Optuna (Bayesian search)

```python
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves':        trial.suggest_int('num_leaves', 15, 128),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 300),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'scale_pos_weight':  13.0,
    }
    model = lgb.LGBMClassifier(**params, n_jobs=-1)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring='average_precision').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=80, timeout=3600)

print(f'Best: {study.best_params}')
print(f'Best score: {study.best_value:.3f}')
```

Optuna's TPE sampler learns from previous trials — it doesn't waste time exploring regions of param space that aren't promising. Typically gets to a good solution in 30-50% fewer trials than Random.

### Per-model param grids that matter

Not all hyperparameters are equally important. Here's the short list per family:

#### LightGBM / XGBoost

| Param | What it controls | Range |
|---|---|---|
| `n_estimators` | Number of trees | 300-1500 |
| `learning_rate` | Step size per tree | 0.01-0.2 (log-uniform) |
| `num_leaves` / `max_depth` | Tree complexity | 15-128 / 3-12 |
| `min_child_samples` | Min rows per leaf — anti-overfit | 20-300 |
| `reg_alpha` / `reg_lambda` | L1 / L2 regularisation | 1e-3 to 10 (log) |
| `subsample` | Row subsample per tree | 0.7-1.0 |
| `colsample_bytree` | Column subsample per tree | 0.7-1.0 |

`n_estimators` and `learning_rate` interact — high LR + few trees ≈ low LR + many trees. Tune them together.

#### Random Forest

| Param | Range |
|---|---|
| `n_estimators` | 200-1000 |
| `max_depth` | None or 5-30 |
| `min_samples_split` | 2-20 |
| `min_samples_leaf` | 1-10 |
| `max_features` | 'sqrt', 'log2', or 0.3-0.8 |

#### LogReg / Ridge / Lasso

| Param | Range |
|---|---|
| `C` (LogReg) / `alpha` (Ridge/Lasso) | 0.001 to 100 (log-uniform) |
| `penalty` (LogReg) | l1 / l2 / elasticnet |
| `l1_ratio` (ElasticNet only) | 0.0 to 1.0 |

For ElasticNet, `l1_ratio=0.5` is a common starting point; tune downward (toward Ridge) on noisy features or upward (toward Lasso) for selection.

#### SVM (SVC, SVR)

| Param | Range |
|---|---|
| `C` | 0.1-100 (log) |
| `kernel` | linear, rbf, poly |
| `gamma` (RBF) | 'scale', 'auto', or 0.001-1.0 (log) |

SVM training is O(n²) or O(n³). Don't tune SVM on > 10K rows unless you have time.

### Watch out for…

- **Tuning on the test set**: `train_test_split` first, tune on train (using CV inside), evaluate on test ONCE. Tuning on test = inflated metric.
- **Tuning to noise**: if `n_iter=80` and the score range is ± 0.005, you're tuning to noise. Either reduce iter count or expand the search space.
- **Forgetting to set `random_state`**: random search results vary across runs without it. Reproducibility matters.
- **Different hyperparameters with the same outcome**: `learning_rate=0.05, n_estimators=500` ≈ `learning_rate=0.1, n_estimators=250`. Don't tune both axes blindly — fix one and tune the other.
- **n_jobs=-1 on both inner and outer loops**: causes thread contention. Pick one.

### When NOT to tune

- The metric difference between bake-off candidates is < 2% — tuning won't save you. Go back to Phase 2.
- The model is already at the "noise floor" of the task (e.g., recall@k is bound by label noise, not by tuning).
- Phases 1-4 weren't done thoroughly. Tuning amplifies mistakes from earlier phases.

---

## Step 5.2 — Ensemble / Stacking

### What it is

Combining multiple models to outperform any single one. Three approaches:

- **Voting / averaging** — average predictions from N models. Simplest.
- **Stacking** — a meta-model learns how to combine base-model predictions.
- **Blending** — weighted manual combination.

### When to ensemble

Ensembling helps when models make **different kinds of mistakes**:

- Linear model + tree model — linear catches linear patterns, tree catches interactions. Errors are somewhat uncorrelated.
- Same family but different hyperparameters — almost never worth it. Errors are too correlated.

If your top two candidates' errors are highly correlated (they get wrong the same rows), ensembling gives you almost nothing.

```python
# Quick check: are model errors correlated?
import numpy as np
errors_lgbm = (y_val != preds_lgbm).astype(int)
errors_xgb  = (y_val != preds_xgb).astype(int)
print(f'Error correlation: {np.corrcoef(errors_lgbm, errors_xgb)[0, 1]:.2f}')
# > 0.8 → errors highly correlated, ensemble probably won't help.
# < 0.5 → errors complementary, ensemble might help.
```

### When NOT to ensemble

- **Real-time latency budgets**: stacking 3 models = 3× inference cost. If you have a 100ms budget, this often kills ensembling.
- **Explainability requirements**: stacking blurs which model is "the model". Per-row SHAP on a stack is messier.
- **Marginal improvement**: if voting gets +0.5% and you're shipping production, the maintenance cost of multiple models usually isn't worth it.

### Voting / averaging (the simplest ensemble)

```python
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

ensemble = VotingClassifier(
    estimators=[
        ('lgbm',   lgb.LGBMClassifier(scale_pos_weight=13.0, n_estimators=500)),
        ('logreg', LogisticRegression(class_weight='balanced', max_iter=2000)),
    ],
    voting='soft',     # average predicted probabilities (recommended)
)
ensemble.fit(X_train, y_train)
```

For regression:

```python
from sklearn.ensemble import VotingRegressor

ensemble = VotingRegressor(
    estimators=[
        ('lgbm',  lgb.LGBMRegressor(n_estimators=500)),
        ('ridge', Ridge(alpha=1.0)),
    ],
)
```

### Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stack = StackingClassifier(
    estimators=[
        ('lgbm',  lgb.LGBMClassifier(scale_pos_weight=13.0, n_estimators=500)),
        ('xgb',   xgb.XGBClassifier(scale_pos_weight=13.0, n_estimators=500)),
        ('ridge', LogisticRegression(class_weight='balanced', max_iter=2000)),
    ],
    final_estimator=LogisticRegression(),    # simple meta-model
    cv=5,                                     # CV for base predictions (no leakage)
    n_jobs=-1,
)
stack.fit(X_train, y_train)
```

The meta-model learns weights for combining base predictions. Use a **simple** meta-model (LogisticRegression or Ridge) — complex meta-models over-fit.

### Manual blending

```python
# Blend two models manually with weights tuned on validation
import numpy as np

probs_lgbm  = lgbm_model.predict_proba(X_val)[:, 1]
probs_logreg = logreg_model.predict_proba(X_val)[:, 1]

best_score = 0
best_w = 0.5
for w in np.linspace(0, 1, 21):
    blended = w * probs_lgbm + (1 - w) * probs_logreg
    score = average_precision_score(y_val, blended)
    if score > best_score:
        best_score = score
        best_w = w

print(f'Best blend: LGBM = {best_w:.2f}, LogReg = {1 - best_w:.2f}')
print(f'Best PR-AUC: {best_score:.3f}')
```

Often the result is something like `0.7 LGBM + 0.3 LogReg` — the linear model contributes a small but meaningful boost in tail regions.

### Common ensemble shapes from the catalog

| Brief | Stack |
|---|---|
| House price prediction | LightGBM + XGBoost + ElasticNet → Ridge meta. +0.5% RMSE vs single LightGBM. |
| Hospital LOS | 55% two-stage + 45% single LightGBM (errors complementary on long-stay tail). |
| Ride-share surge | 60% two-stage + 40% single LightGBM. |
| Insurance claims | 70% two-stage + 30% Tweedie. |
| Fraud detection | **No stacking** — latency budget rejects it. Single LGBM. |
| Loan default (regulated) | **No stacking** — LogReg only for ECOA compliance. |

Notice the pattern: ensembling shows up most often in regression problems where errors are complementary and latency budgets are generous. Real-time and regulated contexts almost always reject it.

### Watch out for…

- **Stacking with the wrong CV**: base-model predictions for training the meta-model MUST come from out-of-fold predictions, or the meta-model overfits massively. `StackingClassifier` handles this if you set `cv=` correctly.
- **Stacking models that are nearly identical**: two LightGBMs with different seeds are too correlated. Mix model *families*.
- **Inference cost blow-up**: stacking 3 models with 500 trees each = 1500 trees + meta-model inference per row. Profile latency.
- **Calibration**: a stacked model is typically *not* calibrated. Wrap with `CalibratedClassifierCV` after stacking if probabilities matter.

---

## Decision tree for this phase

```
Step 5.1 — Hyperparameter Tuning
  Budget < 30 min and small grid?  → GridSearchCV
  Budget 1-2 hours, medium grid?    → RandomizedSearchCV
  Budget > 2 hours OR production?  → Optuna

  Tune on TRAIN (with CV inside). Never on test.
  Fix random_state for reproducibility.
  Stop if iteration-to-iteration delta < 0.5% on score.

Step 5.2 — Ensemble / Stacking
  Real-time latency budget?         → NO. Single model.
  Regulated / explainability hard?  → NO. Single linear or single tree + SHAP.
  Errors of top 2 models correlated > 0.8?  → NO. Skip.

  Otherwise:
    Quick test → Voting (soft)
    Diverse families → Stacking with simple meta-model
    Manual tuning → blending by weight grid

  After stacking → calibrate if probabilities matter (Phase 6).
```

---

## Common mistakes in this phase

| Mistake | Why bad | Fix |
|---|---|---|
| Tuning on the test set | Inflated metric; fails in production | CV on train; test ONCE in Phase 6 |
| Tuning before bake-off | Wastes budget on wrong model family | Phase 4 picks the family; Phase 5 tunes it |
| GridSearchCV with 10 params × 5 values | 5M+ fits → infeasible | RandomizedSearchCV or Optuna |
| Stacking models with correlated errors | Marginal gain, big complexity cost | Check error correlation first |
| Stacking with `cv=None` | Base predictions overfit; meta-model trains on noise | Always set `cv=5` on Stacking |
| Stacking + real-time | Latency budget blown | Single model for real-time |
| Tuning to noise (range < 1σ of CV) | Win is fake | Either reduce iter count or rethink features |
| Skipping calibration after stacking | Probabilities meaningless | Wrap with CalibratedClassifierCV in Phase 6 |

---

## Worked example from the catalog

[`/scenarios/regression/04_house_pricing`](/scenarios/regression/04_house_pricing) — full Phase 5 walkthrough:

1. **Tuning** — Optuna on LightGBM, 80 trials, CV with `KFold(5)`:
   ```python
   def objective(trial):
       params = {
           'n_estimators':      trial.suggest_int('n_estimators', 500, 1500),
           'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
           'num_leaves':        trial.suggest_int('num_leaves', 16, 96),
           'reg_alpha':         trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
           'reg_lambda':        trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
       }
       ...
   ```
   Best: n_est=900, lr=0.04, num_leaves=48, reg_α=0.5, reg_λ=2.0 → RMSE 0.119 (log).
2. **Ensemble** — StackingRegressor: LightGBM + XGBoost + ElasticNet → Ridge meta. RMSE 0.114 (log) = ~$13K MAE on raw $200K-median target. Single LightGBM was $14K MAE — gain ~$1K per house.
3. **Latency check** — agent tool accepts 50ms per listing; stack = 35ms. Ships.

[`/scenarios/classification/01_fraud_detection`](/scenarios/classification/01_fraud_detection) — opposite outcome:

1. **Tuning** — Optuna on LightGBM, 100 trials. Best PR-AUC: 0.51.
2. **Ensemble experiment** — stacking LightGBM + XGBoost + LogReg → +0.5% PR-AUC. **Rejected:** inference latency went from 0.5ms to 2.4ms — too close to the 100ms budget. Shipped single LightGBM.

The two examples illustrate the recurring pattern: ensembling is a tradeoff between marginal gain and complexity/latency cost. Get the tradeoff right for your brief.

---

## Cheat sheet

```
TUNING
  Strategy by budget:
    GridSearchCV          small grids, < 30 min
    RandomizedSearchCV    default, 1-2 hours
    Optuna (TPE)          large spaces, production

  Params that matter most:
    LGB/XGB: n_estimators, learning_rate, num_leaves/max_depth,
             min_child_samples, reg_alpha, reg_lambda
    RF:      n_estimators, max_depth, min_samples_leaf, max_features
    LogReg:  C, penalty
    SVM:     C, kernel, gamma

  Tune on TRAIN (CV inside). Test set ONCE in Phase 6.
  Stop when iteration delta < 0.5%.

ENSEMBLE
  Check first: error correlation between top 2 models.
    > 0.8 → skip ensemble (won't help)
    < 0.5 → promising

  Shape:
    Voting (soft)         simplest; average probs
    Stacking + meta-model larger gain, more complexity
    Manual blending       weight grid; pick on val

  ALWAYS reject ensemble for:
    real-time latency budget
    regulated single-model requirement
    < 1% marginal gain in CV
```

---

## What comes next

You have a tuned (and maybe ensembled) model. Now you have to verify it works on data it hasn't seen, produce the metrics you'll report to the team, and decide whether the model's decisions are explainable and fair. **Phase 6** is the held-out test evaluation and the explainability layer — both required before deployment.
