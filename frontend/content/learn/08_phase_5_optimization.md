# Phase 5: Optimization

> Tune the winner. Maybe ensemble. Diminishing returns are real — budget ~10% of project time.

---

## Step 5.1 — Hyperparameter tuning

Pick by budget:

| Compute | Param grid | Use |
|---|---|---|
| < 30 min | Small (< 20 combos) | `GridSearchCV` |
| 1–2 hours | Medium (20–200 combos) | `RandomizedSearchCV` |
| > 2 hours / production | Large | `Optuna` (Bayesian TPE) |

**Always tune on train (with CV inside).** Never on test. Fix `random_state` for reproducibility. Stop when iteration delta < 0.5%.

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
        'scale_pos_weight':  13.0,
    }
    return cross_val_score(lgb.LGBMClassifier(**params, n_jobs=-1),
                           X_train, y_train, cv=cv, scoring='average_precision').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=80, timeout=3600)
```

### Params that matter per model

| Model | Tune |
|---|---|
| LightGBM / XGBoost | `n_estimators`, `learning_rate`, `num_leaves` / `max_depth`, `min_child_samples`, `reg_alpha`, `reg_lambda`, `subsample` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| LogReg / Ridge / Lasso | `C` / `alpha`, `penalty`, `l1_ratio` (ElasticNet) |
| SVM | `C`, `kernel`, `gamma` |

---

## Step 5.2 — Ensemble (optional)

Quick test first — are top-2 model errors correlated?

```python
errors_lgbm = (y_val != preds_lgbm).astype(int)
errors_xgb  = (y_val != preds_xgb).astype(int)
print(np.corrcoef(errors_lgbm, errors_xgb)[0, 1])
# > 0.8 → skip ensemble (won't help)
# < 0.5 → promising
```

If promising:

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stack = StackingClassifier(
    estimators=[('lgbm', lgbm_model), ('xgb', xgb_model), ('ridge', ridge_model)],
    final_estimator=LogisticRegression(),
    cv=5, n_jobs=-1,
)
```

**Reject ensemble when:** real-time latency budget, regulated single-model requirement, marginal gain < 2%, or top-2 errors highly correlated.

---

## Common mistakes

| Mistake | Fix |
|---|---|
| Tuning on the test set | CV on train; test once in Phase 6 |
| GridSearch over 10 params × 5 values | Use RandomizedSearch or Optuna at scale |
| Tuning to noise (delta < 1σ of CV) | Reduce iters or revisit features |
| Stacking models with correlated errors | Check correlation first; skip if > 0.8 |
| Stacking + real-time | Latency budget blown; ship single model |
| Forgetting calibration after stacking | Wrap with `CalibratedClassifierCV` in Phase 6 |
