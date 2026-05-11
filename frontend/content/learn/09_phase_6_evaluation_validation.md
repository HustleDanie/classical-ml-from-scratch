# Phase 6: Evaluation & Validation

> The held-out test set. Used **once**. Plus calibration, threshold tuning, and explainability — the layer that turns a model into something a human can trust and ship.

---

## What this phase is for

You have a tuned (and maybe ensembled) model. Phase 6 verifies it works in production-like conditions:

1. **Final evaluation on test set** — used exactly once. Compute the metrics from Phase 1. Compare against baseline.
2. **Threshold tuning** (classification) — pick the decision threshold from cost / capacity / target-recall on validation, evaluate on test.
3. **Calibration** — make sure the probabilities mean what they claim.
4. **Explainability** — global feature importance + per-row attributions for the audience that needs them.
5. **Fairness audit** — per-group metrics if the model touches a consumer-facing decision.

By the end of Phase 6, you can answer four questions about your model:

- Does it beat baseline on the test set, by how much, on the metrics that matter?
- Can it produce a calibrated probability if someone consumes it?
- Can it explain itself per-row if a human needs to know why?
- Is it fair across the groups you have to audit?

If any answer is "no", you're not ready for Phase 7.

---

## Where you are in the pipeline

```
   PHASE 1 — Understand the Problem
   PHASE 2 — Data Exploration & Cleaning
   PHASE 3 — Feature Selection & Preprocessing
   PHASE 4 — Model Selection & Training
   PHASE 5 — Optimization
►► PHASE 6 — Evaluation & Validation         ◄◄  (you are here)
   PHASE 7 — Deployment
```

## How much time to spend here

Roughly **10% of total project time**. Compute is fast; the deliberate parts are picking the threshold and writing the fairness audit. A day on a normal project.

---

## Step 6.1 — Final Evaluation on Test Set

### What it is

Compute the metrics from Phase 1 on the held-out test set. **Once.** You don't get a second chance — every "re-test after a tweak" leaks the test set into your decision-making and the metric stops being honest.

### Classification metrics

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    brier_score_loss,
)

# Probabilities and predictions at the chosen threshold
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= threshold).astype(int)

print(classification_report(y_test, preds))
print('Confusion matrix:')
print(confusion_matrix(y_test, preds))

print(f'PR-AUC:  {average_precision_score(y_test, probs):.3f}')
print(f'ROC-AUC: {roc_auc_score(y_test, probs):.3f}')
print(f'Brier:   {brier_score_loss(y_test, probs):.3f}')

# Capacity-based metric if relevant
import numpy as np
k = 3500
top_k = np.argsort(-probs)[:k]
print(f'Recall@{k}:    {y_test.iloc[top_k].sum() / y_test.sum():.3f}')
print(f'Precision@{k}: {y_test.iloc[top_k].mean():.3f}')
```

### Regression metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

preds = model.predict(X_test)
print(f'MAE:  ${mean_absolute_error(y_test, preds):,.0f}')
print(f'RMSE: ${mean_squared_error(y_test, preds, squared=False):,.0f}')
print(f'R²:   {r2_score(y_test, preds):.3f}')

# MAPE
mape = np.mean(np.abs((y_test - preds) / y_test))
print(f'MAPE: {mape:.2%}')

# Quantile loss (if relevant)
# Already split into models[0.1, 0.5, 0.9] in Phase 4 / 5
```

### The reporting table

A clean test-set report has 5-6 rows:

```
                Baseline    Tuned LGBM    Ensemble    Δ vs baseline
PR-AUC          0.31        0.49           0.51         +0.20
Recall@3500     0.42        0.66           0.68         +0.26
Precision@3500  0.05        0.20           0.21         +0.16
Brier score     0.155       0.061          0.058        -0.097
Cost ($K)       1240        420            395          -845
```

Always show both the absolute number and the delta vs baseline. The delta is what justifies the model's existence.

### Reading the confusion matrix

For binary classification at your chosen threshold:

```
                Predicted 0      Predicted 1
Actual 0        TN = 12,400      FP = 235
Actual 1        FN = 36           TP = 89
```

Translate it into business language:
- TN = transactions correctly cleared (12,400)
- FP = false alarms that wasted an analyst's time (235)
- FN = fraud we missed (36 transactions, potential $180K loss)
- TP = fraud we caught (89, prevented $445K loss)

Always show the dollar / business-impact version next to the raw counts.

### Residual plots (regression)

```python
import matplotlib.pyplot as plt
import seaborn as sns

residuals = y_test - preds
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(preds, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted'); plt.ylabel('Residual')
plt.title('Residuals vs Predicted')

plt.subplot(1, 2, 2)
sns.histplot(residuals, kde=True)
plt.xlabel('Residual')
plt.title('Residual distribution')
plt.tight_layout(); plt.show()
```

Look for:
- **Pattern in residuals vs predicted** — non-flat means heteroscedasticity (try log transform or quantile loss).
- **Non-normal residuals** — long tails mean outliers the model underestimates.
- **Large residuals concentrated at one end** — boundary effects; model fails in the extremes.

### Watch out for…

- **Re-running the test after a tweak**: this contaminates the test set. Either accept the current result, or get a fresh test set.
- **Comparing models without baseline**: a 0.85 ROC-AUC means nothing without knowing the baseline's score.
- **Reporting only a single metric**: always primary + secondary. Saw this rule in Phase 1.
- **Missing the capacity metric**: if the brief specifies capacity (k = 3,500), you MUST report recall@k. PR-AUC alone isn't enough.

---

## Step 6.2 — Threshold Tuning & Calibration

### Two different problems

**Threshold tuning** picks the cutoff at which you call something "positive". **Calibration** makes sure the score itself means what it says — that "0.7" actually means "70% chance".

You usually need both. They are not the same thing.

### When you need threshold tuning

Always, for any imbalanced problem with cost asymmetry or fixed budget. Specifically when:

- The classes are imbalanced (default 0.5 will rarely fire on the rare class).
- A missed positive and a false alarm have different costs.
- The deployment pattern is "rank and act on top-k".
- The brief specifies a recall target.
- The model uses `class_weight` or `scale_pos_weight` (which shifts probabilities).

### When you need calibration

When the **probability itself** is consumed downstream:

- The score multiplies a dollar amount: insurance pricing, expected loss, ad bidding.
- A regulator will ask "what's the probability we cited?" (FOIA, GDPR, ECOA).
- You feed the score into another model or cost calculation.
- A human reads the probability and is supposed to trust it (doctor reading a risk score).
- You used `class_weight` / `scale_pos_weight` / `is_unbalance=True` — these make models over-confident.

### Threshold strategies

#### A. Cost-driven

```python
import numpy as np

def cost_at(threshold, y_true, probs, c_fp, c_fn):
    pred = (probs >= threshold).astype(int)
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    return c_fp * fp + c_fn * fn

probs_val = model.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.001, 0.999, 999)
costs = [cost_at(t, y_val, probs_val, c_fp=15, c_fn=5000) for t in thresholds]
optimal = thresholds[int(np.argmin(costs))]
print(f'Cost-optimal threshold: {optimal:.3f}')
```

Almost always far below 0.5 for imbalanced problems. Fraud detection often lands at 0.10-0.20.

#### B. Capacity-driven (recall@k)

```python
probs_val = model.predict_proba(X_val)[:, 1]
k = 3500
threshold = np.sort(probs_val)[-k]
print(f'Top-{k} threshold: {threshold:.3f}')
```

Use when there's a fixed team capacity. Pick k from the brief, not from intuition.

#### C. Target-recall

```python
from sklearn.metrics import precision_recall_curve

prec, rec, thr = precision_recall_curve(y_val, probs_val)
target_recall = 0.95
mask = rec >= target_recall
threshold = thr[mask][-1] if mask.any() else 0.0
print(f'Threshold for ≥{target_recall:.0%} recall: {threshold:.3f}')
```

Use for compliance contexts, screening (medical, security), regulated risk scoring.

#### D. F1-maximising

```python
prec, rec, thr = precision_recall_curve(y_val, probs_val)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
optimal = thr[int(np.argmax(f1[:-1]))]
```

Fallback when no explicit costs / budget / recall target.

### Calibration

Three diagnostics in order of trust:

**1. Calibration plot.**

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_val, probs_val, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', label='model')
plt.plot([0, 1], [0, 1], 'k--', label='perfect')
plt.xlabel('Predicted probability'); plt.ylabel('Observed frequency')
plt.legend(); plt.show()
```

If the curve deviates noticeably from the diagonal, you need calibration.

**2. Brier score with and without.**

If `CalibratedClassifierCV` reduces Brier by more than ~10%, you needed it.

**3. Per-decile reliability table.**

```python
import pandas as pd
df = pd.DataFrame({'score': probs_val, 'y': y_val})
df['decile'] = pd.qcut(df['score'], 10, labels=False, duplicates='drop')
print(df.groupby('decile').agg(score_mean=('score', 'mean'), pos_rate=('y', 'mean')))
```

If `score_mean` and `pos_rate` track each other, you're calibrated.

### Calibration methods

#### Isotonic regression (default for tree models)

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(base_model, cv='prefit', method='isotonic')
calibrated.fit(X_val, y_val)
probs_calibrated = calibrated.predict_proba(X_test)[:, 1]
```

Use for tree models with `class_weight` / `scale_pos_weight`. Needs ≥ 1,000 validation samples.

#### Platt scaling (sigmoid)

```python
calibrated = CalibratedClassifierCV(base_model, cv='prefit', method='sigmoid')
```

Use for SVMs specifically; or small validation sets (< 500 positives) where isotonic over-fits.

#### No calibration

LogReg without `class_weight` on roughly balanced data is usually already calibrated. Verify with the calibration plot before adding a layer.

### The full sequence

```python
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

# 1. Train base model with imbalance handling
base = lgb.LGBMClassifier(scale_pos_weight=spw, n_estimators=500)
base.fit(X_train, y_train)

# 2. Calibrate FIRST
calibrated = CalibratedClassifierCV(base, cv='prefit', method='isotonic')
calibrated.fit(X_val, y_val)

# 3. THEN tune threshold on calibrated probabilities
probs_val = calibrated.predict_proba(X_val)[:, 1]
threshold = np.sort(probs_val)[-3500]   # capacity-driven

# 4. Evaluate on test
probs_test = calibrated.predict_proba(X_test)[:, 1]
preds_test = (probs_test >= threshold).astype(int)
```

Order matters: **calibrate first, then tune threshold**. Reverse order means the threshold drifts when calibration changes.

### Watch out for…

- **Tuning threshold on the test set**: tune on val, evaluate on test. Same fold = cherry-picking.
- **Calibrating without measuring first**: always check the calibration curve before adding a calibration layer; it's not a free lunch.
- **Wrapping a stacked model in `CalibratedClassifierCV`**: works, but inference cost adds up.

---

## Step 6.3 — Explainability

### Who needs what

Three different audiences need three different kinds of explanation:

| Audience | What they need | Tool |
|---|---|---|
| Data scientist | Global feature importance — "which features drive the model?" | SHAP global; permutation importance; PDP |
| Operator / agent / doctor | Per-row "why did the model say this?" | SHAP per-row; coefficients |
| Regulator / auditor / customer | Audit trail, fairness, recourse | Calibrated probabilities + per-group fairness + counterfactuals |

### SHAP — the workhorse

```python
import shap
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)        # TreeSHAP — fast and exact for trees
shap_values = explainer(X_test)

# Global feature importance — mean absolute SHAP per feature
shap.plots.bar(shap_values, max_display=15)

# Per-row explanation (waterfall plot)
i = 42
shap.plots.waterfall(shap_values[i], max_display=10)

# Programmatic extraction for an agent script
import numpy as np
contribs = shap_values[i].values
features = X_test.columns
top3 = np.argsort(-np.abs(contribs))[:3]
for j in top3:
    sign = '↑' if contribs[j] > 0 else '↓'
    print(f'{sign} {features[j]} = {X_test.iloc[i, j]} (contribution: {contribs[j]:+.3f})')
```

Output for an agent script:
```
Risk score 0.41 — driven by:
  ↑ amount_vs_avg_30d = 8.3   (this transaction is 8x normal for this customer)
  ↑ ip_risk_score = high      (IP has prior fraud association)
  ↓ device_known = True        (customer's regular device — reduces risk)
```

### When to use SHAP

- Tree models (LightGBM, XGBoost, RF) — TreeSHAP is fast and exact.
- Per-row explanations needed (operator dashboards, adverse-action letters).
- Global importance backed by a principled definition.

**Watch out:**
- TreeSHAP scales with depth and number of trees. 500 trees, depth 10 → ~4-7ms per row. Tight for sub-100ms real-time.
- Cache top-3 contributions only for actioned rows (above threshold). Don't compute SHAP for every prediction.

### Coefficients (linear models)

```python
import pandas as pd
coefs = pd.Series(lr.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
print(coefs.head(10))

# Per-row contribution = coefficient × scaled feature value
def explain_row(model, scaler, x_row, feature_names):
    scaled = scaler.transform(x_row.reshape(1, -1)).ravel()
    contribs = model.coef_[0] * scaled
    return pd.Series(contribs, index=feature_names).sort_values(key=abs, ascending=False)
```

**Use when:**
- Brief demands a linear model for regulatory reasons (ECOA Reg-B for lending).
- Adverse-action notices.
- Stakeholder communication where "the model gave 1.4 points to feature X" is more useful than SHAP plots.

**Watch out:**
- Requires proper scaling. Without StandardScaler, a feature 0-1M will look smaller than a feature 0-1.
- Multicollinearity destroys interpretation. Sign of a coefficient can flip with two correlated features.

### Permutation importance

```python
from sklearn.inspection import permutation_importance
result = permutation_importance(
    model, X_val, y_val,
    n_repeats=10, scoring='average_precision', random_state=42, n_jobs=-1,
)
imp = pd.Series(result.importances_mean, index=X_val.columns)
print(imp.sort_values(ascending=False).head(15))
```

**Use as sanity check** against tree-based importance (which is biased toward high-cardinality features). Slow — n_repeats × n_features × inference. Don't run every retraining.

### Partial Dependence Plots (PDP) and ICE

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model, X_val,
    features=['age', 'income', ('age', 'income')],   # 1D and 2D
    kind='both',                                      # PDP + ICE overlay
)
```

**Use when:** you want to understand HOW the model uses a feature (linear, threshold, curve). Stakeholder communication.

**Watch out:** PDP averages over rows including regions of feature space the model isn't trained on. Always check data density before reading the plot.

### Fairness audit — the explainability everyone forgets

```python
from sklearn.metrics import recall_score, precision_score, brier_score_loss

def fairness_audit(probs, preds, y, sensitive_col):
    rows = []
    for group, mask in sensitive_col.groupby(sensitive_col).groups.items():
        rows.append({
            'group': group,
            'n': len(mask),
            'positive_rate': preds[mask].mean(),
            'recall':        recall_score(y[mask], preds[mask]),
            'precision':     precision_score(y[mask], preds[mask]),
            'brier':         brier_score_loss(y[mask], probs[mask]),
        })
    return pd.DataFrame(rows)

audit = fairness_audit(probs_test, preds_test, y_test, X_test['demographic_group'])
print(audit)

# Alert if recall gap > 5pp
recall_gap = audit['recall'].max() - audit['recall'].min()
if recall_gap > 0.05:
    print(f'⚠ Recall gap across groups: {recall_gap:.3f}')
```

**Use when:**
- Brief mentions fairness, disparate impact, equal opportunity, or names a regulator.
- Model touches a consumer-facing decision.

**Watch out:**
- "Group-blind" models are NOT fair. Removing demographic features doesn't remove bias if proxies (zip, last name) carry it. You NEED the demographic to audit, even if you don't use it in the model.
- Equal performance across groups is rarely achievable in full. Pick the criterion the brief actually requires.

### Counterfactual explanations

"What's the smallest change that would flip the decision?" E.g., for a loan denial: "If income were $4K higher OR DTI were 0.32 instead of 0.41, you would have been approved."

```python
import dice_ml

dice_data = dice_ml.Data(dataframe=df, continuous_features=cols_num, outcome_name='approved')
dice_model = dice_ml.Model(model=model, backend='sklearn')
dice_explainer = dice_ml.Dice(dice_data, dice_model, method='random')

cf = dice_explainer.generate_counterfactuals(
    X_test.iloc[0:1], total_CFs=3, desired_class='opposite',
)
cf.visualize_as_dataframe()
```

**Use when:**
- Adverse-action notices that need actionable feedback.
- Customer-facing recourse.
- Debugging — a counterfactual that flips on a tiny change usually exposes a fragile model.

---

## Decision tree for this phase

```
Step 6.1 — Final test evaluation
  Run the model on X_test ONCE.
  Compute primary + secondary metrics from Phase 1.
  Build the reporting table (model vs baseline).
  Cost / business-impact version of confusion matrix.
  Residual plot for regression.

Step 6.2 — Threshold + calibration
  Probabilities consumed downstream?
    Yes → calibrate FIRST (isotonic for trees, Platt for SVM).
  
  Pick threshold strategy:
    Cost given             → cost-driven
    Fixed capacity         → capacity-driven (recall@k)
    Compliance recall      → target-recall
    Balanced, no costs     → F1-maximising
  
  Tune on VAL, evaluate on TEST.

Step 6.3 — Explainability
  Per-row explanation needed?
    Tree model    → SHAP TreeShap
    Linear model  → coefficients
    Anything else → KernelSHAP (slower)
  
  Global feature importance?
    Tree     → mean |SHAP| + permutation as sanity check
    Linear   → coefficients
    Any      → PDP / ICE for feature-shape understanding
  
  Regulated context?
    Linear coefficients OR SHAP + counterfactuals
    Calibrated probabilities mandatory
    Per-group fairness audit mandatory
  
  Any consumer-facing decision?
    Fairness audit mandatory (audit groups, even if "group-blind")
```

---

## Common mistakes in this phase

| Mistake | Why bad | Fix |
|---|---|---|
| Re-running test after a tweak | Contaminates the test set | Test ONCE, then decide |
| Reporting one metric without baseline | Can't tell if model adds value | Always show model vs baseline |
| Tuning threshold on test set | Inflated metric | Tune on val, evaluate on test |
| Calibrating before threshold tuning | Threshold drifts when calibration changes | Calibrate first, then threshold |
| SHAP for every prediction in production | Latency blowup | Cache only for actioned rows |
| "Group-blind" instead of fairness audit | Bias hides in proxies | Need demographic to audit, even if not in features |
| PDP outside data-supported feature range | Meaningless extrapolation | Check data density before reading |
| Skipping calibration check | Probabilities can lie | Always plot calibration curve |
| Stacked model without calibration | Probabilities meaningless | Wrap with CalibratedClassifierCV after stacking |

---

## Worked example from the catalog

[`/scenarios/classification/01_fraud_detection`](/scenarios/classification/01_fraud_detection):

**Test evaluation:**
```
              Baseline rule    Tuned LGBM    Δ
PR-AUC        0.31             0.51          +0.20
Recall@3500   0.42             0.68          +0.26
Precision@3500 0.05            0.21          +0.16
Brier         0.155            0.058         -0.097
Cost ($/Mtxn) 8,400            980           -7,420
```

**Calibration:** isotonic on validation; Brier improved 0.034 → 0.011 (3× improvement).

**Threshold:** cost-driven on validation. Default 0.5 → 0.15 by minimising `15 × FP + 5000 × FN`.

**Explainability:** SHAP precomputed for every row; top-3 cached for rows above 0.15. Agent script template:
> Score 0.41 — three drivers: amount-vs-avg 8x normal, IP risk high, merchant fraud rate elevated. Mitigating: device known.

[`/scenarios/classification/05_loan_fairness`](/scenarios/classification/05_loan_fairness):

**Model:** LogReg + L2 (chosen for ECOA compliance).

**Calibration:** model already calibrated (LogReg without class_weight); Brier 0.118 — verified.

**Threshold:** target-recall at 0.85 to meet legal screen requirement.

**Explainability:** coefficients reported per feature; counterfactual recourse generated for every denial — "Approve if (income +$X) OR (dti < 0.36)".

**Fairness audit:** monthly per-group audit (recall, precision, calibration, decision rate). Threshold post-processed per group when gap > 5pp.

---

## Cheat sheet

```
TEST EVALUATION
  Use the test set ONCE.
  Primary + secondary metric (from Phase 1).
  Always show model vs baseline.
  Confusion matrix in business terms ($).
  Residual plot for regression (heteroscedasticity? long tail?).

THRESHOLD + CALIBRATION
  Order: calibrate FIRST, then tune threshold.
  Threshold strategies:
    cost-driven        c_fp × FP + c_fn × FN
    capacity-driven    np.sort(probs)[-k]
    target-recall      highest threshold with recall ≥ T
    F1-max             fallback
  Calibration:
    tree + class_weight → isotonic (mandatory)
    SVM                  → Platt (sigmoid)
    LogReg balanced      → measure first
  Tune on VAL, evaluate on TEST.

EXPLAINABILITY
  Per-row:
    tree   → SHAP TreeShap; cache top-3 for actioned rows
    linear → coefficients (require scaling)
  Global:
    tree   → mean |SHAP|; permutation as sanity check
    any    → PDP / ICE for feature-shape
  Regulated:
    linear coefficients OR SHAP + counterfactuals
    calibrated probabilities mandatory
    per-group fairness audit mandatory
  Counterfactuals for adverse-action recourse.
```

---

## What comes next

You have a model that beats baseline, with calibrated probabilities, the right threshold, and per-row explanations. **Phase 7** is deployment: saving the model + preprocessor + threshold + calibrator + lookup tables as a versioned bundle, wiring up monitoring for latency / drift / fairness, and writing the rollback plan. The model isn't done until it's been observed in production for the agreed retraining window.
