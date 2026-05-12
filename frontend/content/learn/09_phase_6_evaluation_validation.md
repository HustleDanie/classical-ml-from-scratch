# Phase 6: Evaluation & Validation

> Held-out test (used once). Calibrate. Threshold-tune. Explain. Audit fairness.

---

## Step 6.1 — Final test evaluation

Use the held-out test set **exactly once**. Re-running after a tweak contaminates it.

```python
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score, roc_auc_score, brier_score_loss,
)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= threshold).astype(int)

print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
print(f'PR-AUC: {average_precision_score(y_test, probs):.3f}  '
      f'ROC-AUC: {roc_auc_score(y_test, probs):.3f}  '
      f'Brier: {brier_score_loss(y_test, probs):.3f}')
```

Always show **model vs baseline + delta**. Translate the confusion matrix into business numbers (TP/FP/FN/TN in $).

---

## Step 6.2 — Threshold tuning (calibrate FIRST, then threshold)

Calibrate when probabilities are consumed downstream (pricing, regulators, downstream models):

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(base_model, cv='prefit', method='isotonic')
calibrated.fit(X_val, y_val)
```

- **Isotonic** → tree models with `class_weight` / `scale_pos_weight` (default, needs ≥ 1,000 val samples).
- **Sigmoid (Platt)** → SVMs, or small validation sets (< 500 positives).
- **No calibration** → LogReg without `class_weight` on balanced data (verify with `calibration_curve`).

Then pick the threshold:

| Strategy | When | Code |
|---|---|---|
| Cost-driven | Brief gives $ costs | minimise `c_fp × FP + c_fn × FN` over `thresholds` |
| Capacity (recall@k) | Fixed team budget | `threshold = np.sort(probs)[-k]` |
| Target-recall | Compliance recall target | highest threshold s.t. `recall ≥ T` |
| F1-maximising | Balanced, no costs given | `argmax(2·prec·rec / (prec+rec))` |

**Tune on val, evaluate on test.** Same fold = cherry-picking.

---

## Step 6.3 — Explainability

Match the tool to the audience:

| Audience | Need | Tool |
|---|---|---|
| Data scientist | What features drive the model? | Global mean-\|SHAP\| + permutation importance |
| Operator / agent / doctor | Why is **this** row scored as it is? | SHAP per-row top-3 (TreeSHAP) |
| Regulator / customer | Audit + recourse | Linear coefficients OR SHAP + counterfactuals (`dice-ml`) |

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap.plots.bar(shap_values, max_display=15)             # global
shap.plots.waterfall(shap_values[i], max_display=10)    # per-row
```

Cache top-3 SHAP only for **actioned rows** (above threshold) to keep latency in budget.

---

## Step 6.4 — Fairness audit

Mandatory for any consumer-facing or regulated decision.

```python
def fairness_audit(probs, preds, y, group_col):
    rows = []
    for g, mask in group_col.groupby(group_col).groups.items():
        rows.append({
            'group': g, 'n': len(mask),
            'positive_rate': preds[mask].mean(),
            'recall': recall_score(y[mask], preds[mask]),
            'precision': precision_score(y[mask], preds[mask]),
            'brier': brier_score_loss(y[mask], probs[mask]),
        })
    return pd.DataFrame(rows)
```

You **need the demographic to audit** — "group-blind" models hide bias in proxies (zip, last name).

---

## Common mistakes

| Mistake | Fix |
|---|---|
| Re-running test after a tweak | Test once; if it fails, fix and use a fresh test set |
| Reporting one metric without baseline | Always model vs baseline |
| Threshold tuned on test | Tune on val, evaluate on test |
| Calibrating after threshold tuning | Calibrate **first**, then threshold |
| SHAP on every prediction in real-time | Cache top-3 only for actioned rows |
| "Group-blind" instead of audit | Need the demographic column to audit |
