# Phase 7: Deployment

> Save the bundle, address the 8 production concerns, watch it in prod.

---

## Step 7.1 — Persist the bundle

A production model is **not** just `model.pkl`. Ship a versioned bundle:

- The fitted model (or full `Pipeline`)
- The preprocessor / `ColumnTransformer`
- The calibrator (if used)
- The threshold (separate from the model — thresholds change!)
- Lookup tables (target encodings, etc.)
- Training data hash + training config (hyperparameters, feature list, seed)

```python
import joblib, json, datetime as dt
from pathlib import Path

def save_bundle(model, preprocessor, threshold, calibrator=None, config=None, registry='/models'):
    version = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    path = Path(registry) / version
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,        path / 'model.joblib')
    joblib.dump(preprocessor, path / 'preprocessor.joblib')
    if calibrator is not None:
        joblib.dump(calibrator, path / 'calibrator.joblib')
    with open(path / 'meta.json', 'w') as f:
        json.dump({'version': version, 'threshold': threshold, 'config': config}, f)
    return version

# Production reads /models/current symlink. Rollback = `ln -sf <previous> current`.
```

---

## Step 7.2 — The 8 production concerns

| # | Concern | What to do |
|---|---|---|
| 1 | **Latency** | Measure single-row + batch latency end-to-end (incl. preprocessing). If over budget, shrink the model before swapping. |
| 2 | **Retraining cadence** | Tie to data velocity: weekly (fraud), monthly (churn), quarterly (housing), annually (slow). Evaluate on holdout before promoting. |
| 3 | **Drift** | Daily PSI on top-5 SHAP features. Alert at PSI > 0.25. Rolling perf metric on labelled data when feedback delay permits. |
| 4 | **Fairness** | Monthly per-group recall / precision / Brier. Alert on > 5pp gap. |
| 5 | **Logging** | Every prediction: `ts`, `model_version`, `threshold_version`, `input_hash`, `prob`, `decision`, `shap_top3`. Retention matches the regulator (ECOA 25 mo, HIPAA 6 yr). |
| 6 | **Versioning** | Model + preprocessor + threshold + config + data hash. Test the rollback before going live. |
| 7 | **Edge cases** | Unseen categories, all-zero / all-missing inputs, cold-start entities, adversarial inputs. Validate inputs at entry; emit metrics on failure modes. |
| 8 | **Dashboard** | Five live numbers: throughput, latency P99, score P50, decision rate, rolling perf metric. |

---

## Inference wrapper

```python
def score(x_row, version='current'):
    model, preprocessor, calibrator, meta = load_bundle(version)
    X = preprocessor.transform(pd.DataFrame([x_row]))
    probs = model.predict_proba(X)[:, 1]
    if calibrator is not None:
        probs = calibrator.predict_proba(X)[:, 1]
    decision = int(probs[0] >= meta['threshold'])
    return {
        'probability': float(probs[0]),
        'decision': decision,
        'model_version': meta['version'],
        'threshold': meta['threshold'],
    }
```

---

## Common mistakes

| Mistake | Fix |
|---|---|
| Saving only the model object | Save the whole bundle (model + preprocessor + threshold + …) |
| No rollback plan | Symlink/flag-based rollback; **test it before going live** |
| Retraining without holdout eval | Always evaluate on a held-out fold before promoting |
| Returning `0.5` on validation failure | Emit a metric instead so on-call can see failure-mode growth |
| Cross-version preprocessor mismatch | Pin sklearn version; rebuild bundles on upgrade |
| Forgetting label-collection plan | Plan label collection alongside deployment — concept drift needs it |
