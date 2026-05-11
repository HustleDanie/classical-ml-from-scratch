# Phase 7: Deployment

> A model in a notebook isn't a product. The last 10% of work is what separates "the validation metric looks good" from "the model is making decisions in production". Eight concerns. Skip any and you'll find out which.

---

## What this phase is for

You have a model that beats baseline, with calibrated probabilities, the right threshold, and per-row explanations. Phase 7 turns that into something that runs in production:

1. **Persist the artifacts** — model + preprocessor + threshold + calibrator + lookup tables, versioned.
2. **Address the 8 production concerns** — latency, retraining cadence, drift, fairness, logging, versioning, edge cases, dashboard.
3. **Validate end-to-end** — measure real latency, test the rollback, dry-run the monitoring.

The model isn't "done" until it's been observed in production for the agreed retraining window. Many models get shipped and quietly degrade — Phase 7 sets up the monitoring that catches that.

---

## Where you are in the pipeline

```
   PHASE 1 — Understand the Problem
   PHASE 2 — Data Exploration & Cleaning
   PHASE 3 — Feature Selection & Preprocessing
   PHASE 4 — Model Selection & Training
   PHASE 5 — Optimization
   PHASE 6 — Evaluation & Validation
►► PHASE 7 — Deployment                       ◄◄  (you are here)
```

## How much time to spend here

Roughly **5% of total project time**. Quick if the team has standard MLOps tooling; longer if you're setting up from scratch. The 8 concerns below are NOT optional — what changes is the *depth* per concern based on the brief.

---

## Step 7.1 — Persist the Artifacts

### What it is

A production model is **not just the model object**. You ship a bundle:

- The fitted model (or pipeline).
- The fitted preprocessor (ColumnTransformer with statistics).
- The calibration mapping (if used).
- The chosen threshold (or threshold function).
- Any lookup tables (target encodings, smoothed merchant rates).
- The training data hash (so future you can reproduce).
- The training config (hyperparameters, feature list, seed).

If you ship only `model.pkl`, the production system can't reproduce the inputs you trained on.

### Code

```python
import joblib
import json
import hashlib
import datetime as dt
from pathlib import Path

def save_bundle(
    model, preprocessor, threshold, calibrator=None,
    lookup_tables=None, config=None, registry='/models',
):
    version = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    path = Path(registry) / version
    path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model,         path / 'model.joblib')
    joblib.dump(preprocessor,  path / 'preprocessor.joblib')
    if calibrator is not None:
        joblib.dump(calibrator, path / 'calibrator.joblib')
    if lookup_tables is not None:
        joblib.dump(lookup_tables, path / 'lookup_tables.joblib')
    
    meta = {
        'version': version,
        'threshold': threshold,
        'config': config or {},
        'created_utc': dt.datetime.utcnow().isoformat(),
    }
    with open(path / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f'Saved bundle to {path}')
    return version

# Inference time — load by version
def load_bundle(version, registry='/models'):
    path = Path(registry) / version
    model = joblib.load(path / 'model.joblib')
    preprocessor = joblib.load(path / 'preprocessor.joblib')
    calibrator = joblib.load(path / 'calibrator.joblib') if (path / 'calibrator.joblib').exists() else None
    with open(path / 'meta.json') as f:
        meta = json.load(f)
    return model, preprocessor, calibrator, meta
```

### Production scoring (the inference function)

```python
def score(x_row, version='current'):
    model, preprocessor, calibrator, meta = load_bundle(version)
    
    # 1. Preprocess
    X = preprocessor.transform(pd.DataFrame([x_row]))
    
    # 2. Base model probability
    probs = model.predict_proba(X)[:, 1]
    
    # 3. Calibrate
    if calibrator is not None:
        probs = calibrator.predict_proba(X)[:, 1]
    
    # 4. Threshold to decision
    decision = (probs >= meta['threshold']).astype(int)
    
    return {
        'probability': float(probs[0]),
        'decision':    int(decision[0]),
        'model_version': meta['version'],
        'threshold':   meta['threshold'],
    }
```

This is the canonical shape for a production scoring API. Wrap it in Flask, FastAPI, or whatever your platform uses.

### Watch out for…

- **The threshold is part of the model bundle**. A decision is `(prob >= threshold)`. Both must move together — if the threshold changes (e.g., team capacity goes up), it's a new bundle version.
- **Lookup tables can drift** — merchant fraud rates from 6 months ago aren't current rates. Decide whether they're part of the model (frozen) or fresh (recomputed at inference).
- **Pickle / joblib are not version-safe across sklearn versions**. Pin sklearn version in production; rebuild bundles on upgrade.

---

## Step 7.2 — The 8 Production Concerns

### 1. Inference latency

**The signal.** Brief mentions a latency budget: "<100ms per transaction", "<2s per page load", "<24h daily batch".

**Measure first, then constrain.**

```python
import time

# Warm up
score(X_test.iloc[0])

# Measure single-row (representative of production traffic)
n = 1000
t0 = time.perf_counter()
for i in range(n):
    score(X_test.iloc[i % len(X_test)])
single_row_ms = (time.perf_counter() - t0) / n * 1000
print(f'Single-row latency: {single_row_ms:.2f}ms')

# Measure batch
batch_size = 1000
t0 = time.perf_counter()
score_batch(X_test.iloc[:batch_size])
batch_ms = (time.perf_counter() - t0) * 1000
print(f'Batch-{batch_size} latency: {batch_ms:.1f}ms ({batch_ms/batch_size:.2f}ms/row)')
```

**Watch out:**
- Inference includes preprocessing — profile the whole pipeline, not just the model.
- A LightGBM with 500 trees, depth 8, scoring 1 row → ~0.5-2ms. Heavier ensembles or stacking can blow past 50ms.
- If you blow latency: shrink the model first (fewer trees, lower depth). Often drop to 200 trees with negligible PR-AUC loss.

**Examples from the catalog:**
- Fraud detection: <100ms hard limit → single LightGBM, no stacking.
- Ride-share surge: <2s for all 120 zones → LightGBM, 1.2s achieved.
- House price: 50ms/listing → ElasticNet + LightGBM both fit.
- Hospital LOS: 5s/admission (overnight batch) → blended models OK.

### 2. Retraining cadence

**The signal.** How quickly does the world change?

| Data velocity | Cadence |
|---|---|
| Fraud patterns evolve weekly | Retrain weekly |
| User behaviour shifts monthly | Retrain monthly |
| Service / product changes quarterly | Retrain quarterly |
| Static physics, slow drift | Retrain annually or trigger-based |

**Code (scheduled retraining):**

```python
def retrain():
    df = load_recent_data(end=today, days_back=180)
    X, y = prepare(df)
    model = train(X, y)
    score = evaluate_on_holdout(model, df)
    if score >= MIN_THRESHOLD:
        version = save_bundle(model, preprocessor, threshold, ...)
        # promote behind feature flag
    else:
        alert('Retraining produced model below acceptance threshold')
```

**Watch out:**
- Don't retrain blindly. Every retraining is a risk — the new model can be worse.
- Always evaluate on a held-out fold before promoting.
- Keep the previous N versions for instant rollback. Disk is cheap; an outage isn't.

### 3. Drift detection

**Three flavours of drift:**

1. **Feature drift** — input distribution shifts (new merchants, demographic changes).
2. **Concept drift** — feature-target relationship shifts (a previously-safe pattern becomes risky).
3. **Label drift** — target distribution shifts (fraud rate doubles).

**Code (PSI — Population Stability Index):**

```python
import numpy as np

def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """
    PSI < 0.10  : no drift
    PSI 0.10-0.25 : moderate drift, investigate
    PSI > 0.25 : significant drift, alarm
    """
    quantiles = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    ref_p = np.histogram(reference, bins=quantiles)[0] / len(reference) + 1e-6
    cur_p = np.histogram(current,   bins=quantiles)[0] / len(current) + 1e-6
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))

# Daily PSI on top-5 features by SHAP importance
for col in top_5_features:
    score = psi(X_train[col].values, X_today[col].values)
    if score > 0.25:
        alert(f'Feature {col} has drifted (PSI={score:.2f})')
```

**Code (concept drift — rolling PR-AUC on labelled data):**

```python
from sklearn.metrics import average_precision_score
import pandas as pd

scored = pd.DataFrame({'date': dates, 'prob': probs, 'y': y_true})
rolling = scored.set_index('date')\
                .resample('D')\
                .apply(lambda df: average_precision_score(df['y'], df['prob']))
if rolling.iloc[-7:].mean() < BASELINE * 0.9:
    alert(f'Rolling PR-AUC dropped: {rolling.iloc[-7:].mean():.3f} vs baseline {BASELINE:.3f}')
```

**Watch out:**
- PSI on features the model doesn't use is informational, not actionable. Prioritise alarms on top-5 by SHAP.
- Concept drift requires labels — many problems have feedback delays (fraud labels 30+ days late). Build the label-collection plan as part of deployment.
- A drift alarm is a question, not a problem statement. Investigate before reacting.

### 4. Fairness audit (recurring, not one-time)

```python
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

# Run monthly on production decisions
audit = fairness_audit(probs_prod, preds_prod, y_prod, X_prod['demographic_group'])

# Alert if recall gap > 5 percentage points
recall_gap = audit['recall'].max() - audit['recall'].min()
if recall_gap > 0.05:
    alert(f'Recall gap across groups: {recall_gap:.3f}')
```

**Watch out:**
- Fairness audits require demographic data — if you can't access it, you can't audit. Plan data access before deploying.
- Equal performance across groups isn't always achievable. Pick the right fairness criterion (equal opportunity, demographic parity, calibration).

### 5. Logging and audit trail

**The signal.** Regulated, FOIA-able, or customer can appeal.

**For every prediction, log:**

- Timestamp
- Model version (e.g., `v2024-09-12T08:00:00Z`)
- Threshold version (separate from model version!)
- Input features (hashed if sensitive — but recoverable in audit)
- Output score and decision
- Top-3 SHAP contributions (cached for actioned rows only)

```python
import json, hashlib, datetime as dt

def log_decision(model_version, threshold_version, x_row, prob, decision, shap_top3):
    record = {
        'ts':                dt.datetime.utcnow().isoformat(),
        'model_version':     model_version,
        'threshold_version': threshold_version,
        'input_hash':        hashlib.sha256(json.dumps(x_row.to_dict(), sort_keys=True).encode()).hexdigest(),
        'prob':              float(prob),
        'decision':          int(decision),
        'shap_top3':         shap_top3,
    }
    write_to_log_table(record)
```

**Watch out:**
- Log retention must match the regulatory horizon. ECOA: 25 months. HIPAA: 6 years. Know yours before deploying.
- Logs with PII are themselves a liability. Hash or anonymise where possible; gate access where you can't.

### 6. Versioning and rollback

Every artifact, every time:

- The model file (with cryptographic hash).
- The preprocessor / ColumnTransformer.
- Lookup tables (target encodings, calibration mappings).
- The threshold.
- The training data snapshot (or hash).
- The training config (hyperparameters, feature list, random seed).

```python
def register(model, preprocessor, threshold, config, registry='/models'):
    version = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    path = Path(registry) / version
    path.mkdir(parents=True, exist_ok=True)
    with open(path / 'model.pkl', 'wb') as f: pickle.dump(model, f)
    with open(path / 'preprocessor.pkl', 'wb') as f: pickle.dump(preprocessor, f)
    with open(path / 'meta.json', 'w') as f:
        json.dump({'threshold': threshold, 'config': config, 'version': version}, f)
    return version

# Production reads from /models/current symlink
# Rollback = `ln -sf /models/<previous_version> /models/current`
```

**Watch out:**
- Threshold is part of the model bundle.
- Cross-version preprocessor mismatch is a top-10 production bug. Test the rollback before going live.

### 7. Edge cases

Every brief has them. Read the brief and list at least three.

**Common production edge cases:**

- **Unseen categories** — new merchant, new ZIP, new product. Set `handle_unknown='ignore'` on OneHotEncoder; have a global-mean fallback for target-encoded features.
- **All-zero / all-missing inputs** — data pipeline failure upstream. Validate input ranges on entry.
- **Boundary inputs** — $0 transaction, weight=0, future date. Decide up-front whether to score, refuse, or fall back.
- **Adversarial inputs** — someone gaming the score (fraud, abuse, content moderation).
- **Cold start** — new user / device / patient with no history. Velocity / target-encoded features won't be computable; need a "first-row" code path.

```python
def safe_predict(model, x_row):
    if x_row.isna().sum() > MAX_MISSING:
        return None, 'too_many_missing_values'
    if x_row['amount'] < 0:
        return None, 'negative_amount'
    if x_row['user_history_days'] < MIN_HISTORY:
        return DEFAULT_RISK_FALLBACK, 'cold_start'
    return model.predict_proba(x_row.values.reshape(1, -1))[0, 1], 'ok'
```

**Watch out:** a model that returns `0.5` for everything in failure cases is silently broken. Emit a metric (e.g., `cold_start_count`) so dashboards catch failure-mode growth.

### 8. The on-call dashboard — five numbers you watch

1. **Throughput** — predictions per minute. Sudden drop = upstream failure.
2. **Latency P50 / P95 / P99** — alert on P99 SLA breach.
3. **Score distribution** — P50, P95 of `prob`. Sudden shift = world changed or model broke.
4. **Decision rate** — fraction of predictions above the threshold. Sudden shift = drift, threshold misconfig, feature pipeline break.
5. **Performance metric** — rolling PR-AUC / MAE on a 7-day window (when labels available).

If your dashboard doesn't show these five live, the model is partially deployed.

```python
# Pseudocode for the dashboard backend
prom_metrics = {
    'predictions_per_minute': gauge('predictions_per_minute'),
    'latency_p99_ms':         gauge('latency_p99_ms'),
    'score_p50':              gauge('score_p50'),
    'decision_rate':          gauge('decision_rate'),
    'rolling_pr_auc_7d':      gauge('rolling_pr_auc_7d'),
}
```

---

## Decision tree for this phase

```
For every model going to production:

  1. Latency budget known?
     └── No → Get one. Then measure. Cut model if blown.
  2. Retraining cadence picked?
     └── Tied to data velocity. Document it.
  3. Drift signals defined?
     └── Top 5 features by SHAP, daily PSI.
     └── Concept drift: rolling PR-AUC / MAE on labelled data.
  4. Consumer-facing or regulated?
     └── Yes → Fairness audit cadence + log retention + adverse-action handling.
     └── No  → Light audit; still log model version + threshold per prediction.
  5. Versioning in place?
     └── Model + preprocessor + threshold + config, all hashed and persisted.
     └── Rollback plan: a config flag flips traffic back to vN-1.
  6. Edge cases listed?
     └── Cold start, missing fields, unseen categories, adversarial — at minimum.
  7. Dashboard live?
     └── Throughput, latency P99, score distribution, decision rate, rolling perf.

If any "no" → deployment isn't ready.
```

---

## Common combinations from the catalog

| Brief | Latency | Cadence | Drift | Audit |
|---|---|---|---|---|
| Fraud detection | <100ms real-time | Weekly | Daily PSI on top-5; rolling PR-AUC on confirmed labels | FOIA: model version + SHAP top-3 logged |
| Ride-share surge | <2s all-zones | Daily | PSI on demand-supply, lag features | Internal only |
| Hospital LOS | 5s/admission | Quarterly | Per-ward MAE rolling | HIPAA: 6yr log retention |
| House price | 50ms/listing | Quarterly | PSI on neighborhood, sale month | Internal |
| Loan default | 200ms/application | Monthly | Daily PSI on top-10; per-group fairness rolling | ECOA: 25mo retention; adverse-action notices |
| Restaurant inspection | 24h batch | Quarterly | Per-cuisine drift; reserve 10% random inspections | FOIA: model version + SHAP per dispatch |

---

## Common mistakes in this phase

| Mistake | Why bad | Fix |
|---|---|---|
| Saving only the model object | Production can't reproduce inputs | Save the whole bundle (model + preprocessor + threshold + ...) |
| No rollback plan | Outage when a new model is bad | Symlink-based or feature-flag-based rollback; test it |
| Retraining without holdout evaluation | New model can be worse than old | Evaluate on holdout BEFORE promoting |
| Drift monitoring without prioritisation | Alarms on irrelevant features create fatigue | Monitor top-5 by SHAP importance |
| Returning 0.5 on input validation failure | Silently broken, on-call won't notice | Emit a metric for failure-mode counts |
| Cross-version preprocessor mismatch | Top-10 production bug | Pin sklearn version; rebuild bundle on upgrade |
| Skipping the fairness audit cadence | Drift can introduce bias over time | Monthly per-group metrics |
| Logging PII in cleartext | Liability | Hash or anonymise; gate access |
| Forgetting label-collection plan | Concept drift detection blind | Plan label collection alongside deployment |

---

## Worked example from the catalog

[`/scenarios/classification/01_fraud_detection`](/scenarios/classification/01_fraud_detection) — full Phase 7:

- **Latency**: <100ms hard limit. Single LightGBM, 400 trees, depth 8 → measured 0.6ms per row. Plenty of margin for preprocessing (~3-5ms).
- **Retraining**: weekly. Evaluated on held-out month-6 before promoting; PR-AUC must be ≥ baseline + 0.40.
- **Drift**: daily PSI on top-5 SHAP features (amount, ip_risk, merchant_rate, hour, device). Rolling 7-day PR-AUC on confirmed-fraud labels (30-day delay).
- **Audit log**: every flagged transaction logs model version, threshold version, prob, top-3 SHAP. Retention 25 months.
- **Edge cases**:
  - New merchants → fallback to global merchant_fraud_rate.
  - Cold-start users (history < 30d) → separate "thin-file" model.
  - Adversarial: monitor for unusual feature combinations vs training distribution.
- **Rollback**: feature flag swaps to v(n-1) on >10% rolling-PR-AUC drop.
- **Dashboard**: throughput, latency P99, score-distribution P95, decision-rate (% flagged), 7-day rolling PR-AUC. All live in Grafana.

[`/scenarios/classification/05_loan_fairness`](/scenarios/classification/05_loan_fairness) — regulated deployment:

- **Latency**: 200ms — leaves room for adverse-action generation.
- **Retraining**: monthly. Promotion gated on per-group recall gap < 5pp.
- **Fairness**: monthly audit (recall, precision, calibration, decision rate). Threshold post-processed per group if gap > 5pp.
- **Audit log**: 25 months (ECOA Reg-B). Every denial includes top-3 reasons + counterfactual recourse ("Approve if income +$X OR DTI < 0.36").
- **Edge cases**: thin-file applicants → separate code path with Bayesian-smoothed defaults.

---

## Cheat sheet

```
PERSIST
  Save the BUNDLE (not just model):
    model + preprocessor + threshold + calibrator + lookup tables
    + meta.json (version, config, data hash)
  Inference reads from /models/current symlink.

THE 8 CONCERNS
  1. Latency      → measure single-row + batch; cut model if over budget
  2. Cadence      → weekly / monthly / quarterly per data velocity
  3. Drift        → daily PSI on top-5 SHAP features; rolling perf on labels
  4. Fairness     → cadence per regulator; audit by group, not group-blind
  5. Logging      → model version + threshold + input hash + top-3 SHAP
  6. Versioning   → model + preprocessor + threshold + config + data hash
  7. Edge cases   → cold start, unseen categories, adversarial, missing
  8. Dashboard    → throughput, latency P99, score P50, decision rate, rolling perf

ALWAYS
  - Test the rollback before going live.
  - Validate inputs at entry; emit metrics on failure modes.
  - Pin sklearn version; rebuild bundles on upgrade.
  - Pick log retention from the regulator, not from instinct.
  - Reserve random inspections / hold-outs to keep training data honest.
  - Match fairness audit cadence to your regulator's expectations.

If any of the 8 concerns is "TBD", deployment isn't ready.
```

---

## You've finished the Learn track

You've walked through all seven phases of the classical ML pipeline. The toolkit is now in your head: how to read a brief, when to spend time on features vs models, which technique to reach for given which signal, and what production needs that notebooks never tell you.

**Where to go next:**

- [`/scenarios`](/scenarios) — 155 expert scenarios across 14 industries. Pick one shape that matches a brief you're working on, read the 17-step walkthrough.
- [`/practice`](/practice) — generate a fresh AI brief, write your approach, then compare against a 17-step expert walkthrough plus a critique of your attempt. The fastest way to internalise the toolkit.
- [`/algorithms`](/algorithms) — 12 classical algorithms implemented from scratch in NumPy, verified against scikit-learn. Look here when you want to know *how* a method works under the hood.

Every problem from here on follows the same 7 phases. The skill is recognising which signal in the brief points at which technique in your toolkit — and once you've done it on a dozen real problems, you'll do it on autopilot.
