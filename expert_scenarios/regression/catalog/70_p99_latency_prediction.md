# Expert Scenario 70: P99 Latency Prediction (Quantile Regression)

> **Complexity:** Predict the P99 of a request-latency distribution, not the mean. Quantile loss (pinball), heavy-tailed target with rare extreme values, system-feature engineering matters more than model choice, multiple quantiles must be non-crossing, conformal prediction adds rigorous coverage guarantees.

---

## The Brief

A web platform serving 200M requests/day wants per-endpoint P99 latency forecasts 5 minutes ahead so the SRE team can preemptively scale or shed load before SLO breaches. The data is per-request latency in ms, aggregated to 1-minute windows per endpoint. They want:

- Prediction of the P99 latency for the next 5-minute window for each of ~400 endpoints.
- Coverage guarantee: if you say "P99 will be below 400ms", that should be true 99% of the time across endpoints.
- Inference budget: < 60 seconds total for all 400 predictions every 5 minutes.
- A confidence interval (P10/P90 prediction interval) for capacity-planning use.

Predicting the median latency (P50) would tell you nothing useful — outages happen in the tail. The whole point is the tail.

---

## Step 1: Define the Problem Type

```
Type:           Quantile regression on P99 latency
Primary Metric: Pinball loss at quantile 0.99
Secondary:      Empirical coverage (target = 99%); interval width
Business Goal:  Quantile-99 prediction with empirical coverage 99% +/- 1pp
Constraint:     Non-crossing quantiles (predicted P10 must be < P50 < P90 < P99)
                < 60s total inference for 400 endpoints
Target shape:   Right-skewed; P99 ranges 50-2000ms; tail extends to 30s
```

**Expert thinking:** point regression on the mean would be wrong. We want the 99th percentile, which for skewed latency distributions is dramatically higher than the mean. Quantile regression directly optimizes the right metric (pinball loss) without ever needing to predict the mean.

---

## Step 2: Understand the Data

```
Shape: 1-minute windows × 400 endpoints × 90 days = ~52M rows
Target: latency_p99_next_5min (ms)

Columns (per 1-min window):

REQUEST AGGREGATES (current window):
- endpoint_id (400 unique)
- timestamp (datetime, 1-min granularity)
- request_count (int, 50-500K per minute per endpoint)
- p50_latency_ms (float)
- p90_latency_ms (float)
- p99_latency_ms (float)  <- not the target; current observed value
- mean_latency_ms (float)
- error_rate (float, 0-1)
- bytes_in (sum)
- bytes_out (sum)

ROUTING / STATE:
- service_pod_count (int, 1-200, scaled by autoscaler)
- service_cpu_avg (float, 0-100%)
- service_memory_avg (float, 0-100%)
- db_connection_pool_used (int)
- cache_hit_rate (float, 0-1)
- queue_depth (int, requests waiting)

UPSTREAM:
- upstream_p99 (float, dependency latency)
- upstream_error_rate (float)

CALENDAR:
- hour_of_day (0-23)
- day_of_week (0-6)
- is_business_hours (binary)
- minute_of_hour (0-59)

EVENT FLAGS:
- deployment_within_15min (binary)
- known_dependency_outage (binary)
- promo_traffic_event (binary)

TARGET (constructed):
- latency_p99_next_5min  <- computed by aggregating 5 future minutes per endpoint
```

**Expert thinking:** the target is constructed by looking ahead 5 minutes at the actual latency distribution. At training time, we have ground truth; at inference time, we don't (we're predicting it). Lag features from past latency, recent throughput, and current resource utilization carry most of the signal.

---

## Step 3: EDA on Tail

```python
df['latency_p99_next_5min'].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999])
# count  52M
# mean   180ms
# 50%    72ms
# 90%    340ms
# 95%    580ms
# 99%    1,420ms
# 99.9%  4,860ms

# log-distribution looks more normal
np.log1p(df['latency_p99_next_5min']).describe()
# Skew: 1.4 (still right-skewed but workable)

# Per-endpoint variance: huge
endpoint_stats = df.groupby('endpoint_id')['latency_p99_next_5min'].agg(['mean', 'std', 'median'])
# Some endpoints stably 50-100ms (static content); others 200-2000ms (database queries)

# Tail-event triggers (P99 > 1000ms)
high_tail = df[df['latency_p99_next_5min'] > 1000]
print(high_tail['service_cpu_avg'].mean())  # 78% (vs 42% normally)
print(high_tail['queue_depth'].mean())       # 145 (vs 8 normally)
print(high_tail['db_connection_pool_used'].mean())  # 92% (vs 35%)
```

**Findings:**

| Finding | Implication |
|---------|------------|
| P99 latency span 30ms - 30s across endpoints | Per-endpoint normalization or model-per-endpoint |
| High-tail events correlated with CPU > 70%, queue depth > 50, pool > 80% | Resource pressure features are predictive |
| Deployment within 15min → 3.4x more tail events | Explicit feature for recent deploys |
| `upstream_p99` correlates 0.62 with target | Cascading latency is real; include upstream features |
| 90% of windows are "normal" (P99 < 200ms); 10% are "stressed" | Bimodal mixture; quantile regression handles this naturally |

---

## Step 4: Data Cleaning

```python
# Drop windows with < 50 requests (noise dominates P99 estimation)
df = df[df['request_count'] >= 50]

# Outliers: DO NOT cap. The 30s P99 events ARE the signal.
# But if any timestamp has impossibly high values (sensor error), drop:
df = df[df['latency_p99_next_5min'] < 60000]  # 60s cap; clearly broken if higher

# Missing handling
df['upstream_p99'] = df['upstream_p99'].fillna(df['p99_latency_ms'])  # if upstream missing, use own
df['upstream_error_rate'] = df['upstream_error_rate'].fillna(0)
df['known_dependency_outage'] = df['known_dependency_outage'].fillna(0).astype(int)

# Log-transform target for stability (we'll inverse-transform predictions)
df['target_log'] = np.log1p(df['latency_p99_next_5min'])
```

---

## Step 5: Feature Engineering

```python
# === LAG FEATURES (per endpoint) ===
df = df.sort_values(['endpoint_id', 'timestamp'])
group = df.groupby('endpoint_id')

# Latency lags
df['p99_lag_1m']  = group['p99_latency_ms'].shift(1)
df['p99_lag_5m']  = group['p99_latency_ms'].shift(5)
df['p99_lag_15m'] = group['p99_latency_ms'].shift(15)
df['p99_lag_60m'] = group['p99_latency_ms'].shift(60)
df['p99_lag_24h'] = group['p99_latency_ms'].shift(60*24)
df['p99_lag_7d'] = group['p99_latency_ms'].shift(60*24*7)

# Rolling stats
df['p99_ma_5m']   = group['p99_latency_ms'].shift(1).rolling(5).mean()
df['p99_ma_15m']  = group['p99_latency_ms'].shift(1).rolling(15).mean()
df['p99_ma_60m']  = group['p99_latency_ms'].shift(1).rolling(60).mean()
df['p99_max_5m']  = group['p99_latency_ms'].shift(1).rolling(5).max()
df['p99_max_15m'] = group['p99_latency_ms'].shift(1).rolling(15).max()

# Volatility
df['p99_std_15m'] = group['p99_latency_ms'].shift(1).rolling(15).std()

# Trend
df['p99_trend_1v15'] = df['p99_lag_1m'] / (df['p99_ma_15m'] + 1)
df['p99_growth_rate'] = (df['p99_lag_1m'] - df['p99_lag_5m']) / (df['p99_lag_5m'] + 1)

# === RESOURCE-PRESSURE FEATURES ===
df['cpu_pressure'] = (df['service_cpu_avg'] / 100) ** 2  # quadratic — small CPU below 70% is fine, above 70% explodes
df['queue_pressure'] = np.log1p(df['queue_depth'])
df['pool_pressure'] = df['db_connection_pool_used'] / 100

# === COMPOSITE STRESS SCORE ===
df['stress_score'] = (
    (df['service_cpu_avg'] > 70).astype(int) +
    (df['queue_depth'] > 50).astype(int) +
    (df['db_connection_pool_used'] > 80).astype(int) +
    (df['error_rate'] > 0.005).astype(int) +
    (df['upstream_p99'] > 500).astype(int)
)

# === ENDPOINT-LEVEL TARGET ENCODING (smoothed, on TRAIN ONLY) ===
endpoint_baseline_p99 = df_train.groupby('endpoint_id')['latency_p99_next_5min'].quantile(0.5)
df['endpoint_baseline'] = df['endpoint_id'].map(endpoint_baseline_p99).fillna(df['p99_latency_ms'])
```

---

## Step 6: Feature Selection

```python
# Time-series feature importance
mi_scores = mutual_info_regression(X_train, y_train_log, random_state=42)

# Top 12:
#   p99_lag_1m              0.142
#   p99_max_15m             0.121
#   p99_ma_15m              0.114
#   stress_score            0.098
#   queue_pressure          0.087
#   cpu_pressure            0.082
#   p99_lag_5m              0.071
#   upstream_p99            0.064
#   endpoint_baseline       0.054
#   request_count           0.041
#   deployment_within_15min 0.038
#   db_connection_pool_used 0.029
```

---

## Step 7: Preprocessing

```python
# LightGBM handles categorical natively (endpoint_id has 400 levels — use as cat)
# No scaling needed for tree models
```

---

## Step 8: Train/Test Split — Walk-Forward

```python
# Cardinal rule of time-series: walk-forward, never random
# Train: first 70 days
# Validation: days 71-80
# Test: days 81-90

df = df.sort_values('timestamp')
cutoffs = [df['timestamp'].quantile(0.78), df['timestamp'].quantile(0.89)]
train = df[df['timestamp'] < cutoffs[0]]
val   = df[(df['timestamp'] >= cutoffs[0]) & (df['timestamp'] < cutoffs[1])]
test  = df[df['timestamp'] >= cutoffs[1]]
```

---

## Step 9: Baselines

```python
# Baseline 1: predict current p99_latency_ms as the next P99
# Pinball loss at q=0.99: 87 ms

# Baseline 2: predict the 60-min rolling max p99
# Pinball loss at q=0.99: 64 ms

# Baseline 3: per-endpoint constant (its 95th historical percentile)
# Pinball loss at q=0.99: 71 ms
```

---

## Step 10: Try Multiple Quantile Regression Models

```python
import lightgbm as lgb

# === Model 1: LightGBM with quantile loss at q=0.99 ===
model_99 = lgb.LGBMRegressor(
    objective='quantile',
    alpha=0.99,
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
# Pinball loss at q=0.99 on validation: 41 ms

# === Model 2: Quantile Random Forest ===
# (sklearn's RandomForestRegressor doesn't natively do quantiles; use quantile-forest package)
from quantile_forest import RandomForestQuantileRegressor
qrf = RandomForestQuantileRegressor(n_estimators=300, random_state=42)
qrf.fit(X_train, y_train_log)
preds_99 = qrf.predict(X_val, quantiles=[0.99])
# Pinball loss at q=0.99: 48 ms (worse than LightGBM)

# === Model 3: Linear quantile regression (sklearn 1.0+) ===
from sklearn.linear_model import QuantileRegressor
ql = QuantileRegressor(quantile=0.99, alpha=0.001, solver='highs')
ql.fit(X_train, y_train_log)
# Pinball loss at q=0.99: 73 ms (linear too restrictive for non-linear interactions)

# === Model 4: LightGBM with quantile loss + Tweedie hybrid ===
# Some research suggests training the median (Tweedie/MAE) and adding a calibration step
# Skipped here -- pure quantile loss is cleaner
```

**Top performer:** LightGBM with quantile loss (pinball loss 41 ms at q=0.99).

---

## Step 11: Train Multiple Quantiles for Coverage Guarantees

```python
# We need P10, P50, P90, P99 -- four separate models
# (one model per quantile; LightGBM trains them independently)

quantiles = [0.10, 0.50, 0.90, 0.99]
models = {}
for q in quantiles:
    model = lgb.LGBMRegressor(
        objective='quantile', alpha=q,
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train_log)
    models[q] = model

# Predict each quantile
predictions = {q: m.predict(X_val) for q, m in models.items()}

# CHECK: non-crossing quantiles
crossing_count = ((predictions[0.99] < predictions[0.90]) |
                   (predictions[0.90] < predictions[0.50]) |
                   (predictions[0.50] < predictions[0.10])).sum()
print(f"Crossing quantile rows: {crossing_count} / {len(X_val)}")
# Typically 1-3% cross
```

**Expert insight:** quantile crossings (P50 > P99 for example) are mathematically impossible but happen because each model is trained independently. Solutions:
- **Sort post-hoc:** trivially fix per-row by sorting predictions
- **Joint training:** use a multi-quantile loss that penalizes crossings
- **Monotonic constraints:** force the model to use feature monotonicity

For 1-3% crossings, post-hoc sorting is fine and simplest.

---

## Step 12: Conformal Prediction for Coverage Guarantees

```python
# QUANTILE LOSS doesn't guarantee empirical coverage matches the nominal level.
# A model trained on q=0.99 might actually achieve only 96% coverage on holdout.
# Conformal prediction adds a rigorous coverage guarantee.

from sklearn.linear_model import QuantileRegressor
from mapie.regression import MapieQuantileRegressor

# Use the trained LightGBM as the base; conformal calibration on a held-out set
mapie = MapieQuantileRegressor(estimator=models[0.99], method='quantile', alpha=0.01)
# alpha=0.01 means we want 99% coverage

# Calibrate on a held-out conformal set
mapie.calibrate(X_calib, y_calib_log)

# Now predict with calibrated intervals
y_pred, y_pis = mapie.predict(X_test, alpha=0.01)
# y_pred: point estimate
# y_pis: prediction intervals shape (n, 2, 1) for [lower, upper]

# Empirical coverage check on test set
in_interval = ((y_test_log >= y_pis[:, 0, 0]) & (y_test_log <= y_pis[:, 1, 0]))
empirical_coverage = in_interval.mean()
# Target: 0.99
# Without conformal: 0.962
# With conformal: 0.994  ✓
```

**Expert insight:** conformal prediction provides distribution-free coverage guarantees. It's the gold standard for "I claim P99 — and I mean it" predictions. Without conformal, your "P99 prediction" might empirically only cover 96% of cases, not 99%.

---

## Step 13: Hyperparameter Tuning

```python
import optuna

def objective(trial):
    params = {
        'objective': 'quantile',
        'alpha': 0.99,
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('cbt', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    # Use walk-forward CV (3-fold on training data)
    # Score: pinball loss at q=0.99
    return walk_forward_pinball(params, X_train, y_train_log, alpha=0.99)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=80)
# Best pinball loss: 36 ms
```

---

## Step 14: Per-Endpoint Calibration

```python
# Some endpoints (e.g., payment processing) have different latency distributions
# Their conformal calibration may differ
# Compute per-endpoint conformal scaling

per_endpoint_calib = {}
for endpoint_id in df_train['endpoint_id'].unique():
    mask = (X_calib['endpoint_id'] == endpoint_id)
    if mask.sum() < 100: continue
    # Compute the empirical conformal score for this endpoint
    score = conformal_score(y_calib_log[mask], predictions_99[mask])
    per_endpoint_calib[endpoint_id] = score

# At inference, apply per-endpoint conformal width adjustment
final_p99 = base_p99_prediction + per_endpoint_calib.get(endpoint_id, global_score)
```

---

## Step 15: Final Evaluation on Held-Out Test Set

```python
# Final model: LightGBM quantile regression at q={0.10, 0.50, 0.90, 0.99}
#              + conformal prediction (alpha=0.01)
#              + per-endpoint calibration scaling
#
# Test set: 10 days × 400 endpoints × 1-min windows = ~5.8M predictions
#
# Performance:
#   Pinball loss at q=0.99:        34 ms
#   Empirical coverage (P99):      0.991  ✓ (target 0.99)
#   Empirical coverage (P10-P90):  0.812
#   Mean prediction interval width: 280 ms
#   Per-endpoint coverage spread:  0.97 - 0.998 (some over-coverage on stable endpoints)
#
# Inference time per batch (400 endpoints): 12 seconds  ✓ (target 60s)
```

---

## Step 16: Operational Use

```python
# SRE alerting integration:
# Every 5 minutes:
#   1. Pull recent metrics for all 400 endpoints
#   2. Compute features (lag features, stress scores)
#   3. Predict P10/P50/P90/P99 for each endpoint
#   4. Compare predicted P99 vs SLO threshold
#   5. If P99 prediction > SLO, fire alert OR auto-scale

# SLO breach prediction matrix:
#   For each endpoint, the SLO is e.g. P99 < 500ms
#   When predicted P99 > SLO threshold, trigger:
#     1. Soft alert (slack notification)
#     2. Scale-up signal to autoscaler
#     3. If predicted P99 > 1500ms (3x SLO), wake up oncall

# Avoid alert fatigue: only alert on predictions that are NEW exceedances
# (not when we already alerted 5 min ago for the same condition)
```

---

## Step 17: Deployment

```python
# Inference path:
#   - 4 LightGBM models (one per quantile) -- load once at startup
#   - Conformal calibration parameters (per-endpoint dictionary)
#   - Feature engineering reusable in real-time
#
# Total model size: ~150MB
# Inference per endpoint: 30ms
# Total for 400 endpoints: 12 seconds (parallelizable to 4s on multi-core)

# Retraining cadence: WEEKLY
#   - Latency patterns shift with code deploys, dependency updates, traffic patterns
#   - Weekly retrain on rolling 60-day window
#   - Re-conformalize on most recent 7 days
#   - A/B test: compare new model coverage vs old over next 24h

# Monitoring:
#   - Daily empirical coverage (must stay 0.99 ± 0.01)
#   - Weekly pinball loss drift
#   - Per-endpoint coverage spread
#   - Alert if coverage drops below 0.97 anywhere -- triggers retrain
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Loss function | RMSE | Pinball loss at q=0.99 (directly optimizes the right quantity) |
| Single model | Predict the mean | Four models for P10/P50/P90/P99 |
| Coverage | "I trained at q=0.99 so it must be 99%" | Conformal prediction wraps the model with rigorous coverage guarantee |
| Quantile crossing | Ignore | Post-hoc sort or joint training |
| Train/test split | Random | Walk-forward (latency patterns drift weekly) |
| Outliers | Cap at 99th percentile | Keep — tail events ARE the signal |
| Stress features | "CPU > 70%" linear | Quadratic CPU pressure (small CPU below 70% is fine, above explodes) |
| Per-endpoint variation | One global model | Endpoint as categorical + per-endpoint conformal calibration |
| Latency target distribution | Use raw ms | log1p-transform target |
| Lag features | None | Lag 1/5/15/60min/24h/7d, rolling mean/max/std, growth rates |
| Validation metric | RMSE | Pinball loss + empirical coverage |
| Alert thresholds | Fixed | Predicted P99 > SLO -> auto-scale; P99 > 3x SLO -> wake oncall |
