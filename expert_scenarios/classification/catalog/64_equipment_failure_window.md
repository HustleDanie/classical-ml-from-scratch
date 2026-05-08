# Expert Scenario 64: Equipment Failure Window Prediction (Time-Series Event Classification)

> **Complexity:** Sliding-window features over high-frequency sensor streams, severe imbalance (failures rare), prediction horizon (1-24 hours ahead) is the deliverable, label noise from operator-recovered failures, asset-level CV mandatory, the cost of missing a failure is order-of-magnitude higher than a false alarm.

---

## The Brief

A heavy-equipment fleet operator (mining trucks, compressors, generators) gives you 18 months of high-frequency sensor data from 2,400 assets. Each asset emits ~50 sensor channels at 1Hz (vibration, temperature, pressure, current, oil quality, etc.). Maintenance records mark each "failure event" with a timestamp.

The brief asks for a predictive maintenance model:

- For each asset at each hour, predict the probability of a failure event in the **next 24 hours**.
- A failure costs $40,000 (downtime + emergency repair + production loss).
- A false alarm costs $1,200 (preventive inspection that finds nothing).
- Cost ratio: 33:1 — strongly favor catching failures.
- Sliding scoring: every hour, re-score every asset's next-24h window.
- The maintenance crew has finite capacity: at most 50 inspections per day across all 2,400 assets.

This is harder than typical anomaly detection because: most "failures" have warning signs starting 4-24 hours before; sliding-window labels create temporal leakage if you're not careful; some failures are operator-recovered (small fault, recovered without log); and the maintenance budget creates a top-k constraint.

---

## Step 1: Define the Problem Type

```
Type:           Binary classification on (asset, hour) pairs;
                target = "failure occurred within next 24 hours"
Primary Metric: Recall at fixed daily inspection budget (top-50 per day across fleet)
Secondary:      PR-AUC; Per-asset recall (some assets shouldn't always be flagged)
                Lead time distribution (how many hours BEFORE the actual event?)
Business Goal:  Catch >= 65% of failures with daily inspection budget = 50;
                lead time mean >= 6 hours (enough to schedule maintenance)
Constraint:     Inference at hourly cadence on 2,400 assets ≈ 240 predictions/min
Imbalance:      ~0.1% of (asset-hour) pairs are pre-failure
                ~5% of asset-hours are within the 24h pre-failure window
```

**Expert thinking:** the right framing is NOT "predict if a failure happens at hour t." It's "predict if a failure happens in the [t, t+24) window." This makes the problem much more tractable — most failures have a 6-24 hour warning signature in sensor patterns.

The metric is "recall at top-50 inspections per day" because that's the actual operational constraint. F1 is the wrong metric — it ignores the budget.

---

## Step 2: Understand the Data

```
Shape: 2,400 assets × ~13,000 hours each ≈ 31M asset-hour records

Per asset-hour:
- asset_id (2,400 unique)
- hour_timestamp
- 50 sensor channels (1-second aggregates over the hour: mean, max, std, range)
  Total raw features: 50 × 4 stats = 200 features

Asset metadata:
- asset_type (8 types: truck-A / truck-B / compressor / generator / ...)
- asset_age_years
- manufacturer
- last_maintenance_hours_ago
- operating_condition (Normal / Heavy / Extreme)
- shift_pattern (24-7 / 16-8 / 12-12)

Failure log (separate table):
- asset_id
- failure_timestamp
- failure_type (8 categories: bearing / motor / hydraulic / cooling / electrical / ...)
- severity (Minor / Major / Critical)
- root_cause (post-mortem)
- prior_warnings_logged (binary, was operator already flagging it?)

Aggregate failure stats:
- 1,847 failures across 2,400 assets over 18 months
- Mean time between failures: ~280 days per asset
- Most failures are concentrated in 130 "problem" assets that account for 60% of failures
```

**Expert thinking:** the 130 problem assets create a model that learns "this specific asset is likely to fail" — fine if those assets keep being problems, but breaks down when they're replaced or repaired. The model needs both asset-specific signals AND condition-specific signals.

---

## Step 3: EDA

```python
# Failure rate per asset-hour (sliding 24h target)
df['will_fail_24h'] = compute_sliding_target(df, horizon_hours=24)
df['will_fail_24h'].mean()  # 0.045 (4.5% of asset-hours are within a 24h pre-failure window)

# Lead time distribution (hours from sensor anomaly to actual failure)
# Compute by tracking sensor pattern changes
# Most failures show first sensor anomaly 4-18 hours before event
# Distribution mean: 8.2 hours; median: 6 hours

# Pre-failure sensor patterns (averaged across 1,847 failures)
# Vibration RMS rises ~3-12 hours before mechanical failures
# Bearing temperature rises 6-18 hours before bearing failures
# Oil quality degrades over days for chronic failures
# Some failures show NO sensor warning (electrical fault, sudden cracks) -- ~22% of failures

# Pre-failure window features (24h before failure):
# - vibration_rms.std × 2.4 vs normal
# - temperature_max × 1.6 vs normal
# - current_draw_max × 1.3 vs normal
# These are the model's signal
```

**Findings:**

| Finding | Implication |
|---------|------------|
| 4.5% of asset-hours are pre-failure | Model needs to find the ~5% in 31M rows; severe imbalance |
| 22% of failures have NO sensor warning | Maximum achievable recall ~78% — be honest with stakeholders |
| Lead time median 6h | Predicting 24h ahead is feasible; predicting 1h ahead is harder |
| 130 "problem" assets generate 60% of failures | Model risks anchoring on asset_id — needs careful handling |
| Sensor patterns vary by asset_type | Per-asset-type models or strong asset_type features |

---

## Step 4: Data Cleaning

```python
# === LABEL CONSTRUCTION (THE MOST CRITICAL STEP) ===
# For each asset-hour, set will_fail_24h = 1 if a failure happens within next 24h
# DANGER: don't include the failure-hour itself in the "pre-failure" window

def build_target(asset_records, horizon_hours=24):
    asset_records = asset_records.sort_values('timestamp')
    asset_records['will_fail_24h'] = 0
    for _, fail in failures[failures['asset_id'] == asset_records['asset_id'].iloc[0]].iterrows():
        # Mark the 24h preceding the failure as positive
        window_start = fail['failure_timestamp'] - pd.Timedelta(hours=horizon_hours)
        window_end = fail['failure_timestamp']
        mask = (asset_records['timestamp'] >= window_start) & (asset_records['timestamp'] < window_end)
        asset_records.loc[mask, 'will_fail_24h'] = 1
    return asset_records

# === FILTER POST-FAILURE HOURS ===
# After a failure, the asset is in repair; sensor data is artifact
# Drop the 48 hours after each failure
df = df[~is_within_post_failure_window(df, hours=48)]

# === HANDLE OPERATOR-RECOVERED FAILURES ===
# Some "near misses" -- operator notices warning signs and intervenes -- aren't logged
# These are noise in the negative class (model trained on them as negative learns wrong patterns)
# Heuristic: if a 24h pre-failure window pattern matches an operator log of "manual intervention",
# mark it as ambiguous (drop from training)
```

**Expert insight:** label construction is THE place small bugs cause big problems. A common mistake: marking the failure-hour itself as positive. This creates leakage — sensor patterns at the moment of failure are obviously different. The model learns the failure signature itself rather than predicting it.

---

## Step 5: Feature Engineering — Sliding Windows

```python
# === LAG / WINDOW FEATURES ===
# For each sensor, compute statistics over multiple windows
df = df.sort_values(['asset_id', 'hour_timestamp'])
g = df.groupby('asset_id')

# Different time windows capture different dynamics
windows = {
    '1h':   1,
    '6h':   6,
    '24h':  24,
    '7d':   168,
    '30d':  720
}

# For each sensor and each window, compute mean, max, std
for sensor in sensor_channels:
    for w_name, w_hours in windows.items():
        df[f'{sensor}_mean_{w_name}'] = g[sensor].rolling(w_hours, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f'{sensor}_max_{w_name}'] = g[sensor].rolling(w_hours, min_periods=1).max().reset_index(level=0, drop=True)
        df[f'{sensor}_std_{w_name}'] = g[sensor].rolling(w_hours, min_periods=1).std().reset_index(level=0, drop=True)

# === DEVIATION FROM PERSONAL BASELINE ===
# Each asset has its own normal operating range
# A vibration of 2.4 mm/s is normal for one truck, alarming for another
for sensor in critical_sensors:
    asset_baseline = g[sensor].rolling(720, min_periods=72).mean().reset_index(level=0, drop=True)
    df[f'{sensor}_deviation_from_baseline'] = df[sensor] - asset_baseline
    df[f'{sensor}_relative_deviation'] = df[f'{sensor}_deviation_from_baseline'] / (asset_baseline.abs() + 0.1)

# === RATE OF CHANGE ===
for sensor in critical_sensors:
    df[f'{sensor}_change_6h'] = g[sensor].diff(6)
    df[f'{sensor}_change_24h'] = g[sensor].diff(24)

# === DOMAIN STRESS COMPOSITE ===
df['stress_score'] = (
    (df['vibration_rms_mean_1h'] > df['vibration_rms_mean_30d'] * 1.5).astype(int) +
    (df['temperature_max_1h'] > df['temperature_max_30d'] * 1.2).astype(int) +
    (df['current_draw_max_1h'] > df['current_draw_max_30d'] * 1.15).astype(int) +
    (df['oil_quality_index'] < 0.6).astype(int)
)

# === MAINTENANCE-RELATED FEATURES ===
df['hours_since_last_maintenance'] = (df['hour_timestamp'] - df['last_maintenance_hours_ago']).dt.total_seconds() / 3600
df['maintenance_overdue'] = (df['hours_since_last_maintenance'] > df['recommended_maintenance_interval']).astype(int)
```

**Expert insight:** "deviation from personal baseline" is the killer feature for predictive maintenance. Each asset has its own normal — a vibration of 2.4 mm/s might be excellent for one bulldozer and concerning for another. The model needs to learn relative deviations, not absolute thresholds.

---

## Step 6: Feature Selection

```python
# After windows + deviations + composites: ~600 features
# Use mutual information to keep top 200-300 most relevant

from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X_train.sample(100000), y_train.sample(100000), random_state=42)

# Top 20 features:
#   vibration_rms_mean_24h_relative_deviation        0.062
#   bearing_temperature_max_6h                       0.054
#   stress_score                                     0.048
#   oil_quality_index                                0.041
#   vibration_rms_change_24h                         0.038
#   maintenance_overdue                              0.035
#   ... 14 more

# Drop features with MI < 0.001
# Final: 280 features
```

---

## Step 7: Train/Test Split — Asset-Based and Time-Based

```python
# Two splits required, applied jointly:
#   1. ASSET-BASED: don't put the same asset in both train and test
#      (Otherwise the model learns asset-specific quirks rather than generalizable patterns)
#   2. TIME-BASED: train on earlier data, test on later
#      (Matches deployment reality)

# Hybrid: split assets 70/30, AND split time within those splits 70/30
asset_train_ids, asset_test_ids = train_test_split(asset_list, test_size=0.30, random_state=42)
df_train = df[df['asset_id'].isin(asset_train_ids) & (df['timestamp'] < cutoff)]
df_test = df[df['asset_id'].isin(asset_test_ids) & (df['timestamp'] >= cutoff)]

# Result: model is tested on UNSEEN ASSETS over UNSEEN TIME
# This is the strict generalization test
```

**Expert insight:** without asset-level split, the model can over-fit to the 130 "problem" assets. Random split puts those assets in train AND test; the model learns "asset X is problematic" rather than "this sensor pattern means failure."

---

## Step 8: Try Multiple Models

```python
import lightgbm as lgb

# === Model 1: Logistic Regression with L2 ===
lr = LogisticRegression(C=0.1, penalty='l2', class_weight='balanced', max_iter=500)
# Recall at top-50 budget: 0.42

# === Model 2: Random Forest ===
rf = RandomForestClassifier(n_estimators=500, max_depth=12, class_weight='balanced', n_jobs=-1)
# Recall at top-50: 0.58

# === Model 3: LightGBM ===
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=1000, max_depth=6, num_leaves=63,
    learning_rate=0.05,
    is_unbalance=True,  # handles imbalance
    random_state=42
)
# Recall at top-50: 0.67  ✓ (target 0.65)

# === Model 4: XGBoost ===
xgb = XGBClassifier(scale_pos_weight=20, n_estimators=1000, ...)
# Recall at top-50: 0.66

# === Model 5: Per-Asset-Type Ensemble ===
# Different asset types (trucks, compressors) have different failure modes
# Train 8 separate models, one per asset_type
type_models = {atype: lgb.LGBMClassifier(...).fit(X_train_atype, y_train_atype) for atype in asset_types}

# Recall at top-50: 0.71  <- BEST
```

**Top performer:** Per-asset-type LightGBM ensemble (recall 0.71 at top-50 daily inspections).

---

## Step 9: Imbalance Handling

```python
# is_unbalance=True in LightGBM works well for moderate imbalance
# For more aggressive balancing:
#   scale_pos_weight = (1 - 0.045) / 0.045 ≈ 21
# Tested: roughly equivalent to is_unbalance=True

# SMOTE on time-series sliding windows is INAPPROPRIATE
# Synthetic samples don't preserve temporal continuity within an asset
# Stick with class_weight / scale_pos_weight

# Alternative: train on NEGATIVES from a tighter window
# (e.g., negative_samples = asset-hours where failure happens 48-72h later)
# This trains the model on "early warning" vs "imminent" -- different framing
# Result: higher recall but more lead-time drift
```

---

## Step 10: Threshold = Top-K Daily

```python
# Daily inspection budget: 50
# At each hour, generate predictions for all 2,400 assets
# Aggregate to daily by taking max(probability) per asset per day
# Sort assets by max-probability; top 50 get inspections

def top_k_per_day(predictions_df, k=50):
    daily = predictions_df.groupby(['asset_id', predictions_df['timestamp'].dt.date])['prob'].max().reset_index()
    daily['rank'] = daily.groupby(daily['timestamp'].dt.date)['prob'].rank(ascending=False)
    daily['inspect'] = (daily['rank'] <= k).astype(int)
    return daily

# Recall at top-50: 0.71
# Average lead time: 7.4 hours (target >= 6) ✓
```

---

## Step 11: Lead Time Distribution Analysis

```python
# When the model flags an asset as "will fail in 24h" how many hours BEFORE the actual event was that?
# Maintenance crews need lead time to schedule

# Compute: for each detected failure, the earliest hour of high-prob flag
detected_failures = []
for failure in failures_in_test:
    flag_hours = predictions[
        (predictions['asset_id'] == failure['asset_id']) &
        (predictions['timestamp'] >= failure['timestamp'] - pd.Timedelta(hours=24)) &
        (predictions['timestamp'] < failure['timestamp']) &
        (predictions['inspected'])
    ]
    if len(flag_hours) > 0:
        lead_time = (failure['timestamp'] - flag_hours['timestamp'].min()).total_seconds() / 3600
        detected_failures.append(lead_time)

# Distribution:
# Mean lead time: 7.4 hours
# Median: 6 hours
# 25th percentile: 4 hours
# 75th percentile: 12 hours
# 22% of failures: 0 lead time (no warning, missed)
```

---

## Step 12: Type-Specific Recall

```python
# Different failure types have different predictability
# Bearing:  0.83 recall (clear signature)
# Motor:    0.72 recall (medium signature)
# Hydraulic: 0.68 recall
# Cooling:  0.71 recall
# Electrical: 0.45 recall (very poor — sudden faults)
# Mechanical fracture: 0.31 recall (even sudden)
# ... 

# Realistic: the 22% of failures with no warning are mostly electrical / sudden fracture
# Mitigation: pair model with physical inspection schedule to catch the sudden ones
```

---

## Step 13: Final Evaluation

```python
# Final model: Per-asset-type LightGBM ensemble + top-50 daily threshold
# Test set: 720 unseen assets × 6 months = ~3.1M asset-hours
#
# Performance:
#   Recall (overall) at top-50:        0.71  ✓ (target 0.65)
#   Mean lead time:                    7.4 hours  ✓ (target 6)
#   Mean lead time of caught failures: 8.6 hours
#   PR-AUC:                             0.42
#   Cost saved (avg per month):        ~$1.8M (vs no model)
#
# Per-failure-type recall:
#   Bearing:    0.83
#   Motor:      0.72
#   Hydraulic:  0.68
#   Cooling:    0.71
#   Electrical: 0.45
#   Other:      0.51
#
# Inference: 240 predictions/min easily within budget
```

---

## Step 14: Per-Asset Explainability

```python
# When a maintenance crew gets an inspection alert, they need to know WHY

# Asset-12345 flagged with probability 0.78 for inspection in next 24h
# Top SHAP drivers:
#   1. bearing_temp_max_6h_relative = 1.4    (+0.18 risk)
#   2. vibration_rms_change_24h = +6.2        (+0.14)
#   3. oil_quality_index = 0.51               (+0.11)
#   4. stress_score = 4                       (+0.10)
#   5. maintenance_overdue = 1                (+0.08)
#
# Maintenance dispatcher message:
# "Bearing temperature trending up over past 6 hours; vibration shows 24-hour rise;
#  oil quality below threshold; asset is overdue for maintenance.
#  Likely failure mode: bearing wear. Inspect within 8 hours."
```

---

## Step 15: Deployment

```python
# Production:
#   1. Hourly batch: pull last 30 days of sensor data per asset
#   2. Compute sliding window features
#   3. Per-asset-type LightGBM predictions
#   4. Aggregate to daily; rank top-50 across fleet
#   5. Push alerts to maintenance dispatcher

# Retraining: MONTHLY
#   - Add new failures + corrected labels (operator-recovered events)
#   - Retrain per-asset-type models
#   - Re-fit top-K threshold based on inspection capacity
#   - Validate on held-out fleet segment

# Monitoring:
#   - Daily recall on confirmed failures (rolling 30 days)
#   - Per-failure-type recall drift
#   - Lead time distribution drift
#   - False alarm rate (true positive rate among inspected)
#   - Model staleness alarm if prediction distribution drifts significantly

# Special handling:
#   - New asset (no history): use asset-type baseline patterns; expand for 30 days
#   - Replaced asset: fresh start; clear historical baselines
#   - Unusual maintenance event: flag for re-baselining
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Target framing | "Predict failure now" | "Predict failure in next 24h window" |
| Train/test split | Random | Asset-based + time-based |
| Lead time analysis | Ignore | Track lead time distribution; aim for mean ≥ 6h |
| Sliding windows | Use raw values | Compute mean/max/std at 1h/6h/24h/7d/30d windows |
| Personal baselines | Use absolute thresholds | Per-asset rolling baselines + relative deviations |
| Imbalance | SMOTE | scale_pos_weight; SMOTE inappropriate for sliding windows |
| Threshold | F1 optimum | Top-50 daily inspection budget (operational constraint) |
| Asset diversity | One global model | Per-asset-type ensemble (8 models) |
| Failure types | Treat all the same | Per-failure-type recall analysis; honest about electrical/sudden |
| Operator-recovered failures | Treat as negatives | Label as ambiguous; drop from training |
| Explanation | "model said inspect" | Top SHAP drivers + likely failure mode + recommended urgency |
| Deployment | Direct alert | Maintenance budget enforced; per-asset-type inference |
