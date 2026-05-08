# Expert Scenario 46: Delivery Time Prediction (ETA Regression)

> **Complexity:** Real-time inference at order placement (< 200ms), heavy-tailed delays (most orders on time, some catastrophically late), geospatial features, multi-stage pipeline (pickup + transit + drop-off), ETA must be a calibrated probability distribution not a point estimate, customer-facing accuracy directly impacts NPS.

---

## The Brief

A food-delivery platform completes 4M deliveries per week across 200 cities. Each order's ETA is computed at checkout and shown to the customer. The product team's research:

- 9 minutes late = 0% NPS impact (within tolerance)
- 15 minutes late = -0.3 NPS per affected order
- 25 minutes late = -1.0 NPS (customers actively complain and may churn)
- Conversely, "delivered earlier than promised" delights customers but at smaller magnitude

The goal: a model that predicts ETA accurately AND, equally importantly, **provides a confidence interval** so the UI can show "expected delivery: 25-35 min" rather than a misleading single number "30 min."

Constraints:
- Real-time at checkout: < 200ms inference, including feature pulls.
- Per-city models: each city has its own dynamics (traffic, restaurant density).
- 4M deliveries/week training data available; refresh nightly.
- The output should support a decision: should the UI show "fast" badge? Auto-cancel orders that won't make it? Re-route to a different driver?

This is harder than standard regression because: customers see and react to predictions in real-time, the cost of overpromising is much higher than under-promising, and a single number isn't enough — uncertainty must be surfaced.

---

## Step 1: Define the Problem Type

```
Type:           Regression on minutes (5 - 90 typical, tail to 180+)
Primary Metric: MAPE on point estimate; Pinball loss at q=0.90 for the upper bound
Secondary:      Coverage of P50-P95 prediction intervals
                Per-city MAPE distribution
                NPS-weighted error: heavily penalize underestimates
Business Goal:  MAPE < 12%; P95 prediction interval covers actual at 95%; under-promise rate ≤ 8%
Constraint:     < 200ms inference; per-city granular models possible
Target shape:   Right-skewed (most orders on time, occasional very-late)
```

**Expert thinking:** the asymmetric cost (overpromise > underpromise) means we shouldn't optimize for symmetric MAE. The right framing is: predict the median (P50) for the displayed ETA, but also predict P90 so the UI can show a buffer. Under-promising gives a small NPS lift; over-promising blows it up.

---

## Step 2: Understand the Data

```
Shape: ~200M historical deliveries across 200 cities × ~3 years

Per delivery, features at order-placement time:

ORDER:
- order_id (unique)
- placed_at (timestamp)
- estimated_food_prep_min (restaurant-reported, 5-45)
- num_items
- order_value_usd
- has_alcohol (binary)
- has_special_instructions (binary)

RESTAURANT (15):
- restaurant_id
- restaurant_lat, restaurant_lng
- restaurant_avg_prep_min_30d (rolling)
- restaurant_avg_prep_min_dow (day-of-week specific)
- restaurant_consistency_score_30d (1 - std/mean of prep times)
- restaurant_open_orders_currently (queue depth)
- restaurant_max_capacity_estimate
- num_active_drivers_within_2mi
- num_active_drivers_within_5mi
- restaurant_avg_handoff_min (driver wait at restaurant)
- restaurant_busy_now (binary, P95+ vs typical at this hour)
- ... 4 more

CUSTOMER / GEO:
- customer_id (anonymized)
- delivery_lat, delivery_lng
- haversine_distance_miles  -- straight-line distance
- haversine / 30 * 60 = naive_minutes  -- naive estimate
- street_distance_miles  -- from routing API (NaN at 12% of cells)
- elevation_change_meters
- delivery_address_type (House/Apartment/Office/Other)
- ttl_to_apartment_estimate  (extra time for stairs/concierge)

TRAFFIC / TIME (12):
- hour_of_day
- day_of_week
- is_weekend
- is_lunch_rush  (11:30-13:30 local)
- is_dinner_rush (17:30-20:00 local)
- traffic_index_recent_30min (third-party, 1.0=normal, 1.5=heavy, 2.0=gridlock)
- weather_temp_c
- weather_precip_mm
- weather_severe_event (binary)
- recent_avg_eta_in_grid_30min  -- key feature: what just happened in this area?
- recent_avg_actual_in_grid_30min  -- what's the ACTUAL latency lately?
- recent_late_rate_in_grid_30min  -- proportion of orders late in last 30min in this 1-mile grid

CITY:
- city_id (200 unique)
- city_density_score  (1-5)

TARGET:
- actual_delivery_minutes  (placed → delivered)
```

**Expert thinking:** the most predictive features by far:
1. `recent_avg_actual_in_grid_30min` — what just happened in this exact area is the strongest predictor.
2. `restaurant_open_orders_currently` — busy restaurant = slower prep.
3. `traffic_index_recent_30min` — traffic now predicts traffic in 25 min reasonably well.
4. `restaurant_avg_prep_min_dow` — Tuesday lunch is structurally faster than Friday dinner.

Distance is necessary but not the dominant signal — most ETA variance comes from prep time and traffic, not raw distance.

---

## Step 3: EDA

```python
df['actual_delivery_minutes'].describe()
# count    200,000,000
# mean    32.4
# std     12.8
# min      8
# 25%     24
# 50%     30
# 75%     38
# 95%     56
# 99%     82
# max    320 (extreme outlier)

# Skewness: 1.4 (right-skewed)

# Distribution per city density
df.groupby('city_density_score')['actual_delivery_minutes'].describe()
# Density 1 (rural/spread): mean 38, std 14, P95 64
# Density 5 (downtown):     mean 28, std 10, P95 48
# Cities with high density have shorter ETAs but tighter distributions

# Late rate by hour
df['is_late'] = (df['actual_delivery_minutes'] > df['promised_eta'] + 9)
df.groupby('hour_of_day')['is_late'].mean()
# 11:00-12:30: 18% late (lunch rush spillover)
# 18:00-19:30: 24% late (dinner rush)
# 22:00-02:00: 8% late (low volume, fast)
# 03:00-10:00: 4% late
```

---

## Step 4: Data Cleaning

```python
# === HANDLE TARGET TAIL ===
# Cap at P99.5 for training stability; full distribution preserved at inference
df['actual_log'] = np.log1p(df['actual_delivery_minutes'])

# === MISSING ROUTING DATA ===
df['has_street_distance'] = df['street_distance_miles'].notna().astype(int)
df['street_distance_miles'] = df['street_distance_miles'].fillna(df['haversine_distance_miles'] * 1.4)
# ~1.4x is typical street/haversine ratio; falls back gracefully

# === NIGHT-TIME OUTLIERS ===
# 3am orders sometimes wait 2 hours due to no drivers
# These are technically real but distort training; cap at P99.9 (320min)
df = df[df['actual_delivery_minutes'] <= 200]

# === FILTER ORDERS WITH MISSING FEATURES ===
df = df.dropna(subset=['restaurant_avg_prep_min_30d', 'haversine_distance_miles'])
```

---

## Step 5: Feature Engineering — Real-Time Aware

```python
# === DISTANCE FEATURES ===
df['log_haversine'] = np.log1p(df['haversine_distance_miles'])
df['street_to_haversine_ratio'] = df['street_distance_miles'] / (df['haversine_distance_miles'] + 0.1)

# === RESTAURANT BUSYNESS ===
df['restaurant_load_factor'] = df['restaurant_open_orders_currently'] / (df['restaurant_max_capacity_estimate'] + 1)
df['restaurant_load_high'] = (df['restaurant_load_factor'] > 0.7).astype(int)

# === DRIVER AVAILABILITY ===
df['log_drivers_within_2mi'] = np.log1p(df['num_active_drivers_within_2mi'])
df['driver_scarcity'] = (df['num_active_drivers_within_2mi'] < 3).astype(int)

# === GRID-LEVEL RECENT EXPERIENCE (CRITICAL) ===
# These features capture "what just happened in this area"
# They are computed at order time using the previous 30 minutes of completed deliveries
# in a 1-mile geographic grid

df['eta_in_grid_30min'] = df['recent_avg_actual_in_grid_30min']
df['eta_in_grid_volatility'] = df['recent_avg_actual_in_grid_30min'] - df['recent_avg_eta_in_grid_30min']
df['grid_running_late'] = (df['eta_in_grid_volatility'] > 5).astype(int)

# === COMPOSITE STRESS SCORE ===
df['delivery_stress'] = (
    df['traffic_index_recent_30min'] +
    df['restaurant_load_factor'] * 2 +
    df['weather_severe_event'].astype(int) * 0.5 +
    df['driver_scarcity']
)

# === CYCLICAL TIME ENCODING ===
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
df['minute_of_hour_sin'] = np.sin(2 * np.pi * df['minute_of_hour'] / 60)
df['minute_of_hour_cos'] = np.cos(2 * np.pi * df['minute_of_hour'] / 60)

# === RESTAURANT × CITY INTERACTIONS ===
# Some restaurant chains have very different prep times in different cities
# Use restaurant_id × city_id interaction (will be huge cardinality; target encode)
df['rest_city_avg_prep'] = df.groupby(['restaurant_id', 'city_id'])['actual_delivery_minutes'].transform('mean')
```

**Expert insight:** `recent_avg_actual_in_grid_30min` is the killer feature. It's "what happened to the last 50 deliveries in this area" — a strong real-time signal. If your area is currently slow, your delivery will be slow. This kind of streaming feature is hard to compute at inference time; requires a real-time aggregate service.

---

## Step 6: Per-City Models vs Global Model

```python
# Question: 200 cities; train one global model or one model per city?
# Trade-off:
#   Global model: more data per training, less specialized per city
#   Per-city: specialized but small per-city sample sizes

# Hybrid approach (best of both):
#   1. Train a global LightGBM on all data
#   2. For each major city (~30 cities with > 100K orders), fine-tune with city-specific data
#   3. For minor cities (~170), use the global model

# Per-city CV results:
# Global model MAPE: 11.8%
# Per-city models MAPE: 11.4% (top 30 cities); 12.5% (smaller cities — overfit)
# Hybrid: MAPE 11.2% (global on small, fine-tuned on large)
```

---

## Step 7: Train Multiple Models

```python
import lightgbm as lgb

# Setup: walk-forward CV across 90-day blocks (5 folds)

# === Model 1: Linear Regression (baseline) ===
# MAPE: 16.4%

# === Model 2: Gradient Boosting (sklearn) ===
# MAPE: 12.6%, training takes 4hrs

# === Model 3: LightGBM with regression objective ===
lgb_reg = lgb.LGBMRegressor(
    n_estimators=1000,
    max_depth=8,
    num_leaves=63,
    learning_rate=0.05,
    objective='regression',
    random_state=42
)
# CV MAPE: 11.8%, training: 18min

# === Model 4: LightGBM with HUBER loss (asymmetric) ===
lgb_huber = lgb.LGBMRegressor(
    objective='huber',
    alpha=0.9,  # quantile-like
    n_estimators=1000,
    ...
)
# CV MAPE: 11.6% — slightly more robust to outliers

# === Model 5: LightGBM with Tweedie (right-skewed target) ===
lgb_tweedie = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.5,
    ...
)
# CV MAPE: 11.4%

# === Model 6: Quantile regression at q=0.5 + q=0.9 ===
# For UI: show median ETA + 90th-percentile upper bound
lgb_q50 = lgb.LGBMRegressor(objective='quantile', alpha=0.5, ...)
lgb_q90 = lgb.LGBMRegressor(objective='quantile', alpha=0.9, ...)
```

**Top performer:** LightGBM Tweedie + per-city fine-tuning (MAPE 11.2%).

---

## Step 8: Asymmetric Loss for Under-Promise Bias

```python
# Custom loss: penalize underestimates more than overestimates
# This pushes the model to predict slightly higher (safer for customer expectations)

def asymmetric_huber(y_true, y_pred):
    error = y_pred - y_true
    # If positive (overpredicted ETA), small loss
    # If negative (underpredicted, i.e. delivery LATER than predicted), large loss
    loss = np.where(error >= 0, 0.5 * error**2, 2.0 * error**2)
    return loss

# Implement as LightGBM custom objective:
def asymmetric_objective(y_true, y_pred):
    error = y_pred - y_true
    grad = np.where(error >= 0, error, 4 * error)
    hess = np.where(error >= 0, 1.0, 4.0)
    return grad, hess

lgb_asym = lgb.LGBMRegressor(objective=asymmetric_objective, ...)
# CV MAPE: 11.6% (slightly worse than symmetric)
# But: under-promise rate drops from 8.4% to 6.1%
# Trade-off: slightly worse mean error, much fewer NPS-hit late deliveries
# Decision: use asymmetric for production
```

---

## Step 9: Conformal Prediction Intervals

```python
# Single-quantile training doesn't guarantee empirical coverage matches nominal
# Use conformal prediction for rigorous coverage

from mapie.regression import MapieQuantileRegressor

# Pipeline: train quantile models, then conformalize
mapie = MapieQuantileRegressor(estimator=lgb_q90, method='quantile', alpha=0.05)
mapie.calibrate(X_calib, y_calib)

y_pred_conformal, intervals = mapie.predict(X_test)
# intervals shape: (n, 2, 1) for [lower, upper]

# Empirical coverage of P5-P95 intervals:
in_interval = ((y_test >= intervals[:, 0, 0]) & (y_test <= intervals[:, 1, 0]))
print(f"P5-P95 empirical coverage: {in_interval.mean():.3f}")
# Without conformal: 0.872
# With conformal: 0.952

# UI displays: "Estimated 28-38 minutes" (P5 = 28, P95 = 38)
```

---

## Step 10: Final Evaluation

```python
# Final model: LightGBM Tweedie + asymmetric loss + per-city fine-tune + conformal P95
#
# Test set: most recent 4 weeks, 16M deliveries
#
# Performance:
#   MAPE on point estimate (P50):     11.4%   ✓ (target 12%)
#   Under-promise rate:                6.1%   ✓ (target 8%)
#   Mean prediction interval width:   ~10 minutes
#   P5-P95 empirical coverage:        0.952   ✓
#
# Per-city distribution:
#   Best 10 cities: MAPE 9.4%
#   Worst 10 cities: MAPE 14.2%
#   90th percentile city: MAPE 12.8%
#
# Inference latency: 110ms p99 (LightGBM + conformal lookup)
```

---

## Step 11: Per-Order Explainability (for ops dashboards)

```python
# SHAP per delivery, especially when the prediction differs from the naive (haversine/30 * 60)

# Order #4821: predicted 38 min (naive would say 16 min for 8 miles)
# Top 5 SHAP drivers (positive = adds time):
#   restaurant_load_factor = 0.85          (+8.2 min)  [restaurant is busy]
#   recent_avg_actual_in_grid = 35         (+5.4 min)  [area is running slow]
#   traffic_index_recent = 1.6             (+4.1 min)  [traffic is heavy]
#   driver_scarcity = 1                    (+2.8 min)  [few drivers nearby]
#   has_alcohol = 1                        (+0.6 min)  [verification adds time]
# Naive base 16 min + drivers = ~37 min ≈ predicted 38 min

# Used by:
# - SRE dashboards: identify why orders are slow in a region
# - Ops leads: see when restaurant load drives many late deliveries
# - Auto-action triggers: high load + high traffic + scarcity → recommend re-routing or partial refund
```

---

## Step 12: Deployment

```python
# Production architecture:
#   1. Real-time feature service: pulls grid stats, restaurant load, driver availability
#   2. Batch features: city-level patterns, restaurant historical (refreshed nightly)
#   3. LightGBM serving: 110ms p99
#   4. Conformal calibration parameters: per-city dictionary
#   5. UI shows: "Estimated 25-35 minutes" (P5 - P95 interval)
#
# Retraining: NIGHTLY
#   - Past 7 days of data appended; model retrained
#   - New model A/B tested on 10% of orders for 24h
#   - Fall back to incumbent if new model MAPE > +0.005
#
# Monitoring (per city, per hour):
#   - Daily MAPE on completed deliveries
#   - P95 coverage drift (alert if drops below 0.92)
#   - Under-promise rate by city
#   - Restaurant-load feature staleness (alert if any feature service > 60s lag)

# Special handling:
#   - Severe weather event detected → switch to "weather-aware" model variant
#   - Major holiday → use special holiday-aware model
#   - New restaurant: cold-start with city-typical prep time
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Loss function | RMSE / MSE | Asymmetric loss (underestimates penalized 4x) |
| Output | Single ETA | P50 (display) + P95 (interval upper); shown as "25-35 min" |
| Coverage | "trained at q=0.95 so coverage is 95%" | Conformal prediction for rigorous coverage |
| Distance feature | Just haversine | Haversine + street routing + ratio + city density |
| Feature freshness | Static features | Real-time grid features (recent_avg_actual_in_grid_30min) |
| Per-city handling | One global model | Hybrid: global + city-specific fine-tunes for top 30 cities |
| Train/test split | Random | Time-based with multi-week holdout |
| Outliers | Cap at 99% | Keep the tail (it's the signal); remove only 200+ min impossibles |
| Restaurant load | Mean prep time only | Restaurant load factor + queue depth + driver availability |
| Cold-start | "Need history" | City-typical prep + estimate from haversine |
| Bias for late vs early | Symmetric | Penalize underestimates 4x; biases toward UNDER-promise |
| Monitoring | None | MAPE per city per hour; coverage drift; feature freshness |
